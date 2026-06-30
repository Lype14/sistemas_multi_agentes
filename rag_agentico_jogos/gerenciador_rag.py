import os
import asyncio
import chromadb
from playwright.async_api import async_playwright
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder


CAMINHO_BANCO = os.path.join(os.getcwd(), "vdb_data")
chroma_client = chromadb.PersistentClient(path=CAMINHO_BANCO)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="gemma2", temperature=0.2)


print("Inicializando modelo de Reranking...")
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def criar_base_jogo(nome_do_jogo: str):
    nome_limpo = nome_do_jogo.lower().replace(" ", "-")
    chroma_client.get_or_create_collection(name=nome_limpo)
    print(f"Base vetorial para '{nome_limpo}' criada ou já existente.")
    return nome_limpo


def listar_bases():
    return [col.name for col in chroma_client.list_collections()]

async def _salvar_url_como_pdf(url: str, caminho_pdf: str):
    print(f"Navegador acessando a URL: {url}...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=45000)
        await page.pdf(path=caminho_pdf)
        await browser.close()
    print(f"Página convertida em PDF com sucesso em: {caminho_pdf}")

def anexar_link_a_base(url: str, nome_do_jogo: str):
    nome_limpo = nome_do_jogo.lower().replace(" ", "-")
    caminho_pdf_temporario = os.path.join(os.getcwd(), f"temp_{nome_limpo}.pdf")
    
    try:
        asyncio.run(_salvar_url_como_pdf(url, caminho_pdf_temporario))
        
        print("Iniciando etapa de PARSER do arquivo PDF...")
        loader = PyPDFLoader(caminho_pdf_temporario) if 'PyPDFLoader' in globals() else None
        if not loader:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(caminho_pdf_temporario)
            
        documentos_brutos = loader.load()
        
        print("Iniciando etapa de chunking semântico...")
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        blocos_de_texto = text_splitter.split_documents(documentos_brutos)
        print(f"Texto quebrado em {len(blocos_de_texto)} blocos semânticos.")
        
        for bloco in blocos_de_texto:
            bloco.metadata["fonte_url"] = url
            bloco.metadata["jogo"] = nome_limpo

        print(f"Armazenando os blocos na base do jogo '{nome_limpo}'...")
        db = Chroma(client=chroma_client, collection_name=nome_limpo, embedding_function=embeddings)
        db.add_documents(blocos_de_texto)
        print(f"Ingestão concluída com sucesso para: {nome_limpo}!")
        
    except Exception as e:
        print(f"Erro durante o pipeline de ingestão: {e}")
    finally:
        if os.path.exists(caminho_pdf_temporario):
            os.remove(caminho_pdf_temporario)
            print("Arquivo temporário PDF limpo.")


def consultar_base_jogo(query: str, nome_do_jogo: str):
    nome_limpo = nome_do_jogo.lower().replace(" ", "-")
    
    db = Chroma(client=chroma_client, collection_name=nome_limpo, embedding_function=embeddings)
    
    retriever = db.as_retriever(search_kwargs={"k": 10})
    documentos_candidatos = retriever.invoke(query)
    
    if not documentos_candidatos:
        return "Nenhum contexto encontrado no banco de dados."

    pares_rerank = [[query, doc.page_content] for doc in documentos_candidatos]
    
    print("Calculando reordenação profunda dos blocos de texto (Reranking)...")
    scores = reranker_model.predict(pares_rerank)
    
    doc_scores = list(zip(documentos_candidatos, scores))
    doc_scores_ordenados = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    
    melhores_documentos = [doc for doc, score in doc_scores_ordenados[:3]]
    contexto_filtrado = "\n\n".join([doc.page_content for doc in melhores_documentos])
    
    prompt = ChatPromptTemplate.from_template("""
    Você é um assistente especialista no jogo {jogo}. 
    Use o contexto fornecido abaixo para dar dicas, estratégias e responder à dúvida do jogador.
    Gere respostas ricas, detalhadas e muito bem explicadas baseadas apenas no contexto.
    
    Contexto: {context}
    Pergunta: {input}
    Resposta (em português):""")
    
    cadeia_geracao = prompt | llm | StrOutputParser()
    return cadeia_geracao.invoke({"context": contexto_filtrado, "input": query, "jogo": nome_do_jogo})