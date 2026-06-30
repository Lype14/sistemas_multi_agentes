import os
import sys
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM

from gerenciador_rag import (
    listar_bases,
    criar_base_jogo,
    anexar_link_a_base,
    consultar_base_jogo
)

llm_agente = OllamaLLM(model="gemma2", temperature=0.1)

PROMPT_SISTEMA = """Você é o "Coordenador de Guias de Jogos", um agente inteligente especialista em gerenciamento de conhecimento de games.
Você gerencia bases vetoriais locais para cada jogo de forma isolada.

Bases de dados de jogos atualmente disponíveis no sistema: {bases_disponiveis}

Sua tarefa é analisar o que o usuário deseja e responder com um comando estruturado em formato JSON para o sistema executar a ação correta por baixo dos panos. 

Você deve responder APENAS com o JSON, sem nenhuma outra palavra antes ou depois.

Formatos de decisão permitidos:

1. Se o usuário quiser saber dicas/estratégias sobre um jogo que JÁ EXISTE na lista acima:
{{"acao": "consultar", "jogo": "nome-do-jogo", "argumento": "pergunta do usuario"}}

2. Se o usuário pedir para aprender sobre um jogo que NÃO ESTÁ na lista acima:
{{"acao": "criar", "jogo": "nome-do-jogo", "argumento": ""}}

3. Se o usuário fornecer um link (começando com http) para adicionar conhecimento a um jogo:
{{"acao": "ingerir", "jogo": "nome-do-jogo", "argumento": "url-fornecida"}}

4. Se o usuário apenas cumprimentar ou falar algo genérico que não envolva as ferramentas:
{{"acao": "conversar", "jogo": "", "argumento": "sua resposta amigável e natural aqui"}}


Utilize somente as informações contidas nas bases vetoriais do sistema, caso as informações requisitadas não estejam nas bases de dados fale explicitamente que não sabe ou não tem conhecimento prévio para responder a pergunta do usuário.
"""

def orquestrador_agente(mensagem_usuario: str, historico: list):
    bases = listar_bases()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", PROMPT_SISTEMA),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = prompt_template | llm_agente
    
    resposta_agente = chain.invoke({
        "input": mensagem_usuario,
        "bases_disponiveis": str(bases),
        "chat_history": historico
    }).strip()
    
    import json
    try:
        decisao = json.loads(resposta_agente)
    except Exception:
        if "http" in mensagem_usuario:
            decisao = {"acao": "ingerir", "jogo": bases[0] if bases else "jogo-generico", "argumento": mensagem_usuario}
        else:
            decisao = {"acao": "conversar", "jogo": "", "argumento": resposta_agente}

    acao = decisao.get("acao")
    jogo = decisao.get("jogo")
    arg = decisao.get("argumento")
    
    if acao == "consultar":
        print(f"\n🤖 [Agente] Identifiquei que você quer consultar a base de '{jogo}'. Buscando dados...")
        resposta_rag = consultar_base_jogo(arg, jogo)
        return resposta_rag
        
    elif acao == "criar":
        print(f"\n🤖 [Agente] Entendi! Não tenho o jogo '{jogo}' mapeado. Criando a base agora...")
        criar_base_jogo(jogo)
        return f"Criei uma base de dados vazia para o jogo '{jogo}'. Agora, por favor, envie um link da web com estratégias dele para eu ler e aprender!"
        
    elif acao == "ingerir":
        print(f"\n🤖 [Agente] Iniciando processamento do link para o jogo '{jogo}'...")
        anexar_link_a_base(arg, jogo)
        return f"Concluí a leitura do link! Adicionei esses novos conhecimentos à base do jogo '{jogo}'. O que quer saber sobre ele agora?"
        
    else:
        return arg

if __name__ == "__main__":
    print("\n========================================================")
    print("BEM-VINDO AO SISTEMA DE SUPORTE A JOGOS")
    print("========================================================")
    print("Exemplos de interação:")
    print(" - 'Qual a classe da Jett no Valorant?' (Consulta RAG)")
    print(" - 'Quero aprender a jogar League of Legends' (Criação de Base autônoma)")
    print(" - Enviar um link com dicas para o agente indexar no banco.")
    print("Digite 'sair' para encerrar o programa.\n")
    
    historico_conversa = []
    
    while True:
        entrada = input("\n👤 Você: ")
        if entrada.lower() == "sair":
            print("Encerrando o agente... Até logo!")
            sys.exit()
            
        if not entrada.strip():
            continue
            
        resposta_final = orquestrador_agente(entrada, historico_conversa)
        
        print(f"\n Assistente Agêntico:\n{resposta_final}")
        
        historico_conversa.append(HumanMessage(content=entrada))
        historico_conversa.append(AIMessage(content=resposta_final))