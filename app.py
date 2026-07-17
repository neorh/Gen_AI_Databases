import streamlit as st
import os
from operator import itemgetter
# Importando as classes específicas para o ecossistema Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# CONFIGURAÇÕES SEGURAS VIA STREAMLIT SECRETS
# ==============================================================================
# Certifique-se de que estes campos estão preenchidos no "Secrets" do Streamlit:
ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"] # Nome da implantação do Chat (ex: "gpt-5-mini")
API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

st.set_page_config(page_title="iAutos Bot", page_icon="🤖")
st.title("🤖 Assistente Virtual iAutos")

# Formata os documentos retornados do Chroma para texto simples
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cache do Streamlit para evitar reprocessar o PDF a cada mensagem enviada
@st.cache_resource
def inicializar_bot():
    # Modelo de Chat (LLM) - usa o nome configurado nos secrets
    llm = AzureChatOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        azure_deployment=DEPLOYMENT_NAME,
        api_version=API_VERSION,
        temperature=1
    )
    
    # Modelo de Embedding - usa o nome exato da sua implantação de embedding do Azure
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        azure_deployment="text-embedding-3-small", 
        api_version=API_VERSION,
    )
    
    # Processa o PDF fixo de contexto
    loader = PyPDFLoader("Caso de uso - Marketplace de classificados veículos.pdf")
    paginas = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(paginas)
    
    # Salva os fragmentos no banco vetorial temporário em memória
    vector_db = Chroma.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Define o Prompt de instruções da Persona
    system_prompt = (
        "Você é o assistente virtual inteligente da iAutos...\n"
        "Use estritamente os fragmentos de contexto abaixo para responder:\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # Configura a cadeia moderna de execução (LCEL) com roteamento correto
    rag_chain = (
        {
            # O itemgetter pega apenas o texto da chave "input" e manda para o retriever
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
# Inicialização do Bot
bot_chain = inicializar_bot()

# Histórico de Conversa na tela
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura de input do usuário
if prompt_usuario := st.chat_input("Como posso te ajudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
        
    # Geração da resposta do assistente
    with st.chat_message("assistant"):
        # Formata o histórico mantendo as roles mapeadas para o LangChain
        langchain_history = []
        for msg in st.session_state.messages[:-1]:
            role = "human" if msg["role"] == "user" else "ai"
            langchain_history.append((role, msg["content"]))
            
        # Invoca o pipeline moderno
        response_text = bot_chain.invoke({
            "input": prompt_usuario, 
            "chat_history": langchain_history
        }) 
        
        st.markdown(response_text)
        
    st.session_state.messages.append({"role": "assistant", "content": response_text})
