import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# CONFIGURAÇÕES SEGURAS VIA STREAMLIT SECRETS
# ==============================================================================
ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]

st.set_page_config(page_title="iAutos Bot", page_icon="🤖")
st.title("🤖 Assistente Virtual iAutos")

# Formatar os documentos recuperados em texto simples para o prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Inicializar modelos e processar o PDF (usando cache para otimizar desempenho)
@st.cache_resource
def inicializar_bot():
    llm = ChatOpenAI(
        openai_api_base=ENDPOINT, 
        openai_api_key=API_KEY, 
        model_name=DEPLOYMENT_NAME, 
        temperature=0.2
    )
    
    embeddings = OpenAIEmbeddings(
        openai_api_base=ENDPOINT, 
        openai_api_key=API_KEY, 
        model="text-embedding-3-small"
    )
    
    # Processa o PDF fixo
    loader = PyPDFLoader("Caso de uso - Marketplace de classificados veículos.pdf")
    paginas = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(paginas)
    
    vector_db = Chroma.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Configura o Prompt do Sistema
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
    
    # CONSTRUÇÃO DA CADEIA NO PADRÃO MODERNO (LCEL)
    # 1. Recuperamos os documentos do banco e enviamos para o prompt junto com o histórico
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

bot_chain = inicializar_bot()

# Histórico de Conversa na tela
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt_usuario := st.chat_input("Como posso te ajudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
        
    # Resposta do Bot
    with st.chat_message("assistant"):
        # Formata o histórico do Streamlit no formato que o ChatPromptTemplate espera
        langchain_history = []
        for msg in st.session_state.messages[:-1]: # ignora a última mensagem adicionada
            role = "human" if msg["role"] == "user" else "ai"
            langchain_history.append((role, msg["content"]))
            
        # Invoca a cadeia e retorna o texto diretamente
        response_text = bot_chain.invoke({
            "input": prompt_usuario, 
            "chat_history": langchain_history
        }) 
        
        st.markdown(response_text)
        
    st.session_state.messages.append({"role": "assistant", "content": response_text})
