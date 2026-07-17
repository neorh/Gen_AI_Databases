import streamlit as st
import os
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
# Informações do seu painel do Azure e do Streamlit Secrets
ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"] # Corresponde a "gpt-5-mini" para o chat
API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

st.set_page_config(page_title="iAutos Bot", page_icon="🤖")
st.title("🤖 Assistente Virtual iAutos")

# Formatar os documentos recuperados em texto simples para o prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Inicializar modelos e processar o PDF (usando cache do Streamlit)
@st.cache_resource
def inicializar_bot():
    # Inicialização correta do LLM usando a classe do Azure
    llm = AzureChatOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        azure_deployment=DEPLOYMENT_NAME,  # Usa "gpt-5-mini" dos seus secrets
        api_version=API_VERSION,
        temperature=0.2
    )
    
    # Inicialização correta das Embeddings usando a classe do Azure
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        # ============================= ATENÇÃO AQUI =============================
        # Verifique no seu painel do Azure qual o "Nome da Implantação"
        # que você deu para o seu modelo de embedding (ex: text-embedding-3-small).
        # Se você usou o mesmo nome "gpt-5-mini" para a implantação de embedding,
        # o código abaixo funcionará. Caso contrário, substitua pela correta.
        # ========================================================================
        azure_deployment="gpt-5-mini", 
        api_version=API_VERSION,
    )
    
    # Processa o PDF fixo
    loader = PyPDFLoader("Caso de uso - Marketplace de classificados veículos.pdf")
    paginas = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(paginas)
    
    vector_db = Chroma.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
