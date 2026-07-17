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
ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"] # Deve ser "gpt-5-mini"
API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

st.set_page_config(page_title="iAutos Bot", page_icon="🤖")
st.title("🤖 Assistente Virtual iAutos")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def inicializar_bot():
    # Modelo de Chat (LLM) - usa o nome dos secrets
    llm = AzureChatOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        azure_deployment=DEPLOYMENT_NAME,
        api_version=API_VERSION,
        temperature=0.2
    )
    
    # Modelo de Embedding - usa o nome que vimos na sua imagem
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        # ======================== AJUSTE FINAL AQUI ========================
        # Usando o nome exato da sua implantação de embedding do Azure
        azure_deployment="text-embedding-3-small", 
        # =================================================================
        api_version=API_VERSION,
    )
    
    # O resto do código continua igual...
    loader = PyPDFLoader("Caso de uso - Marketplace de classificados veículos.pdf")
    paginas = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(paginas)
    
    vector_db = Chroma.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
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

# --- O restante do seu código para interface do Streamlit ---

bot_chain = inicializar_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_usuario := st.chat_input("Como posso te ajudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
        
    with st.chat_message("assistant"):
        langchain_history = []
        for msg in st.session_state.messages[:-1]:
            role = "human" if msg["role"] == "user" else "ai"
            langchain_history.append((role, msg["content"]))
            
        response_text = bot_chain.invoke({
            "input": prompt_usuario, 
            "chat_history": langchain_history
        }) 
        
        st.markdown(response_text)
        
    st.session_state.messages.append({"role": "assistant", "content": response_text})

