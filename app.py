import streamlit as st
import os
import langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Configurações da API (As mesmas que você já configurou)
ENDPOINT = "https://gen-ai-database-trabalho-final.openai.azure.com/openai/v1"
DEPLOYMENT_NAME = "gpt-5-mini"
API_KEY = "AxKGHm51EcdbtTy2Jr3bxYQfuMkuTFXg5GS7vSbfE9785DHc06ymJQQJ99CGACHYHv6XJ3w3AAAAACOGesBS."

st.set_page_config(page_title="iAutos Bot", page_icon="🤖")
st.title("🤖 Assistente Virtual iAutos")

# Inicializar modelos e processar o PDF (usando cache do Streamlit para não reprocessar toda hora)
@st.cache_resource
def inicializar_bot():
    llm = ChatOpenAI(openai_api_base=ENDPOINT, openai_api_key=API_KEY, model_name=DEPLOYMENT_NAME, temperature=0.2)
    embeddings = OpenAIEmbeddings(openai_api_base=ENDPOINT, openai_api_key=API_KEY, model="text-embedding-3-small")
    
    # Processa o PDF fixo
    loader = PyPDFLoader("Caso de uso - Marketplace de classificados veículos.pdf")
    paginas = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(paginas)
    
    vector_db = Chroma.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Configura Prompts e Chains
    system_prompt = (
        "Você é o assistente virtual inteligente da iAutos...\n"
        "Use estritamente os fragmentos de contexto abaixo:\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

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
        # Converte o histórico para o formato que o LangChain espera se necessário, ou simula
        resposta = bot_chain.invoke({"input": prompt_usuario, "chat_history": []}) 
        response_text = resposta["answer"]
        st.markdown(response_text)
        
    st.session_state.messages.append({"role": "assistant", "content": response_text})
