from operator import itemgetter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from src.config import settings

def format_docs(docs):
    """Formata os documentos recuperados do banco vetorial em texto simples."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain():
    """Inicializa os modelos, carrega o PDF e constrói o pipeline RAG (LCEL)."""
    
    # 1. Inicializa os Modelos
    llm = AzureChatOpenAI(
        azure_endpoint=settings.ENDPOINT,
        api_key=settings.API_KEY,
        azure_deployment=settings.DEPLOYMENT_NAME,
        api_version=settings.API_VERSION,
        temperature=1.0 
    )
    
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.ENDPOINT,
        api_key=settings.API_KEY,
        azure_deployment=settings.EMBEDDING_DEPLOYMENT, 
        api_version=settings.API_VERSION,
    )
    
    # 2. Processa Documentos e Cria o Banco Vetorial
    loader = PyPDFLoader(settings.PDF_PATH)
    paginas = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(paginas)
    
    vector_db = Chroma.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 3. Configura o Prompt
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
    
    # 4. Constrói a Cadeia (Chain)
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
