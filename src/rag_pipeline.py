from operator import itemgetter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Importação da técnica avançada de fatiamento semântico
from langchain_experimental.text_splitter import SemanticChunker

# Importação das configurações seguras
from src.config import settings

def format_docs(docs):
    """Formata os documentos recuperados do banco vetorial em texto simples."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain():
    """Inicializa os modelos, carrega o PDF usando Semantic Chunking e constrói o pipeline RAG (LCEL)."""
    
    # 1. Inicializa os Modelos
    llm = AzureChatOpenAI(
        azure_endpoint=settings.ENDPOINT,
        api_key=settings.API_KEY,
        azure_deployment=settings.DEPLOYMENT_NAME,
        api_version=settings.API_VERSION,
        temperature=1.0  # Mantendo 1.0 para compatibilidade com modelos o1/gpt-5-mini
    )
    
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.ENDPOINT,
        api_key=settings.API_KEY,
        azure_deployment=settings.EMBEDDING_DEPLOYMENT, 
        api_version=settings.API_VERSION,
    )
    
    # 2. Processa Documentos com SEMANTIC CHUNKING
    loader = PyPDFLoader(settings.PDF_PATH)
    paginas = loader.load()
    
    # Junta todo o texto para o fatiador inteligente poder ler organicamente
    texto_completo = " ".join([pag.page_content for pag in paginas])
    
    # Configura o fatiador semântico usando as embeddings do Azure
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    # Gera os documentos fatiados com base em mudança de assunto
    docs = semantic_splitter.create_documents([texto_completo])
    
    # 3. Cria o Banco Vetorial
    vector_db = Chroma.from_documents(docs, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 4. Configura o Prompt e Persona
    system_prompt = (
        "Você é o assistente virtual oficial de atendimento ao cliente da iAutos, um marketplace de classificados de veículos.\n"
        "Sua missão é ajudar vendedores e compradores a entenderem as regras de publicação e uso da plataforma.\n\n"
        "DIRETRIZES DE COMPORTAMENTO:\n"
        "1. Seja sempre educado, prestativo e utilize uma linguagem clara e profissional.\n"
        "2. Responda APENAS com base nos fragmentos de contexto fornecidos abaixo.\n"
        "3. Se a dúvida do cliente NÃO estiver no contexto, diga educadamente que não possui essa informação e oriente a contatar o suporte humano.\n"
        "4. Formate respostas longas em tópicos (bullet points) para facilitar a leitura.\n\n"
        "CONTEXTO DE CONHECIMENTO:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # 5. Constrói a Cadeia (Chain) LCEL
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
