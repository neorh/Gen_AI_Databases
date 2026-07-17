import streamlit as st

class Settings:
    """Classe para centralizar todas as variáveis de ambiente e configurações."""
    
    # Credenciais Azure OpenAI
    ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
    API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
    DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
    API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    # Configurações de Modelos e Caminhos
    EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
    PDF_PATH = "data/Caso de uso - Marketplace de classificados veículos.pdf"
    
settings = Settings()
