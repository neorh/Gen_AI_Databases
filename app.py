import streamlit as st
from src.rag_pipeline import create_rag_chain

# 1. Configurações da página (Layout expandido fica mais bonito para chat)
st.set_page_config(
    page_title="iAutos Bot", 
    page_icon="🚗", 
    layout="centered" # ou "wide" se preferir tela cheia
)

# 2. Injetando CSS customizado para esconder marcas do Streamlit
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} /* Esconde o menu de hambúrguer */
        footer {visibility: hidden;}    /* Esconde o rodapé */
        header {visibility: hidden;}    /* Esconde o cabeçalho padrão */
    </style>
""", unsafe_allow_html=True)

# Cache para carregar a IA
@st.cache_resource
def load_bot():
    return create_rag_chain()

bot_chain = load_bot()

# 3. Barra Lateral (Sidebar) Elegante
with st.sidebar:
    st.title("🚗 iAutos Bot")
    st.markdown("Seu assistente virtual especialista no marketplace de classificados de veículos.")
    st.divider() # Linha de separação
    
    st.markdown("### Sobre")
    st.info("Este bot utiliza IA Generativa (Azure OpenAI) e a tecnologia RAG para consultar documentos internos da empresa.")
    
    # Botão para limpar a conversa
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun() # Recarrega a página instantaneamente

# Inicializa o histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Tela de Boas-vindas (se o chat estiver vazio)
if len(st.session_state.messages) == 0:
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>Como posso te ajudar hoje?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Faça perguntas sobre o caso de uso do marketplace iAutos.</p>", unsafe_allow_html=True)

# Constantes para os avatares (pode ser emoji ou URL de uma imagem real)
USER_AVATAR = "👤"
BOT_AVATAR = "🚗"

# Renderiza as mensagens anteriores com os novos avatares
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Captura de input do usuário
if prompt_usuario := st.chat_input("Digite sua pergunta aqui..."):
    
    # Adiciona a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt_usuario)
        
    # Geração da resposta
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        # Spinner visual animado enquanto o bot pensa
        with st.spinner("Analisando documentos da iAutos..."):
            langchain_history = [
                ("human" if msg["role"] == "user" else "ai", msg["content"])
                for msg in st.session_state.messages[:-1]
            ]
                
            response_text = bot_chain.invoke({
                "input": prompt_usuario, 
                "chat_history": langchain_history
            }) 
            
        st.markdown(response_text)
        
    st.session_state.messages.append({"role": "assistant", "content": response_text})
