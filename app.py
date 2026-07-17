import streamlit as st
from src.rag_pipeline import create_rag_chain

# 1. Configurações da página
st.set_page_config(
    page_title="iAutos Bot", 
    page_icon="🚗", 
    layout="centered"
)

# Injetando CSS customizado para esconder marcas do Streamlit
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;}    
        header {visibility: hidden;}    
    </style>
""", unsafe_allow_html=True)

# Cache para carregar a IA
@st.cache_resource
def load_bot():
    return create_rag_chain()

bot_chain = load_bot()

# 2. Barra Lateral (Sidebar)
with st.sidebar:
    st.title("🚗 iAutos Bot")
    st.markdown("Seu assistente virtual especialista no marketplace de classificados de veículos.")
    st.divider()
    
    st.markdown("### Sobre")
    st.info("Este bot utiliza IA Generativa (Azure OpenAI) e a tecnologia RAG para consultar documentos internos da empresa.")
    
    # Botão para limpar a conversa
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Inicializa o histórico se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Constantes para os avatares
USER_AVATAR = "👤"
BOT_AVATAR = "🚗"

# Lista de perguntas sugeridas para exibir na tela inicial
PERGUNTAS_SUGERIDAS = [
    "Como funciona o modelo de negócios da iAutos?",
    "Quais são os principais canais de atração de leads?",
    "Como a iAutos se diferencia da concorrência?",
]

# Variável para capturar se uma sugestão foi clicada
pergunta_clicada = None

# 3. Tela de Boas-vindas + Botões de Sugestão (se o chat estiver vazio)
if len(st.session_state.messages) == 0:
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>Como posso te ajudar hoje?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 25px;'>Escolha uma das sugestões abaixo ou digite sua própria pergunta:</p>", unsafe_allow_html=True)
    
    # Criando colunas lado a lado para os botões ficarem alinhados horizontalmente
    cols = st.columns(len(PERGUNTAS_SUGERIDAS))
    for i, pergunta in enumerate(PERGUNTAS_SUGERIDAS):
        with cols[i]:
            # Se o botão for clicado, guardamos o texto dele na variável
            if st.button(pergunta, key=f"sugestao_{i}", use_container_width=True):
                pergunta_clicada = pergunta

# Renderiza as mensagens anteriores do histórico
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 4. Fluxo de Entrada de Dados (Aceita digitação ou o clique do botão)
prompt_usuario = st.chat_input("Digite sua pergunta aqui...")

# Se o usuário digitou algo OU clicou em um botão de sugestão:
if prompt_usuario or pergunta_clicada:
    # Define a pergunta final a ser processada
    pergunta_final = prompt_usuario if prompt_usuario else pergunta_clicada
    
    # Adiciona a mensagem do usuário ao histórico e exibe na tela
    st.session_state.messages.append({"role": "user", "content": pergunta_final})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(pergunta_final)
        
    # Geração da resposta do assistente
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Analisando documentos da iAutos..."):
            langchain_history = [
                ("human" if msg["role"] == "user" else "ai", msg["content"])
                for msg in st.session_state.messages[:-1]
            ]
                
            response_text = bot_chain.invoke({
                "input": pergunta_final, 
                "chat_history": langchain_history
            }) 
            
        st.markdown(response_text)
        
    # Salva a resposta no histórico e força a atualização da tela
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()
