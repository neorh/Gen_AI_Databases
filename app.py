import streamlit as st
from src.rag_pipeline import create_rag_chain

# 1. Configurações da página (Layout focado no chat)
st.set_page_config(
    page_title="iAutos Bot", 
    page_icon="🚗", 
    layout="centered"
)

# Injetando CSS customizado para esconder marcas do Streamlit e melhorar espaçamento
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;}    
        header {visibility: hidden;}    
        /* Adiciona um pequeno espaço no fundo para os botões não ficarem colados no input */
        .block-container {
            padding-bottom: 120px;
        }
    </style>
""", unsafe_allow_html=True)

# Cache para carregar a IA
@st.cache_resource
def load_bot():
    return create_rag_chain()

bot_chain = load_bot()

# Inicializa o histórico se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Constantes para os avatares
USER_AVATAR = "👤"
BOT_AVATAR = "🚗"

# Lista de perguntas sugeridas para exibir
PERGUNTAS_SUGERIDAS = [
    "Como funciona o modelo de negócios da iAutos?",
    "Como funcionam os contratos com a iAutos?",
    "Como a iAutos se diferencia da concorrência?",
]

# Variável para capturar cliques de botões
pergunta_clicada = None

# 2. Barra Lateral (Apenas utilitários, sem as perguntas)
with st.sidebar:
    st.title("🚗 iAutos Bot")
    st.markdown("Seu assistente virtual especialista no marketplace de classificados de veículos.")
    st.divider()
    
    st.markdown("### Opções")
    # Botão para limpar a conversa
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 3. Tela de Boas-vindas (Se o chat estiver totalmente vazio)
if len(st.session_state.messages) == 0:
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>Olá! Eu sou o assistente da iAutos</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Como posso te ajudar hoje?</p>", unsafe_allow_html=True)

# 4. Renderiza as mensagens anteriores do histórico na tela principal
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 5. Seção Fixa de Sugestões de Perguntas na Tela Principal (Abaixo do chat)
st.markdown("---")
st.caption("💡 **Perguntas Rápidas:** Escolha uma opção abaixo para perguntar instantaneamente:")

# Criando colunas no desktop que viram blocos empilhados no mobile automaticamente
cols = st.columns(len(PERGUNTAS_SUGERIDAS))
for i, pergunta in enumerate(PERGUNTAS_SUGERIDAS):
    with cols[i]:
        # Cada botão aciona a pergunta correspondente ao ser clicado
        if st.button(pergunta, key=f"sugestao_tela_{i}", use_container_width=True):
            pergunta_clicada = pergunta

# 6. Entrada de Dados (Campo de digitação fixado na base do Streamlit)
prompt_usuario = st.chat_input("Digite sua pergunta aqui...")

# Se o usuário digitou ou clicou em um dos botões rápidos
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
