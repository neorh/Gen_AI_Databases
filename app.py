import streamlit as st
from src.rag_pipeline import create_rag_chain

# Configurações da página
st.set_page_config(page_title="iAutos Bot", page_icon="🤖")
st.title("🤖 Assistente Virtual iAutos")

# Cache para não recriar a cadeia do LangChain a cada interação
@st.cache_resource
def load_bot():
    return create_rag_chain()

bot_chain = load_bot()

# Inicializa o histórico na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Renderiza as mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura de input do usuário
if prompt_usuario := st.chat_input("Como posso te ajudar hoje?"):
    
    # Adiciona a mensagem do usuário na tela
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
        
    # Geração da resposta
    with st.chat_message("assistant"):
        # Prepara o histórico no padrão LangChain
        langchain_history = [
            ("human" if msg["role"] == "user" else "ai", msg["content"])
            for msg in st.session_state.messages[:-1]
        ]
            
        # Invoca a IA
        response_text = bot_chain.invoke({
            "input": prompt_usuario, 
            "chat_history": langchain_history
        }) 
        st.markdown(response_text)
        
    # Salva a resposta no histórico
    st.session_state.messages.append({"role": "assistant", "content": response_text})
