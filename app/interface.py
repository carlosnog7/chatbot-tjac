import streamlit as st
from main import pergunta_ia 

# Configuração da página
st.set_page_config(page_title="Chatbot TJAC", page_icon="⚖️", layout="centered")

# Estética Customizada
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Protótipo Chatbot TJAC")
st.caption("Interface de Testes")

# Inicialização do histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibição do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Interação do Usuário
if prompt := st.chat_input("Como posso ajudar o Tribunal hoje?"):
    # Adiciona pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Resposta da IA com Spinner de carregamento
    with st.chat_message("assistant"):
        with st.spinner("Gerando resposta..."):
            full_response = pergunta_ia(prompt)
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})