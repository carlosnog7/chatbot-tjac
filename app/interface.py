import streamlit as st
from main import pergunta_ia 

st.set_page_config(page_title="Chatbot TJAC", page_icon="⚖️", layout="centered")

# estetica
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Chatbot TJAC ⚖️")
st.caption("Interface testes")

# inicializacao historico
if "messages" not in st.session_state:
    st.session_state.messages = []

# exibição histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# interação
if prompt := st.chat_input("Como posso ajudar o Tribunal hoje?"):
# pergunta do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# resposta da ia
    with st.chat_message("assistant"):
        with st.spinner("Consultando base de dados..."):
#funcao pergunta 
            full_response = pergunta_ia(prompt)
            
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})