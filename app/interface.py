import streamlit as st

try:
    from app.chatbot_service import responder_pergunta
except ModuleNotFoundError:
    from chatbot_service import responder_pergunta

st.set_page_config(page_title="Chatbot TJAC", page_icon="TJ", layout="centered")

st.markdown(
    """
    <style>
    .main { background: linear-gradient(180deg, #f7f3eb 0%, #ffffff 100%); }
    .stApp {
        font-family: "Segoe UI", sans-serif;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    .stChatInput input {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Protótipo Chatbot TJAC")
st.caption("Assistente virtual para orientar cidadãos com base nos registros oficiais do Tribunal.")


def serialize_sources(sources):
    return [
        {
            "source": source.source,
            "page": source.page,
            "score": source.score,
            "retrieval": source.retrieval,
            "content": source.content,
        }
        for source in sources
    ]


def render_sources(sources):
    if not sources:
        return

    with st.expander("Fontes consultadas"):
        for source in sources:
            page_label = (
                f"página {source['page']}"
                if source["page"] is not None
                else "página não identificada"
            )
            st.markdown(
                f"**{source['source']}** | {page_label} | relevância {source['score']:.2f} | busca {source['retrieval']}"
            )
            st.caption(
                source["content"][:500] + ("..." if len(source["content"]) > 500 else "")
            )


def append_assistant_message(response):
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.text,
            "sources": serialize_sources(response.sources),
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Pode falar comigo do jeito que for mais confortável. "
                "Se quiser, eu também posso buscar informações oficiais do TJAC."
            ),
            "sources": [],
        }
    ]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if prompt := st.chat_input("Escreva sua mensagem"):
    st.session_state.pending_prompt = prompt
    st.rerun()

st.session_state.messages = st.session_state.messages[-50:]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if st.session_state.pending_prompt is not None:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Pensando...")
        response = responder_pergunta(prompt)
        placeholder.empty()
        st.markdown(response.text)
        current_sources = serialize_sources(response.sources)
        render_sources(current_sources)

    append_assistant_message(response)
    st.session_state.messages = st.session_state.messages[-50:]
    st.rerun()
