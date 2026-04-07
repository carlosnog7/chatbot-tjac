import logging

from fastapi import FastAPI, Request

try:
    from app.chatbot_service import responder_pergunta
except ModuleNotFoundError:
    from chatbot_service import responder_pergunta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Servidor ativo"}


@app.post("/webhook")
async def receive_whatsapp_msg(request: Request):
    try:
        payload = await request.json()
        value = payload["entry"][0]["changes"][0]["value"]
        messages = value.get("messages", [])

        if not messages:
            return {"status": "no_messages"}

        text_body = messages[0].get("text", {}).get("body")
        if not text_body:
            return {"status": "no_text"}

        resposta = responder_pergunta(text_body)
        return {
            "status": "success",
            "resposta": resposta.text,
            "fontes": [
                {
                    "documento": source.source,
                    "pagina": source.page,
                    "relevancia": round(source.score, 3),
                    "tipo_busca": source.retrieval,
                }
                for source in resposta.sources
            ],
            "usou_contexto": resposta.used_context,
            "modo_interacao": resposta.interaction_mode,
        }
    except (KeyError, IndexError, TypeError, ValueError):
        logger.exception("Payload inválido recebido no webhook.")
        return {"status": "error"}
