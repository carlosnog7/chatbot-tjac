import os
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI, Request
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"
PROMPT_SISTEMA = (
    """
    Você é a Inteligência Artificial oficial do Tribunal de Justiça do Acre (TJAC). 
    Sua missão é ser o GPS Humano do portal www.tjac.jus.br para o cidadão acreano.\n\n"
    - Use 'Linguagem Simples' (Resolução 332 CNJ). Evite termos técnicos.\n"
    - Se precisar usar um termo jurídico, explique-o logo em seguida.\n"
    - Use o fuso horário do Acre (atualmente 09:17 AM).\n"
    - Sempre que possível, forneça o caminho no site. Ex: 'Início > Cidadão > Certidões'.\n"
    - Comece com uma saudação cordial.\n"
    - Use NEGRITO para termos importantes e links.\n"
    - Use listas para passos a passos.
    """
)
chat_session = client.chats.create(
    model=MODEL_ID,
    config={'system_instruction': PROMPT_SISTEMA}
    )

app = FastAPI()


def pergunta_ia(pergunta_usuario):
    try:
        response = chat_session.send_message(pergunta_usuario)

        return response.text
    except Exception as e:
        if "429" in str(e):
            return "Limite de mensagens gratuitas atingido."
        return f"Desculpe, tive um erro tecnico: {e}"

@app.get("/")
def home():
    return {"status": "Servidor do TJAC Ativo"}

@app.post("/webhook")
async def receive_whatsapp_msg(request: Request):
    payload = await request.json()
    
    try:
        entry = payload['entry'][0]
        changes = entry['changes'][0]
        value = changes['value']
        
        if 'messages' in value:
            message = value['messages'][0]
            sender_number = message['from']
            text_body = message['text']['body']
            
            resposta_ia = pergunta_ia(text_body)
            
            print(f"\nNova mensagem recebida")
            print(f"De: {sender_number}")
            print(f"Texto: {text_body}")
            print(f"Resposta IA: {resposta_ia}")
            print("-" * 30)
            
            return {"status": "success", "resposta": resposta_ia}
            
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    return {"status": "no_messages"}

