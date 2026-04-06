import os
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI, Request

# Bibliotecas RAG
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.5-flash-lite"
PROMPT_SISTEMA = (
    """
    Você é a Inteligência Artificial oficial do Tribunal de Justiça do Acre (TJAC).
    
    DIRETRIZES:
    - Você possui acesso à Base de Conhecimento Oficial do Tribunal (Editais, Resoluções, etc).
    - Jamais diga "baseado no texto que você me enviou" ou "no contexto fornecido".
    - Responda com autoridade: "De acordo com o Edital de Estágio de 2025..." ou "Segundo as normas do Tribunal...".
    - Se a informação não constar na sua base, diga: "Não localizei essa informação específica nos meus registros oficiais, mas geralmente..."
    - Use 'Linguagem Simples' e NEGRITO para dados importantes.
    """
)

chat_session = client.chats.create(
    model=MODEL_ID,
    config={'system_instruction': PROMPT_SISTEMA}
)

def inicializar_base_dados():
    pasta_docs = "docs_tjac/"
    if not os.path.exists(pasta_docs) or not os.listdir(pasta_docs):
        return None
    try:
        loader = DirectoryLoader(pasta_docs, glob="./*.pdf", loader_cls=PyPDFLoader)
        documentos_completos = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documentos_completos)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
    except:
        return None

vector_db = inicializar_base_dados()

def buscar_contexto(pergunta):
    if vector_db:
        docs = vector_db.similarity_search(pergunta, k=10)
        return "\n\n".join([doc.page_content for doc in docs])
    return ""


def pergunta_ia(pergunta_usuario):
    try:
        gatilhos = [
            "edital", "processo", "seletivo", "estágio", "inscrição", 
            "prova", "vaga", "valor", "bolsa", "regra", "documento", 
            "aprovados", "resultado", "lista", "nome", "ti", "tecnologia", "biologia"
        ]
        precisa_de_documento = any(palavra in pergunta_usuario.lower() for palavra in gatilhos)

        if precisa_de_documento:
            contexto_tjac = buscar_contexto(pergunta_usuario)
            prompt_final = (
            f"--- INFORMAÇÕES DOS REGISTROS OFICIAIS ---\n{contexto_tjac}\n"
            f"--- FIM DOS REGISTROS ---\n\n"
            f"PERGUNTA: {pergunta_usuario}\n\n"
            "INSTRUÇÃO: Analise cuidadosamente as informações acima. Se houver uma lista de nomes, "
            "identifique os nomes relacionados à Tecnologia/TI e responda ao usuário. "
            "Se encontrar nomes mas não tiver certeza se são de TI, cite os nomes encontrados e peça confirmação."
        )
        else:
            prompt_final = pergunta_usuario

        response = chat_session.send_message(prompt_final)
        return response.text

    except Exception as e:
        return f"Erro técnico: {e}"


app = FastAPI()

@app.get("/")
def home(): return {"status": "Servidor ativo"}
@app.post("/webhook")
async def receive_whatsapp_msg(request: Request):
    payload = await request.json()
    try:
        value = payload['entry'][0]['changes'][0]['value']
        if 'messages' in value:
            text_body = value['messages'][0]['text']['body']
            return {"status": "success", "resposta": pergunta_ia(text_body)}
    except: return {"status": "error"}
    return {"status": "no_messages"}