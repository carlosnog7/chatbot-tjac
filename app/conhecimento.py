from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def preparar_base_conhecimento():
    caminho_pdf = "docs_tjac/SEI_2117808_Edital_01_2025.pdf" 
    
    if not os.path.exists(caminho_pdf):
        return None
    loader = PyPDFLoader(caminho_pdf)
    paginas = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(paginas)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore

base_tjac = preparar_base_conhecimento()

def buscar_no_edital(pergunta):
    if base_tjac:
        resultados = base_tjac.similarity_search(pergunta, k=3)
        conteudo_extraido = "\n\n".join([doc.page_content for doc in resultados])
        return conteudo_extraido
    return ""