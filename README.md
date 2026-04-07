## Chatbot TJAC

Protótipo de chatbot para o Tribunal de Justiça do Acre, inspirado na proposta da JuLIA, com integração ao Gemini e arquitetura RAG para responder dúvidas de cidadãos com base em documentos oficiais.

## Estrutura

- `app/chatbot_service.py`: camada central da aplicação, com Gemini, RAG, cache FAISS e montagem de prompts.
- `app/main.py`: API FastAPI e webhook.
- `app/interface.py`: interface de testes em Streamlit.
- `docs_tjac/`: PDFs usados como base de conhecimento.

## Dependências

```bash
pip install -r requirements.txt
```

Definir `GOOGLE_API_KEY`.

API:

```bash
uvicorn app.main:app --reload
```

INTERFACE:

```bash
streamlit run app/interface.py
```
