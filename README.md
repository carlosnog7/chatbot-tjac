## Chatbot TJAC

Protótipo de chatbot para o Tribunal de Justiça do Acre, inspirado na proposta da JuLIA, com integração ao Gemini e arquitetura RAG para responder dúvidas de cidadãos com base em documentos oficiais.

## Estrutura

- `app/chatbot_service.py`: camada central da aplicação, com Gemini, RAG, cache FAISS e montagem de prompts.
- `app/main.py`: API FastAPI e webhook.
- `app/interface.py`: interface de testes em Streamlit.
- `docs_tjac/`: PDFs usados como base de conhecimento.

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Defina a variável `GOOGLE_API_KEY` no arquivo `.env`.

Suba a API:

```bash
uvicorn app.main:app --reload
```

Abra a interface de testes:

```bash
streamlit run app/interface.py
```
