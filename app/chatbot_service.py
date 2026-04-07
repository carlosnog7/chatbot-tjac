import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs_tjac"
CACHE_DIR = BASE_DIR / ".cache"
FAISS_DIR = CACHE_DIR / "faiss_tjac"
METADATA_FILE = CACHE_DIR / "faiss_tjac_metadata.json"
WEB_CACHE_DIR = CACHE_DIR / "web_sources"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = "gemini-2.5-flash-lite"
VECTOR_K = 12
LEXICAL_K = 8
FINAL_CONTEXT_K = 6
MIN_RETRIEVAL_SCORE = 1.0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
MAX_RETRIES = 3
HTTP_TIMEOUT_SECONDS = 20
WEB_SOURCES = [
    {
        "name": "Enderecos, Telefones e Balcao Virtual - TJAC",
        "url": "https://www.tjac.jus.br/enderecos-e-telefones/",
    },
    {
        "name": "Orgaos de Composicao Atual - TJAC",
        "url": "https://www.tjac.jus.br/adm/orgaos-composicao-atual/",
    },
]
GREETING_WORDS = {
    "oi",
    "ola",
    "olá",
    "oie",
    "opa",
    "e ai",
    "e aí",
    "bom dia",
    "boa tarde",
    "boa noite",
    "tudo bem",
}
SMALL_TALK_PATTERNS = {
    "greeting": [
        "oi",
        "ola",
        "olá",
        "oie",
        "opa",
        "e ai",
        "e aí",
        "bom dia",
        "boa tarde",
        "boa noite",
    ],
    "gratitude": [
        "obrigado",
        "obrigada",
        "valeu",
        "agradecido",
        "agradecida",
        "muito obrigado",
        "muito obrigada",
    ],
    "wellbeing": [
        "tudo bem",
        "como voce esta",
        "como você está",
        "como vai",
        "ta tudo bem",
        "está tudo bem",
    ],
    "identity": [
        "quem e voce",
        "quem é voce",
        "quem e você",
        "quem é você",
        "o que voce faz",
        "o que você faz",
        "como voce pode ajudar",
        "como você pode ajudar",
    ],
    "help_request": [
        "pode me ajudar",
        "voce pode me ajudar",
        "você pode me ajudar",
        "preciso de ajuda",
        "me ajuda",
        "quero ajuda",
    ],
}
TJAC_HINTS = {
    "tjac",
    "tribunal",
    "tribunal de justica",
    "tribunal de justiça",
    "edital",
    "estagio",
    "estágio",
    "processo seletivo",
    "inscricao",
    "inscrição",
    "resultado",
    "classificacao",
    "classificação",
    "imovel",
    "imóvel",
    "registro",
    "cartorio",
    "cartório",
    "extrajudicial",
    "certidao",
    "certidão",
    "documento",
    "telefone",
    "endereco",
    "endereço",
    "horario",
    "horário",
}
STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "eu",
    "gostaria",
    "me",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "por",
    "qual",
    "quais",
    "que",
    "quem",
    "se",
    "ser",
    "sobre",
    "um",
    "uma",
}
QUERY_EXPANSIONS = {
    "inscrito": ["inscritos", "resultado", "ranking", "classificacao", "nome", "cpf"],
    "inscritos": ["resultado", "ranking", "classificacao", "nome", "cpf"],
    "resultado": ["ranking", "classificacao", "nome", "cpf"],
    "data": ["periodo", "cronograma", "inscricao", "publicacao"],
    "documentos": ["cpf", "rg", "certidao", "escritura", "matricula"],
    "proprietario": ["proprietarios", "comprador", "vendedor"],
    "proprietário": ["proprietarios", "comprador", "vendedor"],
    "imovel": ["imovel", "registro", "escritura", "matricula", "cartorio"],
    "imóvel": ["imovel", "registro", "escritura", "matricula", "cartorio"],
}
PROMPT_SISTEMA = """
Voce e a Inteligencia Artificial oficial do Tribunal de Justica do Acre (TJAC).

DIRETRIZES:
- Sua prioridade e responder com base nos registros oficiais recuperados da base documental do TJAC.
- Responda de forma clara, objetiva, casual e acolhedora, pensando em cidadaos que precisam de orientacao rapida.
- Em conversas simples, cumprimente de forma natural e amigavel, sem linguagem tecnica.
- Nunca diga "baseado no texto que voce me enviou" ou "no contexto fornecido".
- Quando os registros oficiais forem suficientes, responda com autoridade: "De acordo com o edital..." ou "Segundo a cartilha do TJAC...".
- Se os registros recuperados nao forem suficientes para sustentar a resposta, diga explicitamente que nao localizou a informacao especifica nos registros oficiais consultados.
- Nao invente nomes, datas, documentos, etapas ou exigencias.
- Se a pergunta for sobre listas, inscritos, classificacao ou resultados, priorize nomes, datas, paginas e documento de origem.
- Use linguagem simples e destaque em **negrito** os dados mais importantes quando estiver apresentando informacoes oficiais.
""".strip()
PROMPT_CONVERSA = """
Voce e a assistente virtual do TJAC.

Fale de forma acolhedora, natural e humana, como uma atendente paciente.
Converse em portugues do Brasil, com frases simples e calorosas.
Evite soar automatica, repetitiva, rigida ou institucional demais.
Nao use listas nem respostas engessadas quando a pessoa so estiver conversando.
Se a pessoa mandar cumprimento, responda como em uma conversa normal.
Se a pessoa pedir ajuda de forma generica, acolha e convide a explicar a duvida do jeito dela.
Nao invente informacoes oficiais nessa modalidade. Aqui voce so conversa e orienta o proximo passo.
Responda de forma curta, leve e gentil.
""".strip()


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    page: int | None
    score: float
    retrieval: str


@dataclass(frozen=True)
class ChatbotResponse:
    text: str
    sources: list[RetrievedChunk]
    used_context: bool
    retrieval_debug: str
    interaction_mode: str


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr", "td", "th", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = re.findall(r"[a-z0-9]{2,}", normalized)
    return [token for token in tokens if token not in STOPWORDS]


def _expanded_query_terms(question: str) -> list[str]:
    base_terms = _tokenize(question)
    expanded_terms = list(base_terms)
    normalized_question = _normalize_text(question)

    for term in list(base_terms):
        for extra in QUERY_EXPANSIONS.get(term, []):
            expanded_terms.extend(_tokenize(extra))

    if "quem" in normalized_question and ("inscrito" in normalized_question or "resultado" in normalized_question):
        expanded_terms.extend(["nome", "cpf", "ranking", "classificacao"])

    if "data" in normalized_question and ("processo seletivo" in normalized_question or "estagio" in normalized_question):
        expanded_terms.extend(["edital", "periodo", "cronograma", "inscricao"])

    return list(dict.fromkeys(expanded_terms))


def _is_greeting_only(question: str) -> bool:
    normalized = _normalize_text(question).strip()
    return normalized in GREETING_WORDS


def _classify_small_talk(question: str) -> str | None:
    normalized = _normalize_text(question).strip()
    compact = re.sub(r"\s+", " ", normalized)
    for category, patterns in SMALL_TALK_PATTERNS.items():
        if any(pattern in compact for pattern in patterns):
            return category
    return None


def _is_information_request(question: str) -> bool:
    normalized = _normalize_text(question)
    question_tokens = _tokenize(question)
    interrogative_markers = {
        "como",
        "qual",
        "quais",
        "quando",
        "onde",
        "quem",
        "preciso",
        "necessario",
        "necessário",
        "telefone",
        "endereco",
        "endereço",
        "horario",
        "horário",
        "documento",
        "documentos",
        "inscrito",
        "inscritos",
        "resultado",
        "edital",
        "processo",
        "vaga",
        "registro",
        "imovel",
        "imóvel",
    }
    return (
        "?" in question
        or any(marker in normalized for marker in TJAC_HINTS)
        or any(token in interrogative_markers for token in question_tokens)
        or len(question_tokens) >= 6
    )


def _conversation_prompt(user_message: str, category: str | None = None) -> str:
    category_hint = ""
    if category == "greeting":
        category_hint = "A pessoa acabou de cumprimentar voce."
    elif category == "gratitude":
        category_hint = "A pessoa agradeceu."
    elif category == "wellbeing":
        category_hint = "A pessoa esta puxando conversa e perguntando como voce esta."
    elif category == "identity":
        category_hint = "A pessoa quer saber quem voce e ou como voce ajuda."
    elif category == "help_request":
        category_hint = "A pessoa pediu ajuda, mas ainda nao explicou a duvida."

    return (
        f"{category_hint}\n"
        f"Mensagem da pessoa: {user_message}\n\n"
        "Responda de forma acolhedora, casual e breve. "
        "Soe como uma conversa normal, sem parecer texto pronto."
    )


def _looks_like_tjac_query(question: str) -> bool:
    normalized = _normalize_text(question)
    return any(hint in normalized for hint in TJAC_HINTS)


def _format_source_label(doc: RetrievedChunk) -> str:
    page_label = f", pagina {doc.page}" if doc.page is not None else ""
    return f"{doc.source}{page_label}"


def _format_sources_for_answer(sources: list[RetrievedChunk]) -> str:
    if not sources:
        return ""

    unique_labels: list[str] = []
    seen: set[str] = set()
    for chunk in sources:
        label = _format_source_label(chunk)
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)

    formatted = "; ".join(unique_labels[:4])
    return f"\n\nFontes consultadas: {formatted}."


def _fallback_without_context(question: str) -> str:
    if _looks_like_tjac_query(question):
        return (
            "Nao localizei essa informacao especifica nos registros oficiais consultados do TJAC. "
            "Se voce tiver o edital, ato, lista ou documento exato que deseja consultar, posso analisar "
            "assim que ele estiver disponivel na base."
        )

    return (
        "Posso ajudar melhor com informacoes que estejam nos registros oficiais do TJAC, como "
        "**editais, resultados, regras, documentos e orientacoes de servicos**. "
        "Se quiser, faca a pergunta citando o tema ou o documento desejado."
    )


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY nao encontrada nas variaveis de ambiente.")
    return genai.Client(api_key=api_key)


@lru_cache(maxsize=1)
def get_chat_session():
    return get_client().chats.create(
        model=MODEL_ID,
        config={"system_instruction": PROMPT_SISTEMA},
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings | None:
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception:
        logger.exception(
            "Nao foi possivel inicializar o modelo de embeddings. A busca seguira em modo textual."
        )
        return None


def _document_signature() -> list[dict[str, float | str]]:
    signature = []
    if DOCS_DIR.exists():
        for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
            stat = pdf_path.stat()
            signature.append(
                {
                    "type": "pdf",
                    "name": pdf_path.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )

    for source in WEB_SOURCES:
        signature.append(
            {
                "type": "web",
                "name": source["name"],
                "url": source["url"],
            }
        )
    return signature


def _cache_is_valid() -> bool:
    if not FAISS_DIR.exists() or not METADATA_FILE.exists():
        return False

    try:
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return metadata.get("documents") == _document_signature()


def _save_cache_metadata() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(
        json.dumps({"documents": _document_signature()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _enrich_document_metadata(documents: list[Document]) -> list[Document]:
    enriched: list[Document] = []
    for doc in documents:
        metadata = dict(doc.metadata)
        source_path = Path(metadata.get("source", ""))
        metadata["source_name"] = metadata.get("source_name") or (
            source_path.name if source_path.name else "Documento TJAC"
        )
        metadata["page_number"] = (metadata.get("page") + 1) if metadata.get("page") is not None else None
        enriched.append(Document(page_content=doc.page_content, metadata=metadata))
    return enriched


def _sanitize_cache_filename(url: str) -> str:
    filename = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")
    return f"{filename}.json"


def _trim_web_text(text: str, source_name: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()

    if "Enderecos, Telefones e Balcao Virtual" in source_name:
        start_marker = "Endereços, Telefones e Balcão Virtual"
        end_marker = "Última modificação:"
    else:
        start_marker = ""
        end_marker = "Última modificação:"

    if start_marker:
        start_index = normalized.find(start_marker)
        if start_index >= 0:
            normalized = normalized[start_index:]

    end_index = normalized.find(end_marker)
    if end_index >= 0:
        normalized = normalized[:end_index]

    return normalized


def _fetch_web_source(source: dict[str, str]) -> Document | None:
    WEB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = WEB_CACHE_DIR / _sanitize_cache_filename(source["url"])

    try:
        request = Request(
            source["url"],
            headers={"User-Agent": "Mozilla/5.0 (compatible; TJAC-Chatbot/1.0)"},
        )
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            html = response.read().decode("utf-8", errors="ignore")

        parser = _HTMLTextExtractor()
        parser.feed(html)
        extracted_text = _trim_web_text(parser.get_text(), source["name"])

        payload = {
            "url": source["url"],
            "name": source["name"],
            "text": extracted_text,
            "fetched_at": time.time(),
        }
        cache_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Fonte web carregada com sucesso: %s", source["url"])
    except (OSError, URLError, TimeoutError):
        logger.exception("Falha ao carregar fonte web: %s", source["url"])
        if not cache_file.exists():
            return None

        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            extracted_text = payload.get("text", "")
            logger.warning("Usando cache local para a fonte web %s", source["url"])
        except (OSError, json.JSONDecodeError):
            return None

    if not extracted_text:
        return None

    return Document(
        page_content=extracted_text,
        metadata={
            "source": source["url"],
            "source_name": source["name"],
            "page_number": None,
            "source_type": "web",
            "url": source["url"],
        },
    )


@lru_cache(maxsize=1)
def get_web_documents() -> list[Document]:
    documents = []
    for source in WEB_SOURCES:
        document = _fetch_web_source(source)
        if document is not None:
            documents.append(document)
    return documents


@lru_cache(maxsize=1)
def get_split_documents() -> list[Document]:
    raw_documents: list[Document] = []
    if DOCS_DIR.exists() and any(DOCS_DIR.glob("*.pdf")):
        loader = DirectoryLoader(str(DOCS_DIR), glob="*.pdf", loader_cls=PyPDFLoader)
        raw_documents.extend(loader.load())

    raw_documents.extend(get_web_documents())
    if not raw_documents:
        return []

    enriched_documents = _enrich_document_metadata(raw_documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(enriched_documents)
    logger.info("Registros oficiais carregados e divididos em %s chunks.", len(chunks))
    return chunks


def _build_vector_db() -> FAISS | None:
    chunks = get_split_documents()
    if not chunks:
        logger.warning("Nenhum registro oficial encontrado em %s", DOCS_DIR)
        return None

    embeddings = get_embeddings()
    if embeddings is None:
        return None

    vector_db = FAISS.from_documents(chunks, embeddings)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(FAISS_DIR))
    _save_cache_metadata()
    logger.info("Base vetorial criada com %s chunks.", len(chunks))
    return vector_db


@lru_cache(maxsize=1)
def get_vector_db() -> FAISS | None:
    try:
        if _cache_is_valid():
            embeddings = get_embeddings()
            if embeddings is None:
                return None
            logger.info("Carregando base vetorial do cache local.")
            return FAISS.load_local(
                str(FAISS_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )

        logger.info("Cache ausente ou desatualizado. Recriando base vetorial.")
        return _build_vector_db()
    except Exception:
        logger.exception("Falha ao inicializar a base vetorial.")
        return None


def _lexical_score(question: str, content: str, source_name: str) -> float:
    question_tokens = _expanded_query_terms(question)
    if not question_tokens:
        return 0.0

    content_normalized = _normalize_text(content)
    source_normalized = _normalize_text(source_name)
    score = 0.0
    matched_tokens = 0
    for token in question_tokens:
        if token in content_normalized:
            score += 2.0
            matched_tokens += 1
        if token in source_normalized:
            score += 1.5

    phrase = " ".join(question_tokens[:4])
    if phrase and phrase in content_normalized:
        score += 3.0

    question_normalized = _normalize_text(question)
    if ("inscrito" in question_normalized or "resultado" in question_normalized) and re.search(r"\*{3}\.\d{3}\.\d{3}-\*{2}", content):
        score += 6.0

    if "nome cpf" in content_normalized or "ordem nome cpf" in content_normalized:
        score += 3.0

    if "sistema de informacao" in question_normalized and "sistema de informacao" in content_normalized:
        score += 4.0

    if "sistemas de informacao" in question_normalized and "sistemas de informacao" in content_normalized:
        score += 4.0

    coverage_bonus = matched_tokens / max(len(question_tokens), 1)
    score += coverage_bonus * 4.0
    return score


def _chunk_from_document(doc: Document, score: float, retrieval: str) -> RetrievedChunk:
    metadata = doc.metadata
    return RetrievedChunk(
        content=" ".join(doc.page_content.split()),
        source=metadata.get("source_name", "Documento TJAC"),
        page=metadata.get("page_number"),
        score=score,
        retrieval=retrieval,
    )


def _lexical_search(question: str, limit: int = LEXICAL_K) -> list[RetrievedChunk]:
    tokens = _expanded_query_terms(question)
    if not tokens:
        return []

    ranked: list[RetrievedChunk] = []
    for doc in get_split_documents():
        score = _lexical_score(question, doc.page_content, doc.metadata.get("source_name", ""))
        if score > 0:
            ranked.append(_chunk_from_document(doc, score, "lexical"))

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:limit]


def _vector_search(question: str, limit: int = VECTOR_K) -> list[RetrievedChunk]:
    vector_db = get_vector_db()
    if not vector_db:
        return []

    try:
        docs_and_scores = vector_db.similarity_search_with_score(question, k=limit)
    except Exception:
        logger.exception("Falha na busca vetorial.")
        return []

    ranked: list[RetrievedChunk] = []
    for doc, raw_score in docs_and_scores:
        similarity_score = 1.0 / (1.0 + float(raw_score))
        ranked.append(_chunk_from_document(doc, similarity_score, "vector"))
    return ranked


def retrieve_relevant_chunks(question: str) -> list[RetrievedChunk]:
    vector_hits = _vector_search(question)
    lexical_hits = _lexical_search(question)

    combined: dict[tuple[str, int | None, str], RetrievedChunk] = {}
    for chunk in vector_hits + lexical_hits:
        key = (chunk.source, chunk.page, chunk.content[:120])
        existing = combined.get(key)
        if existing is None or chunk.score > existing.score:
            combined[key] = chunk

    ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
    selected = [chunk for chunk in ranked if chunk.score >= MIN_RETRIEVAL_SCORE][:FINAL_CONTEXT_K]
    return selected


def buscar_contexto(pergunta: str, limite: int = FINAL_CONTEXT_K) -> str:
    chunks = retrieve_relevant_chunks(pergunta)[:limite]
    return "\n\n".join(chunk.content for chunk in chunks)


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        page_label = f" | pagina {chunk.page}" if chunk.page is not None else ""
        blocks.append(
            f"[Trecho {index} | fonte: {chunk.source}{page_label} | relevancia: {chunk.score:.2f}]\n"
            f"{chunk.content}"
        )
    return "\n\n".join(blocks)


def montar_prompt_com_contexto(pergunta_usuario: str, chunks: list[RetrievedChunk]) -> str:
    return (
        "--- REGISTROS OFICIAIS RECUPERADOS DO TJAC ---\n"
        f"{_build_context_block(chunks)}\n"
        "--- FIM DOS REGISTROS ---\n\n"
        f"PERGUNTA DO USUARIO: {pergunta_usuario}\n\n"
        "INSTRUCOES DE RESPOSTA:\n"
        "- Responda usando prioritariamente os registros oficiais acima.\n"
        "- Se a resposta estiver nos registros, diga isso de forma direta e objetiva.\n"
        "- Se a pergunta for sobre nomes, datas, documentos, resultados ou requisitos, reproduza apenas o que estiver claramente sustentado nos registros.\n"
        "- Se os registros nao trouxerem a informacao especifica pedida, diga: "
        "\"Nao localizei essa informacao especifica nos registros oficiais consultados.\" \n"
        "- Nao complemente com conhecimento geral quando faltar base documental suficiente.\n"
        "- Ao final, mencione de forma natural o nome do documento mais relevante consultado."
    )


def _send_message_with_retry(prompt: str, conversation_mode: bool = False) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if conversation_mode:
                response = get_client().models.generate_content(
                    model=MODEL_ID,
                    contents=f"{PROMPT_CONVERSA}\n\n{prompt}",
                )
            else:
                response = get_chat_session().send_message(prompt)
            return response.text
        except Exception as exc:
            last_error = exc
            message = str(exc)
            is_temporary = "503" in message or "UNAVAILABLE" in message.upper()
            if attempt < MAX_RETRIES and is_temporary:
                wait_seconds = attempt * 2
                logger.warning("Modelo indisponivel temporariamente. Tentando novamente em %ss.", wait_seconds)
                time.sleep(wait_seconds)
                continue
            raise

    raise RuntimeError(f"Falha ao consultar o modelo: {last_error}")


def _generate_conversation_reply(user_message: str, category: str | None = None) -> str:
    prompt = _conversation_prompt(user_message, category)
    return _send_message_with_retry(prompt, conversation_mode=True).strip()


def responder_pergunta(pergunta_usuario: str) -> ChatbotResponse:
    try:
        if _is_greeting_only(pergunta_usuario):
            return ChatbotResponse(
                text=_generate_conversation_reply(pergunta_usuario, "greeting"),
                sources=[],
                used_context=False,
                retrieval_debug="greeting_only",
                interaction_mode="conversation",
            )

        small_talk_category = _classify_small_talk(pergunta_usuario)
        if small_talk_category and not _is_information_request(pergunta_usuario):
            return ChatbotResponse(
                text=_generate_conversation_reply(pergunta_usuario, small_talk_category),
                sources=[],
                used_context=False,
                retrieval_debug=f"small_talk:{small_talk_category}",
                interaction_mode="conversation",
            )

        chunks = retrieve_relevant_chunks(pergunta_usuario)
        logger.info(
            "Pergunta recebida: %s | chunks recuperados: %s",
            pergunta_usuario,
            len(chunks),
        )

        if not chunks:
            return ChatbotResponse(
                text=_fallback_without_context(pergunta_usuario),
                sources=[],
                used_context=False,
                retrieval_debug="no_relevant_chunks",
                interaction_mode="fallback",
            )

        prompt_final = montar_prompt_com_contexto(pergunta_usuario, chunks)
        answer = _send_message_with_retry(prompt_final).strip()
        answer_with_sources = f"{answer}{_format_sources_for_answer(chunks)}"
        return ChatbotResponse(
            text=answer_with_sources,
            sources=chunks,
            used_context=True,
            retrieval_debug="hybrid_retrieval",
            interaction_mode="retrieval",
        )
    except Exception as exc:
        logger.exception("Erro ao gerar resposta da IA.")
        return ChatbotResponse(
            text=f"Erro tecnico: {exc}",
            sources=[],
            used_context=False,
            retrieval_debug="error",
            interaction_mode="error",
        )


def pergunta_ia(pergunta_usuario: str) -> str:
    return responder_pergunta(pergunta_usuario).text


def obter_fontes_resposta(pergunta_usuario: str) -> list[dict[str, Any]]:
    response = responder_pergunta(pergunta_usuario)
    return [
        {
            "source": chunk.source,
            "page": chunk.page,
            "score": chunk.score,
            "retrieval": chunk.retrieval,
            "content": chunk.content,
        }
        for chunk in response.sources
    ]
