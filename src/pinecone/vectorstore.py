import json
import re
import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langsmith import traceable

from src.exception.exception_handler import AppException
from src.logger.logging import logging


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "local_kb"


def _safe_index_name(index_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", (index_name or "").strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "default"


def _index_path(index_name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{_safe_index_name(index_name)}.json"


def _tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    ascii_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", lowered)
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text or "")

    synonym_map = {
        "rd": ["研发部", "research", "development"],
        "r&d": ["研发部", "research", "development"],
        "research-and-development": ["研发部"],
        "employee": ["员工", "人数"],
        "employees": ["员工", "人数"],
        "count": ["数量", "人数"],
        "headcount": ["人数", "员工"],
        "department": ["部门"],
    }

    ordered: list[str] = []
    seen = set()
    for token in ascii_tokens + chinese_tokens:
        clean = token.strip()
        if len(clean) < 2 or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)

        for synonym in synonym_map.get(clean.lower(), []):
            if synonym not in seen:
                seen.add(synonym)
                ordered.append(synonym)

    lowered_joined = " ".join(ascii_tokens)
    if "r&d" in lowered_joined or "research and development" in (text or "").lower():
        for synonym in ["研发部", "部门", "员工", "人数"]:
            if synonym not in seen:
                seen.add(synonym)
                ordered.append(synonym)

    return ordered


def _score_chunk(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    haystack = (content or "").lower()
    score = 0.0
    for token in query_tokens:
        token_lower = token.lower()
        if token_lower in haystack:
            score += 1.0
            if len(token_lower) > 4:
                score += 0.2
    return score


class LocalRetriever:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.path = _index_path(index_name)

    def _load_chunks(self) -> list[dict]:
        if not self.path.exists():
            return []

        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        chunks = payload.get("chunks", [])
        return chunks if isinstance(chunks, list) else []

    def invoke(self, query: str, k: int = 4):
        chunks = self._load_chunks()
        ranked = []
        for chunk in chunks:
            content = str(chunk.get("page_content", ""))
            score = _score_chunk(query, content)
            if score <= 0:
                continue
            ranked.append((score, content, chunk.get("metadata", {})))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            Document(page_content=content, metadata=metadata or {})
            for _, content, metadata in ranked[:k]
        ]


def get_retriever(index_name: str):
    return LocalRetriever(index_name)


@traceable(name="save_doc_to_local_kb")
def add_doc_to_vectorstore(index_name: str, content: str):
    """
    Saves PDF-derived chunks into a local JSON knowledge base.
    """
    if content is None:
        raise ValueError("No content found to add in the knowledge base")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "],
            add_start_index=True,
        )
        documents = text_splitter.create_documents(texts=[content])

        payload = {
            "index_name": _safe_index_name(index_name),
            "chunks": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
        }

        output_path = _index_path(index_name)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logging.info("Saved document chunks to local knowledge base successfully")

    except Exception as e:
        logging.error(f"Error during document chunking and saving in local knowledge base: {e}")
        raise AppException(e, sys)
