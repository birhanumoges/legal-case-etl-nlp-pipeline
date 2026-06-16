"""rag/retriever.py — Vector search retrieval for RAG."""

from __future__ import annotations
from typing import List, Dict
from vectorstore.similarity_search import SimilaritySearch
from utils.logger import get_logger

logger = get_logger(__name__)


class LegalRetriever:
    def __init__(self):
        self.search = SimilaritySearch()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        return self.search.search(query, top_k=top_k)

    def format_context(self, docs: List[Dict]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            hdr  = f"[Case {i}] ID={doc['id']}  Score={doc['score']:.3f}"
            if meta.get("Case_Type"):
                hdr += f"  Type={meta['Case_Type']}  Verdict={meta.get('Verdict','?')}"
            parts.append(f"{hdr}\n{doc['text']}")
        return "\n\n".join(parts)
