"""extractors/case_chunker.py — Split case text into overlapping chunks for RAG/embeddings."""

from __future__ import annotations
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

logger = get_logger(__name__)


class CaseChunker:
    """
    Split long case opinion text into fixed-size overlapping word chunks
    suitable for vector embedding and RAG retrieval.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, case_id: str, metadata: dict | None = None) -> List[Dict]:
        """
        Returns a list of chunk dicts:
          { chunk_id, case_id, chunk_index, text, word_start, word_end, metadata }
        """
        words = text.split()
        if not words:
            return []

        chunks: List[Dict] = []
        step = max(1, self.chunk_size - self.overlap)
        idx = 0
        chunk_index = 0

        while idx < len(words):
            end = min(idx + self.chunk_size, len(words))
            chunk_words = words[idx:end]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{case_id}_chunk_{chunk_index:04d}",
                "case_id": case_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "word_start": idx,
                "word_end": end,
                "metadata": metadata or {},
            })
            chunk_index += 1
            if end == len(words):
                break
            idx += step

        return chunks

    def chunk_batch(self, records: List[Dict]) -> List[Dict]:
        """Chunk a list of records, each having 'Case_ID', 'Case_Text', and optional metadata."""
        all_chunks: List[Dict] = []
        for rec in records:
            meta = {k: rec.get(k) for k in ("Case_Type", "Sub_Type", "Verdict", "Court", "Year")}
            chunks = self.chunk(rec.get("Case_Text", ""), rec.get("Case_ID", "unknown"), meta)
            all_chunks.extend(chunks)
        logger.info("Chunked %d records → %d chunks", len(records), len(all_chunks))
        return all_chunks
