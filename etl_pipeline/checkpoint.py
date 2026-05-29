"""
pipeline/checkpoint.py
-----------------------
Save and resume ETL progress so large runs can be interrupted
and continued without reprocessing already-done cases.

How it works
------------
After each batch of cases is processed, the list of completed
Case_IDs is written to a JSON checkpoint file.
On the next run, already-completed IDs are skipped.

Public API
----------
    Checkpoint(path)
    checkpoint.is_done(case_id)   -> bool
    checkpoint.mark_done(case_id)
    checkpoint.load()             -> set of done IDs
    checkpoint.reset()            -> clears the checkpoint
    checkpoint.stats()            -> dict
"""

import json
import os
import time
from typing import Set
from utils.logger import get_logger

logger = get_logger(__name__)


class Checkpoint:
    """
    Lightweight file-backed checkpoint for ETL resume.

    Parameters
    ----------
    path : full path to the .json checkpoint file.
           Typically config.CHECKPOINT_DIR / "etl_checkpoint.json".
    """

    def __init__(self, path: str):
        self.path     = path
        self._done:   Set[str] = set()
        self._created = time.time()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.load()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_done(self, case_id: str) -> bool:
        return str(case_id) in self._done

    def mark_done(self, case_id: str) -> None:
        self._done.add(str(case_id))
        self._flush()

    def mark_done_batch(self, case_ids) -> None:
        """Mark multiple IDs as done in one write."""
        for cid in case_ids:
            self._done.add(str(cid))
        self._flush()

    def load(self) -> Set[str]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data      = json.load(f)
                    self._done = set(data.get("done", []))
                logger.info(
                    "Checkpoint loaded: %d cases already processed ← %s",
                    len(self._done), self.path,
                )
            except Exception as exc:
                logger.warning("Could not load checkpoint (%s) — starting fresh.", exc)
                self._done = set()
        return self._done

    def reset(self) -> None:
        """Delete the checkpoint file and clear in-memory state."""
        self._done = set()
        if os.path.exists(self.path):
            os.remove(self.path)
            logger.info("Checkpoint reset: %s deleted.", self.path)

    def stats(self) -> dict:
        return {
            "completed_cases": len(self._done),
            "checkpoint_file": self.path,
            "file_exists":     os.path.exists(self.path),
        }

    def __len__(self) -> int:
        return len(self._done)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"done": list(self._done)}, f)
        except Exception as exc:
            logger.error("Failed to write checkpoint: %s", exc)