"""database/repository.py — CRUD operations for LegalCase."""

from __future__ import annotations
import pandas as pd
from sqlalchemy.orm import Session
from database.models import LegalCase
from database.connection import get_session
from utils.logger import get_logger

logger = get_logger(__name__)


class LegalCaseRepository:

    def __init__(self, session: Session | None = None):
        self.session = session or get_session()

    # ── create ───────────────────────────────────────────────────────
    def bulk_insert_from_df(self, df: pd.DataFrame, batch_size: int = 500):
        col_map = {
            "Case_ID": "case_id", "Case_Name": "case_name",
            "Source_Folder": "source_folder", "Year": "year",
            "Court": "court", "Case_Text": "case_text",
            "Case_Text_Full_Length": "case_text_length",
            "Verdict": "verdict", "Case_Type": "case_type", "Sub_Type": "sub_type",
            "Verdict_Mapped": "verdict_mapped",
            "Case_Type_Mapped": "case_type_mapped", "Sub_Type_Mapped": "sub_type_mapped",
            "Num_Citations": "num_citations", "Legal_Citations": "legal_citations",
            "Statutes": "statutes",
            "text_length": "text_length", "word_count": "word_count",
            "Year_Numeric": "year_numeric",
        }
        records_added = 0
        batch = []
        for _, row in df.iterrows():
            kwargs = {v: row.get(k) for k, v in col_map.items() if k in df.columns}
            batch.append(LegalCase(**kwargs))
            if len(batch) >= batch_size:
                self._flush(batch)
                records_added += len(batch)
                batch = []
        if batch:
            self._flush(batch)
            records_added += len(batch)
        logger.info("Inserted %d records into DB", records_added)

    def _flush(self, batch: list):
        try:
            self.session.bulk_save_objects(batch)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error("DB insert error: %s", e)

    # ── read ─────────────────────────────────────────────────────────
    def get_by_id(self, case_id: str) -> LegalCase | None:
        return self.session.query(LegalCase).filter_by(case_id=case_id).first()

    def get_all(self) -> list[LegalCase]:
        return self.session.query(LegalCase).all()

    def count(self) -> int:
        return self.session.query(LegalCase).count()

    def close(self):
        self.session.close()
