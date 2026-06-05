"""database/models.py — SQLAlchemy ORM models for legal cases."""

from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Index
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class LegalCase(Base):
    __tablename__ = "legal_cases"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    case_id              = Column(String(64),  nullable=False, index=True, unique=True)
    case_name            = Column(String(512), nullable=True)
    source_folder        = Column(String(128), nullable=True)
    year                 = Column(String(8),   nullable=True, index=True)
    court                = Column(String(256), nullable=True, index=True)
    case_text            = Column(Text,        nullable=True)
    case_text_length     = Column(Integer,     nullable=True)

    # Raw extracted labels
    verdict              = Column(String(128), nullable=True)
    case_type            = Column(String(64),  nullable=True)
    sub_type             = Column(String(64),  nullable=True)

    # Mapped / canonical labels
    verdict_mapped       = Column(String(32),  nullable=True, index=True)
    case_type_mapped     = Column(String(32),  nullable=True, index=True)
    sub_type_mapped      = Column(String(64),  nullable=True, index=True)

    # Features
    num_citations        = Column(Integer,     nullable=True, default=0)
    legal_citations      = Column(Text,        nullable=True)
    statutes             = Column(Text,        nullable=True)

    # Engineered
    text_length          = Column(Integer,     nullable=True)
    word_count           = Column(Integer,     nullable=True)
    year_numeric         = Column(Float,       nullable=True)

    created_at           = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_year_type", "year", "case_type_mapped"),
    )

    def __repr__(self):
        return f"<LegalCase id={self.case_id} type={self.case_type_mapped} verdict={self.verdict_mapped}>"
