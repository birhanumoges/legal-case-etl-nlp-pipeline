"""utils/db_utils.py — Thin SQLAlchemy helpers."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from config import DB_URL
from utils.logger import get_logger

logger = get_logger(__name__)


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True, echo=False)


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal()


def execute_sql(sql: str, params: dict | None = None):
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        conn.commit()
        return result
