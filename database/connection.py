"""database/connection.py — PostgreSQL connection via SQLAlchemy."""

from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config import DB_URL
from database.models import Base
from utils.logger import get_logger

logger = get_logger(__name__)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True, echo=False, pool_size=5)
        logger.info("DB engine created: %s", DB_URL.split("@")[-1])
    return _engine


def get_session() -> Session:
    SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return SessionLocal()


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=get_engine())
    logger.info("Database tables created.")
