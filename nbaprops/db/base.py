"""Engine and session helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from nbaprops.settings import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, pool_pre_ping=True)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on error."""
    session = sessionmaker(bind=get_engine(url))()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
