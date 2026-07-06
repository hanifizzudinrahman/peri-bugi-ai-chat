"""
PostgresSaver checkpointer untuk LangGraph (Phase 1).

Pattern:
- Singleton instance per process (lazy init di first call).
- Connection pool terpisah dari main SQLAlchemy pool (avoid contention).
- Setup tables: dilakukan SEKALI di startup hook ai-chat via ensure_tables_exist().
  LangGraph .setup() idempotent — aman dipanggil berkali-kali.
- Graceful degradation: kalau init gagal, log warning dan return None
  sehingga graph compile tetap jalan (tanpa persistence).

Usage:
    from app.agents.memory.checkpointer import get_checkpointer
    cp = await get_checkpointer()
    if cp:
        graph = builder.compile(checkpointer=cp)
    else:
        graph = builder.compile()  # in-memory only

Phase 1: digunakan sebagai persistence layer untuk per-thread state.
Phase 2: Phase 3+ akan leverage `aget_state()` untuk resume conversation.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)


# Module-level singletons
_checkpointer_instance: Optional[object] = None
_pool_instance: Optional[object] = None
_init_attempted: bool = False
_init_succeeded: bool = False
_tables_setup_done: bool = False


def _build_conn_string() -> str:
    """Build psycopg connection string from settings.

    Pakai libpq conninfo format (key=value) — lebih robust dari URL format
    untuk username/password yang mengandung karakter spesial (seperti `.`
    di `postgres.project_ref` Supabase pooler, atau `+` `/` di base64 password).

    Semua value di-strip() untuk hindari trailing whitespace/newline yang
    kadang muncul saat secret mount ke env var di Cloud Run.

    Format: postgresql://user:pass@host:port/dbname
    NOTE: tidak pakai +asyncpg suffix — langgraph PostgresSaver pakai psycopg.
    """
    # Strip semua value — defensive terhadap trailing newline dari secret mount
    user = settings.DB_USER.strip()
    password = settings.DB_PASSWORD.strip()
    host = settings.DB_HOST.strip()
    port = settings.DB_PORT
    dbname = settings.DB_NAME.strip()

    # Libpq conninfo format — key=value, value di-escape kalau ada spesial chars
    # Lihat: https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
    def _escape(val: str) -> str:
        """Escape value untuk libpq connstring (wrap in quotes kalau ada special chars)."""
        val = str(val)
        if not val:
            return "''"
        if any(c in val for c in " \\'"):
            escaped = val.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"
        return val

    return (
        f"postgresql://{_escape(user)}:{_escape(password)}"
        f"@{_escape(host)}:{port}/{_escape(dbname)}"
    )


async def _ensure_tables_exist() -> bool:
    """
    Ensure LangGraph checkpoint tables exist in DB.

    Idempotent: aman dipanggil berkali-kali. LangGraph PostgresSaver.setup()
    handle versioning via checkpoint_migrations table internal.

    Returns True kalau setup sukses (atau tables sudah ada dari sebelumnya).
    """
    global _tables_setup_done
    if _tables_setup_done:
        return True

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as e:
        logger.error(
            f"[checkpointer] Package langgraph-checkpoint-postgres belum terinstall: {e}. "
            f"Rebuild Docker image dengan requirements yang updated."
        )
        return False

    try:
        conn_str = _build_conn_string()

        # Pakai context manager from_conn_string untuk setup ringan (no pool)
        async with AsyncPostgresSaver.from_conn_string(conn_str) as saver:
            await saver.setup()

        _tables_setup_done = True
        logger.info(
            "[checkpointer] LangGraph checkpoint tables ready "
            "(checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations)"
        )
        return True

    except Exception as e:
        logger.error(
            f"[checkpointer] Setup tables failed: {type(e).__name__}: {e}",
            exc_info=True,
        )
        return False


async def get_checkpointer():
    """
    Return AsyncPostgresSaver singleton.

    Returns None kalau init gagal — caller handle graceful (compile without
    checkpointer, in-memory only). Tidak raise.

    Behavior:
    - First call: ensure tables exist → init pool + saver
    - Subsequent calls: return cached instance
    - Init gagal: log warning + return None (idempotent — won't retry every call)
    """
    global _checkpointer_instance, _pool_instance, _init_attempted, _init_succeeded

    if _init_attempted:
        return _checkpointer_instance

    _init_attempted = True

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool
    except ImportError as e:
        logger.error(
            f"[checkpointer] Required packages missing: {e}. "
            f"Install: pip install langgraph-checkpoint-postgres 'psycopg[binary,pool]'"
        )
        return None

    # Step 1: ensure tables exist (idempotent)
    tables_ok = await _ensure_tables_exist()
    if not tables_ok:
        logger.warning(
            "[checkpointer] Tables setup gagal — Tanya Peri akan jalan tanpa persistence."
        )
        return None

    # Step 2: build connection pool + saver
    try:
        conn_str = _build_conn_string()

        _pool_instance = AsyncConnectionPool(
            conninfo=conn_str,
            min_size=2,
            max_size=10,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,  # untuk pgbouncer compat (kalau di-pakai future)
                "row_factory": _get_row_factory(),
            },
            open=False,
        )
        await _pool_instance.open()

        _checkpointer_instance = AsyncPostgresSaver(_pool_instance)
        _init_succeeded = True

        logger.info(
            f"[checkpointer] AsyncPostgresSaver ready — "
            f"pool min=2 max=10 db={settings.DB_NAME}"
        )
        return _checkpointer_instance

    except Exception as e:
        logger.warning(
            f"[checkpointer] Init failed: {type(e).__name__}: {e}. "
            f"Tanya Peri akan jalan tanpa checkpointer (in-memory only). "
            f"Cek: (1) DB reachable? (2) Migration 029 sudah jalan? "
            f"(3) langgraph-checkpoint-postgres installed?"
        )
        return None


def _get_row_factory():
    """psycopg row factory — dict_row supaya kompatibel dengan langgraph."""
    try:
        from psycopg.rows import dict_row
        return dict_row
    except ImportError:
        return None


async def shutdown_checkpointer():
    """Cleanup — call dari FastAPI shutdown event."""
    global _pool_instance, _checkpointer_instance
    if _pool_instance is not None:
        try:
            await _pool_instance.close()
        except Exception as e:
            logger.warning(f"[checkpointer] Pool close error: {e}")
    _pool_instance = None
    _checkpointer_instance = None


async def checkpointer_healthy() -> bool:
    """Health check — return True kalau pool ready & DB reachable."""
    if _pool_instance is None:
        return False
    try:
        async with _pool_instance.connection() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False
