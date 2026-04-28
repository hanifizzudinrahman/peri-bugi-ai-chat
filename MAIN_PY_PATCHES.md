# PATCH: app/main.py untuk Phase 1

File `app/main.py` (954 baris) **TIDAK perlu di-replace full**. Cuma 2 patch kecil berikut:

---

## Patch 1: Update `/health/agents` endpoint (line 103-106)

**Cari**:
```python
@app.get("/health/agents")
async def health_agents():
    from app.agents.graph import _AGENT_REGISTRY
    return {"status": "ok", "registered_agents": list(_AGENT_REGISTRY.keys()), "agent_count": len(_AGENT_REGISTRY)}
```

**Ganti dengan**:
```python
@app.get("/health/agents")
async def health_agents():
    from app.agents.nodes.agent_dispatcher import _AGENT_REGISTRY
    return {"status": "ok", "registered_agents": list(_AGENT_REGISTRY.keys()), "agent_count": len(_AGENT_REGISTRY)}
```

**Why**: `_AGENT_REGISTRY` sekarang ada di `nodes/agent_dispatcher.py` (Phase 1 refactor),
bukan lagi di `graph.py` (yang sekarang slim — hanya entry points).

---

## Patch 2: Tambahin startup/shutdown hooks + checkpointer health (di akhir file, sebelum `if __name__ == "__main__":` jika ada, atau di mana saja sebelum end of file)

**Tambahin block ini** (cari posisi yang cocok, idealnya setelah `/health/llm` endpoint):

```python
# =============================================================================
# Phase 1 — LangGraph checkpointer lifecycle hooks
# =============================================================================


@app.on_event("startup")
async def startup_checkpointer():
    """Warm up LangGraph PostgresSaver checkpointer pool."""
    try:
        from app.agents.memory.checkpointer import get_checkpointer
        cp = await get_checkpointer()
        if cp is not None:
            log.info("checkpointer_ready", status="ok")
        else:
            log.warning("checkpointer_unavailable",
                       status="degraded",
                       msg="Tanya Peri akan jalan tanpa state persistence")
    except Exception as e:
        log.error("checkpointer_init_failed", error=str(e))


@app.on_event("shutdown")
async def shutdown_checkpointer_hook():
    """Cleanly close checkpointer pool."""
    try:
        from app.agents.memory.checkpointer import shutdown_checkpointer
        await shutdown_checkpointer()
        log.info("checkpointer_shutdown", status="ok")
    except Exception as e:
        log.error("checkpointer_shutdown_failed", error=str(e))


@app.get("/health/checkpointer")
async def health_checkpointer():
    """Health check khusus untuk LangGraph checkpointer (Phase 1)."""
    from app.agents.memory.checkpointer import checkpointer_healthy
    healthy = await checkpointer_healthy()
    return {
        "status": "ok" if healthy else "degraded",
        "checkpointer_healthy": healthy,
        "note": (
            "Checkpointer healthy = state persistence aktif."
            if healthy
            else "Checkpointer not available — chat masih jalan tanpa persistence."
        ),
    }
```

---

## Verifikasi

Setelah apply 2 patch di atas, restart container:

```bash
docker compose restart peri_bugi_ai_chat
```

Expected logs di startup:
```
[checkpointer] AsyncPostgresSaver ready — pool min=2 max=10 db=peri_bugi
checkpointer_ready  status=ok
```

Test health endpoints:
```bash
curl http://localhost:8003/health/checkpointer
# Expected: {"status":"ok","checkpointer_healthy":true,...}

curl http://localhost:8003/health/agents
# Expected: {"status":"ok","registered_agents":["kb_dental","user_profile",...],"agent_count":6}
```

---

## Rollback

Kalau ada masalah, revert kedua patch:
- Patch 1: balik import ke `from app.agents.graph import _AGENT_REGISTRY`
- Patch 2: hapus 3 functions `startup_checkpointer`, `shutdown_checkpointer_hook`, `health_checkpointer`
