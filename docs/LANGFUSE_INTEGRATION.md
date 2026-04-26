# Langfuse Observability Integration

Dokumentasi integrasi Langfuse di `peri-bugi-ai-chat`.

## Apa Itu Langfuse di Sini

Langfuse adalah platform observability untuk LLM apps. Di Peri Bugi, Langfuse
digunakan untuk:

- **Trace setiap message di Tanya Peri** — visualisasi multi-agent flow
  (router → supervisor → sub_agents → generate)
- **Token usage + cost** per message, per user, per session
- **Latency breakdown** per node di graph
- **Replay & debug** trace yang error
- **Filter & search** trace berdasarkan user_id, session_id, agent name, tags

## Architecture: Factory Wrap Pattern

Setiap LLM yang di-return dari `get_llm()` (di `app/config/llm.py`) **otomatis
di-wrap** dengan Langfuse callback handler kalau `LANGFUSE_ENABLED=true`.
Tidak perlu modifikasi setiap LLM call site untuk basic tracing.

```
┌─────────────────┐
│ get_llm()       │  → ChatOllama / ChatGemini / ChatOpenAI (raw)
│  (factory)      │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────┐
│ _attach_observability(llm)      │  → llm.with_config({"callbacks": [handler]})
│   (silent if disabled)          │
└────────┬────────────────────────┘
         │
         ↓
┌─────────────────┐
│ Caller code     │  → llm.ainvoke([msg], config=trace_config)
│ di agent        │     trace_config = build_trace_config(state, agent_name)
└─────────────────┘
```

**Result**: Setiap LLM call **otomatis ke-trace**. Plus per-call metadata
(session_id, user_id, agent name) di-attach via `build_trace_config()`.

## Activation

### Step 1: Set Environment Variables

Edit `.env`:

```env
# Aktifkan
LANGFUSE_ENABLED=true

# Keys dari Langfuse UI (Settings → API Keys → Create new keys)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx

# Internal Docker network URL (untuk container ai-chat)
# Tidak perlu diubah untuk dev local — sudah benar
LANGFUSE_HOST=http://langfuse-web:3000
```

### Step 2: Restart Container

```powershell
cd peri-bugi-ai-chat
docker compose up -d --build ai-chat
```

⚠️ Pakai `--build` kalau ini install langfuse package pertama kali.
Build sekitar 3-5 menit (download package).

### Step 3: Verify

Send 1 test message ke Tanya Peri (misal via UI di `peri-bugi-web` atau via
Swagger di `localhost:8003/docs`). Lalu buka Langfuse dashboard:

```
http://localhost:3100 → Project tanya-peri-dev → Tab "Traces"
```

Akan muncul trace baru dalam beberapa detik.

## Deactivation

3 cara, pilih yang paling fit:

### A. Disable Total (eksplisit, restart)

```env
LANGFUSE_ENABLED=false
```

```powershell
docker compose restart ai-chat
```

Trace stop, zero overhead. **Recommended saat focus debug agent logic.**

### B. Stop Langfuse Stack

```powershell
cd peri-bugi-langfuse
docker compose down
```

ai-chat **tetap jalan normal** — graceful degradation. Trace di-drop diam-diam.
Berguna untuk hemat RAM saat tidak butuh observability.

> **Note**: Langfuse SDK menjalankan retry async di background. Hanif akan lihat
> warning log ringan tiap request, tapi tidak block app.

### C. Hapus Keys

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

Saat keys empty, `get_langfuse_handler()` return None → no-op. Sama efek-nya
dengan A, tapi A lebih eksplisit.

## What Gets Traced

| Component | Auto-trace? | Notes |
|---|---|---|
| `llm.ainvoke()` di router | ✅ | agent="router", metadata: session_id, user_id |
| `llm.astream()` di generate | ✅ | agent="generate", per-token streaming |
| `llm.ainvoke()` di supervisor | ✅ | agent="supervisor" |
| `llm.ainvoke()` di view_hint_detector | ✅ | agent="view_hint_detector" |
| `llm.ainvoke()` di memory_job | ✅ | agent="memory_summary" (background task) |
| `retriever.ainvoke()` Qdrant kb_dental | ✅ | agent="kb_dental_retriever" |
| `retriever.ainvoke()` Qdrant app_faq | ✅ | agent="app_faq_retriever" |
| HTTP calls ke peri-bugi-api (phase2 agents) | ❌ | Skip Phase 1 — bisa ditambah nanti |

## Per-Call Metadata Attached

Setiap trace dapat metadata berikut (kalau state available):

| Field | Value | Source |
|---|---|---|
| `session_id` | UUID chat session | `state["session_id"]` |
| `user_id` | UUID user | `state["user_context"]["user_id"]` |
| `trace_id` | X-Request-ID dari header | `state["trace_id"]` |
| `chat_message_id` | UUID chat_messages.id | `state["chat_message_id"]` |
| `response_mode` | simple/medium/detailed | `state["response_mode"]` |
| `agent` | router/supervisor/dll | hardcoded di each call site |
| Tags | `agent:<name>`, `mode:<mode>` | auto-build |

Filter trace di Langfuse UI berdasarkan field ini untuk debug specific scenario.

## Adding Tracing untuk LLM Call Site Baru

Pattern untuk LLM call yang baru ditambahkan setelah ini:

### Auto-Tracing (Minimum, Recommended)

```python
from app.config.llm import get_llm

# Cuma get_llm() — otomatis ke-wrap kalau Langfuse enabled
llm = get_llm(temperature=0.5, max_tokens=500)
result = await llm.ainvoke([HumanMessage(content=prompt)])
```

Trace akan masuk dengan agent name = empty (cuma muncul sebagai "ChatModel").

### Auto-Tracing + Per-Call Metadata (Recommended untuk Agent)

```python
from app.config.llm import get_llm
from app.config.observability import build_trace_config

llm = get_llm(temperature=0.5, max_tokens=500)
trace_config = build_trace_config(state=state, agent_name="my_new_agent")
result = await llm.ainvoke([HumanMessage(content=prompt)], config=trace_config)
```

Trace masuk dengan metadata lengkap (session, user, agent name, tags).
Pattern ini yang dipakai di semua existing call sites.

### Pattern untuk Function Tanpa State

Kalau function tidak punya `AgentState` (e.g. background job, util function),
build pseudo-state:

```python
trace_config = build_trace_config(
    state={"session_id": session_id, "user_context": {"user_id": user_id}},
    agent_name="my_background_job",
)
```

Lihat `app/agents/memory_job.py` sebagai contoh.

## Troubleshooting

### Trace tidak muncul di Langfuse UI

Cek urutan ini:

1. **Verify enabled**:
   ```powershell
   docker exec peri_bugi_ai_chat python -c "from app.config.settings import settings; print('enabled:', settings.LANGFUSE_ENABLED, 'keys:', bool(settings.LANGFUSE_PUBLIC_KEY), bool(settings.LANGFUSE_SECRET_KEY))"
   ```
   Expected: `enabled: True keys: True True`

2. **Verify network reach**:
   ```powershell
   docker exec peri_bugi_ai_chat python -c "import urllib.request; print(urllib.request.urlopen('http://langfuse-web:3000/api/public/health', timeout=5).read().decode())"
   ```
   Expected: `{"status":"OK","version":"3.x.x"}`

3. **Verify init success**:
   ```powershell
   docker compose logs ai-chat --tail=30 | Select-String "Langfuse"
   ```
   Expected log: `INFO Langfuse observability ENABLED — host=http://langfuse-web:3000`

4. **Verify keys correct di Langfuse UI** — coba regenerate keys, update .env, restart.

### Error "ImportError: No module named 'langfuse.langchain'"

Kemungkinan version langfuse < 3.0. Update:

```powershell
docker compose up -d --build ai-chat
```

Build akan reinstall dari `requirements-base.txt` (`langfuse>=3.0.0,<4.0.0`).

### App lambat / latency naik banyak

Langfuse SDK pakai async batching di background, tidak block request.
Kalau Hanif lihat latency naik:

1. Check log warning — kemungkinan Langfuse unreachable (timeout retry)
2. Disable sementara dengan `LANGFUSE_ENABLED=false`
3. Verify `LANGFUSE_HOST` benar (gunakan `http://langfuse-web:3000` untuk
   container, BUKAN `http://localhost:3100`)

### Warning log spam saat Langfuse offline

Itu normal — graceful degradation. Untuk silence:

```env
LANGFUSE_ENABLED=false
```

## Reference

- Langfuse SDK v3 docs: https://langfuse.com/docs/observability/sdk/python
- Langchain integration: https://langfuse.com/integrations/frameworks/langchain
- Self-hosted setup: https://langfuse.com/self-hosting
- Hanif's Langfuse instance: http://localhost:3100
