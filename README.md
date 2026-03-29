# peri-bugi-ai-chat

AI Chat service untuk fitur **Tanya Peri** — chatbot kesehatan gigi anak dari Peri Bugi.

---

## Arsitektur

```
peri-bugi-web
     │  POST /api/v1/chat/message  (Bearer token)
     ▼
peri-bugi-api  ── auth + inject user context ──►  peri-bugi-ai-chat  (port 8003)
     │                                                   │
     │  ◄──── SSE stream (thinking/token/done) ──────────┘
     ▼
peri-bugi-web (render real-time)
```

`peri-bugi-ai-chat` adalah **private service** — tidak diakses langsung dari web/mobile.
Semua auth dan data user dihandle oleh `peri-bugi-api`.

---

## Stack

| Komponen | Teknologi |
|---|---|
| Framework | FastAPI + Uvicorn |
| Agent | LangGraph (simplified graph runner) |
| LLM | Pluggable: Ollama / Gemini / OpenAI |
| Vector Store | Qdrant |
| Embedding | sentence-transformers (local dev) |
| Deployment | Google Cloud Run |

---

## Struktur Repo

```
peri-bugi-ai-chat/
├── app/
│   ├── main.py                    # FastAPI app + /chat/stream endpoint
│   ├── config/
│   │   ├── settings.py            # Config via .env (pluggable LLM, Qdrant, dll)
│   │   └── llm.py                 # LLM factory — swap provider tanpa ubah agent
│   ├── schemas/
│   │   └── chat.py                # ChatRequest, AgentState, SSE event helpers
│   ├── agents/
│   │   ├── peri_agent.py          # Graph runner utama (router → tools → generate)
│   │   ├── nodes/
│   │   │   ├── router.py          # Hybrid intent classifier (rule-based + LLM fallback)
│   │   │   └── generate.py        # LLM response generation + streaming + clarification
│   │   └── tools/
│   │       ├── retrieve.py        # RAG dari Qdrant knowledge base
│   │       └── image.py           # Forward gambar ke peri-bugi-ai-cv
├── scripts/
│   └── ingest_pdf.py              # Upload PDF ke Qdrant vector store
├── Dockerfile
├── docker-compose.yml             # Local dev: ai-chat + Qdrant
├── requirements.txt
└── .env.example
```

---

## Setup Lokal

### 1. Copy .env

```bash
cp .env.example .env
```

Edit `.env`:

```env
INTERNAL_SECRET=<sama dengan AI_CHAT_INTERNAL_SECRET di peri-bugi-api>
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=gemma2:2b
```

### 2. Jalankan Docker

```bash
docker compose up -d
```

Ini jalankan:
- `ai-chat` di port 8003
- `qdrant` di port 6333

### 3. Update peri-bugi-api .env

```env
AI_CHAT_URL=http://host.docker.internal:8003
```

Lalu restart: `docker compose restart api`

### 4. Verifikasi

```bash
curl http://localhost:8003/health
# → {"status":"ok","service":"tanya-peri-ai-chat","provider":"ollama","model":"gemma2:2b"}
```

---

## LLM Provider

Ganti provider cukup dengan ubah `.env` — tidak perlu ubah code agent:

| Provider | Config |
|---|---|
| Ollama (local dev) | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=gemma2:2b` |
| Gemini (production) | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...` |
| OpenAI (production) | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` |

### Rekomendasi Model per Fase

| Fase | Provider | Model | Keterangan |
|---|---|---|---|
| Local dev | Ollama | `gemma2:2b` | Cepat, ringan, tidak ada thinking overhead |
| Local dev | Ollama | `qwen3.5` | Kualitas lebih baik, thinking mode auto-disabled |
| Staging | Gemini | `gemini-1.5-flash` | Gratis tier tersedia, cepat |
| Production | Gemini | `gemini-1.5-pro` | Kualitas terbaik untuk dental QA |

> **Note:** Model Ollama bertipe "thinking" (qwen3.5, deepseek-r1) sudah di-handle dengan
> `think: False` — thinking mode di-disable otomatis supaya TTFT tetap cepat.

---

## Agent Flow

```
[START]
   │
   ▼
[router]          ← Hybrid: rule-based keyword → fallback LLM classify
   │
   ├── dental_qa ──────► [retrieve] → [check_clarification] → [generate]
   ├── context_query ──► [check_clarification] → [generate]
   ├── image ──────────► [image_tool] → [generate]
   ├── clarification_answer → [generate]
   └── smalltalk ─────────────────────────────► [generate]
                                                      │
                                              needs_clarification?
                                              ├── yes → [emit_clarify] → END
                                              └── no  → [emit_done] → END
```

### Intent Types

| Intent | Trigger | Contoh |
|---|---|---|
| `dental_qa` | Kata kunci gigi, karies, sikat, dll | "Berapa kali sikat gigi?" |
| `context_query` | Tanya data user sendiri | "Berapa streak anak saya?" |
| `image` | Ada image_url atau kata "foto/gambar" | Kirim foto gigi |
| `clarification_answer` | Menjawab checkpoint klarifikasi | Pilih opsi a/b/c |
| `smalltalk` | Sapaan, basa-basi | "Halo", "Terima kasih" |

---

## SSE Event Types

Response di-stream sebagai Server-Sent Events. Format tiap event:
```json
data: {"event": "<type>", "data": <payload>}
```

| Event | Kapan | Payload |
|---|---|---|
| `thinking` | Saat agent processing | `{step, label, done}` |
| `token` | Tiap token LLM | `string` |
| `clarify` | Agent butuh klarifikasi | `{question, options, allow_multiple}` |
| `tool` | Log tool call | `{tool, input, result}` |
| `done` | Response selesai + tersimpan DB | `ChatMessageResponse` |
| `error` | Ada error | `string` |

---

## Prompt Management

Semua prompt disimpan di DB `peri-bugi-api` (tabel `prompt_templates`) — bisa diupdate tanpa redeploy.

| Key | Fungsi |
|---|---|
| `persona_system` | Karakter dan aturan dasar Tanya Peri |
| `router_classify` | Klasifikasi intent (fallback LLM) |
| `generate_dental_qa` | Generate jawaban dari knowledge base |
| `generate_with_context` | Generate jawaban dengan data user |
| `clarify_decision` | Tentukan kapan perlu klarifikasi |
| `web_search_query` | Buat search query dari pertanyaan user |

Update prompt via SQL:
```sql
UPDATE prompt_templates
SET content = '<prompt baru>', updated_at = NOW()
WHERE key = 'persona_system' AND is_active = true;
```

---

## Knowledge Base (RAG)

Ingest PDF ke Qdrant:

```bash
# Dari dalam folder peri-bugi-ai-chat
python scripts/ingest_pdf.py --pdf path/to/dental_knowledge.pdf

# Dengan custom chunk size
python scripts/ingest_pdf.py --pdf dental.pdf --chunk-size 400 --chunk-overlap 40
```

Qdrant collection: `peri_bugi_dental` (configurable via `QDRANT_COLLECTION` di .env)

---

## Deployment (Cloud Run)

```bash
# Build image
docker build -t asia-southeast2-docker.pkg.dev/peri-bugi-491218/peri-bugi/ai-chat:latest .

# Push
docker push asia-southeast2-docker.pkg.dev/peri-bugi-491218/peri-bugi/ai-chat:latest

# Deploy
gcloud run deploy peri-bugi-ai-chat \
  --image asia-southeast2-docker.pkg.dev/peri-bugi-491218/peri-bugi/ai-chat:latest \
  --region asia-southeast2 \
  --no-allow-unauthenticated \
  --min-instances 1 \
  --set-env-vars LLM_PROVIDER=gemini,GEMINI_MODEL=gemini-1.5-flash
```

> **Penting:** `--min-instances 1` untuk menghindari cold start (2-4 detik) yang terasa di chat.
> `--no-allow-unauthenticated` karena service ini private — hanya bisa diakses dari peri-bugi-api.

---

## Next Improvements

### Prioritas Tinggi

- [ ] **LLM Call Logging** — kirim metadata (token count, latency, model) ke `peri-bugi-api` untuk disimpan di tabel `llm_call_logs`. Data ini penting untuk monitoring cost dan performance.
- [ ] **Web Search Tool** — implementasi node web search untuk pertanyaan dental yang spesifik dan tidak ada di knowledge base. Pakai Google Search API atau SerpAPI.
- [ ] **Ingest PDF knowledge base** — upload dan ingest PDF panduan kesehatan gigi anak ke Qdrant supaya RAG aktif. Saat ini RAG berjalan tapi collection masih kosong.

### Prioritas Menengah

- [ ] **Write Actions** — arsitektur sudah siap (`tool_calls` di-log ke DB), tinggal implementasi. Use case pertama: booking Janji Peri dari chat. Agent emit tool call → `peri-bugi-api` eksekusi.
- [ ] **Image Input** — user bisa kirim foto gigi → agent forward ke `peri-bugi-ai-cv` → narasi hasil. `image.py` sudah ada tapi perlu disambungkan ke flow upload gambar di FE.
- [ ] **Session Context Persistence** — saat ini context window 20 pesan terakhir. Untuk conversation panjang, perlu summary/compression supaya context tidak hilang.
- [ ] **Streaming lebih stabil** — handle `GeneratorExit` dengan lebih graceful. Saat ini kalau FE tutup koneksi di tengah stream, response tersimpan dari `done` event (sudah fix) tapi ada edge case kalau `done` belum sempat dikirim.

### Prioritas Rendah / Future

- [ ] **Multi-language** — saat ini prompt dan response dalam Bahasa Indonesia. Support Bahasa Inggris untuk diaspora.
- [ ] **Voice input** — integrasi dengan Whisper untuk voice-to-text sebelum dikirim ke agent.
- [ ] **Proactive notifications** — agent bisa kirim reminder atau insight berdasarkan data brushing (misal: "Streak hampir putus!").
- [ ] **Doctor mode** — konteks dan persona berbeda untuk user dengan role `doctor`. Lebih klinis, bisa akses data pasien (dengan izin).
- [ ] **Feedback loop** — user bisa kasih thumbs up/down untuk response. Data ini dipakai untuk fine-tuning prompt.
- [ ] **A/B testing prompts** — tabel `prompt_templates` sudah support multi-version, tinggal implementasi logic untuk serve versi berbeda ke user yang berbeda.

---

## Troubleshooting

### Response kosong / tidak muncul di FE
1. Cek apakah Ollama jalan: `curl http://localhost:11434/api/tags`
2. Cek apakah model tersedia: nama model di `.env` harus sama persis dengan output `ollama list`
3. Model thinking (qwen3.5, deepseek-r1) butuh Ollama >= 0.6.0 untuk support `think: False`

### "All connection attempts failed" di FE
- ai-chat container mati → `docker compose up -d` dari folder `peri-bugi-ai-chat`
- `AI_CHAT_URL` di `peri-bugi-api/.env` salah → pastikan `http://host.docker.internal:8003`

### Response terpotong
- Sudah di-fix: content diambil dari `done` event bukan akumulasi proxy
- Kalau masih terjadi, cek `LLM_MAX_TOKENS` di `.env` (default 1024)

### Qdrant "Api key is used with an insecure connection" warning
- Normal untuk local dev — Qdrant lokal tidak pakai HTTPS
- Di production (Qdrant Cloud), set `QDRANT_API_KEY` dan gunakan URL HTTPS

---

## Environment Variables

```env
# App
APP_ENV=development              # development | production
APP_PORT=8003

# Security (wajib sama dengan peri-bugi-api)
INTERNAL_SECRET=<shared_secret>

# LLM
LLM_PROVIDER=ollama              # ollama | gemini | openai
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=gemma2:2b
GEMINI_API_KEY=
GEMINI_MODEL=gemini-1.5-pro
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024
LLM_TIMEOUT_SECONDS=60

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                  # kosong untuk local, isi untuk Qdrant Cloud
QDRANT_COLLECTION=peri_bugi_dental

# Embedding
EMBEDDING_PROVIDER=local         # local | gemini | openai
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# External
AI_CV_URL=                       # URL peri-bugi-ai-cv untuk image inference
```
