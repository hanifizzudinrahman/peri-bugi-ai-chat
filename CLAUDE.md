# Peri Bugi — AI Chat (`peri-bugi-ai-chat`)

FastAPI + **LangGraph** multi-agent backend di balik **Tanya Peri** — asisten
kesehatan gigi anak untuk orang tua. Menjawab pertanyaan, membaca hasil scan
Mata Peri, dan menjaga nada bicara ramah-orang-tua dalam Bahasa Indonesia.

Dipanggil **hanya oleh `peri-bugi-api`**, tidak pernah langsung dari browser/mobile.

## Stack
- **FastAPI** + Uvicorn, async, response **SSE streaming**
- **LangGraph** — graph node/edge, checkpointer di PostgreSQL
- **Gemini 2.5 Flash** sebagai LLM utama
- **Qdrant** untuk knowledge base dental (RAG)
- **Langfuse** untuk tracing LLM
- Pydantic v2 · pytest (`asyncio_mode=auto`)

## Commands
```powershell
docker compose up -d --build      # ai-chat + qdrant
docker compose logs ai-chat -f
pytest
```
Port lokal **8003**, Qdrant **6333**. `/docs` untuk Swagger.

## Project layout
```
app/
  main.py                 app factory, SSE endpoint, middleware
  agents/
    graph.py              definisi graph LangGraph — mulai dari sini
    builder.py            perakitan graph
    peri_agent.py         agen utama
    nodes/
      pre_router.py       klasifikasi awal sebelum routing
      router.py           menentukan cabang
      agent.py            (~31KB) node agen
      agent_dispatcher.py memanggil sub-agent
      generate.py         (~79KB) penyusunan jawaban akhir — file terbesar
      tools_node.py       eksekusi tool
      tool_bridge.py      jembatan ke tool eksternal
    sub_agents/           agen khusus per domain
    tools/                definisi tool
    memory/               memori percakapan
  services/llm_logger.py  pencatatan input/output token per panggilan LLM
  config/ middleware/ schemas/
```

**Mulai dari `agents/graph.py`** untuk memahami alur. `generate.py` sangat besar —
baca bagian yang relevan saja, jangan dibaca utuh.

## Konvensi (verifiable)
- **Streaming SSE**: event `thinking`, `tool`, `token`, `clarify`, `done`.
  Payload `done` berisi pesan final lengkap dengan `metadata`.
- **`metadata.image_artifacts` harus ikut di event `done`** — dari situ web
  mengambil URL gambar overlay hasil analisa foto. Kalau hilang, gambar tidak
  muncul di chat walaupun analisanya sukses.
- **`metadata.llm_call_logs`** mencatat input/output token untuk dashboard biaya.
  Jangan dihapus saat merapikan payload.
- **Checkpointer LangGraph** memakai PostgreSQL. Perhatikan penyusunan connection
  string — karakter khusus pada password pernah memutus koneksi psycopg.
- Nada jawaban: Bahasa Indonesia, ramah orang tua, panggil "Bunda". **Disclaimer
  wajib**: hasil AI adalah screening awal, bukan diagnosis dokter. Angka dan
  disclaimer tidak boleh diubah LLM — hanya boleh di-*rephrase*.

## Integrasi dengan Mata Peri
Tool `analyze_chat_image` meneruskan foto ke `peri-bugi-ai-cv` lewat `peri-bugi-api`.
Hasilnya masuk sebagai konteks terstruktur (`structured_report`) yang di-*rephrase*
LLM. Kalau foto tidak jelas tampak bagian mana, graph mengembalikan event
`clarify` dengan pilihan (depan/atas/bawah/kiri/kanan) — bukan langsung menjawab.

## Codebase queries — pakai graphify dulu (hemat token)
`graphify-out/graph.json` ada di repo ini (code knowledge-graph via AST, di-gitignore).
Untuk pertanyaan arsitektur / "apa yang connect ke X" / "apa yang manggil Y" / cek
**blast-radius sebelum ngubah sesuatu** — query graph dulu, jangan langsung grep:
`graphify explain "<Nama>"` · `graphify path "<A>" "<B>"` · atau skill `/graphify query "..."`.
Berguna buat maping alur LangGraph (`agents/graph.py`, `generate.py` yang besar).
Rebuild kalau kode berubah / curiga stale (nol token, nol LLM): `graphify update .`.
Doc lengkap + benchmark + batasan: workspace `docs/GRAPHIFY.md`.

## Non-negotiable working rules
1. **Don't change what's already stable and working** without explicit approval.
2. **Read before write** — jangan menebak nama node, state key, atau bentuk event.
3. **Additive-only patches** — tambah di samping, jangan memutus alur stabil.
4. **Confirm scope before coding**; tunggu "gas"/"lanjut".
5. **Investigate before coding** — konfirmasi dengan log/trace Langfuse, bukan tebakan.
6. **Jangan ubah angka atau disclaimer medis.** LLM boleh mengubah kalimat,
   tidak boleh mengubah fakta.

## Deploy
Baca `../docs/BRANCHING_AND_DEPLOY.md`. Repo ini pernah punya branch
`langfuse-integration` yang hidup 37 commit di depan `main` sementara produksi
jalan dari branch itu — sudah disatukan 2026-07-18. Jangan ulangi: kerja di
branch pendek, merge ke `main`, deploy dari `main`.

## Secrets — never commit
`.env` (kunci Gemini, Qdrant, Langfuse, DB). Gitignored, biarkan begitu.
