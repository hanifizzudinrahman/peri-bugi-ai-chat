# Peri Bugi — AI Chat (`peri-bugi-ai-chat`)

FastAPI + **LangGraph** multi-agent backend di balik **Tanya Peri** — asisten
kesehatan gigi anak untuk orang tua. Menjawab pertanyaan, membaca hasil scan
Mata Peri, dan menjaga nada bicara ramah-orang-tua dalam Bahasa Indonesia.

Dipanggil **hanya oleh `peri-bugi-api`**, tidak pernah langsung dari browser/mobile.

## Stack
- **FastAPI** + Uvicorn, async, response **SSE streaming**
- **LangGraph** — graph node/edge, checkpointer di PostgreSQL
- **Gemini** sebagai LLM utama — model aktifnya dari `GEMINI_MODEL` di `.env`,
  saat ini `gemini-3.1-flash-lite` (menang telak di perbandingan 2 Agustus 2026:
  19/20 lawan 16/20 `gemini-3.6-flash`, separuh token, dua pertiga waktu)
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

## Text-to-SQL (`data_query`)
Selain 14 tool tetap, ada satu tool analitik `query_family_data`
(`agents/tools/data_query.py`) yang menulis SQL dari pertanyaan orang tua. SQL-nya
divalidasi dan dijalankan di `peri-bugi-api` — service ini tidak pernah menyentuh
tabel bisnis. Dua kunci: izin agent `data_query` **dan** `data_strategy`
(`tools` | `hybrid` | `sql`) yang dikirim api per request. Default `tools` =
perilaku lama persis.

Kalau menambah dataset, yang diedit **katalog di peri-bugi-api**, bukan prompt di
sini. Eval: `evals/nl_query/run.py`. Selengkapnya: workspace `docs/TEXT2SQL.md`.

## Tanya Data Founder (`/founder-analytics/stream`)
Alur KEDUA di repo ini, **terpisah total** dari graph chat: endpoint sendiri,
state sendiri, prompt sendiri, nol node dipakai bersama
(`app/agents/founder_analytics/`). Bentuknya generator async, bukan
`StateGraph` — pipanya lurus dengan satu simpul perbaikan, tanpa checkpointer,
dan tiap langkah harus memancarkan event SSE saat itu juga.

    plan -> sql -> exec -> (perbaiki, maks 3) -> chart -> jawab

Model **tidak menulis Vega-Lite**, cuma niat berbatas (`ChartIntent`); server
yang mengompilasinya, dan hasilnya tanpa `data`/`config`/`width` — ketiganya
ditambahkan `peri-bugi-web` yang memang punya token brand.

Kalau menambah node: `llm_call_logs` wajib tetap sampai ke event `done`, kalau
tidak angka dashboard Pusat Biaya diam-diam mengecil. Selengkapnya: workspace
`docs/FOUNDER_ANALYTICS.md`.

### Dua jalur LLM di satu service — sengaja, sementara
Jalur founder memanggil `app/config/gemini_direct.py` (SDK modern `google.genai`),
bukan `get_llm()`. Alasannya: kendali `thinking` **tidak bisa** diungkapkan lewat
`langchain-google-genai` 2.0.8 yang terpasang — `model_kwargs` dibuang diam-diam
oleh pydantic (`extra="ignore"`) dan SDK lamanya tidak punya medan
`thinking_config` sama sekali. Tanpa kendali itu, model keluarga Flash mencetak
isi penalarannya ke dalam SQL sampai gagal di-parse.

Aturannya: **`thinking_level` untuk Gemini 3.x, `thinking_budget` untuk 2.5,
jangan pernah keduanya** (Google menjawab 400). Bagian jawaban bertanda `thought`
dibuang di tingkat *part*.

Node plan, sql, dan chart memakai **`response_schema`** (model Pydantic di
`state.py`), jadi bentuk keluarannya dijamin di sisi Google — bukan diminta lewat
prompt lalu ditebak parser. Kalau menambah node yang butuh keluaran terstruktur,
pakai jalan yang sama; jangan menulis parser JSON baru.

Tangga jalur-mundurnya berurutan dan urutannya penting: **skema dilepas duluan,
kendali penalaran paling akhir**, baru `get_llm()`. Kehilangan kendali penalaran
adalah kegagalan yang tidak kelihatan sampai tagihannya datang; bentuk keluaran
yang longgar cuma menyulitkan parser.

`gemini_direct.generate()` melempar `TeksTerpotong` kalau `finish_reason` =
`MAX_TOKENS`. Jangan ditelan jadi string kosong — token penalaran memakan
`max_output_tokens` yang sama, dan kalau disamarkan, sebabnya terbaca sebagai
"model menolak menjawab" lalu orang mencari kekurangan di katalog.

Jalur Tanya Peri **tidak** disentuh dan masih memakai LangChain. Penyatuannya
menunggu upgrade tumpukan — utang tercatat di workspace `docs/OPEN_ITEMS.md`.

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

### ⚠️ `docker build .` polos SALAH untuk repo ini

`Dockerfile` (default) adalah varian **GPU** — base `nvidia/cuda`, torch cu121,
sentence-transformers. **Cloud Run tidak mendukung GPU sama sekali.**

```
docker build .                      -> 15,5 GB   TIDAK BISA jalan di Cloud Run
docker build -f Dockerfile.cpu .    ->  807 MB   yang benar
```

Kenapa gampang terjebak, dan kenapa ini bukan kelalaian sesaat:
`docker-compose.yml` untuk dev lokal memakai `dockerfile: Dockerfile`, jadi
varian GPU-lah yang dibangun ratusan kali sehari-hari. `Dockerfile.cpu` **tidak
pernah disentuh kecuali saat deploy**, dan deploy jarang. Nama berkas default
justru yang salah untuk produksi.

Yang bikin mahal: `docker build` sukses, `docker push` sukses, exit code nol.
Kegagalannya baru muncul di Cloud Run, jauh dari sebabnya. Terjadi 2 Agustus
2026 — dua push besar terbuang sebelum ketahuan.

```powershell
$SHA = git rev-parse --short HEAD
$IMG = "asia-southeast2-docker.pkg.dev/peri-bugi-491218/peri-bugi"
docker build -f Dockerfile.cpu -t "$IMG/ai-chat:$SHA" .
docker push "$IMG/ai-chat:$SHA"
gcloud run deploy peri-bugi-ai-chat --image="$IMG/ai-chat:$SHA" `
  --region=asia-southeast2 --project=peri-bugi-491218
```

Selebihnya baca `../docs/BRANCHING_AND_DEPLOY.md`. Repo ini pernah punya branch
`langfuse-integration` yang hidup 37 commit di depan `main` sementara produksi
jalan dari branch itu — sudah disatukan 2026-07-18. Jangan ulangi: kerja di
branch pendek, merge ke `main`, deploy dari `main`.

## Secrets — never commit
`.env` (kunci Gemini, Qdrant, Langfuse, DB). Gitignored, biarkan begitu.
