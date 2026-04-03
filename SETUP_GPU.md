# Panduan GPU Docker + RnD Endpoint — peri-bugi-ai-chat

## Bagian 1: Setup GPU di Docker (Windows + WSL2)

### Kenapa Docker perlu setup khusus untuk GPU?

Docker container secara default tidak bisa akses GPU host.
Yang dibutuhkan:
1. **NVIDIA Container Toolkit** — bridge antara Docker dan driver NVIDIA di host
2. **Base image yang punya CUDA runtime** — `nvidia/cuda:...` bukan `python:3.11-slim`
3. **docker-compose dengan `deploy.resources.reservations.devices`** — cara request GPU ke toolkit

---

### Step 1: Cek apakah NVIDIA Container Toolkit sudah terinstall

Buka PowerShell (bukan WSL), jalankan:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

**Kalau berhasil** → keluar output `nvidia-smi` → toolkit sudah ada, lanjut ke Step 3.

**Kalau error** `could not select device driver "nvidia"` → lanjut Step 2.

---

### Step 2: Install NVIDIA Container Toolkit

**Di Windows dengan Docker Desktop:**

Docker Desktop versi terbaru sudah include NVIDIA Container Toolkit.
Yang perlu dilakukan hanya enable di settings:

1. Buka **Docker Desktop**
2. Klik **Settings** (gear icon)
3. Klik **Resources** → **WSL Integration**
4. Pastikan WSL integration aktif untuk distro Ubuntu kamu
5. Klik **Apply & Restart**

Kalau masih error, update Docker Desktop ke versi terbaru, lalu coba lagi Step 1.

**Cara verify toolkit aktif:**
```powershell
docker info | findstr -i "nvidia"
# Harusnya muncul: Runtimes: nvidia runc
```

---

### Step 3: Apply file baru ke repo

Copy semua file dari zip ini ke folder `peri-bugi-ai-chat`:

```
peri-bugi-ai-chat/
├── Dockerfile                          ← REPLACE (versi baru, pakai nvidia base image)
├── docker-compose.yml                  ← REPLACE (tambah GPU config)
├── requirements.txt                    ← REPLACE (untuk conda/manual install)
├── requirements-base.txt               ← NEW (library tanpa torch, untuk Docker layer)
├── requirements-gpu.txt                ← NEW (hanya torch+cuda, untuk Docker layer)
├── requirements-sentence-transformers.txt  ← NEW (sentence-transformers, untuk Docker layer)
└── .env.example                        ← REPLACE (tambah EMBEDDING_DEVICE, PERI_API_URL, RND_MODE)
```

---

### Step 4: Update .env ai-chat

Buka `.env` (bukan `.env.example`) dan tambahkan:

```env
# GPU / Device config
EMBEDDING_DEVICE=auto

# URL peri-bugi-api untuk kirim LLM call logs
# Pakai host.docker.internal karena ai-chat jalan di Docker
PERI_API_URL=http://host.docker.internal:8000

# RnD mode — aktifkan untuk development/research
RND_MODE=True
```

---

### Step 5: Build image pertama kali

**PENTING:** Build pertama kali akan download torch ~2.5 GB.
Ini normal dan hanya terjadi sekali. Setelah itu, layer di-cache.

```powershell
cd path\to\peri-bugi-ai-chat
docker compose build
```

Proses build:
- Layer torch: ~5-15 menit tergantung koneksi (download 2.5 GB)
- Layer sentence-transformers: ~2 menit
- Layer requirements-base: ~3 menit
- Layer app code: < 10 detik

**Kalau build gagal di tengah jalan (koneksi putus, timeout, dll):**

```powershell
# Jalankan lagi perintah yang sama — Docker akan lanjut dari layer terakhir yang berhasil
docker compose build
```

Docker TIDAK mengulang dari awal kalau ada cache. Layer yang sudah berhasil tidak didownload ulang.

**Kapan torch didownload ulang?**
Hanya kalau kamu ubah `requirements-gpu.txt` (misal ganti versi torch).
Kalau kamu ubah `requirements-base.txt` atau app code → torch TIDAK didownload ulang.

---

### Step 6: Jalankan container

```powershell
docker compose up -d
```

---

### Step 7: Verify GPU aktif di dalam container

```powershell
# Cek apakah GPU terdeteksi di dalam container
docker exec peri_bugi_ai_chat python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# Atau via endpoint health/gpu
curl http://localhost:8003/health/gpu
```

**Output yang diharapkan:**
```json
{
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3070 Laptop GPU",
  "embedding_device_setting": "auto"
}
```

**Kalau `cuda_available: false` padahal GPU ada:**
- Cek apakah `deploy.resources.reservations.devices` ada di docker-compose.yml
- Jalankan: `docker compose down && docker compose up -d` (bukan hanya restart)
- Verify toolkit: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

---

### Step 8: Verify HuggingFace model cache

Volume `huggingface_cache` menyimpan model sentence-transformers.
Model didownload saat **pertama kali endpoint embedding dipanggil** (bukan saat build).
Download sekali ~500MB, setelah itu pakai cache dari volume.

Cek apakah volume ada:
```powershell
docker volume ls | findstr huggingface
# Harusnya muncul: peri-bugi-ai-chat_huggingface_cache
```

---

## Bagian 2: Endpoint RnD — `/chat/rnd`

### Apa itu endpoint ini?

Endpoint khusus untuk research dan eksperimen. Tidak perlu login atau internal secret.
Bisa override semua parameter LLM, embedding, dan prompt per-request.

Aktif kalau `RND_MODE=True` di `.env`. **Matikan di production** (`RND_MODE=False`).

### Akses via Swagger

Buka: `http://localhost:8003/docs`
Cari: `POST /chat/rnd`

---

### Contoh-contoh penggunaan

#### Contoh 1: Pertanyaan sederhana (paling minimal)

```json
{
  "message": "berapa kali anak harus sikat gigi sehari?",
  "stream": false
}
```

Response akan pakai model default dari `.env` (sekarang `gemma2:2b` via Ollama).

---

#### Contoh 2: Test model berbeda — bandingkan gemma2:2b vs qwen3.5

Request pertama:
```json
{
  "message": "apa penyebab gigi berlubang pada anak?",
  "model": "gemma2:2b",
  "stream": false,
  "experiment_id": "compare-model-001",
  "experiment_tags": ["baseline", "gemma2"]
}
```

Request kedua (sama persis, beda model):
```json
{
  "message": "apa penyebab gigi berlubang pada anak?",
  "model": "qwen3.5",
  "stream": false,
  "experiment_id": "compare-model-001",
  "experiment_tags": ["baseline", "qwen3.5"]
}
```

Bandingkan field `metrics.latency_ms`, `metrics.ttft_ms`, `metrics.output_tokens_approx`.

---

#### Contoh 3: Test dengan expected output untuk evaluasi otomatis

```json
{
  "message": "apakah anak usia 3 tahun boleh pakai pasta gigi berfluoride?",
  "model": "gemma2:2b",
  "stream": false,
  "experiment_id": "fluoride-test",
  "expected_intent": "dental_qa",
  "expected_keywords": ["fluoride", "pasta gigi", "dokter gigi", "sedikit"],
  "temperature": 0.3
}
```

Response akan include:
```json
"metrics": {
  "intent_correct": true,
  "keyword_hit_rate": 0.75,
  "keywords_found": ["fluoride", "pasta gigi", "dokter gigi"],
  "keywords_missing": ["sedikit"]
}
```

`keyword_hit_rate` = 3/4 = 0.75 artinya 75% keyword yang diharapkan muncul di response.

---

#### Contoh 4: Test custom system prompt

```json
{
  "message": "halo, siapa kamu?",
  "system_prompt": "Kamu adalah asisten bernama Denti, robot gigi dari Planet Fluoride. Jawab dengan gaya yang sangat formal dan ilmiah.",
  "stream": false,
  "experiment_id": "prompt-test-001"
}
```

System prompt ini menggantikan `persona_system` dari DB untuk request ini saja.
DB tidak berubah.

---

#### Contoh 5: Skip router, paksa intent tertentu

```json
{
  "message": "cerita tentang sikat gigi dong",
  "force_intent": "dental_qa",
  "stream": false,
  "experiment_id": "force-intent-test"
}
```

Dengan `force_intent`, node router di-skip sepenuhnya.
Berguna untuk test node generate secara isolated tanpa noise dari router.

---

#### Contoh 6: Test dengan context user terisi (simulasi user nyata)

```json
{
  "message": "bagaimana hasil sikat gigi anak saya?",
  "force_intent": "context_query",
  "stream": false,
  "user_context": {
    "user": {
      "full_name": "Budi Santoso",
      "nickname": "Pak Budi"
    },
    "child": {
      "full_name": "Arif Santoso",
      "nickname": "Arif",
      "age_years": 7
    },
    "brushing": {
      "current_streak": 5,
      "best_streak": 14,
      "last_complete_date": "2026-04-02"
    },
    "mata_peri_last_result": null
  }
}
```

---

#### Contoh 7: Test dengan Gemini (ganti provider)

```json
{
  "message": "tips menyikat gigi yang benar untuk anak usia 5 tahun",
  "provider": "gemini",
  "model": "gemini-1.5-flash",
  "temperature": 0.5,
  "max_tokens": 512,
  "stream": false,
  "experiment_id": "gemini-vs-ollama",
  "experiment_tags": ["gemini", "flash"]
}
```

Pastikan `GEMINI_API_KEY` sudah diset di `.env`.

---

#### Contoh 8: Test RAG dengan top_k berbeda

```json
{
  "message": "bagaimana cara mencegah karies pada anak balita?",
  "top_k_docs": 5,
  "stream": false,
  "experiment_id": "rag-topk-test"
}
```

Response akan include `metrics.rag_docs_retrieved` = jumlah dokumen yang berhasil diambil.
Kalau Qdrant collection masih kosong, nilainya = 0 (tidak error, hanya tidak dapat dokumen).

---

#### Contoh 9: Debug — lihat full prompt yang dikirim ke LLM

```json
{
  "message": "gigi anak saya berlubang, apa yang harus saya lakukan?",
  "stream": false,
  "include_prompt_in_response": true
}
```

Response akan include field `prompt_debug.system` (full system prompt) dan
`prompt_debug.messages` (semua pesan yang dikirim ke LLM).
Berguna untuk debug kenapa LLM memberikan jawaban tertentu.

---

#### Contoh 10: Streaming mode (sama seperti production, tapi tanpa auth)

```json
{
  "message": "berapa lama waktu ideal sikat gigi anak?",
  "stream": true
}
```

Dengan `stream: true`, response adalah SSE stream sama persis seperti endpoint `/chat/stream`.
Bisa test di Swagger tapi hasilnya tidak di-render real-time (hanya text).
Lebih baik test streaming via curl:

```powershell
curl -X POST http://localhost:8003/chat/rnd `
  -H "Content-Type: application/json" `
  -d '{"message": "berapa lama sikat gigi anak?", "stream": true}' `
  --no-buffer
```

---

### Batch experiment via Python script

Untuk compare banyak model sekaligus, buat script Python sederhana:

```python
import httpx
import json

BASE_URL = "http://localhost:8003"

pertanyaan = [
    "berapa kali anak harus sikat gigi sehari?",
    "apa penyebab gigi berlubang pada anak?",
    "kapan anak harus pertama kali ke dokter gigi?",
    "apakah aman pakai pasta gigi fluoride untuk anak 2 tahun?",
]

models = [
    {"provider": "ollama", "model": "gemma2:2b"},
    {"provider": "ollama", "model": "qwen3.5"},
]

results = []

for pertanyaan_item in pertanyaan:
    for model_config in models:
        resp = httpx.post(
            f"{BASE_URL}/chat/rnd",
            json={
                "message": pertanyaan_item,
                "provider": model_config["provider"],
                "model": model_config["model"],
                "stream": False,
                "experiment_id": "batch-001",
                "experiment_tags": [model_config["model"]],
                "expected_intent": "dental_qa",
            },
            timeout=60,
        )
        data = resp.json()
        results.append({
            "pertanyaan": pertanyaan_item,
            "model": model_config["model"],
            "response": data["response"][:200],  # truncate
            "latency_ms": data["metrics"]["latency_ms"],
            "ttft_ms": data["metrics"]["ttft_ms"],
            "output_tokens": data["metrics"]["output_tokens_approx"],
            "intent_correct": data["metrics"].get("intent_correct"),
        })
        print(f"✓ {model_config['model']} — {pertanyaan_item[:40]}...")

# Print hasil
print("\n=== HASIL EKSPERIMEN ===")
for r in results:
    print(f"[{r['model']}] {r['pertanyaan'][:40]}...")
    print(f"  latency: {r['latency_ms']}ms | ttft: {r['ttft_ms']}ms | tokens: {r['output_tokens']}")
    print(f"  intent_correct: {r['intent_correct']}")
    print()

# Simpan ke JSON untuk analisis lebih lanjut
with open("experiment_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Hasil disimpan ke experiment_results.json")
```

---

## Bagian 3: Troubleshooting

### Error: `could not select device driver "nvidia"`

Artinya NVIDIA Container Toolkit belum aktif atau Docker Desktop perlu restart.

```powershell
# Test toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Kalau masih error: update Docker Desktop ke versi terbaru.

---

### Error saat build: `pip install` timeout atau connection error

Docker build otomatis retry kalau jalankan `docker compose build` lagi.
Layer yang sudah berhasil tidak didownload ulang.

Kalau mau paksa mulai dari awal (hati-hati, akan download ulang semua):
```powershell
docker compose build --no-cache
```

---

### Embedding tetap jalan di CPU padahal GPU ada

Cek dua hal:

1. `EMBEDDING_DEVICE=auto` sudah ada di `.env`
2. Container dibuat ulang (bukan hanya restart): `docker compose down && docker compose up -d`

Verify:
```powershell
curl http://localhost:8003/health/gpu
```

---

### Model HuggingFace didownload ulang setiap container recreate

Pastikan volume `huggingface_cache` ada di `docker-compose.yml`:
```yaml
volumes:
  - huggingface_cache:/root/.cache/huggingface
```

Dan di bagian bawah file:
```yaml
volumes:
  huggingface_cache:
```

Cek volume ada:
```powershell
docker volume ls | findstr huggingface
```
