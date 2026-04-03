"""
Script untuk keep Ollama model tetap warm di VRAM.

Cara pakai:
    # Jalankan manual (ping sekali)
    python scripts/keep_alive_ollama.py --once

    # Jalankan terus menerus (ping setiap 5 menit)
    python scripts/keep_alive_ollama.py

    # Custom interval dan model
    python scripts/keep_alive_ollama.py --interval 3 --model qwen3.5

    # Ping semua model yang ada di list
    python scripts/keep_alive_ollama.py --all-models

Background:
    Ollama secara default unload model dari VRAM setelah 5 menit idle.
    Unload = cold start ~3 detik untuk request berikutnya.
    Script ini kirim ping kecil setiap N menit supaya model tidak di-unload.

Best practice untuk dev:
    Jalankan script ini di terminal terpisah saat sedang development aktif.
    Tidak perlu jalan terus-menerus saat tidak sedang kerja.

Best practice untuk production:
    Kalau deploy Ollama di server, set environment variable:
        OLLAMA_KEEP_ALIVE=-1   (model tidak pernah di-unload)
    Atau buat systemd service dari script ini.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def ping_ollama(base_url: str, model: str, keep_alive: str = "10m") -> bool:
    """
    Kirim request kecil ke Ollama untuk keep model tetap di VRAM.
    Menggunakan /api/generate dengan prompt kosong — tidak generate apapun,
    hanya memastikan model ter-load.
    """
    import urllib.request
    import json

    payload = json.dumps({
        "model": model,
        "keep_alive": keep_alive,
        "prompt": "",  # kosong = tidak generate, hanya load/keep model
    }).encode()

    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"   ⚠️  Gagal ping {model}: {e}")
        return False


def get_loaded_models(base_url: str) -> list[dict]:
    """Ambil list model yang sedang di-load di VRAM."""
    import urllib.request
    import json

    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("models", [])
    except Exception:
        return []


def get_available_models(base_url: str) -> list[str]:
    """Ambil semua model yang tersedia di Ollama."""
    import urllib.request
    import json

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def run(
    base_url: str,
    models: list[str],
    interval_minutes: int,
    keep_alive: str,
    once: bool,
    all_models: bool,
):
    print(f"🔥 Ollama Keep-Alive")
    print(f"   URL      : {base_url}")
    print(f"   Interval : setiap {interval_minutes} menit")
    print(f"   Keep-alive: {keep_alive}")

    if all_models:
        available = get_available_models(base_url)
        if available:
            models = available
            print(f"   Models   : {', '.join(models)} (semua)")
        else:
            print("   ⚠️  Tidak bisa ambil daftar model dari Ollama")

    if not models:
        print("   ❌ Tidak ada model yang akan di-ping. Gunakan --model atau --all-models")
        sys.exit(1)

    print(f"   Models   : {', '.join(models)}")
    print()

    def do_ping():
        now = datetime.now().strftime("%H:%M:%S")
        loaded = get_loaded_models(base_url)
        loaded_names = [m.get("name", "").split(":")[0] for m in loaded]

        for model in models:
            model_short = model.split(":")[0]
            in_vram = any(model_short in name for name in loaded_names)
            status = "🔥 warm" if in_vram else "❄️  cold"
            success = ping_ollama(base_url, model, keep_alive)
            result = "✅" if success else "❌"
            print(f"[{now}] {result} {model:<20} {status}")

    if once:
        do_ping()
        return

    print("Tekan Ctrl+C untuk berhenti.\n")
    while True:
        do_ping()
        print(f"   → Ping berikutnya dalam {interval_minutes} menit...\n")
        try:
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\nDihentikan.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keep Ollama model warm di VRAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python scripts/keep_alive_ollama.py                    # ping gemma2:2b setiap 5 menit
  python scripts/keep_alive_ollama.py --once             # ping sekali lalu keluar
  python scripts/keep_alive_ollama.py --model qwen3.5   # ping model spesifik
  python scripts/keep_alive_ollama.py --all-models      # ping semua model yang tersedia
  python scripts/keep_alive_ollama.py --interval 3      # ping setiap 3 menit
        """,
    )
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument(
        "--model", action="append", dest="models",
        default=None, metavar="MODEL",
        help="Model yang di-ping (default: gemma2:2b, bisa diulang untuk banyak model)",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Ping semua model yang tersedia di Ollama",
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Interval ping dalam menit (default: 5)",
    )
    parser.add_argument(
        "--keep-alive", default="10m",
        help="Berapa lama model di-keep di VRAM setelah ping (default: 10m)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Ping sekali lalu keluar (tidak loop)",
    )

    args = parser.parse_args()

    models = args.models or ["gemma2:2b"]

    run(
        base_url=args.url,
        models=models,
        interval_minutes=args.interval,
        keep_alive=args.keep_alive,
        once=args.once,
        all_models=args.all_models,
    )
