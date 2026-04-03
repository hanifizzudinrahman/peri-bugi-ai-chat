"""
Script untuk cek data di database peri-bugi via Docker.
Dijalankan dari luar container, tidak perlu masuk ke dalam.

Cara pakai (dari folder peri-bugi-api):

    # Lihat LLM call logs terbaru
    python scripts/check_db.py llm-logs

    # Lihat semua tabel yang ada
    python scripts/check_db.py tables

    # Lihat chat sessions terbaru
    python scripts/check_db.py chat-sessions

    # Custom SQL query
    python scripts/check_db.py query "SELECT count(*) FROM llm_call_logs"

Atau via Docker exec langsung (tanpa script ini):
    docker exec peri_bugi_db psql -U peri_bugi_user -d peri_bugi -c "SELECT * FROM llm_call_logs ORDER BY created_at DESC LIMIT 5;"
"""
import argparse
import subprocess
import sys


# Config — sesuaikan dengan docker-compose.yml peri-bugi-api
DB_CONTAINER = "peri_bugi_db"
DB_USER = "peri_bugi_user"
DB_NAME = "peri_bugi"


def run_query(sql: str, title: str | None = None) -> bool:
    """Jalankan SQL query di container DB via docker exec."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)

    cmd = [
        "docker", "exec", DB_CONTAINER,
        "psql", "-U", DB_USER, "-d", DB_NAME,
        "-c", sql,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error: {result.stderr.strip()}")
        if "No such container" in result.stderr:
            print(f"\nContainer '{DB_CONTAINER}' tidak ditemukan.")
            print("Pastikan peri-bugi-api sedang jalan:")
            print("  cd peri-bugi-api && docker compose up -d")
        return False

    print(result.stdout)
    return True


def cmd_llm_logs(args):
    """Cek LLM call logs terbaru."""
    limit = getattr(args, 'limit', 10)

    run_query(
        f"""
        SELECT
            to_char(created_at AT TIME ZONE 'Asia/Jakarta', 'MM-DD HH24:MI:SS') AS waktu,
            model,
            provider,
            node,
            latency_ms,
            ttft_ms,
            output_tokens,
            success
        FROM llm_call_logs
        ORDER BY created_at DESC
        LIMIT {limit};
        """,
        title=f"LLM Call Logs — {limit} terbaru"
    )

    # Summary per model
    run_query(
        """
        SELECT
            model,
            provider,
            count(*) AS total_calls,
            round(avg(latency_ms)) AS avg_latency_ms,
            round(avg(ttft_ms)) AS avg_ttft_ms,
            round(avg(output_tokens)) AS avg_tokens,
            sum(CASE WHEN success THEN 1 ELSE 0 END) AS success_count,
            sum(CASE WHEN NOT success THEN 1 ELSE 0 END) AS error_count
        FROM llm_call_logs
        GROUP BY model, provider
        ORDER BY total_calls DESC;
        """,
        title="Summary per Model"
    )


def cmd_tables(args):
    """Tampilkan semua tabel dan jumlah row-nya."""
    run_query(
        """
        SELECT
            tablename AS tabel,
            (xpath('/row/c/text()',
                query_to_xml(format('select count(*) as c from %I', tablename), false, true, ''))
            )[1]::text::int AS jumlah_row
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
        """,
        title="Semua Tabel di Database"
    )


def cmd_chat_sessions(args):
    """Cek chat sessions dan messages terbaru."""
    run_query(
        """
        SELECT
            to_char(cs.created_at AT TIME ZONE 'Asia/Jakarta', 'MM-DD HH24:MI') AS dibuat,
            to_char(cs.last_message_at AT TIME ZONE 'Asia/Jakarta', 'MM-DD HH24:MI') AS pesan_terakhir,
            cs.message_count,
            left(cs.title, 40) AS judul
        FROM chat_sessions cs
        ORDER BY cs.last_message_at DESC NULLS LAST
        LIMIT 10;
        """,
        title="Chat Sessions Terbaru"
    )

    run_query(
        """
        SELECT
            cm.role,
            to_char(cm.created_at AT TIME ZONE 'Asia/Jakarta', 'MM-DD HH24:MI:SS') AS waktu,
            left(cm.content, 60) AS isi,
            cm.metadata->>'model' AS model,
            (cm.metadata->>'latency_ms')::int AS latency_ms
        FROM chat_messages cm
        ORDER BY cm.created_at DESC
        LIMIT 10;
        """,
        title="Chat Messages Terbaru"
    )


def cmd_query(args):
    """Jalankan custom SQL query."""
    run_query(args.sql, title=f"Query: {args.sql[:50]}...")


def cmd_qdrant_test(args):
    """
    Cek koneksi Qdrant dan isi collection.
    Qdrant tidak pakai psql — pakai curl ke HTTP API.
    """
    import json

    qdrant_url = getattr(args, 'qdrant_url', 'http://localhost:6333')
    collection = getattr(args, 'collection', 'peri_bugi_dental')

    print(f"\n{'=' * 60}")
    print(f"  Qdrant Status — {qdrant_url}")
    print('=' * 60)

    # Cek health
    cmd_health = ["curl", "-s", f"{qdrant_url}/healthz"]
    result = subprocess.run(cmd_health, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Qdrant jalan di {qdrant_url}")
    else:
        print(f"❌ Qdrant tidak bisa diakses di {qdrant_url}")
        print("Pastikan container Qdrant jalan:")
        print("  cd peri-bugi-ai-chat && docker compose ps")
        return

    # Cek collections
    cmd_cols = ["curl", "-s", f"{qdrant_url}/collections"]
    result = subprocess.run(cmd_cols, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            collections = data.get("result", {}).get("collections", [])
            if collections:
                print(f"\nCollections yang ada:")
                for col in collections:
                    print(f"  - {col['name']}")
            else:
                print("\n⚠️  Belum ada collection di Qdrant")
                print("Jalankan ingest_pdf.py untuk mengisi knowledge base")
        except json.JSONDecodeError:
            print(result.stdout)

    # Cek collection spesifik
    cmd_col = ["curl", "-s", f"{qdrant_url}/collections/{collection}"]
    result = subprocess.run(cmd_col, capture_output=True, text=True)
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            if data.get("status") == "ok":
                col_info = data.get("result", {})
                vectors_count = col_info.get("vectors_count", 0)
                status = col_info.get("status", "unknown")
                print(f"\nCollection '{collection}':")
                print(f"  Status  : {status}")
                print(f"  Vectors : {vectors_count}")
                if vectors_count == 0:
                    print(f"\n  ⚠️  Collection kosong — belum ada dokumen diingest")
                    print(f"  Jalankan: python scripts/ingest_pdf.py --pdf your_file.pdf")
                else:
                    print(f"\n  ✅ Knowledge base sudah terisi ({vectors_count} vectors)")
            else:
                print(f"\n⚠️  Collection '{collection}' belum ada")
                print(f"Jalankan: python scripts/ingest_pdf.py --pdf your_file.pdf")
        except json.JSONDecodeError:
            print(result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cek data peri-bugi via Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  llm-logs       Lihat LLM call logs terbaru + summary per model
  tables         Lihat semua tabel dan jumlah row-nya
  chat-sessions  Lihat chat sessions dan messages terbaru
  qdrant         Cek status Qdrant dan isi collection
  query          Jalankan custom SQL query

Contoh:
  python scripts/check_db.py llm-logs
  python scripts/check_db.py llm-logs --limit 20
  python scripts/check_db.py qdrant
  python scripts/check_db.py query "SELECT count(*) FROM users"
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # llm-logs
    p_logs = subparsers.add_parser("llm-logs", help="Lihat LLM call logs terbaru")
    p_logs.add_argument("--limit", type=int, default=10, help="Jumlah baris (default: 10)")
    p_logs.set_defaults(func=cmd_llm_logs)

    # tables
    p_tables = subparsers.add_parser("tables", help="Lihat semua tabel dan jumlah row")
    p_tables.set_defaults(func=cmd_tables)

    # chat-sessions
    p_chat = subparsers.add_parser("chat-sessions", help="Lihat chat sessions terbaru")
    p_chat.set_defaults(func=cmd_chat_sessions)

    # qdrant
    p_qdrant = subparsers.add_parser("qdrant", help="Cek status Qdrant dan collection")
    p_qdrant.add_argument("--qdrant-url", default="http://localhost:6333")
    p_qdrant.add_argument("--collection", default="peri_bugi_dental")
    p_qdrant.set_defaults(func=cmd_qdrant_test)

    # query
    p_query = subparsers.add_parser("query", help="Custom SQL query")
    p_query.add_argument("sql", help="SQL query yang akan dijalankan")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)
