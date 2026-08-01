"""Tanya Data Founder — graph analitik lintas-pengguna.

Terpisah total dari graph chat orang tua (`app/agents/graph.py`): endpoint
sendiri, state sendiri, prompt sendiri, katalog sendiri. Tidak ada satu pun
node yang dipakai bersama.

Itu disengaja. Graph chat sudah live, dipakai orang tua setiap hari, dan
menambah cabang di dalamnya berarti tiap perubahan di jalur founder punya
peluang menjatuhkan jalur yang sudah jalan. Jalur ini juga berbentuk lain:
tidak ada pemilihan tool, tidak ada klarifikasi gambar, tidak ada memori
jangka panjang — yang ada pipa lurus dengan satu simpul perbaikan.

    plan  -> pilih dataset + rentang waktu
    sql   -> tulis SQL dari katalog dataset terpilih
    exec  -> kirim ke peri-bugi-api untuk divalidasi dan dijalankan
             gagal? balik ke sql dengan pesan galatnya (maks. N kali)
    chart -> keluarkan NIAT grafik; server yang mengompilasinya

Jawabannya sendiri di-stream di luar graph, mengikuti pola yang sudah dipakai
`generate_node` di jalur orang tua.

Selengkapnya: workspace `docs/FOUNDER_ANALYTICS.md`.
"""
from app.agents.founder_analytics.graph import run_founder_analytics
from app.agents.founder_analytics.state import FounderAnalyticsState

__all__ = ["FounderAnalyticsState", "run_founder_analytics"]
