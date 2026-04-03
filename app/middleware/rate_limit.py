"""
Rate Limiter untuk endpoint RnD.

Menggunakan in-memory sliding window counter per IP.
Cukup untuk dev/research use case — tidak butuh Redis.

Di production (RND_MODE=False), endpoint /chat/rnd sudah disabled,
jadi rate limiter ini hanya aktif di development/staging.
"""
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request


class InMemoryRateLimiter:
    """
    Sliding window rate limiter berbasis in-memory.

    Cara kerja:
    - Setiap IP punya queue timestamp request
    - Setiap request: hapus timestamp yang sudah lewat window
    - Kalau jumlah request dalam window > limit → 429

    Thread-safe dengan Lock karena FastAPI bisa concurrent.
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        """
        Args:
            max_requests   : max request per window per IP (default: 20)
            window_seconds : ukuran window dalam detik (default: 60)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, ip: str) -> tuple[bool, int, int]:
        """
        Cek apakah request dari IP ini diizinkan.

        Returns:
            (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            q = self._requests[ip]

            # Hapus request yang sudah keluar dari window
            while q and q[0] < window_start:
                q.popleft()

            count = len(q)

            if count >= self.max_requests:
                # Berapa detik lagi sampai request tertua keluar dari window
                retry_after = int(q[0] + self.window_seconds - now) + 1
                return False, 0, retry_after

            q.append(now)
            remaining = self.max_requests - count - 1
            return True, remaining, 0


# Instance global — satu untuk /chat/rnd
# Config berbeda antara development dan production tidak perlu,
# karena /chat/rnd sudah di-disable di production via RND_MODE=False
rnd_rate_limiter = InMemoryRateLimiter(
    max_requests=30,   # 30 request per menit per IP
    window_seconds=60,
)

benchmark_rate_limiter = InMemoryRateLimiter(
    max_requests=5,    # 5 benchmark per menit (lebih berat, lebih ketat)
    window_seconds=60,
)


def get_client_ip(request: Request) -> str:
    """
    Ambil IP client dari request.
    Cek X-Forwarded-For dulu (kalau di belakang proxy/nginx).
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For bisa berisi beberapa IP: "client, proxy1, proxy2"
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rnd_rate_limit(request: Request) -> None:
    """
    Dependency untuk rate limit /chat/rnd.
    Raise HTTPException 429 jika limit terlampaui.
    """
    ip = get_client_ip(request)
    allowed, remaining, retry_after = rnd_rate_limiter.is_allowed(ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Terlalu banyak request. Coba lagi dalam {retry_after} detik.",
            headers={"Retry-After": str(retry_after)},
        )


def check_benchmark_rate_limit(request: Request) -> None:
    """
    Dependency untuk rate limit /chat/rnd/benchmark.
    Lebih ketat karena benchmark mengirim N request sekaligus.
    """
    ip = get_client_ip(request)
    allowed, remaining, retry_after = benchmark_rate_limiter.is_allowed(ip)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Benchmark limit tercapai. Coba lagi dalam {retry_after} detik.",
            headers={"Retry-After": str(retry_after)},
        )
