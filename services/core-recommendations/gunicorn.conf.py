import multiprocessing
import os

bind    = "0.0.0.0:5000"
workers = int(os.environ.get("GUNICORN_WORKERS", min(multiprocessing.cpu_count() * 2 + 1, 4)))
threads = int(os.environ.get("GUNICORN_THREADS", 4))
worker_class = "sync"   # keep sync — ML scoring path is CPU-bound
timeout          = int(os.environ.get("GUNICORN_TIMEOUT", 120))
graceful_timeout = 30
keepalive        = 5

# Prevent unbounded memory growth from TF/numpy allocations per-request
max_requests        = int(os.environ.get("GUNICORN_MAX_REQUESTS", 500))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", 50))

accesslog = "-"
errorlog  = "-"
loglevel  = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# NOTE: preload_app is intentionally NOT set — CoreRecommendationsService initializes
# TensorFlow and Redis connections at module load time. Preloading would share these
# across forked workers, corrupting file descriptors and TF state.
