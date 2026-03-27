import fnmatch
import logging
import threading
import time

logger = logging.getLogger("mock-redis")


class MockRedis:
    """Simple in-memory mock for Redis when no real Redis is available."""

    def __init__(self, decode_responses=True):
        self.data = {}
        self.expiry = {}
        self.client_name = "default"
        self._lock = threading.Lock()
        logger.info("Initialized MockRedis for local development")

    def get(self, key):
        """Get value for key, respecting expiration times."""
        with self._lock:
            self._check_expiry(key)
            value = self.data.get(key)
        logger.debug(f"GET {key} -> {value if value is None else '(data)'}")
        return value

    def mget(self, keys):
        """Get values for multiple keys in a single call."""
        return [self.get(key) for key in keys]

    def set(self, key, value):
        """Set key to value without expiration."""
        logger.debug(f"SET {key}")
        with self._lock:
            self.data[key] = value
        return True

    def setex(self, key, ttl, value):
        """Set key to value with expiration time in seconds."""
        logger.debug(f"SETEX {key} {ttl}s")
        with self._lock:
            self.data[key] = value
            self.expiry[key] = time.time() + ttl
        return True

    def delete(self, *keys):
        """Delete one or more keys."""
        count = 0
        with self._lock:
            for key in keys:
                logger.debug(f"DEL {key}")
                if key in self.data:
                    del self.data[key]
                    count += 1
                    if key in self.expiry:
                        del self.expiry[key]
        return count

    def keys(self, pattern):
        """Find all keys matching pattern."""
        with self._lock:
            self._check_all_expiry()
            matched_keys = [key for key in self.data.keys() if fnmatch.fnmatch(key, pattern)]
        logger.debug(f"KEYS {pattern} -> {len(matched_keys)} matches")
        return matched_keys

    def ping(self):
        """Test connectivity - always returns True."""
        logger.debug("PING -> PONG")
        return True

    def client_setname(self, name):
        """Set the client name."""
        logger.debug(f"CLIENT SETNAME {name}")
        self.client_name = name
        return True

    def client_getname(self):
        """Get the client name."""
        return self.client_name

    def _check_expiry(self, key):
        """Check if a key has expired and remove it if so. Caller must hold self._lock."""
        if key in self.expiry and self.expiry[key] < time.time():
            del self.data[key]
            del self.expiry[key]

    def _check_all_expiry(self):
        """Check all keys for expiration. Caller must hold self._lock."""
        now = time.time()
        expired_keys = [k for k, exp in self.expiry.items() if exp < now]
        for key in expired_keys:
            if key in self.data:
                del self.data[key]
            del self.expiry[key]