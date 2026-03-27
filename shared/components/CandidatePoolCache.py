import json
import logging
import time
from typing import Dict, List, Set, Optional
import numpy as np

logger = logging.getLogger("candidate-pool-cache")

POOL_CAP = 200
POOL_TTL_DEFAULT = 86400  # 24 hours


class CandidatePoolCache:
    """
    Per-user candidate pool backed by Redis.

    Stores up to POOL_CAP post vectors per user per content type so the ML
    model has a richer candidate set to rank without requiring a fresh API
    call on every request.

    TMDB Compliance: only post IDs and Two-Tower vectors are persisted.
    No TMDB metadata is stored in the pool. EligibilityFilter is applied
    downstream at scoring time using live metadata.
    """

    def __init__(self, redis_client, pool_ttl: int = POOL_TTL_DEFAULT, pool_cap: int = POOL_CAP):
        self.redis_client = redis_client
        self.pool_ttl = pool_ttl
        self.pool_cap = pool_cap

    # ------------------------------------------------------------------ keys

    def _pool_key(self, user_id: str, content_type: str) -> str:
        return f"pool:vectors:{user_id}:{content_type}"

    def _shown_key(self, user_id: str, content_type: str) -> str:
        return f"pool:shown:{user_id}:{content_type}"

    def _cursor_key(self, user_id: str, content_type: str) -> str:
        return f"pool:cursor:{user_id}:{content_type}"

    # --------------------------------------------------------- pool vectors

    def _load_pool(self, user_id: str, content_type: str) -> List[Dict]:
        """Load pool entries from Redis. Returns [] on any failure."""
        try:
            raw = self.redis_client.get(self._pool_key(user_id, content_type))
            if raw:
                data = json.loads(raw)
                return data.get("entries", [])
        except Exception as e:
            logger.debug(f"Pool load failed for user {user_id} [{content_type}]: {e}")
        return []

    def _save_pool(self, user_id: str, content_type: str, entries: List[Dict]) -> bool:
        """Persist pool entries to Redis. Returns False on failure."""
        try:
            payload = json.dumps({"entries": entries, "saved_at": time.time()})
            self.redis_client.setex(self._pool_key(user_id, content_type), self.pool_ttl, payload)
            return True
        except Exception as e:
            logger.debug(f"Pool save failed for user {user_id} [{content_type}]: {e}")
            return False

    # ------------------------------------------------------- shown IDs

    def _load_shown(self, user_id: str, content_type: str) -> Set[int]:
        """Load shown post IDs from Redis. Returns empty set on failure."""
        try:
            raw = self.redis_client.get(self._shown_key(user_id, content_type))
            if raw:
                return set(json.loads(raw))
        except Exception as e:
            logger.debug(f"Shown IDs load failed for user {user_id} [{content_type}]: {e}")
        return set()

    def _save_shown(self, user_id: str, content_type: str, shown_ids: Set[int]) -> bool:
        """Persist shown post IDs to Redis. Returns False on failure."""
        try:
            payload = json.dumps(list(shown_ids))
            self.redis_client.setex(self._shown_key(user_id, content_type), self.pool_ttl, payload)
            return True
        except Exception as e:
            logger.debug(f"Shown IDs save failed for user {user_id} [{content_type}]: {e}")
            return False

    # --------------------------------------------------------- public API

    def insert_candidates(self, user_id: str, content_type: str,
                          new_vectors: Dict[int, np.ndarray]) -> int:
        """
        Insert freshly fetched candidates into the pool.

        Deduplicates against existing entries, then enforces the cap by
        FIFO tail-truncation (oldest entries dropped first).

        Returns the pool size after insertion (0 on Redis failure).
        """
        if not new_vectors:
            return self.get_pool_size(user_id, content_type)

        try:
            entries = self._load_pool(user_id, content_type)
            existing_ids = {e["post_id"] for e in entries}

            for post_id, vector in new_vectors.items():
                if post_id not in existing_ids:
                    entries.append({
                        "post_id": int(post_id),
                        "vector": vector.tolist()
                    })
                    existing_ids.add(post_id)

            # Enforce cap — keep the most recently inserted entries
            if len(entries) > self.pool_cap:
                entries = entries[-self.pool_cap:]

            self._save_pool(user_id, content_type, entries)
            size = len(entries)
            logger.debug(f"Pool insert: user {user_id} [{content_type}] → {size} entries "
                         f"({len(new_vectors)} new candidates offered)")
            return size

        except Exception as e:
            logger.warning(f"Pool insert failed (non-fatal) for user {user_id}: {e}")
            return 0

    def pull_candidates(self, user_id: str, content_type: str) -> Dict[int, np.ndarray]:
        """
        Return pooled candidates, excluding already-shown post IDs.

        Vectors are deserialized back to np.float32 arrays.
        Returns empty dict if pool is empty or Redis is unavailable.
        """
        try:
            pool_key  = self._pool_key(user_id, content_type)
            shown_key = self._shown_key(user_id, content_type)
            raw_pool, raw_shown = self.redis_client.mget([pool_key, shown_key])

            if not raw_pool:
                return {}
            entries   = json.loads(raw_pool).get("entries", [])
            shown_ids = set(json.loads(raw_shown)) if raw_shown else set()

            result = {}
            for entry in entries:
                post_id = entry["post_id"]
                if post_id not in shown_ids:
                    result[post_id] = np.array(entry["vector"], dtype=np.float32)

            logger.debug(f"Pool pull: user {user_id} [{content_type}] → "
                         f"{len(result)} available (pool={len(entries)}, shown={len(shown_ids)})")
            return result

        except Exception as e:
            logger.warning(f"Pool pull failed (non-fatal) for user {user_id}: {e}")
            return {}

    def mark_shown(self, user_id: str, content_type: str, post_ids: List[int]) -> None:
        """
        Record post IDs that were surfaced in a recommendation response.
        These will be excluded from future pool pulls for this user.
        """
        if not post_ids:
            return
        try:
            shown_ids = self._load_shown(user_id, content_type)
            shown_ids.update(post_ids)
            self._save_shown(user_id, content_type, shown_ids)
            logger.debug(f"Marked {len(post_ids)} posts shown for user {user_id} [{content_type}]. "
                         f"Total shown: {len(shown_ids)}")
        except Exception as e:
            logger.warning(f"mark_shown failed (non-fatal) for user {user_id}: {e}")

    def save_cursor(self, user_id: str, content_type: str,
                    cursor: Optional[str], high_quality_cursor: Optional[str]) -> None:
        """Persist cursor state to Redis so it survives service restarts."""
        try:
            payload = json.dumps({"cursor": cursor, "highQualityCursor": high_quality_cursor})
            self.redis_client.setex(self._cursor_key(user_id, content_type), self.pool_ttl, payload)
        except Exception as e:
            logger.debug(f"Cursor save failed for user {user_id} [{content_type}]: {e}")

    def load_cursor(self, user_id: str, content_type: str) -> Dict:
        """Load persisted cursor state from Redis. Returns empty dict if not found."""
        try:
            raw = self.redis_client.get(self._cursor_key(user_id, content_type))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug(f"Cursor load failed for user {user_id} [{content_type}]: {e}")
        return {"cursor": None, "highQualityCursor": None}

    def filter_shown(self, user_id: str, content_type: str,
                     candidates: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Return only candidates whose post IDs are not in the shown set.
        Used to strip already-recommended posts from a fresh API fetch before scoring.
        """
        if not candidates:
            return candidates
        shown_ids = self._load_shown(user_id, content_type)
        if not shown_ids:
            return candidates
        filtered = {pid: vec for pid, vec in candidates.items() if pid not in shown_ids}
        removed = len(candidates) - len(filtered)
        if removed:
            logger.debug(f"filter_shown: removed {removed} already-shown candidates "
                         f"for user {user_id} [{content_type}]")
        return filtered

    def get_pool_size(self, user_id: str, content_type: str) -> int:
        """Return number of entries currently in the pool. Returns 0 on failure."""
        try:
            entries = self._load_pool(user_id, content_type)
            return len(entries)
        except Exception:
            return 0

    def invalidate(self, user_id: str, content_type: str) -> None:
        """Delete pool, shown-IDs, and cursor keys for a user (e.g. after cursor reset)."""
        for key_fn in (self._pool_key, self._shown_key, self._cursor_key):
            try:
                self.redis_client.delete(key_fn(user_id, content_type))
            except Exception:
                pass
        logger.debug(f"Pool invalidated for user {user_id} [{content_type}]")
