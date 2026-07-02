"""
AvoidedSignalTracker: per-user negative-signal counts derived from not_interested
interactions, backed by Redis.

Complements declared preferences (e.g. avoidGenres - EligibilityFilter hard filter)
with *behaviorally inferred* graduated soft penalties. Deliberately generic over
signal category - a single not_interested click carries more than one avoidance
signal (genre, cast/crew, potentially more later), so this tracks each category
under its own Redis key rather than being hardcoded to one dimension like genre.

Currently used signal_type values:
- "genre": TMDB genre names classified from overview text (GenreTextClassifier)
- "person": TMDB cast/crew person IDs (exact ID matching, no text mining)

TMDB ToS Compliant: signal values are derived from TMDB data (overview text,
cast/crew IDs), but what's stored and applied here is a behavioral interaction
count used as a graduated score multiplier - the same category of signal as the
existing behavioral engagement boost in MetadataEnhancer, not a TMDB feature fed
into Two-Tower/RL scoring or training.
"""

import json
import logging
from typing import Dict, List

logger = logging.getLogger("avoided-signal-tracker")

SIGNAL_TTL_DEFAULT = 30 * 86400  # 30-day rolling window - old signal decays


class AvoidedSignalTracker:
    """Redis-backed per-user, per-signal-category value -> not_interested count map."""

    def __init__(self, redis_client, ttl: int = SIGNAL_TTL_DEFAULT):
        self.redis_client = redis_client
        self.ttl = ttl

    def _key(self, user_id: str, signal_type: str) -> str:
        return f"avoided_signals:{signal_type}:{user_id}"

    def record(self, user_id: str, signal_type: str, values: List) -> None:
        """Increment counts for each value (genre name, person id, ...) under signal_type."""
        if not values:
            return
        try:
            key = self._key(user_id, signal_type)
            raw = self.redis_client.get(key)
            counts: Dict[str, int] = json.loads(raw) if raw else {}
            for value in values:
                str_value = str(value)
                counts[str_value] = counts.get(str_value, 0) + 1
            self.redis_client.setex(key, self.ttl, json.dumps(counts))
            logger.debug(f"Recorded avoided {signal_type} for user {user_id}: {values}")
        except Exception as e:
            logger.warning(f"Failed to record avoided {signal_type} for user {user_id}: {e}")

    def get_counts(self, user_id: str, signal_type: str) -> Dict[str, int]:
        """Return this user's value -> not_interested count map for signal_type. Empty on failure."""
        try:
            raw = self.redis_client.get(self._key(user_id, signal_type))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug(f"Failed to load avoided {signal_type} for user {user_id}: {e}")
        return {}
