import time
import logging
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger("metadata-enhancer")


class MetadataEnhancer:
    """
    Enhances recommendation scores using metadata from user and post vectors.
    Handles language preferences, candidate types, genre matching, and other metadata-based boosting.
    """

    def __init__(self, api_base_url: str, redis_client=None, cache_ttl: int = 3600):
        """
        Initialize the metadata enhancer.

        Args:
            api_base_url: Base URL for the Spring API
            redis_client: Redis client for caching (optional)
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_base_url = api_base_url
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl

        # In-memory cache for when Redis is not available
        self.memory_cache = {}
        self.cache_timestamps = {}

        # Boost factors for different enhancement types
        self.language_boost_factor = 0.3  # Up to 30% boost for language preferences
        self.genre_boost_factor = 0.2  # Up to 20% boost for genre preferences
        self.popularity_boost_factor = 0.15  # Up to 15% boost for popular content
        self.recency_boost_factor = 0.1  # Up to 10% boost for recent content

    def enhance_scores(self, user_id: str, post_ids: List[int], base_scores: np.ndarray,
                       candidates: List[Dict] = None, content_type: str = "posts") -> np.ndarray:
        """
        Enhance base ML scores using metadata (excluding candidate boosting which is handled in Two-Tower model).

        Args:
            user_id: User ID
            post_ids: List of post IDs
            base_scores: Base scores from ML model (already include candidate boosting)
            candidates: Candidate metadata (for reference only)
            content_type: Type of content (posts/trailers)

        Returns:
            Enhanced scores array
        """
        try:
            enhanced_scores = base_scores.copy()

            # Get user metadata for preference-based enhancement
            user_metadata = self._get_cached_metadata(f"user:{user_id}")

            if user_metadata:
                # Apply language preference boosting
                enhanced_scores = self._apply_language_boosting(
                    enhanced_scores, post_ids, user_metadata
                )

                # Apply genre preference boosting
                enhanced_scores = self._apply_genre_boosting(
                    enhanced_scores, post_ids, user_metadata
                )

                # Apply demographic boosting (region-based)
                enhanced_scores = self._apply_demographic_boosting(
                    enhanced_scores, post_ids, user_metadata
                )

            # Apply content-specific boosting (popularity, engagement, etc.)
            enhanced_scores = self._apply_content_boosting(
                enhanced_scores, post_ids, content_type
            )

            # Ensure scores stay in valid range [0, 1]
            enhanced_scores = np.clip(enhanced_scores, 0.0, 1.0)

            return enhanced_scores

        except Exception as e:
            logger.warning(f"Error enhancing scores: {e}")
            return base_scores

    def _apply_language_boosting(self, scores: np.ndarray, post_ids: List[int],
                                 user_metadata: Dict) -> np.ndarray:
        """Apply language preference boosting to scores."""
        try:
            # Extract language preferences from user metadata
            language_weights = user_metadata.get('languageWeights', {})
            if not language_weights:
                return scores

            user_language_prefs = language_weights.get('weights', {})
            if not user_language_prefs:
                return scores

            enhanced_scores = scores.copy()

            # Get post languages and apply boosts
            for i, post_id in enumerate(post_ids):
                post_metadata = self._get_cached_metadata(f"post:{post_id}")
                if not post_metadata:
                    continue

                # Get post language
                categorical_features = post_metadata.get('categoricalFeatures', {})
                post_language = categorical_features.get('language', 'en')

                # Apply language boost if user has preference for this language
                if post_language in user_language_prefs:
                    language_weight = user_language_prefs[post_language]
                    boost_factor = 1.0 + (language_weight * self.language_boost_factor)
                    enhanced_scores[i] *= boost_factor

            return enhanced_scores

        except Exception as e:
            logger.warning(f"Error applying language boosting: {e}")
            return scores

    def _apply_genre_boosting(self, scores: np.ndarray, post_ids: List[int],
                              user_metadata: Dict) -> np.ndarray:
        """Apply genre preference boosting to scores."""
        try:
            # Extract user genre preferences
            interest_weights = user_metadata.get('interestWeights', {})
            if not interest_weights:
                return scores

            enhanced_scores = scores.copy()

            # Get post genres and apply boosts
            for i, post_id in enumerate(post_ids):
                post_metadata = self._get_cached_metadata(f"post:{post_id}")
                if not post_metadata:
                    continue

                # Get post genre weights
                genre_weights = post_metadata.get('genreWeights', {})
                if not genre_weights:
                    continue

                # Calculate genre alignment score
                genre_alignment = 0.0
                total_weight = 0.0

                for genre, post_weight in genre_weights.items():
                    if genre in interest_weights:
                        user_preference = interest_weights[genre]
                        genre_alignment += user_preference * post_weight
                        total_weight += post_weight

                # Apply boost based on genre alignment
                if total_weight > 0:
                    alignment_score = genre_alignment / total_weight
                    boost_factor = 1.0 + (alignment_score * self.genre_boost_factor)
                    enhanced_scores[i] *= boost_factor

            return enhanced_scores

        except Exception as e:
            logger.warning(f"Error applying genre boosting: {e}")
            return scores

    def _apply_demographic_boosting(self, scores: np.ndarray, post_ids: List[int],
                                    user_metadata: Dict) -> np.ndarray:
        """Apply demographic-based boosting (region, age group, etc.)."""
        try:
            # Extract user region
            categorical_features = user_metadata.get('categoricalFeatures', {})
            user_region = categorical_features.get('region')

            if not user_region:
                return scores

            enhanced_scores = scores.copy()

            # Apply region-based boosting
            for i, post_id in enumerate(post_ids):
                post_metadata = self._get_cached_metadata(f"post:{post_id}")
                if not post_metadata:
                    continue

                # Check if this content is popular in user's region
                region_weights = post_metadata.get('regionWeights', {})
                if user_region in region_weights:
                    region_popularity = region_weights[user_region]
                    # Boost by up to 10% for content popular in user's region
                    boost_factor = 1.0 + (region_popularity * 0.1)
                    enhanced_scores[i] *= boost_factor

            return enhanced_scores

        except Exception as e:
            logger.warning(f"Error applying demographic boosting: {e}")
            return scores

    def _apply_content_boosting(self, scores: np.ndarray, post_ids: List[int],
                                content_type: str) -> np.ndarray:
        """Apply content-specific boosting (popularity, recency, etc.)."""
        try:
            enhanced_scores = scores.copy()

            for i, post_id in enumerate(post_ids):
                post_metadata = self._get_cached_metadata(f"post:{post_id}")
                if not post_metadata:
                    continue

                # Apply popularity boosting
                vote_average = post_metadata.get('voteAverage', 0)
                if isinstance(vote_average, (int, float)) and vote_average > 7.0:
                    popularity_boost = 1.0 + ((vote_average - 7.0) / 3.0 * self.popularity_boost_factor)
                    enhanced_scores[i] *= popularity_boost

                # Apply engagement boosting (info button clicks, like counts)
                info_clicks = post_metadata.get('infoButtonClicks', {})
                if isinstance(info_clicks, dict):
                    click_count = info_clicks.get('count', 0)
                    if click_count > 10:  # Threshold for popular content
                        engagement_boost = 1.0 + min(0.1, click_count / 1000.0)  # Up to 10% boost
                        enhanced_scores[i] *= engagement_boost

                # Apply recency boosting for new content
                # (This would require release date analysis)

            return enhanced_scores

        except Exception as e:
            logger.warning(f"Error applying content boosting: {e}")
            return scores

    def _get_cached_metadata(self, key: str) -> Optional[Dict]:
        """Get metadata from cache or API with caching."""
        now = time.time()

        # Check in-memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if now - entry['timestamp'] < self.cache_ttl:
                return entry['data']

        # Check Redis cache if available
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(f"metadata:{key}")
                if cached_data:
                    import json
                    data = json.loads(cached_data)
                    # Update in-memory cache
                    self.memory_cache[key] = {'data': data, 'timestamp': now}
                    return data
            except Exception as e:
                logger.warning(f"Error reading from Redis cache: {e}")

        # Fetch from API
        try:
            data = self._fetch_metadata_from_api(key)
            if data:
                # Cache in memory
                self.memory_cache[key] = {'data': data, 'timestamp': now}

                # Cache in Redis if available
                if self.redis_client:
                    try:
                        import json
                        self.redis_client.setex(
                            f"metadata:{key}",
                            self.cache_ttl,
                            json.dumps(data)
                        )
                    except Exception as e:
                        logger.warning(f"Error writing to Redis cache: {e}")

                return data
        except Exception as e:
            logger.warning(f"Error fetching metadata for {key}: {e}")

        # Cleanup old in-memory cache entries
        self._cleanup_memory_cache(now)

        return None

    def _fetch_metadata_from_api(self, key: str) -> Optional[Dict]:
        """Fetch metadata from the Spring API."""
        try:
            parts = key.split(':')
            if len(parts) != 2:
                return None

            entity_type, entity_id = parts

            if entity_type == 'user':
                url = f"{self.api_base_url}/api/recommendations/users/{entity_id}/metadata"
            elif entity_type == 'post':
                url = f"{self.api_base_url}/api/recommendations/posts/{entity_id}/metadata"
            else:
                return None

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API returned status {response.status_code} for {key}")
                return None

        except Exception as e:
            logger.warning(f"Error fetching metadata from API for {key}: {e}")
            return None

    def _cleanup_memory_cache(self, current_time: float):
        """Clean up old entries from memory cache."""
        try:
            # Only cleanup if cache is getting large
            if len(self.memory_cache) > 1000:
                # Remove entries older than cache_ttl
                keys_to_remove = []
                for key, entry in self.memory_cache.items():
                    if current_time - entry['timestamp'] > self.cache_ttl:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.memory_cache[key]

                if keys_to_remove:
                    logger.debug(f"Cleaned up {len(keys_to_remove)} old cache entries")

        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get statistics about metadata enhancement usage."""
        return {
            "cache_size": len(self.memory_cache),
            "redis_available": self.redis_client is not None,
            "boost_factors": {
                "language": self.language_boost_factor,
                "genre": self.genre_boost_factor,
                "popularity": self.popularity_boost_factor,
                "recency": self.recency_boost_factor
            }
        }

    def clear_cache(self):
        """Clear all cached metadata."""
        self.memory_cache.clear()
        if self.redis_client:
            try:
                # Clear all metadata keys from Redis
                keys = self.redis_client.keys("metadata:*")
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} metadata entries from Redis")
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")

        logger.info("Metadata cache cleared")