"""
MetadataEnhancer: TMDB ToS Compliant Score Enhancement

This module enhances recommendation scores using ONLY behavioral data from the app.
TMDB data is used for:
- Boolean eligibility filtering (pass/fail, not score modification)
- Diversity reranking (reordering, not score modification)

ML features come ONLY from user behavior within the app (clicks, likes, saves, etc.).

Architecture:
    Two-Tower ML → EligibilityFilter → BehavioralBoost → DiversityEnforcer → Final
                       ↑                                        ↑
                  TMDB (boolean)                           TMDB (reorder only)
"""

import time
import logging
import os
import threading
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

# Shared executor for parallel metadata prefetch calls
_meta_executor = ThreadPoolExecutor(max_workers=4)

from EligibilityFilter import EligibilityFilter, create_eligibility_filter
from DiversityEnforcer import DiversityEnforcer, create_diversity_enforcer

logger = logging.getLogger("metadata-enhancer")


class MetadataEnhancer:
    """
    TMDB ToS Compliant MetadataEnhancer.

    Enhances recommendation scores using ONLY behavioral data from the app.
    TMDB metadata is used for:
    - Boolean eligibility filtering (not score modification)
    - Diversity reranking (reordering, not score modification)

    Features:
    - Engagement velocity boosting for trending content
    - Behavioral genre preferences for personalization
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

        # JWT token for API authentication (can be updated via set_jwt_token)
        self._jwt_token = None

        # In-memory cache for when Redis is not available
        self.memory_cache = {}
        self.cache_timestamps = {}
        self._cache_lock = threading.Lock()

        # Initialize TMDB ToS compliant components
        self.eligibility_filter = create_eligibility_filter(
            quality_threshold=5.0,
            min_vote_count=10
        )
        self.diversity_enforcer = create_diversity_enforcer(
            max_consecutive_same_genre=2,
            new_release_positions=[0, 5, 10],
            hidden_gem_positions=[3, 8]
        )

        # Behavioral boost factors (app data ONLY - not TMDB)
        self.engagement_boost_factor = 0.1  # Up to 10% boost for app engagement
        self.click_normalization = 1000.0  # Normalization factor for click counts

        # Engagement Velocity Configuration
        self.velocity_boost_enabled = os.environ.get('VELOCITY_BOOST_ENABLED', 'true').lower() == 'true'
        self.velocity_trending_threshold = float(os.environ.get('VELOCITY_TRENDING_THRESHOLD', '1.5'))
        self.velocity_trending_boost = float(os.environ.get('VELOCITY_TRENDING_BOOST', '1.2'))
        self.velocity_cache_ttl = int(os.environ.get('VELOCITY_CACHE_TTL', '300'))

        # Behavioral Genres Configuration
        self.behavioral_genres_enabled = os.environ.get('BEHAVIORAL_GENRES_ENABLED', 'true').lower() == 'true'
        self.behavioral_genres_cache_ttl = int(os.environ.get('BEHAVIORAL_GENRES_CACHE_TTL', '3600'))

        logger.info(f"TMDB ToS Compliant MetadataEnhancer initialized")
        logger.info(f"Velocity boost: enabled={self.velocity_boost_enabled}, "
                   f"threshold={self.velocity_trending_threshold}, "
                   f"boost={self.velocity_trending_boost}")
        logger.info(f"Behavioral genres: enabled={self.behavioral_genres_enabled}")

    def set_jwt_token(self, token: str):
        """Update the JWT token used for API authentication."""
        self._jwt_token = token
        logger.debug(f"MetadataEnhancer JWT token updated (length: {len(token) if token else 0})")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {}
        auth_token = self._jwt_token or os.environ.get('SERVICE_AUTH_TOKEN', '')
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
            headers['X-Service-Role'] = 'SERVICE'
        return headers

    def enhance_scores(self, user_id: str, post_ids: List[int], base_scores: np.ndarray,
                       candidates: List[Dict] = None, content_type: str = "posts") -> np.ndarray:
        """
        TMDB ToS Compliant score enhancement.

        Enhancement pipeline:
        1. Eligibility filtering (TMDB data - boolean pass/fail)
        2. Behavioral genre soft filter (penalize non-matching genres)
        3. Behavioral engagement boost (app data ONLY)
        4. Engagement velocity boost (trending content)

        Diversity reranking is applied separately via apply_diversity_reranking().

        Args:
            user_id: User ID
            post_ids: List of post IDs
            base_scores: Base scores from ML model
            candidates: Candidate metadata (for reference only)
            content_type: Type of content (posts/trailers)

        Returns:
            Enhanced scores array (filtered items have score 0.0)
        """
        try:
            enhanced_scores = base_scores.copy()

            # Batch prefetch all metadata in parallel (post meta, velocity, user meta are independent)
            future_post_meta = _meta_executor.submit(self._prefetch_post_metadata, post_ids)
            future_velocity = _meta_executor.submit(
                self._prefetch_engagement_velocity, post_ids
            ) if self.velocity_boost_enabled else None
            future_user_meta = _meta_executor.submit(
                self._get_cached_metadata, f"user:{user_id}"
            )

            future_post_meta.result()   # writes into self.memory_cache
            if future_velocity is not None:
                future_velocity.result()
            user_metadata = future_user_meta.result()

            for i, post_id in enumerate(post_ids):
                post_metadata = self._get_cached_metadata(f"post:{post_id}")

                # Step 1: Eligibility filter (boolean - TMDB data)
                if post_metadata and user_metadata:
                    if not self.eligibility_filter.check_eligibility(post_metadata, user_metadata):
                        enhanced_scores[i] = 0.0  # Filter out
                        continue

                    # Step 2: Behavioral genre soft filter (penalize but don't reject)
                    genre_penalty = self.eligibility_filter.calculate_soft_filters(
                        post_metadata, user_metadata
                    )
                    enhanced_scores[i] *= genre_penalty

                # Step 3: Behavioral engagement boost ONLY (app data, not TMDB)
                if post_metadata:
                    enhanced_scores[i] = self._apply_behavioral_boost(
                        enhanced_scores[i], post_metadata, int(post_id)
                    )

            # Ensure scores stay in valid range [0, 1]
            enhanced_scores = np.clip(enhanced_scores, 0.0, 1.0)

            return enhanced_scores

        except Exception as e:
            logger.warning(f"Error enhancing scores: {e}")
            return base_scores

    def _apply_behavioral_boost(self, score: float, post_metadata: Dict, post_id: int = None) -> float:
        """
        Apply behavioral boost based on app engagement data ONLY.

        This uses ONLY data from user interactions within the app,
        NOT TMDB data like voteAverage, popularity, etc.

        Behavioral signals from app:
        - infoButtonClicks: How often users click for more info
        - likeCount: Number of likes in the app
        - saveCount: Number of saves in the app
        - shareCount: Number of shares in the app
        - viewCount: Number of views in the app

        Args:
            score: Base score
            post_metadata: Post metadata
            post_id: Post ID for velocity lookup

        Returns:
            Boosted score (app engagement only)
        """
        try:
            boost_factor = 1.0

            # Info button clicks (app engagement metric)
            info_clicks = post_metadata.get('infoButtonClicks', {})
            if isinstance(info_clicks, dict):
                click_count = info_clicks.get('count', 0)
                if click_count > 10:
                    engagement_boost = min(
                        self.engagement_boost_factor,
                        click_count / self.click_normalization
                    )
                    boost_factor += engagement_boost

            # App-specific engagement ratios (if available)
            view_count = post_metadata.get('viewCount', 0)
            if view_count > 0:
                like_count = post_metadata.get('likeCount', 0)
                like_ratio = like_count / view_count
                if like_ratio > 0.1:
                    boost_factor += min(0.05, like_ratio * 0.2)

                save_count = post_metadata.get('saveCount', 0)
                save_ratio = save_count / view_count
                if save_ratio > 0.05:
                    boost_factor += min(0.05, save_ratio * 0.3)

            # Apply engagement velocity boost for trending content
            if self.velocity_boost_enabled and post_id is not None:
                velocity_data = self._get_cached_metadata(f"velocity:{post_id}")
                if velocity_data:
                    velocity_boost = self._calculate_velocity_boost(velocity_data)
                    boost_factor *= velocity_boost

            return score * boost_factor

        except Exception as e:
            logger.warning(f"Error applying behavioral boost: {e}")
            return score

    def _calculate_velocity_boost(self, velocity_data: Dict) -> float:
        """
        Calculate boost factor based on engagement velocity data.

        Args:
            velocity_data: Engagement velocity data from API

        Returns:
            Boost factor (1.0 = no change, >1.0 = boost, <1.0 = penalty)
        """
        try:
            if not velocity_data:
                return 1.0

            is_trending = velocity_data.get('isTrending', False)
            hourly_velocity = velocity_data.get('hourlyVelocity', 0.0)
            daily_velocity = velocity_data.get('dailyVelocity', 0.0)

            # Trending content gets a boost
            if is_trending:
                return self.velocity_trending_boost

            # Non-trending: scale boost based on velocity
            if hourly_velocity > 0:
                normalized_velocity = min(hourly_velocity / self.velocity_trending_threshold, 2.0)
                if normalized_velocity > 1.0:
                    return 1.0 + (normalized_velocity - 1.0) * 0.1
                elif normalized_velocity < 0.5:
                    return 0.95  # Slight penalty for very low velocity

            return 1.0

        except Exception as e:
            logger.warning(f"Error calculating velocity boost: {e}")
            return 1.0

    def _prefetch_engagement_velocity(self, post_ids: List[int]) -> None:
        """
        Batch prefetch engagement velocity for posts.
        Uses shorter TTL since velocity changes frequently.

        Args:
            post_ids: List of post IDs to fetch velocity for
        """
        now = time.time()

        posts_to_fetch = []
        for post_id in post_ids:
            key = f"velocity:{post_id}"
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if now - entry['timestamp'] < self.velocity_cache_ttl:
                    continue
            posts_to_fetch.append(post_id)

        if not posts_to_fetch:
            return

        # Check Redis for velocity cache
        if self.redis_client:
            try:
                import json
                keys = [f"metadata:velocity:{post_id}" for post_id in posts_to_fetch]
                values = self.redis_client.mget(keys)
                still_need_fetch = []
                for post_id, cached_data in zip(posts_to_fetch, values):
                    if cached_data:
                        self.memory_cache[f"velocity:{post_id}"] = {
                            'data': json.loads(cached_data), 'timestamp': now
                        }
                    else:
                        still_need_fetch.append(post_id)
                posts_to_fetch = still_need_fetch
            except Exception as e:
                logger.warning(f"Error checking Redis for velocity cache: {e}")

        if not posts_to_fetch:
            return

        # Batch fetch velocity from API
        try:
            batch_velocity = self._fetch_batch_engagement_velocity(posts_to_fetch)
            if batch_velocity:
                import json
                for post_id, velocity_data in batch_velocity.items():
                    if velocity_data is not None:
                        key = f"velocity:{post_id}"
                        self.memory_cache[key] = {'data': velocity_data, 'timestamp': now}

                        if self.redis_client:
                            try:
                                self.redis_client.setex(
                                    f"metadata:velocity:{post_id}",
                                    self.velocity_cache_ttl,
                                    json.dumps(velocity_data)
                                )
                            except Exception as e:
                                logger.warning(f"Error caching velocity to Redis: {e}")

                logger.debug(f"Batch fetched velocity for {len(batch_velocity)} posts")
        except Exception as e:
            logger.warning(f"Error batch fetching engagement velocity: {e}")

    def _fetch_batch_engagement_velocity(self, post_ids: List[int]) -> Optional[Dict[int, Dict]]:
        """
        Fetch engagement velocity data for multiple posts in a single API call.

        Args:
            post_ids: List of post IDs

        Returns:
            Dict mapping post_id to velocity data
        """
        try:
            if not post_ids:
                return {}

            url = f"{self.api_base_url}/api/recommendations/posts/engagement-velocity/batch"

            headers = {'Content-Type': 'application/json'}
            headers.update(self._get_auth_headers())
            response = requests.post(url, json=post_ids, headers=headers, timeout=10)

            if response.status_code == 200:
                raw_response = response.json()
                return {int(k): v for k, v in raw_response.items()}
            else:
                logger.warning(f"Velocity batch API returned status {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"Error fetching batch engagement velocity: {e}")
            return None

    def apply_diversity_reranking(self, post_ids: List[int],
                                  scores: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Apply diversity reranking to scored posts.

        This reorders posts for diversity WITHOUT modifying scores.
        Uses TMDB genre data for reordering decisions only.

        Args:
            post_ids: List of post IDs
            scores: Corresponding scores

        Returns:
            Tuple of (reranked_post_ids, reranked_scores)
        """
        try:
            # Filter out zero-scored (filtered) items
            non_zero_mask = scores > 0.0
            valid_post_ids = [pid for i, pid in enumerate(post_ids) if non_zero_mask[i]]
            valid_scores = scores[non_zero_mask]

            if len(valid_post_ids) == 0:
                return [], np.array([])

            # Build metadata map for diversity decisions
            metadata_map = {}
            for post_id in valid_post_ids:
                metadata = self._get_cached_metadata(f"post:{post_id}")
                if metadata:
                    metadata_map[post_id] = metadata

            # Create ranked list
            ranked_posts = list(zip(valid_post_ids, valid_scores.tolist()))
            ranked_posts.sort(key=lambda x: x[1], reverse=True)

            # Apply diversity reranking
            reranked = self.diversity_enforcer.apply_diversity(ranked_posts, metadata_map)

            # Extract results
            reranked_ids = [pid for pid, _ in reranked]
            reranked_scores = np.array([score for _, score in reranked], dtype=np.float32)

            return reranked_ids, reranked_scores

        except Exception as e:
            logger.warning(f"Error applying diversity reranking: {e}")
            # Return original order on error
            return list(post_ids), scores

    def get_filtered_posts(self, post_ids: List[int], scores: np.ndarray) -> List[int]:
        """
        Get list of posts that were filtered out (score = 0).

        Args:
            post_ids: List of post IDs
            scores: Corresponding scores

        Returns:
            List of filtered post IDs
        """
        return [pid for i, pid in enumerate(post_ids) if scores[i] == 0.0]

    def _prefetch_post_metadata(self, post_ids: List[int]) -> None:
        """
        Batch prefetch metadata for all posts to avoid N+1 API calls.
        Only fetches posts that are not already cached.
        """
        now = time.time()

        # Find posts that need fetching (not in cache or expired)
        posts_to_fetch = []
        for post_id in post_ids:
            key = f"post:{post_id}"
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if now - entry['timestamp'] < self.cache_ttl:
                    continue  # Already cached and valid
            posts_to_fetch.append(post_id)

        if not posts_to_fetch:
            return  # All posts already cached

        # Check Redis for any missing posts
        if self.redis_client:
            try:
                import json
                keys = [f"metadata:post:{post_id}" for post_id in posts_to_fetch]
                values = self.redis_client.mget(keys)
                still_need_fetch = []
                for post_id, cached_data in zip(posts_to_fetch, values):
                    if cached_data:
                        self.memory_cache[f"post:{post_id}"] = {
                            'data': json.loads(cached_data), 'timestamp': now
                        }
                    else:
                        still_need_fetch.append(post_id)
                posts_to_fetch = still_need_fetch
            except Exception as e:
                logger.warning(f"Error checking Redis cache for batch: {e}")

        if not posts_to_fetch:
            return  # All posts found in Redis

        # Batch fetch from API
        try:
            batch_metadata = self._fetch_batch_post_metadata(posts_to_fetch)
            if batch_metadata:
                import json
                for post_id, metadata in batch_metadata.items():
                    if metadata is not None:
                        key = f"post:{post_id}"
                        self.memory_cache[key] = {'data': metadata, 'timestamp': now}

                        # Also cache in Redis
                        if self.redis_client:
                            try:
                                self.redis_client.setex(
                                    f"metadata:{key}",
                                    self.cache_ttl,
                                    json.dumps(metadata)
                                )
                            except Exception as e:
                                logger.warning(f"Error caching to Redis: {e}")

                logger.debug(f"Batch fetched metadata for {len(batch_metadata)} posts")
        except Exception as e:
            logger.warning(f"Error batch fetching post metadata: {e}")

    def _fetch_batch_post_metadata(self, post_ids: List[int]) -> Optional[Dict[int, Dict]]:
        """Fetch metadata for multiple posts in a single batch API call."""
        try:
            if not post_ids:
                return {}

            url = f"{self.api_base_url}/api/recommendations/posts/metadata/batch"

            headers = {'Content-Type': 'application/json'}
            headers.update(self._get_auth_headers())
            response = requests.post(url, json=post_ids, headers=headers, timeout=10)

            if response.status_code == 200:
                # Response is Map<Int, Map<String, Any?>?> - convert string keys to int
                raw_response = response.json()
                return {int(k): v for k, v in raw_response.items()}
            else:
                logger.warning(f"Batch API returned status {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"Error fetching batch metadata from API: {e}")
            return None

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

        # Fetch from API (for user metadata or individual post fallback)
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
        """Fetch metadata from the Spring API (for user metadata only, posts use batch)."""
        try:
            parts = key.split(':')
            if len(parts) != 2:
                return None

            entity_type, entity_id = parts

            if entity_type == 'user':
                url = f"{self.api_base_url}/api/recommendations/users/{entity_id}/metadata"
                headers = self._get_auth_headers()
                response = requests.get(url, headers=headers, timeout=5)

                if response.status_code == 200:
                    metadata = response.json()
                    if self.behavioral_genres_enabled:
                        behavioral_genres = self._fetch_behavioral_genres(entity_id)
                        if behavioral_genres:
                            metadata['behavioralGenres'] = behavioral_genres
                    return metadata
                else:
                    logger.warning(f"API returned status {response.status_code} for {key}")
                    return None

            elif entity_type == 'post':
                # Individual post fetch as fallback (batch should be used normally)
                url = f"{self.api_base_url}/api/recommendations/posts/{entity_id}/metadata"
                headers = self._get_auth_headers()
                response = requests.get(url, headers=headers, timeout=5)

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"API returned status {response.status_code} for {key}")
                    return None
            else:
                return None

        except Exception as e:
            logger.warning(f"Error fetching metadata from API for {key}: {e}")
            return None

    def _fetch_behavioral_genres(self, user_id: str) -> List[str]:
        """
        Fetch behavioral genre preferences for a user from the API.

        Args:
            user_id: User ID

        Returns:
            List of top genre names based on user's behavioral interactions
        """
        try:
            url = f"{self.api_base_url}/api/users/{user_id}/behavioral-genres"
            headers = self._get_auth_headers()
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                top_genres = data.get('topGenres', [])
                logger.debug(f"Fetched {len(top_genres)} behavioral genres for user {user_id}")
                return top_genres
            else:
                logger.debug(f"Behavioral genres API returned {response.status_code} for user {user_id}")
                return []

        except Exception as e:
            logger.debug(f"Error fetching behavioral genres for user {user_id}: {e}")
            return []

    def _cleanup_memory_cache(self, current_time: float):
        """Clean up old entries from memory cache."""
        try:
            # Only cleanup if cache is getting large
            if len(self.memory_cache) > 1000:
                with self._cache_lock:
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
            "tmdb_tos_compliant": True,
            "eligibility_filter_stats": self.eligibility_filter.get_stats(),
            "diversity_enforcer_stats": self.diversity_enforcer.get_stats(),
            "boost_factors": {
                "engagement": self.engagement_boost_factor,
                "note": "TMDB boosts removed for ToS compliance"
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

    # =========================================================================
    # DEPRECATED METHODS (kept for backward compatibility during migration)
    # These will log warnings and return neutral values
    # =========================================================================

    @property
    def language_boost_factor(self) -> float:
        """DEPRECATED: TMDB language boosting removed for ToS compliance."""
        return 0.0

    @property
    def genre_boost_factor(self) -> float:
        """DEPRECATED: TMDB genre boosting removed for ToS compliance."""
        return 0.0

    @property
    def popularity_boost_factor(self) -> float:
        """DEPRECATED: TMDB popularity boosting removed for ToS compliance."""
        return 0.0

    @property
    def recency_boost_factor(self) -> float:
        """DEPRECATED: TMDB recency boosting removed for ToS compliance."""
        return 0.0

    @property
    def cast_crew_boost_factor(self) -> float:
        """DEPRECATED: TMDB cast/crew boosting removed for ToS compliance."""
        return 0.0
