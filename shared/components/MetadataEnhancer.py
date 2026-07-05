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
from GenreTextClassifier import classify_overview_genres
from AvoidedSignalTracker import AvoidedSignalTracker


def extract_person_ids(post_metadata: Dict, max_cast: int = 5) -> List[int]:
    """
    Extract cast/crew person IDs from post metadata for exact-ID avoidance matching.

    Cast is limited to the top-billed entries (list order, as returned by the API)
    rather than the full cast list, so the signal isn't diluted by minor/background
    roles. All listed crew departments (director, etc.) are included in full since
    crew lists are typically short and each entry is meaningfully influential.

    Args:
        post_metadata: Post metadata (cast, crew - TMDB data)
        max_cast: Number of top-billed cast entries to include

    Returns:
        List of person IDs (cast + crew), possibly with duplicates if someone
        appears in both (e.g. actor-director) - callers treat this as a bag.
    """
    person_ids = []

    cast = post_metadata.get('cast', []) or []
    for entry in cast[:max_cast]:
        person_id = entry.get('id')
        if person_id is not None:
            person_ids.append(int(person_id))

    crew = post_metadata.get('crew', {}) or {}
    for department_entries in crew.values():
        for entry in (department_entries or []):
            person_id = entry.get('id')
            if person_id is not None:
                person_ids.append(int(person_id))

    return person_ids

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

        # Avoid Genres Configuration (declared user preference, service-gated endpoint)
        self.avoid_genres_enabled = os.environ.get('AVOID_GENRES_ENABLED', 'true').lower() == 'true'

        # Inferred Avoided Signals (behaviorally derived from not_interested interactions,
        # covers multiple categories - genre, cast/crew, ... - see AvoidedSignalTracker)
        self.avoided_signal_tracker = AvoidedSignalTracker(redis_client)
        self.avoided_signal_penalty_enabled = os.environ.get('AVOIDED_SIGNAL_PENALTY_ENABLED', 'true').lower() == 'true'
        self.avoided_signal_penalty_per_count = float(os.environ.get('AVOIDED_SIGNAL_PENALTY_PER_COUNT', '0.15'))
        self.avoided_signal_min_penalty_factor = float(os.environ.get('AVOIDED_SIGNAL_MIN_PENALTY_FACTOR', '0.2'))

        # Resurfaced-favorite discount: candidates liked/saved long enough ago that the
        # DB query's rare-resurface gate let them back into the pool (see
        # PostLanguagesRepository.findScoredCandidates) still compete, just at a
        # disadvantage, rather than relying solely on how infrequently they're included.
        self.resurfaced_favorite_discount = float(os.environ.get('RESURFACED_FAVORITE_DISCOUNT', '0.5'))

        logger.info(f"TMDB ToS Compliant MetadataEnhancer initialized")
        logger.info(f"Velocity boost: enabled={self.velocity_boost_enabled}, "
                   f"threshold={self.velocity_trending_threshold}, "
                   f"boost={self.velocity_trending_boost}")
        logger.info(f"Behavioral genres: enabled={self.behavioral_genres_enabled}")
        logger.info(f"Avoid genres: enabled={self.avoid_genres_enabled}")
        logger.info(f"Avoided signal penalty: enabled={self.avoided_signal_penalty_enabled}, "
                   f"per_count={self.avoided_signal_penalty_per_count}, "
                   f"min_factor={self.avoided_signal_min_penalty_factor}")

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
        2b. Avoided-genre soft penalty (graduated, based on not_interested history)
        2c. Avoided-person soft penalty (graduated, cast/crew, based on not_interested history)
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

            # Fetched fresh every call (not through the hourly user-metadata cache) so a
            # not_interested interaction affects the very next recommendation, not the
            # next cache refresh up to an hour later.
            avoided_genre_counts = {}
            avoided_person_counts = {}
            if self.avoided_signal_penalty_enabled:
                avoided_genre_counts = self.avoided_signal_tracker.get_counts(user_id, 'genre')
                avoided_person_counts = self.avoided_signal_tracker.get_counts(user_id, 'person')

            # Candidates the user liked/saved long enough ago that they passed the
            # rare-resurface gate in the candidate query (see
            # PostLanguagesRepository.findScoredCandidates) are flagged, not excluded -
            # apply a discount here so they compete at a disadvantage instead of at full
            # strength, rather than relying on the low resurface probability alone to
            # keep them out of view.
            candidates_by_id = {c.get('postId'): c for c in candidates} if candidates else {}

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

                # Step 2b: Avoided-genre soft penalty (behaviorally inferred, graduated)
                if post_metadata and avoided_genre_counts:
                    enhanced_scores[i] *= self._calculate_avoided_genre_penalty(
                        post_metadata, avoided_genre_counts
                    )

                # Step 2c: Avoided-person soft penalty (cast/crew, behaviorally inferred, graduated)
                if post_metadata and avoided_person_counts:
                    enhanced_scores[i] *= self._calculate_avoided_person_penalty(
                        post_metadata, avoided_person_counts
                    )

                # Step 3: Behavioral engagement boost ONLY (app data, not TMDB)
                if post_metadata:
                    enhanced_scores[i] = self._apply_behavioral_boost(
                        enhanced_scores[i], post_metadata, int(post_id)
                    )

                # Step 3b: Resurfaced-favorite discount (liked/saved a long time ago,
                # let back into the candidate pool by the rare-resurface gate)
                candidate_meta = candidates_by_id.get(int(post_id)) or candidates_by_id.get(post_id)
                if candidate_meta:
                    is_resurfaced = candidate_meta.get('metadata', {}).get('sourceDetails', {}).get(
                        'isResurfacedFavorite', False
                    )
                    if is_resurfaced:
                        enhanced_scores[i] *= self.resurfaced_favorite_discount

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

    def _graduated_penalty(self, matched_counts: List[int]) -> float:
        """
        Shared graduated-penalty formula for behaviorally inferred avoidance signals.

        Scales with repetition rather than rejecting outright on a single
        occurrence - one not_interested click is noisy signal, a repeated
        pattern is not. Used by both the avoided-genre and avoided-person checks.

        Args:
            matched_counts: not_interested counts for each signal value the post matched

        Returns:
            Penalty factor (1.0 = no penalty, floored at avoided_signal_min_penalty_factor)
        """
        if not matched_counts:
            return 1.0
        worst_count = max(matched_counts)
        penalty = 1.0 - (self.avoided_signal_penalty_per_count * worst_count)
        return max(self.avoided_signal_min_penalty_factor, penalty)

    def _calculate_avoided_genre_penalty(self, post_metadata: Dict,
                                         avoided_genre_counts: Dict[str, int]) -> float:
        """
        Calculate a graduated soft penalty for posts whose overview classifies into
        a genre the user has repeatedly marked not_interested on.

        Unlike the hard avoidGenres eligibility filter (declared preference), this
        is inferred from behavior - see _graduated_penalty.

        Args:
            post_metadata: Post metadata (overview - TMDB data)
            avoided_genre_counts: User's genre -> not_interested count map

        Returns:
            Penalty factor (1.0 = no penalty, floored at avoided_signal_min_penalty_factor)
        """
        try:
            overview = post_metadata.get('overview', '')
            if not overview:
                return 1.0

            classified_genres = classify_overview_genres(overview)
            matched_counts = [
                avoided_genre_counts[g] for g in classified_genres if g in avoided_genre_counts
            ]
            return self._graduated_penalty(matched_counts)

        except Exception as e:
            logger.warning(f"Error calculating avoided-genre penalty: {e}")
            return 1.0

    def _calculate_avoided_person_penalty(self, post_metadata: Dict,
                                          avoided_person_counts: Dict[str, int]) -> float:
        """
        Calculate a graduated soft penalty for posts whose cast/crew includes a
        person the user has repeatedly marked not_interested on.

        Exact person-ID matching (cast + crew, see extract_person_ids) - no text
        mining involved, so no dependency on overview text being present.

        Args:
            post_metadata: Post metadata (cast, crew - TMDB data)
            avoided_person_counts: User's person_id (str) -> not_interested count map

        Returns:
            Penalty factor (1.0 = no penalty, floored at avoided_signal_min_penalty_factor)
        """
        try:
            person_ids = extract_person_ids(post_metadata)
            if not person_ids:
                return 1.0

            matched_counts = [
                avoided_person_counts[str(pid)] for pid in person_ids
                if str(pid) in avoided_person_counts
            ]
            return self._graduated_penalty(matched_counts)

        except Exception as e:
            logger.warning(f"Error calculating avoided-person penalty: {e}")
            return 1.0

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

            url = f"{self.api_base_url}/api/internal/posts/metadata/batch"

            headers = {'Content-Type': 'application/json'}
            headers.update(self._get_auth_headers())
            response = requests.post(url, json=post_ids, headers=headers, timeout=10)

            if response.status_code == 200:
                # Response is Map<Int, {"more_info": Map<String, Any?>}?> - convert
                # string keys to int and unwrap/normalize the more_info payload
                raw_response = response.json()
                return {int(k): self._normalize_post_metadata(v) for k, v in raw_response.items()}
            else:
                logger.warning(f"Batch API returned status {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"Error fetching batch metadata from API: {e}")
            return None

    def _normalize_post_metadata(self, raw: Optional[Dict]) -> Optional[Dict]:
        """
        Normalize a raw post-metadata API response into the flat, camelCase
        shape every consumer (EligibilityFilter, DiversityEnforcer,
        RecommendationExplainer, behavioral boost) expects.

        The API wraps the actual fields under a "more_info" key, and Postgres
        folds some unquoted column aliases to lowercase (voteaverage, releasedate).
        """
        if raw is None:
            return None

        metadata = raw.get('more_info', raw) if isinstance(raw, dict) else raw

        if 'voteaverage' in metadata and 'voteAverage' not in metadata:
            metadata['voteAverage'] = metadata['voteaverage']
        if 'releasedate' in metadata and 'releaseDate' not in metadata:
            metadata['releaseDate'] = metadata['releasedate']

        return metadata

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
                url = f"{self.api_base_url}/api/internal/users/{entity_id}/metadata"
                headers = self._get_auth_headers()
                response = requests.get(url, headers=headers, timeout=5)

                if response.status_code == 200:
                    metadata = response.json()
                    if self.behavioral_genres_enabled:
                        behavioral_genres = self._fetch_behavioral_genres(entity_id)
                        if behavioral_genres:
                            metadata['behavioralGenres'] = behavioral_genres
                    if self.avoid_genres_enabled:
                        avoid_genres = self._fetch_avoid_genres(entity_id)
                        if avoid_genres:
                            metadata['avoidGenres'] = avoid_genres
                    return metadata
                else:
                    logger.warning(f"API returned status {response.status_code} for {key}")
                    return None

            elif entity_type == 'post':
                # Individual post fetch as fallback (batch should be used normally)
                url = f"{self.api_base_url}/api/internal/posts/{entity_id}/metadata"
                headers = self._get_auth_headers()
                response = requests.get(url, headers=headers, timeout=5)

                if response.status_code == 200:
                    return self._normalize_post_metadata(response.json())
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

    def _fetch_avoid_genres(self, user_id: str) -> List[str]:
        """
        Fetch a user's declared avoid-genres from the service-gated Spring API endpoint.

        Args:
            user_id: User ID

        Returns:
            List of genre names the user wants excluded from recommendations
        """
        try:
            url = f"{self.api_base_url}/api/internal/users/{user_id}/avoidGenres"
            headers = self._get_auth_headers()
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                avoid_genres = [g.get('genre_name') for g in data if g.get('genre_name')]
                logger.debug(f"Fetched {len(avoid_genres)} avoid-genres for user {user_id}")
                return avoid_genres
            else:
                logger.debug(f"Avoid-genres API returned {response.status_code} for user {user_id}")
                return []

        except Exception as e:
            logger.debug(f"Error fetching avoid-genres for user {user_id}: {e}")
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
