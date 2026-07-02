"""
EligibilityFilter: TMDB ToS Compliant Boolean Include/Exclude Filter

This module provides boolean eligibility filtering based on TMDB data.
TMDB data is used ONLY for filtering decisions (pass/fail), NOT for ML scoring/training.

Compliance: TMDB data affects eligibility (boolean), not recommendation scores.

Features:
- Boolean filters: language, quality, providers
- Soft filters: behavioral genres (penalize but don't reject)
"""

import logging
from typing import Dict, List, Optional, Set

from GenreTextClassifier import classify_overview_genres

logger = logging.getLogger("eligibility-filter")


class EligibilityFilter:
    """
    TMDB ToS Compliant - Boolean include/exclude only.

    This filter uses TMDB metadata to determine if content is eligible
    for a specific user, but does NOT use TMDB data as ML features.

    Filtering is boolean:
    - True = content is eligible, passes to scoring
    - False = content is filtered out, score set to 0

    Soft filters (behavioral genres):
    - Return penalty factor (0.0 to 1.0) applied to score
    """

    def __init__(self, quality_threshold: float = 5.0, min_vote_count: int = 10,
                 behavioral_genres_enabled: bool = True):
        """
        Initialize the eligibility filter.

        Args:
            quality_threshold: Minimum vote average for quality filter (default 5.0)
            min_vote_count: Minimum votes required to apply quality filter (default 10)
            behavioral_genres_enabled: Enable behavioral genre soft filter (default True)
        """
        self.quality_threshold = quality_threshold
        self.min_vote_count = min_vote_count
        self.behavioral_genres_enabled = behavioral_genres_enabled

        # Soft filter penalty factor for behavioral genres
        self.behavioral_genre_penalty = 0.8  # 20% penalty if no genre match

        # Statistics tracking
        self.stats = {
            'total_checks': 0,
            'filtered_language': 0,
            'filtered_quality': 0,
            'filtered_providers': 0,
            'filtered_avoid_genres': 0,
            'passed': 0,
            'behavioral_genre_checks': 0,
            'behavioral_genre_matches': 0
        }

        logger.info(f"EligibilityFilter initialized: quality_threshold={quality_threshold}, "
                   f"min_vote_count={min_vote_count}, "
                   f"behavioral_genres_enabled={behavioral_genres_enabled}")

    def check_eligibility(self, post_metadata: Dict, user_metadata: Dict) -> bool:
        """
        Check if a post is eligible for a user.

        Returns True if post passes all eligibility filters.
        Returns False if post should be filtered out.

        Note: Behavioral genre filtering is a SOFT filter - use calculate_soft_filters()
        to apply genre-based penalties.

        Args:
            post_metadata: Post metadata including TMDB data
            user_metadata: User metadata including preferences

        Returns:
            Boolean: True if eligible, False if filtered
        """
        self.stats['total_checks'] += 1

        # Language filter
        if not self._check_language(post_metadata, user_metadata):
            self.stats['filtered_language'] += 1
            return False

        # Quality threshold filter
        if not self._check_quality(post_metadata):
            self.stats['filtered_quality'] += 1
            return False

        # Provider availability filter
        if not self._check_providers(post_metadata, user_metadata):
            self.stats['filtered_providers'] += 1
            return False

        # Avoid-genres filter (declared preference, matched via structured genres
        # and via overview text classification for genres missing from genreWeights)
        if not self._check_avoid_genres(post_metadata, user_metadata):
            self.stats['filtered_avoid_genres'] += 1
            return False

        self.stats['passed'] += 1
        return True

    def calculate_soft_filters(self, post_metadata: Dict, user_metadata: Dict) -> float:
        """
        Calculate soft filter penalty factor based on behavioral genres.

        This is a SOFT filter - it returns a penalty factor (0.0 to 1.0)
        that should be multiplied with the score, NOT a boolean filter.

        Args:
            post_metadata: Post metadata
            user_metadata: User metadata

        Returns:
            Penalty factor (1.0 = no penalty, <1.0 = penalty applied)
        """
        if not self.behavioral_genres_enabled:
            return 1.0

        self.stats['behavioral_genre_checks'] += 1

        user_genres = user_metadata.get('behavioralGenres', [])
        if not user_genres:
            # No behavioral preferences = no penalty
            self.stats['behavioral_genre_matches'] += 1
            return 1.0

        post_genres = set(post_metadata.get('genreWeights', {}).keys())
        if not post_genres:
            # Unknown post genres = no penalty
            self.stats['behavioral_genre_matches'] += 1
            return 1.0

        # Check for overlap between user preferences and post genres
        user_genre_set = set(user_genres)
        overlap = user_genre_set & post_genres

        if overlap:
            # Match found - no penalty
            self.stats['behavioral_genre_matches'] += 1
            return 1.0
        else:
            # No match - apply soft penalty
            logger.debug(f"Behavioral genre penalty applied: user={user_genres}, post={list(post_genres)}")
            return self.behavioral_genre_penalty

    def check_eligibility_batch(self, post_ids: List[int],
                                post_metadata_map: Dict[int, Dict],
                                user_metadata: Dict) -> Dict[int, bool]:
        """
        Check eligibility for a batch of posts.

        Args:
            post_ids: List of post IDs to check
            post_metadata_map: Dict mapping post_id to metadata
            user_metadata: User metadata

        Returns:
            Dict mapping post_id to eligibility (True/False)
        """
        results = {}
        for post_id in post_ids:
            post_metadata = post_metadata_map.get(post_id)
            if post_metadata:
                results[post_id] = self.check_eligibility(post_metadata, user_metadata)
            else:
                # If no metadata, assume eligible (don't filter unknown content)
                results[post_id] = True
        return results

    def _check_language(self, post_meta: Dict, user_meta: Dict) -> bool:
        """
        Check if post language matches user preferences.

        Boolean logic: If user has language preferences, post must match one of them.
        If user has no preferences, all languages are acceptable.

        Args:
            post_meta: Post metadata
            user_meta: User metadata

        Returns:
            Boolean: True if language is acceptable
        """
        # Extract user language preferences
        language_weights = user_meta.get('languageWeights', {})
        user_languages = set(language_weights.get('weights', {}).keys())

        # No preference = accept all languages
        if not user_languages:
            return True

        # Get post language
        categorical_features = post_meta.get('categoricalFeatures', {})
        post_language = categorical_features.get('language', 'en')

        # Check if post language is in user's accepted languages
        return post_language in user_languages

    def _check_quality(self, post_meta: Dict) -> bool:
        """
        Check if post meets minimum quality threshold.

        Boolean logic: Posts with sufficient votes must meet quality threshold.
        Posts with insufficient votes are not filtered (benefit of doubt).

        Args:
            post_meta: Post metadata

        Returns:
            Boolean: True if quality is acceptable
        """
        vote_avg = post_meta.get('voteAverage', 0)
        vote_count = post_meta.get('voteCount', 0)

        # Not enough data to judge quality - accept
        if vote_count < self.min_vote_count:
            return True

        # Check against quality threshold
        return vote_avg >= self.quality_threshold

    def _check_providers(self, post_meta: Dict, user_meta: Dict) -> bool:
        """
        Check if post is available on user's subscribed streaming services.

        Boolean logic: If user has provider preferences and post has provider data,
        there must be at least one overlap. If either is missing, accept.

        Args:
            post_meta: Post metadata
            user_meta: User metadata

        Returns:
            Boolean: True if provider availability is acceptable
        """
        # Get user's subscribed providers
        user_providers = set(user_meta.get('subscribedProviders', []))

        # No subscription preferences = accept all
        if not user_providers:
            return True

        # Get content's available providers
        content_providers = set(post_meta.get('availableProviders', []))

        # No provider info = accept (don't filter unknown availability)
        if not content_providers:
            return True

        # Check for overlap between user subscriptions and content availability
        return bool(user_providers & content_providers)

    def _check_avoid_genres(self, post_meta: Dict, user_meta: Dict) -> bool:
        """
        Check if post's overview text implies one of the user's declared avoid-genres.

        Structured genreWeights-based avoid-genre filtering already happens
        upstream (candidate generation) - this check exists specifically to catch
        posts TMDB's structured genre tags miss but the overview text makes clear
        (e.g. a film not tagged "horror" whose synopsis is unambiguously horror).

        Args:
            post_meta: Post metadata (overview - TMDB data)
            user_meta: User metadata (avoidGenres - user's declared preference)

        Returns:
            Boolean: True if post is acceptable (overview doesn't imply an avoided genre)
        """
        avoid_genres = user_meta.get('avoidGenres', [])
        overview = post_meta.get('overview', '')
        if not avoid_genres or not overview:
            return True

        avoid_genres_lower = {g.lower() for g in avoid_genres}
        classified_genres = classify_overview_genres(overview)
        return not any(g.lower() in avoid_genres_lower for g in classified_genres)

    def get_filter_reasons(self, post_metadata: Dict, user_metadata: Dict) -> List[str]:
        """
        Get detailed reasons why a post was filtered (for debugging/logging).

        Args:
            post_metadata: Post metadata
            user_metadata: User metadata

        Returns:
            List of filter reason strings (empty if eligible)
        """
        reasons = []

        if not self._check_language(post_metadata, user_metadata):
            post_lang = post_metadata.get('categoricalFeatures', {}).get('language', 'unknown')
            user_langs = list(user_metadata.get('languageWeights', {}).get('weights', {}).keys())
            reasons.append(f"Language mismatch: post={post_lang}, user_prefs={user_langs}")

        if not self._check_quality(post_metadata):
            vote_avg = post_metadata.get('voteAverage', 0)
            vote_count = post_metadata.get('voteCount', 0)
            reasons.append(f"Quality below threshold: vote_avg={vote_avg}, "
                          f"vote_count={vote_count}, threshold={self.quality_threshold}")

        if not self._check_providers(post_metadata, user_metadata):
            user_providers = user_metadata.get('subscribedProviders', [])
            content_providers = post_metadata.get('availableProviders', [])
            reasons.append(f"No provider match: user={user_providers}, content={content_providers}")

        if not self._check_avoid_genres(post_metadata, user_metadata):
            avoid_genres = user_metadata.get('avoidGenres', [])
            overview = post_metadata.get('overview', '')
            classified = classify_overview_genres(overview)
            reasons.append(f"Avoid-genre match via overview: user_avoids={avoid_genres}, "
                          f"overview_classified_as={sorted(classified)}")

        return reasons

    def get_stats(self) -> Dict[str, any]:
        """Get filter statistics."""
        total = self.stats['total_checks']
        if total == 0:
            return {**self.stats, 'pass_rate': 0.0}

        return {
            **self.stats,
            'pass_rate': self.stats['passed'] / total,
            'language_filter_rate': self.stats['filtered_language'] / total,
            'quality_filter_rate': self.stats['filtered_quality'] / total,
            'provider_filter_rate': self.stats['filtered_providers'] / total,
            'avoid_genre_filter_rate': self.stats['filtered_avoid_genres'] / total,
            'behavioral_genre_match_rate': (
                self.stats['behavioral_genre_matches'] / self.stats['behavioral_genre_checks']
                if self.stats['behavioral_genre_checks'] > 0 else 0.0
            )
        }

    def reset_stats(self):
        """Reset filter statistics."""
        self.stats = {
            'total_checks': 0,
            'filtered_language': 0,
            'filtered_quality': 0,
            'filtered_providers': 0,
            'filtered_avoid_genres': 0,
            'passed': 0,
            'behavioral_genre_checks': 0,
            'behavioral_genre_matches': 0
        }


def create_eligibility_filter(quality_threshold: float = 5.0,
                              min_vote_count: int = 10,
                              behavioral_genres_enabled: bool = True) -> EligibilityFilter:
    """
    Factory function to create an eligibility filter.

    Args:
        quality_threshold: Minimum vote average for quality filter
        min_vote_count: Minimum votes required to apply quality filter
        behavioral_genres_enabled: Enable behavioral genre soft filter

    Returns:
        Configured EligibilityFilter instance
    """
    return EligibilityFilter(
        quality_threshold=quality_threshold,
        min_vote_count=min_vote_count,
        behavioral_genres_enabled=behavioral_genres_enabled
    )
