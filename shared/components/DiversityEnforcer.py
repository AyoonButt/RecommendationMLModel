"""
DiversityEnforcer: TMDB ToS Compliant Diversity Reranking

This module reorders recommendation results for diversity without modifying scores.
TMDB data is used for genre spreading and slot allocation, NOT for ML training.

Compliance: TMDB data affects ordering only (reranking), not the underlying scores.
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

logger = logging.getLogger("diversity-enforcer")


class DiversityEnforcer:
    """
    Reorders recommendation results for diversity - does NOT modify scores.

    TMDB ToS Compliant: Uses TMDB genre/release data for reordering only,
    not as ML features for scoring.

    Diversity rules:
    - No more than max_consecutive_same_genre items in a row
    - Reserved slots for trending content at specified positions (top 3)
    - Reserved slots for new releases at specified positions
    - Reserved slots for hidden gems (high quality, low popularity)
    """

    def __init__(self,
                 max_consecutive_same_genre: int = 2,
                 new_release_positions: List[int] = None,
                 hidden_gem_positions: List[int] = None,
                 new_release_days: int = 30,
                 hidden_gem_threshold: float = 7.0,
                 trending_positions: List[int] = None):
        """
        Initialize the diversity enforcer.

        Args:
            max_consecutive_same_genre: Max items of same primary genre in a row (default 2)
            new_release_positions: Positions reserved for new releases (default [0, 5, 10])
            hidden_gem_positions: Positions reserved for hidden gems (default [3, 8])
            new_release_days: Days since release to qualify as "new" (default 30)
            hidden_gem_threshold: Minimum vote average for hidden gem (default 7.0)
            trending_positions: Positions reserved for trending content (default [0, 1, 2])
        """
        self.max_consecutive_same_genre = max_consecutive_same_genre
        self.new_release_positions = new_release_positions or [0, 5, 10]
        self.hidden_gem_positions = hidden_gem_positions or [3, 8]
        self.trending_positions = trending_positions or [0, 1, 2]
        self.new_release_days = new_release_days
        self.hidden_gem_threshold = hidden_gem_threshold

        # Statistics tracking
        self.stats = {
            'reranking_calls': 0,
            'items_reordered': 0,
            'new_release_slots_filled': 0,
            'hidden_gem_slots_filled': 0,
            'trending_slots_filled': 0,
            'genre_spread_adjustments': 0
        }

        logger.info(f"DiversityEnforcer initialized: max_consecutive={max_consecutive_same_genre}, "
                   f"new_release_positions={self.new_release_positions}, "
                   f"hidden_gem_positions={self.hidden_gem_positions}, "
                   f"trending_positions={self.trending_positions}")

    def apply_diversity(self, ranked_posts: List[Tuple[int, float]],
                       metadata_map: Dict[int, Dict]) -> List[Tuple[int, float]]:
        """
        Reorder posts to ensure diversity without modifying scores.

        Args:
            ranked_posts: List of (post_id, score) tuples sorted by score descending
            metadata_map: Dict mapping post_id to metadata

        Returns:
            Reordered list of (post_id, score) tuples with original scores preserved
        """
        self.stats['reranking_calls'] += 1

        if not ranked_posts:
            return []

        if len(ranked_posts) <= 3:
            # Too few items to reorder meaningfully
            return list(ranked_posts)

        result = []
        remaining = list(ranked_posts)
        genre_history: List[str] = []

        # Pre-allocate reserved slots
        slot_allocations = self._allocate_slots(remaining, metadata_map)

        total_positions = len(ranked_posts)
        original_order = {post_id: idx for idx, (post_id, _) in enumerate(ranked_posts)}

        for position in range(total_positions):
            if not remaining:
                break

            # Check if this position has a reserved allocation
            if position in slot_allocations:
                post_id = slot_allocations[position]
                entry = self._find_and_remove(remaining, post_id)
                if entry:
                    result.append(entry)
                    # Update genre history
                    genres = metadata_map.get(post_id, {}).get('genreWeights', {})
                    primary = self._get_primary_genre(genres)
                    genre_history = self._update_genre_history(genre_history, primary)
                    continue

            # Find next post respecting genre spread rules
            candidate = self._find_diverse_candidate(remaining, metadata_map, genre_history)

            if candidate:
                # Track if we reordered
                candidate_original_pos = original_order.get(candidate[0], position)
                if candidate_original_pos != position:
                    self.stats['items_reordered'] += 1

                result.append(candidate)
                remaining.remove(candidate)

                # Update genre history
                genres = metadata_map.get(candidate[0], {}).get('genreWeights', {})
                primary = self._get_primary_genre(genres)
                genre_history = self._update_genre_history(genre_history, primary)
            elif remaining:
                # Fallback: take first remaining item
                result.append(remaining.pop(0))

        return result

    def _allocate_slots(self, ranked_posts: List[Tuple[int, float]],
                       metadata_map: Dict[int, Dict]) -> Dict[int, int]:
        """
        Pre-allocate reserved slots for trending, new releases, and hidden gems.

        Slot priority: Trending > New Releases > Hidden Gems

        Args:
            ranked_posts: Ranked list of (post_id, score) tuples
            metadata_map: Metadata dictionary

        Returns:
            Dict mapping position to post_id for reserved slots
        """
        allocations = {}
        used_post_ids: Set[int] = set()

        # Find trending posts first (highest priority)
        trending_posts = self._find_trending_posts(ranked_posts, metadata_map)

        # Allocate trending slots
        for position in self.trending_positions:
            if position < len(ranked_posts) and trending_posts:
                for post_id, _ in trending_posts:
                    if post_id not in used_post_ids:
                        allocations[position] = post_id
                        used_post_ids.add(post_id)
                        self.stats['trending_slots_filled'] += 1
                        break

        # Find new releases (sorted by recency, then by score)
        new_releases = self._find_new_releases(ranked_posts, metadata_map)

        # Allocate new release slots
        for position in self.new_release_positions:
            if position < len(ranked_posts) and new_releases:
                for post_id, _ in new_releases:
                    if post_id not in used_post_ids:
                        allocations[position] = post_id
                        used_post_ids.add(post_id)
                        self.stats['new_release_slots_filled'] += 1
                        break

        # Find hidden gems (high quality, lower popularity)
        hidden_gems = self._find_hidden_gems(ranked_posts, metadata_map, used_post_ids)

        # Allocate hidden gem slots
        for position in self.hidden_gem_positions:
            if position < len(ranked_posts) and hidden_gems:
                for post_id, _ in hidden_gems:
                    if post_id not in used_post_ids:
                        allocations[position] = post_id
                        used_post_ids.add(post_id)
                        self.stats['hidden_gem_slots_filled'] += 1
                        break

        return allocations

    def _find_trending_posts(self, ranked_posts: List[Tuple[int, float]],
                           metadata_map: Dict[int, Dict]) -> List[Tuple[int, float]]:
        """
        Find trending posts based on engagement velocity data.

        Args:
            ranked_posts: Ranked list of posts
            metadata_map: Metadata dictionary containing velocity data

        Returns:
            List of (post_id, score) tuples for trending posts
        """
        trending_posts = []

        for post_id, score in ranked_posts:
            metadata = metadata_map.get(post_id, {})
            velocity_data = metadata.get('velocity', {})

            if velocity_data:
                is_trending = velocity_data.get('isTrending', False)
                hourly_velocity = velocity_data.get('hourlyVelocity', 0.0)
                daily_velocity = velocity_data.get('dailyVelocity', 0.0)

                if is_trending or hourly_velocity > 1.5 or daily_velocity > 10.0:
                    trending_posts.append((post_id, score))

        # Sort by score (higher score first among trending)
        trending_posts.sort(key=lambda x: x[1], reverse=True)
        return trending_posts

    def _find_new_releases(self, ranked_posts: List[Tuple[int, float]],
                          metadata_map: Dict[int, Dict]) -> List[Tuple[int, float]]:
        """
        Find new releases from ranked posts.

        Args:
            ranked_posts: Ranked list of posts
            metadata_map: Metadata dictionary

        Returns:
            List of (post_id, score) tuples for new releases
        """
        import time
        current_time = time.time()
        new_release_threshold = current_time - (self.new_release_days * 24 * 60 * 60)

        new_releases = []
        for post_id, score in ranked_posts:
            metadata = metadata_map.get(post_id, {})

            # Check release date
            release_timestamp = metadata.get('releaseTimestamp', 0)
            if release_timestamp > new_release_threshold:
                new_releases.append((post_id, score))

            # Alternative: check recencyBoost from API
            recency_boost = metadata.get('recencyBoost', 1.0)
            if recency_boost > 1.1 and (post_id, score) not in new_releases:
                new_releases.append((post_id, score))

        # Sort by score (higher score first among new releases)
        new_releases.sort(key=lambda x: x[1], reverse=True)
        return new_releases

    def _find_hidden_gems(self, ranked_posts: List[Tuple[int, float]],
                         metadata_map: Dict[int, Dict],
                         exclude_ids: Set[int]) -> List[Tuple[int, float]]:
        """
        Find hidden gems (high quality, lower popularity).

        Args:
            ranked_posts: Ranked list of posts
            metadata_map: Metadata dictionary
            exclude_ids: Post IDs to exclude

        Returns:
            List of (post_id, score) tuples for hidden gems
        """
        hidden_gems = []

        for post_id, score in ranked_posts:
            if post_id in exclude_ids:
                continue

            metadata = metadata_map.get(post_id, {})

            vote_average = metadata.get('voteAverage', 0)
            popularity = metadata.get('popularity', 100)

            # Hidden gem criteria: high quality, lower popularity
            if vote_average >= self.hidden_gem_threshold and popularity < 50:
                hidden_gems.append((post_id, score))

        # Sort by vote average (quality) descending
        hidden_gems.sort(key=lambda x: metadata_map.get(x[0], {}).get('voteAverage', 0),
                        reverse=True)
        return hidden_gems

    def _find_diverse_candidate(self, remaining: List[Tuple[int, float]],
                                metadata_map: Dict[int, Dict],
                                genre_history: List[str]) -> Optional[Tuple[int, float]]:
        """
        Find next candidate that respects genre diversity rules.

        Args:
            remaining: Remaining posts to consider
            metadata_map: Metadata dictionary
            genre_history: Recent genre history

        Returns:
            Best candidate (post_id, score) tuple or None
        """
        if not remaining:
            return None

        # If we don't have enough history, just return the highest scored
        if len(genre_history) < self.max_consecutive_same_genre:
            return remaining[0]

        # Check for consecutive same genre
        recent_genres = genre_history[-self.max_consecutive_same_genre:]
        if len(set(recent_genres)) != 1:
            # Not all same genre, safe to continue
            return remaining[0]

        blocked_genre = recent_genres[0]

        # Need to find a different genre
        for entry in remaining:
            post_id, score = entry
            metadata = metadata_map.get(post_id, {})
            genres = metadata.get('genreWeights', {})
            primary_genre = self._get_primary_genre(genres)

            if primary_genre != blocked_genre:
                self.stats['genre_spread_adjustments'] += 1
                return entry

        # Couldn't find different genre, return highest scored anyway
        return remaining[0]

    def _find_and_remove(self, posts: List[Tuple[int, float]],
                        target_id: int) -> Optional[Tuple[int, float]]:
        """
        Find and remove a post by ID from the list.

        Args:
            posts: List of (post_id, score) tuples
            target_id: Post ID to find

        Returns:
            The found entry or None
        """
        for i, (post_id, score) in enumerate(posts):
            if post_id == target_id:
                return posts.pop(i)
        return None

    def _get_primary_genre(self, genre_weights: Dict[str, float]) -> str:
        """
        Get the primary (highest weighted) genre.

        Args:
            genre_weights: Dict mapping genre to weight

        Returns:
            Primary genre name or 'unknown'
        """
        if not genre_weights:
            return 'unknown'

        return max(genre_weights.items(), key=lambda x: x[1])[0]

    def _update_genre_history(self, history: List[str], genre: str) -> List[str]:
        """
        Update genre history, keeping only recent entries.

        Args:
            history: Current genre history
            genre: New genre to add

        Returns:
            Updated genre history
        """
        updated = history + [genre]
        # Keep only last N entries where N is max_consecutive + 1
        max_history = self.max_consecutive_same_genre + 1
        return updated[-max_history:]

    def get_stats(self) -> Dict[str, any]:
        """Get diversity enforcement statistics."""
        calls = self.stats['reranking_calls']
        if calls == 0:
            return {**self.stats, 'avg_reordered_per_call': 0.0}

        return {
            **self.stats,
            'avg_reordered_per_call': self.stats['items_reordered'] / calls,
            'avg_new_release_fills_per_call': self.stats['new_release_slots_filled'] / calls,
            'avg_hidden_gem_fills_per_call': self.stats['hidden_gem_slots_filled'] / calls,
            'avg_trending_fills_per_call': self.stats['trending_slots_filled'] / calls,
            'avg_genre_adjustments_per_call': self.stats['genre_spread_adjustments'] / calls
        }

    def reset_stats(self):
        """Reset diversity statistics."""
        self.stats = {
            'reranking_calls': 0,
            'items_reordered': 0,
            'new_release_slots_filled': 0,
            'hidden_gem_slots_filled': 0,
            'trending_slots_filled': 0,
            'genre_spread_adjustments': 0
        }


def create_diversity_enforcer(max_consecutive_same_genre: int = 2,
                              new_release_positions: List[int] = None,
                              hidden_gem_positions: List[int] = None,
                              trending_positions: List[int] = None) -> DiversityEnforcer:
    """
    Factory function to create a diversity enforcer.

    Args:
        max_consecutive_same_genre: Max items of same genre in a row
        new_release_positions: Positions for new releases
        hidden_gem_positions: Positions for hidden gems
        trending_positions: Positions for trending content

    Returns:
        Configured DiversityEnforcer instance
    """
    return DiversityEnforcer(
        max_consecutive_same_genre=max_consecutive_same_genre,
        new_release_positions=new_release_positions,
        hidden_gem_positions=hidden_gem_positions,
        trending_positions=trending_positions
    )
