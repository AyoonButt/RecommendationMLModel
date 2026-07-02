"""
RLEnhancedMetadataEnhancer: TMDB ToS Compliant RL Enhancement

This module extends MetadataEnhancer with reinforcement learning capabilities.
RL can ONLY adjust exploration/exploitation and ranking positions,
NOT TMDB-based boost factors.

TMDB ToS Compliant:
- RL does NOT adjust TMDB boosts (removed for compliance)
- RL can adjust ranking order for exploration
- RL learns from behavioral signals only
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# Add the rl-agent service to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../services/rl-agent'))

from MetadataEnhancer import MetadataEnhancer, create_diversity_enforcer, extract_person_ids
from RLIntegration import RLIntegrationManager, create_rl_integration_manager
from GenreTextClassifier import classify_overview_genres

logger = logging.getLogger("rl-enhanced-metadata-enhancer")


class RLEnhancedMetadataEnhancer(MetadataEnhancer):
    """
    RL-Enhanced MetadataEnhancer - TMDB ToS Compliant.

    Extends MetadataEnhancer with reinforcement learning capabilities.
    RL can ONLY adjust exploration/exploitation behavior and ranking positions,
    NOT TMDB-based boost factors (removed for ToS compliance).

    Key constraints:
    - RL cannot use TMDB data as features
    - RL cannot adjust TMDB-based boosts (they don't exist anymore)
    - RL can adjust ranking order for exploration purposes
    - RL learns from app behavioral signals only
    """

    def __init__(self, api_base_url: str, redis_client=None, cache_ttl: int = 3600,
                 rl_config: Dict[str, Any] = None):
        """Initialize the RL-enhanced metadata enhancer."""
        # Initialize parent MetadataEnhancer (TMDB ToS compliant version)
        super().__init__(api_base_url, redis_client, cache_ttl)

        # Initialize diversity enforcer if parent didn't (import resolution issue)
        if not hasattr(self, 'diversity_enforcer'):
            self.diversity_enforcer = create_diversity_enforcer(
                max_consecutive_same_genre=2,
                new_release_positions=[0, 5, 10],
                hidden_gem_positions=[3, 8]
            )

        # Initialize RL integration manager
        self.rl_manager = create_rl_integration_manager(rl_config)

        # RL statistics (TMDB boost adjustments removed)
        self.rl_stats = {
            'rl_requests': 0,
            'rl_ranking_modifications': 0,
            'rl_exploration_actions': 0
        }

        logger.info("RL-Enhanced MetadataEnhancer initialized (TMDB ToS Compliant)")

    def enhance_scores(self, user_id: str, post_ids: List[int], base_scores: np.ndarray,
                       candidates: List[Dict] = None, content_type: str = "posts") -> np.ndarray:
        """
        TMDB ToS Compliant RL enhancement.

        Enhancement pipeline:
        1. Apply compliant base enhancement (eligibility filter + behavioral boost)
        2. RL adjusts ranking order for exploration (NOT boost factors)
        """
        user_id_int = int(user_id)
        self.rl_stats['rl_requests'] += 1

        # Step 1: Apply compliant base enhancement
        enhanced_scores = super().enhance_scores(
            user_id, post_ids, base_scores, candidates, content_type
        )

        # Step 2: Prepare candidate posts for RL processing
        candidate_posts = self._prepare_candidates_for_rl(
            post_ids, enhanced_scores, candidates, content_type
        )

        # Step 3: Build context for RL decision making
        context = self._build_rl_context(user_id_int, candidate_posts, content_type)

        # Step 4: Get RL actions (exploration/exploitation decisions only)
        rl_result = self.rl_manager.enhance_recommendations(
            user_id=user_id_int,
            candidate_posts=candidate_posts,
            context=context
        )

        # Step 5: Apply RL ranking modifications only (NO boost adjustments)
        final_scores = self._apply_rl_ranking_only(
            enhanced_scores, post_ids, rl_result.actions_taken
        )

        # Store RL context for future reward calculation
        self._store_rl_context(user_id_int, post_ids, rl_result, context)

        logger.debug(f"Applied RL enhancement for user {user_id}")

        return final_scores

    def _apply_rl_ranking_only(self, scores: np.ndarray, post_ids: List[int],
                               actions_taken: List) -> np.ndarray:
        """
        Apply RL ranking modifications only - NO TMDB boost adjustments.

        TMDB ToS Compliant: RL can reorder for exploration, but cannot
        adjust TMDB-based boost factors (they don't exist anymore).
        """
        modified_scores = scores.copy()

        for action in actions_taken:
            if action.action_type == "ranking_modification":
                params = action.parameters
                position_adjustment = params.get('position_adjustment', 0)
                rerank_scope = params.get('rerank_scope', 10)

                if position_adjustment != 0:
                    self.rl_stats['rl_ranking_modifications'] += 1

                    for i in range(min(rerank_scope, len(modified_scores))):
                        if position_adjustment > 0:
                            # Boost top items for exploitation
                            adjustment_factor = 1.0 + (position_adjustment * 0.02 *
                                                      (rerank_scope - i) / rerank_scope)
                        else:
                            # Boost lower items for exploration
                            adjustment_factor = 1.0 + (position_adjustment * 0.02 *
                                                      i / rerank_scope)

                        modified_scores[i] *= adjustment_factor

            elif action.action_type == "exploration":
                # Track exploration actions
                self.rl_stats['rl_exploration_actions'] += 1

                # Exploration: slightly boost items in lower positions
                params = action.parameters
                explore_boost = params.get('explore_boost', 0.05)
                explore_range = params.get('explore_range', 5)

                # Apply small boost to positions outside top range
                for i in range(explore_range, min(explore_range * 2, len(modified_scores))):
                    modified_scores[i] *= (1.0 + explore_boost)

            # NOTE: "boost_adjustment" action type is REMOVED for TMDB ToS compliance
            # RL cannot adjust TMDB-based boost factors

        return modified_scores

    def process_user_interaction(self, user_id: str, post_id: int, interaction_type: str,
                                 additional_context: Dict[str, Any] = None):
        """Process user interaction feedback for RL learning."""
        user_id_int = int(user_id)

        if interaction_type == 'not_interested':
            self._record_avoided_signals(user_id, post_id)

        # Get stored RL context
        rl_context = self._get_stored_rl_context(user_id_int)

        # Prepare interaction context
        interaction_context = {
            'post_id': post_id,
            'interaction_type': interaction_type,
            'timestamp': time.time(),
            **(additional_context or {}),
            **(rl_context or {})
        }

        # Process through RL manager
        self.rl_manager.process_user_interaction(
            user_id_int, post_id, interaction_type, interaction_context
        )

        logger.debug(f"Processed RL feedback: user {user_id}, post {post_id}, "
                    f"interaction {interaction_type}")

    def _record_avoided_signals(self, user_id: str, post_id: int) -> None:
        """
        On a not_interested interaction, record avoidance signals for this user
        (consumed as graduated soft penalties in enhance_scores):
        - genre: classified from the post's overview text
        - person: cast/crew IDs, exact matching, no text mining
        """
        try:
            post_metadata = self._get_cached_metadata(f"post:{post_id}")
            if not post_metadata:
                return

            overview = post_metadata.get('overview', '')
            if overview:
                classified_genres = classify_overview_genres(overview)
                if classified_genres:
                    self.avoided_signal_tracker.record(user_id, 'genre', list(classified_genres))
                    logger.debug(f"Recorded not_interested genres for user {user_id}, "
                               f"post {post_id}: {classified_genres}")

            person_ids = extract_person_ids(post_metadata)
            if person_ids:
                self.avoided_signal_tracker.record(user_id, 'person', person_ids)
                logger.debug(f"Recorded not_interested cast/crew for user {user_id}, "
                           f"post {post_id}: {person_ids}")

        except Exception as e:
            logger.warning(f"Error recording avoided signals for user {user_id}, post {post_id}: {e}")

    def _prepare_candidates_for_rl(self, post_ids: List[int], enhanced_scores: np.ndarray,
                                   candidates: List[Dict], content_type: str) -> List[Dict]:
        """Prepare candidate posts for RL processing."""
        candidate_posts = []

        for i, post_id in enumerate(post_ids):
            post_metadata = self._get_cached_metadata(f"post:{post_id}")

            candidate_post = {
                'id': post_id,
                'score': float(enhanced_scores[i]),
                'original_score': float(enhanced_scores[i]),
                'content_type': content_type,
                # Only include behavioral metadata, not TMDB metadata
                'behavioral_metadata': self._extract_behavioral_metadata(post_metadata or {}, int(post_id)),
                'position': i
            }

            if candidates and i < len(candidates):
                candidate_post.update(candidates[i])

            candidate_posts.append(candidate_post)

        return candidate_posts

    def _extract_behavioral_metadata(self, post_metadata: Dict, post_id: int = None) -> Dict:
        """
        Extract only behavioral metadata (app data, not TMDB).

        TMDB ToS Compliant: Only include metrics from app interactions.
        """
        behavioral = {
            'infoButtonClicks': post_metadata.get('infoButtonClicks', {}),
            'likeCount': post_metadata.get('likeCount', 0),
            'saveCount': post_metadata.get('saveCount', 0),
            'shareCount': post_metadata.get('shareCount', 0),
            'viewCount': post_metadata.get('viewCount', 0),
            'commentCount': post_metadata.get('commentCount', 0),
            'recentEngagement': post_metadata.get('recentEngagement', {})
        }

        # Get engagement velocity data from cache if post_id provided
        if post_id is not None and hasattr(self, '_get_cached_metadata'):
            velocity_data = self._get_cached_metadata(f"velocity:{post_id}")
            if velocity_data:
                behavioral['engagementVelocity'] = velocity_data
            else:
                behavioral['engagementVelocity'] = {}
        else:
            behavioral['engagementVelocity'] = post_metadata.get('engagementVelocity', {})

        return behavioral

    def _build_rl_context(self, user_id: int, candidate_posts: List[Dict],
                          content_type: str) -> Dict[str, Any]:
        """Build context for RL decision making."""
        user_metadata = self._get_cached_metadata(f"user:{user_id}")
        user_embedding = self._extract_user_embedding(user_metadata)

        avg_score = np.mean([p['score'] for p in candidate_posts])

        return {
            # Only include behavioral user context, not TMDB preferences
            'user_behavioral_context': self._extract_user_behavioral_context(user_metadata or {}),
            'user_embedding': user_embedding,
            'content_type': content_type,
            'candidate_count': len(candidate_posts),
            'avg_candidate_score': avg_score,
            'request_timestamp': time.time(),
            # NOTE: TMDB boost factors removed for ToS compliance
            'tmdb_tos_compliant': True
        }

    def _extract_user_behavioral_context(self, user_metadata: Dict) -> Dict:
        """
        Extract only behavioral user context (app data, not TMDB preferences).

        TMDB ToS Compliant: Only include metrics from app interactions.
        """
        return {
            'totalInteractions': user_metadata.get('totalInteractions', 0),
            'recentActivity': user_metadata.get('recentActivity', {}),
            'sessionCount': user_metadata.get('sessionCount', 0),
            'avgSessionDuration': user_metadata.get('avgSessionDuration', 0)
        }

    def _extract_user_embedding(self, user_metadata: Optional[Dict]) -> List[float]:
        """Extract user embedding vector (behavioral signals only)."""
        if not user_metadata:
            return [0.0] * 32

        embedding = [0.0] * 32

        # Behavioral embedding based on app interaction patterns
        # NOT based on TMDB interestWeights or languageWeights

        # Recent interaction types (behavioral)
        recent_activity = user_metadata.get('recentActivity', {})
        interaction_counts = recent_activity.get('interactionCounts', {})

        # Map interaction types to embedding dimensions
        interaction_type_to_dim = {
            'like': 0, 'save': 1, 'share': 2, 'skip': 3,
            'not_interested': 4, 'more_info': 5, 'comment': 6, 'view': 7
        }

        for interaction_type, dim in interaction_type_to_dim.items():
            count = interaction_counts.get(interaction_type, 0)
            # Normalize by expected max
            embedding[dim] = min(1.0, count / 100.0)

        return embedding

    def _store_rl_context(self, user_id: int, post_ids: List[int],
                          rl_result, context: Dict[str, Any]):
        """Store RL context for future reward calculation."""
        rl_context = {
            'user_id': user_id,
            'post_ids': post_ids,
            'actions_taken': [action.to_dict() for action in rl_result.actions_taken],
            'original_scores': rl_result.original_scores,
            'rl_adjusted_scores': rl_result.rl_adjusted_scores,
            'timestamp': time.time(),
            'context': context
        }

        if self.redis_client:
            try:
                import json
                self.redis_client.setex(
                    f"rl_context:{user_id}",
                    300,  # 5 minute TTL
                    json.dumps(rl_context, default=str)
                )
            except Exception as e:
                logger.warning(f"Error storing RL context: {e}")

    def _get_stored_rl_context(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve stored RL context for user."""
        if not self.redis_client:
            return None

        try:
            import json
            context_data = self.redis_client.get(f"rl_context:{user_id}")
            if context_data:
                return json.loads(context_data)
        except Exception as e:
            logger.warning(f"Error retrieving RL context: {e}")

        return None

    def get_rl_stats(self) -> Dict[str, Any]:
        """Get RL enhancement statistics."""
        base_stats = super().get_enhancement_stats()

        rl_stats = {
            "rl_requests": self.rl_stats['rl_requests'],
            "rl_ranking_modifications": self.rl_stats['rl_ranking_modifications'],
            "rl_exploration_actions": self.rl_stats['rl_exploration_actions'],
            # TMDB boost adjustments removed for ToS compliance
            "rl_boost_adjustments": 0,  # Always 0 now - ToS compliant
            "avg_boost_change": 0.0,  # Always 0 now - ToS compliant
            "tmdb_tos_compliant": True,
            "rl_integration_stats": self.rl_manager.get_integration_stats() if self.rl_manager else {}
        }

        return {**base_stats, "rl_enhancement": rl_stats}

    def get_filtered_posts(self, post_ids: List[int], scores: np.ndarray) -> List[int]:
        """Get list of posts that were filtered out (score = 0)."""
        return [pid for i, pid in enumerate(post_ids) if scores[i] == 0.0]

    def apply_diversity_reranking(self, post_ids: List[int],
                                  scores: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Apply diversity reranking to scored posts."""
        try:
            non_zero_mask = scores > 0.0
            valid_post_ids = [pid for i, pid in enumerate(post_ids) if non_zero_mask[i]]
            valid_scores = scores[non_zero_mask]

            if len(valid_post_ids) == 0:
                return [], np.array([])

            metadata_map = {}
            for post_id in valid_post_ids:
                metadata = self._get_cached_metadata(f"post:{post_id}")
                if metadata:
                    metadata_map[post_id] = metadata

            ranked_posts = list(zip(valid_post_ids, valid_scores.tolist()))
            ranked_posts.sort(key=lambda x: x[1], reverse=True)

            reranked = self.diversity_enforcer.apply_diversity(ranked_posts, metadata_map)

            reranked_ids = [pid for pid, _ in reranked]
            reranked_scores = np.array([score for _, score in reranked], dtype=np.float32)

            return reranked_ids, reranked_scores

        except Exception as e:
            logger.warning(f"Error applying diversity reranking: {e}")
            return list(post_ids), scores


def create_rl_enhanced_metadata_enhancer(api_base_url: str, redis_client=None,
                                         cache_ttl: int = 3600,
                                         rl_config: Dict[str, Any] = None) -> RLEnhancedMetadataEnhancer:
    """Create RL-enhanced metadata enhancer (TMDB ToS Compliant)."""
    return RLEnhancedMetadataEnhancer(
        api_base_url=api_base_url,
        redis_client=redis_client,
        cache_ttl=cache_ttl,
        rl_config=rl_config
    )
