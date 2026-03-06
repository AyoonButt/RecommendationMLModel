import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os

# Add the rl-agent service to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../services/rl-agent'))

from MetadataEnhancer import MetadataEnhancer
from RLIntegration import RLIntegrationManager, create_rl_integration_manager

logger = logging.getLogger("rl-enhanced-metadata-enhancer")


class RLEnhancedMetadataEnhancer(MetadataEnhancer):
    """
    RL-Enhanced MetadataEnhancer for development.
    Extends MetadataEnhancer with reinforcement learning capabilities.
    """
    
    def __init__(self, api_base_url: str, redis_client=None, cache_ttl: int = 3600,
                 rl_config: Dict[str, Any] = None):
        """Initialize the RL-enhanced metadata enhancer."""
        # Initialize parent MetadataEnhancer
        super().__init__(api_base_url, redis_client, cache_ttl)
        
        # Initialize RL integration manager
        self.rl_manager = create_rl_integration_manager(rl_config)
        
        # RL statistics
        self.rl_stats = {
            'rl_requests': 0,
            'rl_boost_adjustments': 0,
            'total_boost_changes': 0.0
        }
        
        logger.info("RL-Enhanced MetadataEnhancer initialized")
    
    def enhance_scores(self, user_id: str, post_ids: List[int], base_scores: np.ndarray,
                       candidates: List[Dict] = None, content_type: str = "posts") -> np.ndarray:
        """Enhanced version that uses RL to adaptively adjust boost factors."""
        user_id_int = int(user_id)
        self.rl_stats['rl_requests'] += 1
        
        # Prepare candidate posts for RL processing
        candidate_posts = self._prepare_candidates_for_rl(post_ids, base_scores, candidates, content_type)
        
        # Build context for RL decision making
        context = self._build_rl_context(user_id_int, candidate_posts, content_type)
        
        # Get RL action to adjust boost factors
        rl_result = self.rl_manager.enhance_recommendations(
            user_id=user_id_int,
            candidate_posts=candidate_posts,
            context=context
        )
        
        # Apply RL-adjusted boost factors
        enhanced_scores = self._apply_rl_adjusted_boosting(
            user_id_int, post_ids, base_scores, rl_result, candidates, content_type
        )
        
        # Store RL context for future reward calculation
        self._store_rl_context(user_id_int, post_ids, rl_result, context)
        
        logger.debug(f"Applied RL enhancement for user {user_id}")
        
        return enhanced_scores
    
    def process_user_interaction(self, user_id: str, post_id: int, interaction_type: str,
                               additional_context: Dict[str, Any] = None):
        """Process user interaction feedback for RL learning."""
        user_id_int = int(user_id)
        
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
        
        logger.debug(f"Processed RL feedback: user {user_id}, post {post_id}, interaction {interaction_type}")
    
    def _prepare_candidates_for_rl(self, post_ids: List[int], base_scores: np.ndarray,
                                  candidates: List[Dict], content_type: str) -> List[Dict]:
        """Prepare candidate posts for RL processing."""
        # Batch prefetch all post metadata upfront
        self._prefetch_post_metadata(post_ids)

        candidate_posts = []

        for i, post_id in enumerate(post_ids):
            post_metadata = self._get_cached_metadata(f"post:{post_id}")

            candidate_post = {
                'id': post_id,
                'score': float(base_scores[i]),
                'original_score': float(base_scores[i]),
                'content_type': content_type,
                'metadata': post_metadata or {},
                'position': i
            }

            if candidates and i < len(candidates):
                candidate_post.update(candidates[i])

            candidate_posts.append(candidate_post)

        return candidate_posts
    
    def _build_rl_context(self, user_id: int, candidate_posts: List[Dict], content_type: str) -> Dict[str, Any]:
        """Build context for RL decision making."""
        user_metadata = self._get_cached_metadata(f"user:{user_id}")
        user_embedding = self._extract_user_embedding(user_metadata)
        
        avg_score = np.mean([p['score'] for p in candidate_posts])
        
        return {
            'user_metadata': user_metadata or {},
            'user_embedding': user_embedding,
            'content_type': content_type,
            'candidate_count': len(candidate_posts),
            'avg_candidate_score': avg_score,
            'request_timestamp': time.time(),
            'current_boost_factors': {
                'language': self.language_boost_factor,
                'genre': self.genre_boost_factor,
                'popularity': self.popularity_boost_factor,
                'recency': self.recency_boost_factor,
                'cast_crew': self.cast_crew_boost_factor
            }
        }
    
    def _extract_user_embedding(self, user_metadata: Optional[Dict]) -> List[float]:
        """Extract user embedding vector."""
        if not user_metadata:
            return [0.0] * 32
        
        embedding = [0.0] * 32
        
        # Use interest weights to create embedding
        interest_weights = user_metadata.get('interestWeights', {})
        genre_to_dim = {
            'action': 0, 'comedy': 1, 'drama': 2, 'horror': 3,
            'romance': 4, 'thriller': 5, 'sci-fi': 6, 'fantasy': 7
        }
        
        for genre, weight in interest_weights.items():
            if genre.lower() in genre_to_dim:
                dim = genre_to_dim[genre.lower()]
                embedding[dim] = float(weight)
        
        # Add language preferences
        lang_weights = user_metadata.get('languageWeights', {}).get('weights', {})
        lang_to_dim = {'en': 8, 'es': 9, 'fr': 10, 'de': 11, 'it': 12}
        for lang, weight in lang_weights.items():
            if lang in lang_to_dim:
                dim = lang_to_dim[lang]
                embedding[dim] = float(weight)
        
        return embedding
    
    def _apply_rl_adjusted_boosting(self, user_id: int, post_ids: List[int], base_scores: np.ndarray,
                                   rl_result, candidates: List[Dict], content_type: str) -> np.ndarray:
        """Apply boosting with RL-adjusted factors."""
        enhanced_scores = base_scores.copy()
        
        # Extract boost adjustments from RL actions
        boost_adjustments = self._extract_boost_adjustments(rl_result.actions_taken)
        
        # Temporarily adjust boost factors
        original_factors = self._temporarily_adjust_boost_factors(boost_adjustments)
        
        try:
            # Get user metadata
            user_metadata = self._get_cached_metadata(f"user:{user_id}")
            
            if user_metadata:
                # Apply all boosting methods with adjusted factors
                enhanced_scores = self._apply_language_boosting(enhanced_scores, post_ids, user_metadata)
                enhanced_scores = self._apply_genre_boosting(enhanced_scores, post_ids, user_metadata)
                enhanced_scores = self._apply_demographic_boosting(enhanced_scores, post_ids, user_metadata)
                enhanced_scores = self._apply_cast_crew_boosting(enhanced_scores, post_ids, user_metadata)
            
            # Apply content-specific boosting
            enhanced_scores = self._apply_content_boosting(enhanced_scores, post_ids, content_type)
            
            # Apply ranking modifications from RL
            enhanced_scores = self._apply_rl_ranking_modifications(enhanced_scores, post_ids, rl_result.actions_taken)
            
            # Ensure scores stay in valid range
            enhanced_scores = np.clip(enhanced_scores, 0.0, 1.0)
            
        finally:
            # Restore original boost factors
            self._restore_boost_factors(original_factors)
        
        return enhanced_scores
    
    def _extract_boost_adjustments(self, actions_taken: List) -> Dict[str, float]:
        """Extract boost factor adjustments from RL actions."""
        adjustments = {
            'recency': 0.0,
            'cast_crew': 0.0,
            'genre': 0.0,
            'language': 0.0,
            'popularity': 0.0
        }
        
        for action in actions_taken:
            if action.action_type == "boost_adjustment":
                params = action.parameters
                adjustments['recency'] += params.get('recency_boost_adjustment', 0.0)
                adjustments['cast_crew'] += params.get('cast_crew_boost_adjustment', 0.0)
                adjustments['genre'] += params.get('genre_boost_adjustment', 0.0)
                adjustments['language'] += params.get('language_boost_adjustment', 0.0)
                adjustments['popularity'] += params.get('popularity_boost_adjustment', 0.0)
        
        # Track statistics
        total_adjustment = sum(abs(adj) for adj in adjustments.values())
        if total_adjustment > 0:
            self.rl_stats['rl_boost_adjustments'] += 1
            self.rl_stats['total_boost_changes'] += total_adjustment
        
        return adjustments
    
    def _temporarily_adjust_boost_factors(self, adjustments: Dict[str, float]) -> Dict[str, float]:
        """Temporarily adjust boost factors and return original values."""
        original_factors = {
            'language': self.language_boost_factor,
            'genre': self.genre_boost_factor,
            'popularity': self.popularity_boost_factor,
            'recency': self.recency_boost_factor,
            'cast_crew': self.cast_crew_boost_factor
        }
        
        # Apply adjustments with clipping
        self.language_boost_factor = np.clip(self.language_boost_factor + adjustments['language'], 0.0, 0.6)
        self.genre_boost_factor = np.clip(self.genre_boost_factor + adjustments['genre'], 0.0, 0.4)
        self.popularity_boost_factor = np.clip(self.popularity_boost_factor + adjustments['popularity'], 0.0, 0.3)
        self.recency_boost_factor = np.clip(self.recency_boost_factor + adjustments['recency'], 0.0, 0.5)
        self.cast_crew_boost_factor = np.clip(self.cast_crew_boost_factor + adjustments['cast_crew'], 0.0, 0.4)
        
        return original_factors
    
    def _restore_boost_factors(self, original_factors: Dict[str, float]):
        """Restore original boost factors."""
        self.language_boost_factor = original_factors['language']
        self.genre_boost_factor = original_factors['genre']
        self.popularity_boost_factor = original_factors['popularity']
        self.recency_boost_factor = original_factors['recency']
        self.cast_crew_boost_factor = original_factors['cast_crew']
    
    def _apply_rl_ranking_modifications(self, scores: np.ndarray, post_ids: List[int], actions_taken: List) -> np.ndarray:
        """Apply ranking modifications from RL actions."""
        modified_scores = scores.copy()
        
        for action in actions_taken:
            if action.action_type == "ranking_modification":
                params = action.parameters
                position_adjustment = params.get('position_adjustment', 0)
                rerank_scope = params.get('rerank_scope', 10)
                
                if position_adjustment != 0:
                    for i in range(min(rerank_scope, len(modified_scores))):
                        if position_adjustment > 0:
                            adjustment_factor = 1.0 + (position_adjustment * 0.02 * (rerank_scope - i) / rerank_scope)
                        else:
                            adjustment_factor = 1.0 + (position_adjustment * 0.02 * i / rerank_scope)
                        
                        modified_scores[i] *= adjustment_factor
        
        return modified_scores
    
    def _store_rl_context(self, user_id: int, post_ids: List[int], rl_result, context: Dict[str, Any]):
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
            "rl_boost_adjustments": self.rl_stats['rl_boost_adjustments'],
            "avg_boost_change": (self.rl_stats['total_boost_changes'] / max(1, self.rl_stats['rl_boost_adjustments'])),
            "rl_integration_stats": self.rl_manager.get_integration_stats() if self.rl_manager else {}
        }
        
        return {**base_stats, "rl_enhancement": rl_stats}


def create_rl_enhanced_metadata_enhancer(api_base_url: str, redis_client=None, 
                                        cache_ttl: int = 3600, rl_config: Dict[str, Any] = None) -> RLEnhancedMetadataEnhancer:
    """Create RL-enhanced metadata enhancer."""
    return RLEnhancedMetadataEnhancer(
        api_base_url=api_base_url,
        redis_client=redis_client,
        cache_ttl=cache_ttl,
        rl_config=rl_config
    )