import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import math

logger = logging.getLogger("rl-reward-engineer")

@dataclass
class RewardComponents:
    """Components that make up the final reward signal."""
    immediate_reward: float  # Direct user feedback
    exploration_bonus: float  # Reward for exploring diverse content
    engagement_bonus: float  # Reward based on time spent/depth of interaction
    long_term_penalty: float  # Penalty for poor long-term outcomes
    diversity_bonus: float  # Reward for content diversity
    novelty_bonus: float  # Reward for showing novel content
    comment_bonus: float  # Reward adjustment based on comment analysis
    final_reward: float  # Combined final reward
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'immediate': self.immediate_reward,
            'exploration': self.exploration_bonus,
            'engagement': self.engagement_bonus,
            'long_term': self.long_term_penalty,
            'diversity': self.diversity_bonus,
            'novelty': self.novelty_bonus,
            'comment': self.comment_bonus,
            'final': self.final_reward
        }


class RLRewardEngineer:
    """
    Advanced reward engineering for RL-based recommendations.
    Handles multi-objective optimization, reward shaping, and temporal discounting.
    """
    
    def __init__(self, reward_config: Dict[str, Any] = None, 
                 api_base_url: str = "http://localhost:8080"):
        """
        Initialize the reward engineer.
        
        Args:
            reward_config: Configuration for reward calculation
            api_base_url: URL for Spring API (where comment analysis is stored)
        """
        self.config = reward_config or self._get_default_config()
        
        # Track user interaction patterns for adaptive rewards
        self.user_interaction_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.user_content_diversity: Dict[int, set] = defaultdict(set)
        self.user_exploration_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Global statistics for normalization
        self.global_engagement_stats = {'mean': 30.0, 'std': 20.0}  # seconds
        self.global_position_stats = {'mean': 5.0, 'std': 3.0}
        
        # Initialize comment analysis integration
        from RLCommentAnalysisIntegration import RLCommentAnalysisIntegrator
        self.comment_integrator = RLCommentAnalysisIntegrator(api_base_url)
        
        logger.info("RL Reward Engineer initialized with comment analysis integration (fetching from API)")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default reward configuration."""
        return {
            # Base interaction rewards (maps to your existing -0.9 to +1.2 scale)
            'base_rewards': {
                'like': 0.6,
                'save': 1.0,
                'share': 0.8,
                'comment': 0.4,
                'more_info': 0.3,
                'not_interested': -0.9,
                'skip': -0.2,
                'block': -1.0,
                'report': -1.2,
                'long_view': 0.5,
                'return_visit': 0.7,
                'download': 0.9
            },
            
            # Reward shaping weights
            'shaping_weights': {
                'exploration': 0.15,  # Weight for exploration bonus
                'engagement': 0.20,   # Weight for engagement bonus
                'diversity': 0.10,    # Weight for diversity bonus
                'novelty': 0.10,      # Weight for novelty bonus
                'long_term': 0.25     # Weight for long-term factors
            },
            
            # Exploration parameters
            'exploration': {
                'position_bonus_threshold': 5,  # Rank position beyond which exploration bonus applies
                'max_position_bonus': 0.3,
                'low_score_threshold': 0.4,     # ML score below which exploration bonus applies
                'max_score_bonus': 0.2
            },
            
            # Engagement parameters
            'engagement': {
                'time_thresholds': [10, 30, 60, 120, 300],  # seconds
                'time_bonuses': [0.0, 0.1, 0.2, 0.3, 0.4],
                'interaction_depth_bonus': 0.15  # Bonus for deep interactions (comments, shares)
            },
            
            # Diversity parameters
            'diversity': {
                'genre_diversity_window': 10,   # Look at last 10 recommendations
                'max_diversity_bonus': 0.2,
                'content_type_bonus': 0.1      # Bonus for mixing posts/trailers
            },
            
            # Long-term parameters
            'long_term': {
                'session_length_target': 15,    # Target interactions per session
                'return_visit_window_hours': 24,
                'satisfaction_decay_rate': 0.95  # How quickly satisfaction decays
            }
        }
    
    def calculate_reward(self, interaction_type: str, user_id: int, post_id: int,
                        context: Dict[str, Any] = None) -> RewardComponents:
        """
        Calculate comprehensive reward for an interaction.
        
        Args:
            interaction_type: Type of user interaction
            user_id: User ID
            post_id: Post ID
            context: Additional context information
            
        Returns:
            RewardComponents with detailed reward breakdown
        """
        context = context or {}
        
        # Get base reward
        immediate_reward = self._calculate_immediate_reward(interaction_type, context)
        
        # Calculate bonus components
        exploration_bonus = self._calculate_exploration_bonus(user_id, context)
        engagement_bonus = self._calculate_engagement_bonus(interaction_type, context)
        diversity_bonus = self._calculate_diversity_bonus(user_id, post_id, context)
        novelty_bonus = self._calculate_novelty_bonus(user_id, post_id, context)
        long_term_penalty = self._calculate_long_term_adjustment(user_id, interaction_type, context)
        
        # Apply comment-based reward adjustment
        comment_adjusted_reward = self.comment_integrator.calculate_comment_based_reward_adjustment(
            post_id, interaction_type, immediate_reward
        )
        comment_bonus = comment_adjusted_reward - immediate_reward
        
        # Combine components with weights
        weights = self.config['shaping_weights']
        final_reward = (
            comment_adjusted_reward +  # Use comment-adjusted immediate reward
            exploration_bonus * weights['exploration'] +
            engagement_bonus * weights['engagement'] +
            diversity_bonus * weights['diversity'] +
            novelty_bonus * weights['novelty'] +
            long_term_penalty * weights['long_term']
        )
        
        # Clip to reasonable range
        final_reward = np.clip(final_reward, -2.0, 2.0)
        
        # Update user tracking
        self._update_user_tracking(user_id, post_id, interaction_type, context)
        
        return RewardComponents(
            immediate_reward=immediate_reward,
            exploration_bonus=exploration_bonus,
            engagement_bonus=engagement_bonus,
            long_term_penalty=long_term_penalty,
            diversity_bonus=diversity_bonus,
            novelty_bonus=novelty_bonus,
            comment_bonus=comment_bonus,
            final_reward=final_reward
        )
    
    def _calculate_immediate_reward(self, interaction_type: str, context: Dict[str, Any]) -> float:
        """Calculate immediate reward from user interaction."""
        base_reward = self.config['base_rewards'].get(interaction_type, 0.0)
        
        # Apply context-based adjustments
        adjusted_reward = base_reward
        
        # Adjust based on interaction intensity
        engagement_time = context.get('engagement_time_seconds', 0)
        if engagement_time > 0:
            # Longer engagement increases positive rewards and decreases negative ones
            time_factor = min(1.5, 1.0 + engagement_time / 120.0)  # Cap at 1.5x
            if base_reward > 0:
                adjusted_reward *= time_factor
            else:
                adjusted_reward /= time_factor  # Make negative rewards less negative for longer engagement
        
        # Adjust based on user's historical pattern
        user_satisfaction = context.get('user_satisfaction_score', 0.5)
        if user_satisfaction < 0.3:  # Struggling user
            # Be more generous with rewards to encourage engagement
            if base_reward > 0:
                adjusted_reward *= 1.2
        elif user_satisfaction > 0.8:  # Highly satisfied user
            # Be more conservative to avoid over-optimization
            if base_reward > 0:
                adjusted_reward *= 0.9
        
        return adjusted_reward
    
    def _calculate_exploration_bonus(self, user_id: int, context: Dict[str, Any]) -> float:
        """Calculate bonus for exploration (showing diverse/low-confidence content)."""
        bonus = 0.0
        config = self.config['exploration']
        
        # Position-based exploration bonus
        ranking_position = context.get('ranking_position', 0)
        if ranking_position > config['position_bonus_threshold']:
            # Reward for showing lower-ranked items
            position_bonus = min(
                config['max_position_bonus'],
                (ranking_position - config['position_bonus_threshold']) * 0.05
            )
            bonus += position_bonus
        
        # Score-based exploration bonus
        ml_score = context.get('original_score', 0.5)
        if ml_score < config['low_score_threshold']:
            # Reward for showing items the model is less confident about
            score_bonus = min(
                config['max_score_bonus'],
                (config['low_score_threshold'] - ml_score) * 0.5
            )
            bonus += score_bonus
        
        # Cold start bonus
        post_interaction_count = context.get('post_interaction_count', float('inf'))
        if post_interaction_count < 10:  # Cold content
            bonus += 0.1
        
        user_interaction_count = context.get('user_interaction_count', float('inf'))
        if user_interaction_count < 20:  # Cold user
            bonus += 0.1
        
        return bonus
    
    def _calculate_engagement_bonus(self, interaction_type: str, context: Dict[str, Any]) -> float:
        """Calculate bonus based on depth of engagement."""
        bonus = 0.0
        config = self.config['engagement']
        
        # Time-based engagement bonus
        engagement_time = context.get('engagement_time_seconds', 0)
        for i, threshold in enumerate(config['time_thresholds']):
            if engagement_time >= threshold:
                bonus = config['time_bonuses'][i]
        
        # Interaction depth bonus
        if interaction_type in ['comment', 'share', 'download']:
            bonus += config['interaction_depth_bonus']
        
        # Sequential interaction bonus (multiple interactions with same content)
        interaction_sequence_length = context.get('interaction_sequence_length', 1)
        if interaction_sequence_length > 1:
            bonus += min(0.2, (interaction_sequence_length - 1) * 0.05)
        
        return bonus
    
    def _calculate_diversity_bonus(self, user_id: int, post_id: int, context: Dict[str, Any]) -> float:
        """Calculate bonus for content diversity."""
        bonus = 0.0
        config = self.config['diversity']
        
        # Get recent content genres/types shown to user
        recent_interactions = list(self.user_interaction_history[user_id])[-config['genre_diversity_window']:]
        
        if recent_interactions:
            # Get genres of recent content
            recent_genres = set()
            recent_content_types = set()
            
            for interaction in recent_interactions:
                post_genres = interaction.get('genres', [])
                recent_genres.update(post_genres)
                recent_content_types.add(interaction.get('content_type', 'unknown'))
            
            # Check if current content adds diversity
            current_genres = set(context.get('post_genres', []))
            current_content_type = context.get('content_type', 'unknown')
            
            # Genre diversity bonus
            new_genres = current_genres - recent_genres
            if new_genres:
                genre_diversity_ratio = len(new_genres) / max(1, len(current_genres))
                bonus += config['max_diversity_bonus'] * genre_diversity_ratio
            
            # Content type diversity bonus
            if current_content_type not in recent_content_types:
                bonus += config['content_type_bonus']
        
        return bonus
    
    def _calculate_novelty_bonus(self, user_id: int, post_id: int, context: Dict[str, Any]) -> float:
        """Calculate bonus for showing novel content to user."""
        bonus = 0.0
        
        # Check if user has seen this content before
        user_history = self.user_interaction_history[user_id]
        seen_posts = {interaction.get('post_id') for interaction in user_history}
        
        if post_id not in seen_posts:
            # Novel content bonus
            bonus += 0.1
            
            # Additional bonus based on content recency
            content_age_days = context.get('content_age_days', float('inf'))
            if content_age_days < 7:  # Content less than a week old
                bonus += 0.05
            elif content_age_days < 30:  # Content less than a month old
                bonus += 0.03
        
        # Temporal novelty (content from different time periods)
        recent_content_years = {
            interaction.get('content_year', 2023) 
            for interaction in list(user_history)[-10:]
        }
        current_year = context.get('content_year', 2023)
        if current_year not in recent_content_years:
            bonus += 0.05
        
        return bonus
    
    def _calculate_long_term_adjustment(self, user_id: int, interaction_type: str, 
                                      context: Dict[str, Any]) -> float:
        """Calculate long-term satisfaction adjustment."""
        adjustment = 0.0
        config = self.config['long_term']
        
        # Session length consideration
        current_session_length = context.get('session_length', 0)
        target_length = config['session_length_target']
        
        if current_session_length < target_length / 2:
            # Early in session, encourage continued engagement
            if interaction_type in ['like', 'save', 'share']:
                adjustment += 0.1
        elif current_session_length > target_length * 1.5:
            # Long session, avoid fatigue
            if interaction_type in ['not_interested', 'skip']:
                adjustment -= 0.1  # Don't penalize as much for negative feedback in long sessions
        
        # User satisfaction trend
        user_satisfaction_trend = context.get('user_satisfaction_trend', 0.0)
        if user_satisfaction_trend < -0.1:  # Declining satisfaction
            # Be more generous with positive rewards
            if interaction_type in ['like', 'save']:
                adjustment += 0.05
        elif user_satisfaction_trend > 0.1:  # Improving satisfaction
            # Current strategy is working, maintain course
            adjustment += 0.02
        
        # Return visit consideration
        time_since_last_visit = context.get('hours_since_last_visit', 0)
        if time_since_last_visit > 72:  # User returning after 3+ days
            # Encourage re-engagement
            if interaction_type in ['like', 'save', 'share']:
                adjustment += 0.15
        
        return adjustment
    
    def _update_user_tracking(self, user_id: int, post_id: int, interaction_type: str, 
                            context: Dict[str, Any]):
        """Update user tracking for reward calculation."""
        # Record interaction
        interaction_record = {
            'post_id': post_id,
            'interaction_type': interaction_type,
            'timestamp': time.time(),
            'genres': context.get('post_genres', []),
            'content_type': context.get('content_type', 'unknown'),
            'content_year': context.get('content_year', 2023),
            'ml_score': context.get('original_score', 0.5),
            'engagement_time': context.get('engagement_time_seconds', 0)
        }
        
        self.user_interaction_history[user_id].append(interaction_record)
        
        # Update content diversity tracking
        post_genres = set(context.get('post_genres', []))
        self.user_content_diversity[user_id].update(post_genres)
        
        # Update exploration tracking
        if context.get('ranking_position', 0) > 5:
            self.user_exploration_history[user_id].append({
                'post_id': post_id,
                'position': context.get('ranking_position'),
                'timestamp': time.time()
            })
    
    def get_user_satisfaction_score(self, user_id: int, lookback_interactions: int = 20) -> float:
        """Calculate current user satisfaction score based on recent interactions."""
        recent_interactions = list(self.user_interaction_history[user_id])[-lookback_interactions:]
        
        if not recent_interactions:
            return 0.5  # Neutral for new users
        
        # Calculate weighted satisfaction score
        total_score = 0.0
        total_weight = 0.0
        
        for i, interaction in enumerate(recent_interactions):
            # More recent interactions have higher weight
            weight = (i + 1) / len(recent_interactions)
            
            interaction_type = interaction['interaction_type']
            base_reward = self.config['base_rewards'].get(interaction_type, 0.0)
            
            # Normalize to 0-1 scale
            normalized_score = (base_reward + 1.2) / 2.4  # Maps [-1.2, 1.2] to [0, 1]
            
            total_score += normalized_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def get_user_satisfaction_trend(self, user_id: int, window_size: int = 10) -> float:
        """Calculate trend in user satisfaction (positive = improving, negative = declining)."""
        recent_interactions = list(self.user_interaction_history[user_id])[-window_size * 2:]
        
        if len(recent_interactions) < window_size:
            return 0.0  # Not enough data
        
        # Split into early and recent periods
        early_period = recent_interactions[:window_size]
        recent_period = recent_interactions[-window_size:]
        
        def period_satisfaction(interactions):
            scores = []
            for interaction in interactions:
                interaction_type = interaction['interaction_type']
                base_reward = self.config['base_rewards'].get(interaction_type, 0.0)
                normalized_score = (base_reward + 1.2) / 2.4
                scores.append(normalized_score)
            return np.mean(scores) if scores else 0.5
        
        early_satisfaction = period_satisfaction(early_period)
        recent_satisfaction = period_satisfaction(recent_period)
        
        return recent_satisfaction - early_satisfaction
    
    def get_reward_stats(self) -> Dict[str, Any]:
        """Get statistics about reward calculation."""
        return {
            'config': self.config,
            'users_tracked': len(self.user_interaction_history),
            'total_interactions': sum(len(history) for history in self.user_interaction_history.values()),
            'avg_interactions_per_user': np.mean([len(history) for history in self.user_interaction_history.values()]) if self.user_interaction_history else 0,
            'global_stats': {
                'engagement': self.global_engagement_stats,
                'position': self.global_position_stats
            }
        }