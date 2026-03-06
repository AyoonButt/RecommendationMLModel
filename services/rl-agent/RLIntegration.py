import time
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

from RLExperienceCollector import RLExperienceCollector, RLExperience
from RLRewardEngineer import RLRewardEngineer, RewardComponents
from RLStateRepresentation import RLStateBuilder, RLState
from RLContextualBandit import RLContextualBandit, Action, ActionResult

logger = logging.getLogger("rl-integration")

@dataclass
class RLRecommendationResult:
    """Result of RL-enhanced recommendation."""
    post_ids: List[int]
    original_scores: List[float]
    rl_adjusted_scores: List[float]
    actions_taken: List[Action]
    metadata: Dict[str, Any]
    processing_time: float

class RLIntegrationManager:
    """
    Main integration manager that coordinates RL components with existing recommendation system.
    Provides hooks for PostStream and MetadataEnhancer integration.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RL integration manager.

        Args:
            config: Configuration dictionary (merged with defaults)
        """
        self.config = self._merge_config(config)
        
        # Initialize RL components
        self.experience_collector = RLExperienceCollector(
            experience_buffer_size=self.config['experience']['buffer_size'],
            session_timeout_minutes=self.config['experience']['session_timeout'],
            reward_config=self.config['reward']['mapping']
        )
        
        self.reward_engineer = RLRewardEngineer(
            reward_config=self.config['reward'],
            api_base_url=self.config.get('api_base_url', 'http://localhost:8080')
        )
        
        self.state_builder = RLStateBuilder(
            user_embedding_dim=self.config['embeddings']['user_dim'],
            post_embedding_dim=self.config['embeddings']['post_dim'],
            api_base_url=self.config.get('api_base_url', 'http://localhost:8080')
        )
        
        self.contextual_bandit = RLContextualBandit(
            state_dim=self.state_builder.get_state_dimension(),
            config=self.config['bandit']
        )
        
        # Integration state
        self.is_enabled = self.config['integration']['enabled']
        self.learning_mode = self.config['integration']['learning_mode']  # 'online' or 'offline'
        self.a_b_test_ratio = self.config['integration']['a_b_test_ratio']
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.integration_stats = defaultdict(int)
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=self.config['performance']['max_workers'])
        
        logger.info("RL Integration Manager initialized")
        logger.info(f"RL enabled: {self.is_enabled}, Learning mode: {self.learning_mode}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'integration': {
                'enabled': True,
                'learning_mode': 'online',  # 'online' or 'offline'
                'a_b_test_ratio': 0.2,  # 20% of users get RL treatment
                'safety_threshold': 0.1,  # Stop RL if performance drops by more than 10%
                'warmup_interactions': 50  # Minimum interactions before RL kicks in
            },
            'experience': {
                'buffer_size': 10000,
                'session_timeout': 30
            },
            'reward': {
                'mapping': {
                    'like': 0.6, 'save': 1.0, 'not_interested': -0.9,
                    'more_info': 0.3, 'skip': -0.2, 'share': 0.8
                },
                'shaping_weights': {
                    'exploration': 0.15, 'engagement': 0.20,
                    'diversity': 0.10, 'novelty': 0.10, 'long_term': 0.25
                }
            },
            'embeddings': {
                'user_dim': 32,
                'post_dim': 32
            },
            'bandit': {
                'action_space': {'total_actions': 20, 'action_encoding_dim': 10},
                'exploration': {'strategy': 'epsilon_greedy', 'epsilon': 0.1}
            },
            'performance': {
                'max_workers': 4,
                'max_processing_time': 0.1,  # 100ms max for RL processing
                'fallback_enabled': True
            }
        }

    def _merge_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Merge provided config with defaults to ensure all required fields exist."""
        defaults = self._get_default_config()

        if config is None:
            return defaults

        # Deep merge config with defaults
        merged = {}
        all_keys = set(defaults.keys()) | set(config.keys())

        for key in all_keys:
            if key in defaults and key in config:
                if isinstance(defaults[key], dict) and isinstance(config[key], dict):
                    # Recursively merge nested dicts
                    merged[key] = {**defaults[key], **config[key]}
                else:
                    # Use provided value
                    merged[key] = config[key]
            elif key in defaults:
                merged[key] = defaults[key]
            else:
                merged[key] = config[key]

        return merged
    
    def enhance_recommendations(self, user_id: int, candidate_posts: List[Dict[str, Any]], 
                              context: Dict[str, Any]) -> RLRecommendationResult:
        """
        Main entry point for RL-enhanced recommendations.
        Called by PostStream to enhance recommendation results.
        
        Args:
            user_id: User ID
            candidate_posts: List of candidate posts with metadata and scores
            context: Request context (embeddings, user metadata, etc.)
            
        Returns:
            RL-enhanced recommendation result
        """
        start_time = time.time()
        
        try:
            # Check if RL should be applied
            if not self._should_apply_rl(user_id, context):
                return self._create_passthrough_result(candidate_posts, start_time)
            
            # Build state representation
            state = self._build_comprehensive_state(user_id, candidate_posts, context)
            
            # Select RL action
            action = self.contextual_bandit.select_action(
                state.state_vector, user_id, context
            )
            
            # Apply action to recommendations
            enhanced_posts = self._apply_action_to_recommendations(
                action, candidate_posts, context
            )
            
            # Extract results
            post_ids = [post['id'] for post in enhanced_posts]
            original_scores = [post.get('original_score', 0.5) for post in candidate_posts]
            rl_scores = [post.get('rl_adjusted_score', post.get('score', 0.5)) for post in enhanced_posts]
            
            # Store state and action for future reward calculation
            self._store_pending_experience(user_id, action, state, context)
            
            processing_time = time.time() - start_time
            
            # Track performance
            self.integration_stats['rl_recommendations_served'] += 1
            self.performance_metrics['processing_time'].append(processing_time)
            
            logger.debug(f"RL enhancement completed for user {user_id} in {processing_time:.3f}s")
            
            return RLRecommendationResult(
                post_ids=post_ids,
                original_scores=original_scores,
                rl_adjusted_scores=rl_scores,
                actions_taken=[action],
                metadata={
                    'state_dim': len(state.state_vector),
                    'action_type': action.action_type,
                    'rl_enabled': True
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in RL enhancement: {e}")
            return self._create_fallback_result(candidate_posts, start_time)
    
    def process_user_interaction(self, user_id: int, post_id: int, interaction_type: str,
                               context: Dict[str, Any] = None) -> None:
        """
        Process user interaction and update RL components.
        Called when user interacts with recommended content.
        
        Args:
            user_id: User ID
            post_id: Post ID that was interacted with
            interaction_type: Type of interaction (like, save, not_interested, etc.)
            context: Additional context
        """
        try:
            # Collect experience asynchronously
            if self.learning_mode == 'online':
                self.executor.submit(
                    self._process_interaction_async,
                    user_id, post_id, interaction_type, context or {}
                )
            else:
                # Store for offline processing
                self._store_offline_interaction(user_id, post_id, interaction_type, context)
            
        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
    
    def _process_interaction_async(self, user_id: int, post_id: int, 
                                 interaction_type: str, context: Dict[str, Any]):
        """Process interaction asynchronously."""
        try:
            # Collect RL experience
            experience = self.experience_collector.collect_interaction_experience(
                user_id=user_id,
                post_id=post_id,
                interaction_type=interaction_type,
                additional_context=context
            )
            
            # Calculate detailed reward
            reward_components = self.reward_engineer.calculate_reward(
                interaction_type=interaction_type,
                user_id=user_id,
                post_id=post_id,
                context=context
            )
            
            # Update experience with detailed reward
            experience.reward = reward_components.final_reward
            
            # Check if we have a pending action for this user/post
            pending_action = self._get_pending_action(user_id, post_id)
            if pending_action:
                # Create action result
                action_result = ActionResult(
                    action=pending_action['action'],
                    actual_reward=reward_components.final_reward,
                    user_id=user_id,
                    post_id=post_id,
                    timestamp=time.time(),
                    metadata={
                        'state': pending_action['state'],
                        'reward_components': reward_components.to_dict(),
                        'interaction_type': interaction_type
                    }
                )
                
                # Update contextual bandit
                self.contextual_bandit.update_action_result(action_result)
                
                # Clear pending action
                self._clear_pending_action(user_id, post_id)
            
            # Update state builder with interaction
            self.state_builder.update_user_interaction(user_id, {
                'interaction_type': interaction_type,
                'post_id': post_id,
                'reward': reward_components.final_reward,
                'position': context.get('ranking_position', 0)
            })
            
            # Track metrics
            self.integration_stats['interactions_processed'] += 1
            self.performance_metrics['reward_values'].append(reward_components.final_reward)
            
        except Exception as e:
            logger.error(f"Error in async interaction processing: {e}")
    
    def _should_apply_rl(self, user_id: int, context: Dict[str, Any]) -> bool:
        """Determine if RL should be applied for this request."""
        if not self.is_enabled:
            return False
        
        # A/B testing logic
        if user_id % 100 < (self.a_b_test_ratio * 100):
            return True
        
        # Check warmup period
        user_interactions = len(self.experience_collector.user_sequences.get(user_id, []))
        if user_interactions < self.config['integration']['warmup_interactions']:
            return False
        
        # Safety check - disable RL if performance is poor
        if self._check_safety_threshold():
            return False
        
        return True
    
    def _build_comprehensive_state(self, user_id: int, candidate_posts: List[Dict], 
                                 context: Dict[str, Any]) -> RLState:
        """Build comprehensive state representation."""
        # For demo, use first post as primary context
        primary_post = candidate_posts[0] if candidate_posts else {}
        post_id = primary_post.get('id', 0)
        
        # Enhance context with candidate post information
        enhanced_context = {
            **context,
            'user_embedding': context.get('user_embedding', [0.0] * 32),
            'post_embedding': primary_post.get('embedding', [0.0] * 32),
            'post_metadata': primary_post.get('metadata', {}),
            'candidate_count': len(candidate_posts),
            'avg_candidate_score': np.mean([p.get('score', 0.5) for p in candidate_posts])
        }
        
        return self.state_builder.build_state(user_id, post_id, enhanced_context)
    
    def _apply_action_to_recommendations(self, action: Action, candidate_posts: List[Dict], 
                                       context: Dict[str, Any]) -> List[Dict]:
        """Apply RL action to modify recommendations."""
        enhanced_posts = []
        
        for i, post in enumerate(candidate_posts):
            enhanced_post = post.copy()
            original_score = post.get('score', 0.5)
            enhanced_post['original_score'] = original_score
            
            # Apply action based on type
            if action.action_type == "boost_adjustment":
                # Apply metadata boost adjustments
                recency_adj = action.parameters.get('recency_boost_adjustment', 0.0)
                cast_crew_adj = action.parameters.get('cast_crew_boost_adjustment', 0.0)
                
                boost_factor = 1.0 + recency_adj + cast_crew_adj
                enhanced_score = original_score * boost_factor
                
            elif action.action_type == "ranking_modification":
                # Apply position-based adjustments
                position_adj = action.parameters.get('position_adjustment', 0)
                if i < 10:  # Only adjust top 10 results
                    adjustment_factor = 1.0 + (position_adj * 0.05)  # 5% per position
                    enhanced_score = original_score * adjustment_factor
                else:
                    enhanced_score = original_score
                    
            elif action.action_type == "exploration":
                # Apply exploration-based adjustments
                exploration_prob = action.parameters.get('exploration_probability', 0.0)
                diversity_weight = action.parameters.get('diversity_weight', 0.0)
                
                # Add randomness for exploration
                if np.random.random() < exploration_prob:
                    exploration_boost = 1.0 + (diversity_weight * 0.1)
                    enhanced_score = original_score * exploration_boost
                else:
                    enhanced_score = original_score
            else:
                enhanced_score = original_score
            
            # Ensure score stays in valid range
            enhanced_score = np.clip(enhanced_score, 0.0, 1.0)
            enhanced_post['score'] = enhanced_score
            enhanced_post['rl_adjusted_score'] = enhanced_score
            enhanced_post['rl_action'] = action.to_dict()
            
            enhanced_posts.append(enhanced_post)
        
        # Re-sort by enhanced scores
        enhanced_posts.sort(key=lambda x: x['score'], reverse=True)
        
        return enhanced_posts
    
    def _store_pending_experience(self, user_id: int, action: Action, 
                                state: RLState, context: Dict[str, Any]):
        """Store pending experience for reward calculation."""
        if not hasattr(self, '_pending_actions'):
            self._pending_actions = {}
        
        self._pending_actions[f"{user_id}"] = {
            'action': action,
            'state': state.state_vector,
            'timestamp': time.time(),
            'context': context
        }
    
    def _get_pending_action(self, user_id: int, post_id: int) -> Optional[Dict]:
        """Get pending action for user/post."""
        if not hasattr(self, '_pending_actions'):
            return None
        
        return self._pending_actions.get(f"{user_id}")
    
    def _clear_pending_action(self, user_id: int, post_id: int):
        """Clear pending action."""
        if hasattr(self, '_pending_actions'):
            self._pending_actions.pop(f"{user_id}", None)
    
    def _create_passthrough_result(self, candidate_posts: List[Dict], 
                                 start_time: float) -> RLRecommendationResult:
        """Create result that passes through original recommendations."""
        post_ids = [post['id'] for post in candidate_posts]
        scores = [post.get('score', 0.5) for post in candidate_posts]
        
        return RLRecommendationResult(
            post_ids=post_ids,
            original_scores=scores,
            rl_adjusted_scores=scores,
            actions_taken=[],
            metadata={'rl_enabled': False, 'reason': 'passthrough'},
            processing_time=time.time() - start_time
        )
    
    def _create_fallback_result(self, candidate_posts: List[Dict], 
                              start_time: float) -> RLRecommendationResult:
        """Create fallback result when RL fails."""
        self.integration_stats['rl_fallbacks'] += 1
        return self._create_passthrough_result(candidate_posts, start_time)
    
    def _check_safety_threshold(self) -> bool:
        """Check if RL performance has degraded below safety threshold."""
        if len(self.performance_metrics['reward_values']) < 100:
            return False  # Not enough data
        
        recent_rewards = self.performance_metrics['reward_values'][-100:]
        baseline_rewards = self.performance_metrics['reward_values'][-500:-100] if len(self.performance_metrics['reward_values']) >= 500 else recent_rewards
        
        if not baseline_rewards:
            return False
        
        recent_avg = np.mean(recent_rewards)
        baseline_avg = np.mean(baseline_rewards)
        
        performance_drop = (baseline_avg - recent_avg) / abs(baseline_avg) if baseline_avg != 0 else 0
        
        return performance_drop > self.config['integration']['safety_threshold']
    
    def _store_offline_interaction(self, user_id: int, post_id: int, 
                                 interaction_type: str, context: Dict[str, Any]):
        """Store interaction for offline processing."""
        # Implementation for offline storage (e.g., to database or file)
        pass
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics."""
        stats = dict(self.integration_stats)
        
        if self.performance_metrics['processing_time']:
            stats['avg_processing_time'] = np.mean(self.performance_metrics['processing_time'])
            stats['max_processing_time'] = np.max(self.performance_metrics['processing_time'])
        
        if self.performance_metrics['reward_values']:
            stats['avg_reward'] = np.mean(self.performance_metrics['reward_values'])
            stats['reward_std'] = np.std(self.performance_metrics['reward_values'])
        
        stats['rl_enabled'] = self.is_enabled
        stats['learning_mode'] = self.learning_mode
        stats['safety_check_passed'] = not self._check_safety_threshold()
        
        return stats
    
    def reset_performance_metrics(self):
        """Reset performance tracking metrics."""
        self.performance_metrics.clear()
        self.integration_stats.clear()
        logger.info("Reset RL integration performance metrics")
    
    def enable_rl(self):
        """Enable RL processing."""
        self.is_enabled = True
        logger.info("RL processing enabled")
    
    def disable_rl(self):
        """Disable RL processing."""
        self.is_enabled = False
        logger.info("RL processing disabled")
    
    def set_learning_mode(self, mode: str):
        """Set learning mode ('online' or 'offline')."""
        if mode in ['online', 'offline']:
            self.learning_mode = mode
            logger.info(f"Learning mode set to {mode}")
        else:
            logger.warning(f"Invalid learning mode: {mode}")


# Convenience function for easy integration
def create_rl_integration_manager(config: Dict[str, Any] = None) -> RLIntegrationManager:
    """
    Create and configure RL integration manager.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RLIntegrationManager instance
    """
    return RLIntegrationManager(config)


# Integration hooks for existing components
class RLEnhancedMetadataEnhancer:
    """
    Wrapper for MetadataEnhancer that includes RL integration.
    Can be used as drop-in replacement for existing MetadataEnhancer.
    """
    
    def __init__(self, original_enhancer, rl_manager: RLIntegrationManager):
        self.original_enhancer = original_enhancer
        self.rl_manager = rl_manager
    
    def enhance_candidate_posts(self, user_id: int, candidate_posts: List[Dict], 
                              context: Dict[str, Any]) -> List[Dict]:
        """Enhanced version of candidate post enhancement with RL."""
        # First apply original metadata enhancement
        enhanced_posts = self.original_enhancer.enhance_candidate_posts(
            user_id, candidate_posts, context
        )
        
        # Then apply RL enhancement
        rl_result = self.rl_manager.enhance_recommendations(
            user_id, enhanced_posts, context
        )
        
        # Merge results
        final_posts = []
        for i, post_id in enumerate(rl_result.post_ids):
            # Find corresponding enhanced post
            enhanced_post = next((p for p in enhanced_posts if p['id'] == post_id), None)
            if enhanced_post:
                enhanced_post['score'] = rl_result.rl_adjusted_scores[i]
                enhanced_post['rl_metadata'] = {
                    'original_score': rl_result.original_scores[i],
                    'rl_adjusted_score': rl_result.rl_adjusted_scores[i],
                    'actions_taken': [a.to_dict() for a in rl_result.actions_taken]
                }
                final_posts.append(enhanced_post)
        
        return final_posts