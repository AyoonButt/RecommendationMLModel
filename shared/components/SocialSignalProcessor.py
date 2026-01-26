#!/usr/bin/env python3
"""
Social Signal Processor for ML Recommendation Model
Handles social influence, sentiment analysis, and dual preference boosting
"""

import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time
from functools import lru_cache
import threading

# Configure logging
logger = logging.getLogger(__name__)

class InteractionType(Enum):
    LIKE = "like"
    SAVE = "save"
    NOT_INTERESTED = "not_interested"
    COMMENT_POSITIVE = "comment_positive"
    COMMENT_NEGATIVE = "comment_negative"
    VIEW_TIME_HIGH = "view_time_high"
    VIEW_TIME_LOW = "view_time_low"

@dataclass
class SocialSignal:
    """Represents a social signal for a user-post interaction"""
    user_id: int
    post_id: int
    signal_type: InteractionType
    strength: float  # 0.0 to 1.0
    timestamp: float
    metadata: Optional[Dict] = None

@dataclass
class SentimentSignal:
    """Represents sentiment analysis results"""
    post_id: int
    sentiment_label: str  # POSITIVE, NEGATIVE, NEUTRAL
    confidence_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    comment_count: int

@dataclass
class UserSocialProfile:
    """User's social influence profile"""
    user_id: int
    following_influence: Dict[int, float]  # followed_user_id -> influence_weight
    social_activity_level: str
    influence_score: float
    preference_alignment: Dict[int, float]  # followed_user_id -> alignment_score

class SocialSignalProcessor:
    """
    Processes social signals and integrates them into ML recommendations
    """
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8080/api",
                 bert_service_url: str = "http://localhost:8080",
                 cache_size: int = 1000,
                 social_weight: float = 0.25):
        """
        Initialize the social signal processor
        
        Args:
            api_base_url: Base URL for the Kotlin API
            bert_service_url: URL for BERT sentiment service
            cache_size: Size of the LRU cache for social data
            social_weight: Weight for social signals (0.0 to 1.0)
        """
        self.api_base_url = api_base_url
        self.bert_service_url = bert_service_url
        self.social_weight = social_weight
        self.personal_weight = 1.0 - social_weight
        
        # Caching
        self._social_cache = {}
        self._sentiment_cache = {}
        self._cache_lock = threading.Lock()
        self.cache_size = cache_size
        
        # Signal weights for dual preference boosting
        self.positive_weights = {
            InteractionType.LIKE: 0.8,
            InteractionType.SAVE: 1.0,
            InteractionType.COMMENT_POSITIVE: 0.6,
            InteractionType.VIEW_TIME_HIGH: 0.4
        }
        
        self.negative_weights = {
            InteractionType.NOT_INTERESTED: -0.9,
            InteractionType.COMMENT_NEGATIVE: -0.5,
            InteractionType.VIEW_TIME_LOW: -0.2
        }
        
        logger.info(f"Initialized SocialSignalProcessor with social_weight={social_weight}")

    def process_social_boost(self, 
                           user_id: int, 
                           post_ids: List[int],
                           user_embedding: np.ndarray,
                           post_embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply social boosting to recommendation scores
        
        Args:
            user_id: User ID
            post_ids: List of post IDs
            user_embedding: User embedding vector
            post_embeddings: Post embedding vectors
            
        Returns:
            Tuple of (boosted_scores, metadata)
        """
        try:
            # Get social influence data
            social_profile = self._get_user_social_profile(user_id)
            
            # Get sentiment data for posts
            sentiment_data = self._get_posts_sentiment_data(post_ids)
            
            # Calculate social boost factors
            social_boosts = self._calculate_social_boosts(user_id, post_ids, social_profile)
            
            # Calculate sentiment boosts
            sentiment_boosts = self._calculate_sentiment_boosts(post_ids, sentiment_data)
            
            # Apply dual preference boosting
            preference_boosts = self._calculate_preference_boosts(user_id, post_ids)
            
            # Combine all boosts
            combined_boosts = self._combine_boost_factors(
                social_boosts, sentiment_boosts, preference_boosts
            )
            
            # Apply boosts to embeddings
            boosted_scores = self._apply_boosts_to_scores(
                user_embedding, post_embeddings, combined_boosts
            )
            
            # Prepare metadata
            metadata = {
                "social_boosts": social_boosts,
                "sentiment_boosts": sentiment_boosts,
                "preference_boosts": preference_boosts,
                "combined_boosts": combined_boosts,
                "social_profile_strength": social_profile.influence_score if social_profile else 0.0,
                "avg_sentiment_score": np.mean(list(sentiment_boosts.values())) if sentiment_boosts else 0.0
            }
            
            logger.info(f"Applied social boosting for user {user_id} on {len(post_ids)} posts")
            return boosted_scores, metadata
            
        except Exception as e:
            logger.error(f"Error in social boost processing: {e}")
            # Return original scores if social processing fails
            base_scores = np.dot(user_embedding.reshape(1, -1), post_embeddings.T)
            return base_scores.flatten(), {"error": str(e)}

    def get_social_recommendations(self, 
                                 user_id: int, 
                                 limit: int = 20,
                                 exclude_post_ids: List[int] = None) -> List[Dict]:
        """
        Get purely social recommendations based on followed users
        
        Args:
            user_id: User ID
            limit: Maximum number of recommendations
            exclude_post_ids: Post IDs to exclude
            
        Returns:
            List of social recommendation candidates
        """
        try:
            url = f"{self.api_base_url}/social/users/{user_id}/recommendations"
            params = {"limit": limit}
            if exclude_post_ids:
                params["excludePostIds"] = exclude_post_ids
                
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("recommendations", [])
            
            logger.warning(f"Failed to get social recommendations for user {user_id}")
            return []
            
        except Exception as e:
            logger.error(f"Error getting social recommendations: {e}")
            return []

    def update_user_interaction(self, 
                               user_id: int, 
                               post_id: int, 
                               interaction_type: InteractionType,
                               strength: float = 1.0,
                               metadata: Dict = None):
        """
        Update user interaction for future social signal processing
        
        Args:
            user_id: User ID
            post_id: Post ID
            interaction_type: Type of interaction
            strength: Interaction strength (0.0 to 1.0)
            metadata: Additional metadata
        """
        try:
            signal = SocialSignal(
                user_id=user_id,
                post_id=post_id,
                signal_type=interaction_type,
                strength=strength,
                timestamp=time.time(),
                metadata=metadata
            )
            
            # Store signal for processing (you might want to persist this)
            logger.info(f"Updated interaction: user {user_id}, post {post_id}, type {interaction_type.value}")
            
            # Clear cache for this user to get fresh data
            self._invalidate_user_cache(user_id)
            
        except Exception as e:
            logger.error(f"Error updating user interaction: {e}")

    def _get_user_social_profile(self, user_id: int) -> Optional[UserSocialProfile]:
        """Get user's social profile with caching"""
        cache_key = f"social_profile_{user_id}"
        
        with self._cache_lock:
            if cache_key in self._social_cache:
                cached_data, timestamp = self._social_cache[cache_key]
                if time.time() - timestamp < 3600:  # 1 hour cache
                    return cached_data
        
        try:
            url = f"{self.api_base_url}/social/users/{user_id}/influence"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Create social profile (simplified - you might want more data)
                profile = UserSocialProfile(
                    user_id=user_id,
                    following_influence={},  # Would need additional API call
                    social_activity_level=data.get("influenceLevel", "LOW"),
                    influence_score=data.get("socialMetrics", {}).get("influenceToOthers", 0.0),
                    preference_alignment={}  # Would need additional API call
                )
                
                # Cache the result
                with self._cache_lock:
                    if len(self._social_cache) >= self.cache_size:
                        # Remove oldest entry
                        oldest_key = min(self._social_cache.keys(), 
                                       key=lambda k: self._social_cache[k][1])
                        del self._social_cache[oldest_key]
                    
                    self._social_cache[cache_key] = (profile, time.time())
                
                return profile
            
        except Exception as e:
            logger.warning(f"Failed to get social profile for user {user_id}: {e}")
        
        return None

    def _get_posts_sentiment_data(self, post_ids: List[int]) -> Dict[int, SentimentSignal]:
        """Get sentiment data for posts with caching"""
        result = {}
        uncached_posts = []
        
        # Check cache first
        with self._cache_lock:
            for post_id in post_ids:
                cache_key = f"sentiment_{post_id}"
                if cache_key in self._sentiment_cache:
                    cached_data, timestamp = self._sentiment_cache[cache_key]
                    if time.time() - timestamp < 7200:  # 2 hour cache
                        result[post_id] = cached_data
                    else:
                        uncached_posts.append(post_id)
                else:
                    uncached_posts.append(post_id)
        
        # Fetch uncached data
        if uncached_posts:
            try:
                for post_id in uncached_posts:
                    url = f"{self.api_base_url}/comments/sentiment/posts/{post_id}"
                    response = requests.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        sentiment = SentimentSignal(
                            post_id=post_id,
                            sentiment_label=data.get("overallSentiment", "NEUTRAL"),
                            confidence_score=data.get("confidenceScore", 0.5),
                            positive_score=data.get("sentimentDistribution", {}).get("positive", 0) / max(data.get("totalComments", 1), 1),
                            negative_score=data.get("sentimentDistribution", {}).get("negative", 0) / max(data.get("totalComments", 1), 1),
                            neutral_score=data.get("sentimentDistribution", {}).get("neutral", 0) / max(data.get("totalComments", 1), 1),
                            comment_count=data.get("totalComments", 0)
                        )
                        
                        result[post_id] = sentiment
                        
                        # Cache the result
                        with self._cache_lock:
                            cache_key = f"sentiment_{post_id}"
                            if len(self._sentiment_cache) >= self.cache_size:
                                # Remove oldest entry
                                oldest_key = min(self._sentiment_cache.keys(), 
                                               key=lambda k: self._sentiment_cache[k][1])
                                del self._sentiment_cache[oldest_key]
                            
                            self._sentiment_cache[cache_key] = (sentiment, time.time())
                    
            except Exception as e:
                logger.warning(f"Failed to get sentiment data: {e}")
        
        return result

    def _calculate_social_boosts(self, 
                               user_id: int, 
                               post_ids: List[int], 
                               social_profile: Optional[UserSocialProfile]) -> Dict[int, float]:
        """Calculate social boost factors for posts"""
        boosts = {}
        
        if not social_profile:
            return {post_id: 1.0 for post_id in post_ids}
        
        try:
            # Get social boost factors from API
            for post_id in post_ids:
                url = f"{self.api_base_url}/social/users/{user_id}/posts/{post_id}/social-boost"
                response = requests.get(url, timeout=3)
                
                if response.status_code == 200:
                    data = response.json()
                    boosts[post_id] = data.get("socialBoostFactor", 1.0)
                else:
                    boosts[post_id] = 1.0
                    
        except Exception as e:
            logger.warning(f"Error calculating social boosts: {e}")
            boosts = {post_id: 1.0 for post_id in post_ids}
        
        return boosts

    def _calculate_sentiment_boosts(self, 
                                  post_ids: List[int], 
                                  sentiment_data: Dict[int, SentimentSignal]) -> Dict[int, float]:
        """Calculate sentiment-based boost factors"""
        boosts = {}
        
        for post_id in post_ids:
            sentiment = sentiment_data.get(post_id)
            
            if not sentiment or sentiment.comment_count < 3:
                boosts[post_id] = 1.0
                continue
            
            # Calculate boost based on sentiment
            if sentiment.sentiment_label == "POSITIVE" and sentiment.confidence_score > 0.7:
                boost = 1.0 + (sentiment.confidence_score * 0.15)  # Up to 15% boost
            elif sentiment.sentiment_label == "NEGATIVE" and sentiment.confidence_score > 0.7:
                boost = 1.0 - (sentiment.confidence_score * 0.1)   # Up to 10% penalty
            else:
                boost = 1.0
            
            boosts[post_id] = max(0.5, min(1.3, boost))  # Clamp between 0.5 and 1.3
        
        return boosts

    def _calculate_preference_boosts(self, user_id: int, post_ids: List[int]) -> Dict[int, float]:
        """Calculate dual preference boosting (likes/saves vs not interested)"""
        boosts = {}
        
        try:
            # This would typically query user interaction history
            # For now, we'll use a simplified approach
            for post_id in post_ids:
                # Default neutral boost
                boost = 1.0
                
                # In a real implementation, you'd check:
                # - User's previous interactions with similar content
                # - User's explicit positive/negative feedback
                # - User's "not interested" signals
                
                # Placeholder logic - would be replaced with actual interaction data
                boosts[post_id] = boost
                
        except Exception as e:
            logger.warning(f"Error calculating preference boosts: {e}")
            boosts = {post_id: 1.0 for post_id in post_ids}
        
        return boosts

    def _combine_boost_factors(self, 
                             social_boosts: Dict[int, float],
                             sentiment_boosts: Dict[int, float],
                             preference_boosts: Dict[int, float]) -> Dict[int, float]:
        """Combine different boost factors"""
        combined = {}
        
        all_post_ids = set(social_boosts.keys()) | set(sentiment_boosts.keys()) | set(preference_boosts.keys())
        
        for post_id in all_post_ids:
            social_boost = social_boosts.get(post_id, 1.0)
            sentiment_boost = sentiment_boosts.get(post_id, 1.0)
            preference_boost = preference_boosts.get(post_id, 1.0)
            
            # Weighted combination
            combined_boost = (
                social_boost * 0.4 +           # 40% social signals
                sentiment_boost * 0.3 +        # 30% sentiment signals  
                preference_boost * 0.3         # 30% preference signals
            )
            
            # Clamp final boost
            combined[post_id] = max(0.3, min(1.8, combined_boost))
        
        return combined

    def _apply_boosts_to_scores(self, 
                              user_embedding: np.ndarray,
                              post_embeddings: np.ndarray,
                              boost_factors: Dict[int, float]) -> np.ndarray:
        """Apply boost factors to recommendation scores"""
        try:
            # Calculate base similarity scores
            user_norm = np.linalg.norm(user_embedding)
            post_norms = np.linalg.norm(post_embeddings, axis=1)
            
            user_normalized = user_embedding / max(user_norm, 1e-8)
            post_normalized = post_embeddings / np.maximum(post_norms.reshape(-1, 1), 1e-8)
            
            # Compute cosine similarity
            base_scores = np.dot(post_normalized, user_normalized)
            
            # Apply boost factors
            boosted_scores = base_scores.copy()
            for i, (post_id, boost) in enumerate(boost_factors.items()):
                if i < len(boosted_scores):
                    boosted_scores[i] *= boost
            
            return boosted_scores
            
        except Exception as e:
            logger.error(f"Error applying boosts to scores: {e}")
            # Return base scores if boosting fails
            return np.dot(post_embeddings, user_embedding)

    def _invalidate_user_cache(self, user_id: int):
        """Invalidate cache entries for a specific user"""
        with self._cache_lock:
            keys_to_remove = [key for key in self._social_cache.keys() 
                            if key.endswith(f"_{user_id}")]
            for key in keys_to_remove:
                del self._social_cache[key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                "social_cache_size": len(self._social_cache),
                "sentiment_cache_size": len(self._sentiment_cache),
                "max_cache_size": self.cache_size,
                "social_weight": self.social_weight
            }

    def clear_caches(self):
        """Clear all caches"""
        with self._cache_lock:
            self._social_cache.clear()
            self._sentiment_cache.clear()
        logger.info("Cleared all caches")


# Utility functions for integration
def create_social_signal_processor(config: Dict[str, Any]) -> SocialSignalProcessor:
    """Factory function to create social signal processor"""
    return SocialSignalProcessor(
        api_base_url=config.get("api_base_url", "http://localhost:8080/api"),
        bert_service_url=config.get("bert_service_url", "http://localhost:8080"),
        cache_size=config.get("cache_size", 1000),
        social_weight=config.get("social_weight", 0.25)
    )


def apply_dual_preference_boosting(scores: np.ndarray, 
                                 user_interactions: List[SocialSignal]) -> np.ndarray:
    """
    Apply dual preference boosting to scores
    
    Args:
        scores: Base recommendation scores
        user_interactions: List of user interaction signals
        
    Returns:
        Boosted scores
    """
    try:
        boosted_scores = scores.copy()
        
        # Create interaction lookup
        interaction_lookup = {}
        for signal in user_interactions:
            if signal.post_id not in interaction_lookup:
                interaction_lookup[signal.post_id] = []
            interaction_lookup[signal.post_id].append(signal)
        
        # Apply boosts based on interactions
        for i, score in enumerate(scores):
            if i in interaction_lookup:
                signals = interaction_lookup[i]
                
                # Calculate cumulative boost from all signals
                total_boost = 1.0
                for signal in signals:
                    if signal.signal_type in [InteractionType.LIKE, InteractionType.SAVE]:
                        total_boost *= (1.0 + signal.strength * 0.2)  # Positive boost
                    elif signal.signal_type == InteractionType.NOT_INTERESTED:
                        total_boost *= (1.0 - signal.strength * 0.3)  # Negative boost
                
                boosted_scores[i] = score * total_boost
        
        return boosted_scores
        
    except Exception as e:
        logger.error(f"Error applying dual preference boosting: {e}")
        return scores