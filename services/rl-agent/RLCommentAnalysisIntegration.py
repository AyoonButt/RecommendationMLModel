#!/usr/bin/env python3
"""
RL Comment Analysis Integration
Integrates comment sentiment analysis results into RL decision-making process
"""

import time
import logging
import os
import requests
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import json

logger = logging.getLogger("rl-comment-analysis")

@dataclass
class CommentAnalysisFeatures:
    """Comment analysis features for RL state representation."""
    # Sentiment scores
    positive_sentiment_ratio: float
    negative_sentiment_ratio: float
    neutral_sentiment_ratio: float
    average_sentiment_confidence: float
    
    # Toxicity metrics
    average_toxicity_score: float
    high_toxicity_count: int
    hate_speech_count: int
    spam_count: int
    
    # Engagement metrics
    total_comments: int
    comment_engagement_score: float
    comment_quality_score: float
    
    # Temporal features
    recent_comment_trend: float  # Positive = improving sentiment
    comment_velocity: float      # Comments per hour
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for RL state."""
        return np.array([
            self.positive_sentiment_ratio,
            self.negative_sentiment_ratio, 
            self.neutral_sentiment_ratio,
            self.average_sentiment_confidence,
            self.average_toxicity_score,
            min(1.0, self.high_toxicity_count / 10.0),  # Normalize
            min(1.0, self.hate_speech_count / 5.0),     # Normalize
            min(1.0, self.spam_count / 5.0),            # Normalize
            min(1.0, self.total_comments / 100.0),      # Normalize
            self.comment_engagement_score,
            self.comment_quality_score,
            self.recent_comment_trend,
            self.comment_velocity
        ], dtype=np.float32)

class RLCommentAnalysisIntegrator:
    """
    Integrates comment analysis results into RL agent decision-making.
    Provides features and rewards based on comment sentiment and engagement.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        """
        Initialize the comment analysis integrator.
        
        Args:
            api_base_url: URL of the Spring API (where comment analysis is stored)
        """
        self.api_base_url = api_base_url.rstrip('/')
        
        # Cache comment analysis results
        self.comment_cache: Dict[int, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Track comment patterns for users
        self.user_comment_patterns: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Feature dimension for RL state
        self.feature_dim = 13
        
        logger.info(f"RL Comment Analysis Integrator initialized with API URL: {api_base_url}")
    
    def get_comment_analysis_features(self, post_id: int, user_id: int = None) -> CommentAnalysisFeatures:
        """
        Get comment analysis features for a post.
        
        Args:
            post_id: Post ID to analyze
            user_id: Optional user ID for personalized features
            
        Returns:
            CommentAnalysisFeatures object
        """
        try:
            # Get comment analysis from service
            analysis_data = self._fetch_comment_analysis(post_id)
            
            if not analysis_data or "error" in analysis_data:
                return self._create_default_features()
            
            # Extract sentiment features
            sentiment_features = self._extract_sentiment_features(analysis_data)
            
            # Extract toxicity features
            toxicity_features = self._extract_toxicity_features(analysis_data)
            
            # Extract engagement features
            engagement_features = self._extract_engagement_features(analysis_data, post_id)
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features(post_id, user_id)
            
            return CommentAnalysisFeatures(
                **sentiment_features,
                **toxicity_features,
                **engagement_features,
                **temporal_features
            )
            
        except Exception as e:
            logger.error(f"Error getting comment analysis features for post {post_id}: {e}")
            return self._create_default_features()
    
    def _fetch_comment_analysis(self, post_id: int) -> Optional[Dict]:
        """Fetch comment analysis from the Spring API using the correct endpoint."""
        # Check cache first
        cache_key = f"post_{post_id}"
        if cache_key in self.comment_cache:
            cached_data, timestamp = self.comment_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Use the actual API endpoint for post sentiment analysis
            url = f"{self.api_base_url}/api/comments/sentiment/posts/{post_id}"
            
            # Use service authentication if available
            headers = {}
            auth_token = os.getenv('SERVICE_AUTH_TOKEN', '')
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
                headers['X-Service-Role'] = 'SERVICE'
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache the result
                self.comment_cache[cache_key] = (data, time.time())
                
                return data
            elif response.status_code == 404:
                # No comment analysis available yet for this post
                logger.debug(f"No comment analysis found for post {post_id}")
                return None
            else:
                logger.warning(f"API returned status {response.status_code} for post {post_id} comment analysis")
                
        except Exception as e:
            logger.warning(f"Error fetching comment analysis from API for post {post_id}: {e}")
        
        return None
    
    def fetch_batch_comment_analysis(self, post_ids: List[int]) -> Dict[int, Dict]:
        """
        Fetch comment analysis for multiple posts using the batch endpoint.
        This is more efficient for processing multiple posts at once.
        
        Args:
            post_ids: List of post IDs to analyze (max 100)
            
        Returns:
            Dictionary mapping post_id to analysis data
        """
        if not post_ids:
            return {}
        
        # Limit to API maximum
        if len(post_ids) > 100:
            logger.warning(f"Batch size {len(post_ids)} exceeds API limit of 100, truncating")
            post_ids = post_ids[:100]
        
        # Check cache for all posts
        uncached_posts = []
        cached_results = {}
        
        for post_id in post_ids:
            cache_key = f"post_{post_id}"
            if cache_key in self.comment_cache:
                cached_data, timestamp = self.comment_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    cached_results[post_id] = cached_data
                else:
                    uncached_posts.append(post_id)
            else:
                uncached_posts.append(post_id)
        
        # Fetch uncached posts using batch endpoint
        if uncached_posts:
            try:
                # Use the batch posts sentiment endpoint
                url = f"{self.api_base_url}/api/comments/sentiment/batch-posts"
                
                # Use service authentication
                headers = {'Content-Type': 'application/json'}
                auth_token = os.getenv('SERVICE_AUTH_TOKEN', '')
                if auth_token:
                    headers['Authorization'] = f'Bearer {auth_token}'
                    headers['X-Service-Role'] = 'SERVICE'
                
                payload = {"postIds": uncached_posts}
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    batch_data = response.json()
                    sentiment_data = batch_data.get('sentimentData', [])
                    
                    # Cache and organize results
                    for analysis in sentiment_data:
                        post_id = analysis.get('postId')
                        if post_id:
                            cache_key = f"post_{post_id}"
                            self.comment_cache[cache_key] = (analysis, time.time())
                            cached_results[post_id] = analysis
                    
                    logger.debug(f"Fetched batch comment analysis for {len(sentiment_data)} posts")
                else:
                    logger.warning(f"Batch API returned status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error fetching batch comment analysis: {e}")
        
        return cached_results
    
    def _extract_sentiment_features(self, analysis_data: Dict) -> Dict[str, float]:
        """Extract sentiment-related features from API response."""
        # API endpoint: GET /api/comments/sentiment/posts/{postId}
        # Response format matches what the comment analysis service returns
        
        return {
            'positive_sentiment_ratio': analysis_data.get('positiveScore', 0.33),
            'negative_sentiment_ratio': analysis_data.get('negativeScore', 0.33),
            'neutral_sentiment_ratio': analysis_data.get('neutralScore', 0.34),
            'average_sentiment_confidence': analysis_data.get('confidenceScore', 0.5)
        }
    
    def _extract_toxicity_features(self, analysis_data: Dict) -> Dict[str, Any]:
        """Extract toxicity-related features from API response."""
        # API response includes individual comment data if available
        individual_comments = analysis_data.get('individualComments', [])
        
        if individual_comments:
            toxicity_scores = [comment.get('toxicityScore', 0.0) for comment in individual_comments]
            hate_speech_scores = [comment.get('hateSpeechScore', 0.0) for comment in individual_comments]
            spam_scores = [comment.get('spamScore', 0.0) for comment in individual_comments]
            
            avg_toxicity = np.mean(toxicity_scores) if toxicity_scores else 0.0
            high_toxicity_count = sum(1 for score in toxicity_scores if score > 0.7)
            hate_speech_count = sum(1 for score in hate_speech_scores if score > 0.5)
            spam_count = sum(1 for score in spam_scores if score > 0.5)
        else:
            # Use aggregate data from API
            avg_toxicity = analysis_data.get('averageToxicity', 0.0)
            high_toxicity_count = 0
            hate_speech_count = 0
            spam_count = 0
        
        return {
            'average_toxicity_score': avg_toxicity,
            'high_toxicity_count': high_toxicity_count,
            'hate_speech_count': hate_speech_count,
            'spam_count': spam_count
        }
    
    def _extract_engagement_features(self, analysis_data: Dict, post_id: int) -> Dict[str, float]:
        """Extract engagement-related features from API response."""
        total_comments = analysis_data.get('totalComments', 0)
        
        # Calculate engagement score based on comment volume and sentiment
        positive_ratio = analysis_data.get('positiveScore', 0.33)
        negative_ratio = analysis_data.get('negativeScore', 0.33)
        
        # Higher positive sentiment = better engagement
        engagement_score = positive_ratio - (negative_ratio * 0.5)
        engagement_score = max(0.0, min(1.0, engagement_score + 0.5))  # Normalize to 0-1
        
        # Calculate quality score (low toxicity + high confidence + good sentiment balance)
        avg_toxicity = analysis_data.get('averageToxicity', 0.0)
        confidence = analysis_data.get('confidenceScore', 0.5)
        
        quality_score = (1.0 - avg_toxicity) * confidence * (positive_ratio + 0.5)
        quality_score = max(0.0, min(1.0, quality_score))
        
        return {
            'total_comments': total_comments,
            'comment_engagement_score': engagement_score,
            'comment_quality_score': quality_score
        }
    
    def _extract_temporal_features(self, post_id: int, user_id: int = None) -> Dict[str, float]:
        """Extract temporal comment patterns."""
        # For now, return defaults - in production, this would analyze comment timestamps
        # and track sentiment trends over time
        
        return {
            'recent_comment_trend': 0.0,  # Would track sentiment improvement/decline
            'comment_velocity': 0.0       # Would track comments per hour
        }
    
    def _create_default_features(self) -> CommentAnalysisFeatures:
        """Create default features when analysis is unavailable."""
        return CommentAnalysisFeatures(
            positive_sentiment_ratio=0.33,
            negative_sentiment_ratio=0.33,
            neutral_sentiment_ratio=0.34,
            average_sentiment_confidence=0.5,
            average_toxicity_score=0.0,
            high_toxicity_count=0,
            hate_speech_count=0,
            spam_count=0,
            total_comments=0,
            comment_engagement_score=0.5,
            comment_quality_score=0.5,
            recent_comment_trend=0.0,
            comment_velocity=0.0
        )
    
    def calculate_comment_based_reward_adjustment(self, post_id: int, 
                                                interaction_type: str,
                                                base_reward: float) -> float:
        """
        Calculate reward adjustment based on comment analysis.
        
        Args:
            post_id: Post ID
            interaction_type: Type of user interaction
            base_reward: Base reward from interaction
            
        Returns:
            Adjusted reward value
        """
        try:
            features = self.get_comment_analysis_features(post_id)
            
            # Positive adjustments for good comment environments
            if interaction_type in ['like', 'save', 'share']:
                # Reward more for engaging with high-quality content
                quality_bonus = features.comment_quality_score * 0.2
                engagement_bonus = features.comment_engagement_score * 0.1
                
                # Bonus for content with positive community sentiment
                if features.positive_sentiment_ratio > 0.6:
                    sentiment_bonus = 0.1
                else:
                    sentiment_bonus = 0.0
                
                total_bonus = quality_bonus + engagement_bonus + sentiment_bonus
                
            # Negative adjustments for toxic content interactions
            elif interaction_type in ['not_interested', 'skip']:
                # Less penalty if avoiding toxic content
                if features.average_toxicity_score > 0.5 or features.negative_sentiment_ratio > 0.6:
                    toxicity_adjustment = 0.2  # Reduce penalty
                else:
                    toxicity_adjustment = 0.0
                
                total_bonus = toxicity_adjustment
                
            else:
                total_bonus = 0.0
            
            # Apply adjustment
            adjusted_reward = base_reward + total_bonus
            
            # Clip to reasonable range
            return np.clip(adjusted_reward, -2.0, 2.0)
            
        except Exception as e:
            logger.error(f"Error calculating comment-based reward adjustment: {e}")
            return base_reward
    
    def get_comment_features_for_state(self, post_id: int, user_id: int = None) -> np.ndarray:
        """
        Get comment analysis features as vector for RL state representation.
        
        Args:
            post_id: Post ID
            user_id: Optional user ID
            
        Returns:
            Feature vector for RL state
        """
        features = self.get_comment_analysis_features(post_id, user_id)
        return features.to_vector()
    
    def update_user_comment_interaction(self, user_id: int, post_id: int, 
                                      interaction_type: str, comment_data: Dict = None):
        """
        Update user's comment interaction patterns.
        
        Args:
            user_id: User ID
            post_id: Post ID
            interaction_type: Type of interaction
            comment_data: Optional comment analysis data
        """
        try:
            # Get comment features for this post
            if comment_data:
                # Use provided data
                features = comment_data
            else:
                # Fetch from service
                analysis_data = self._fetch_comment_analysis(post_id)
                features = analysis_data or {}
            
            # Record interaction with comment context
            interaction_record = {
                'timestamp': time.time(),
                'post_id': post_id,
                'interaction_type': interaction_type,
                'positive_sentiment': features.get('positiveScore', 0.33),
                'negative_sentiment': features.get('negativeScore', 0.33),
                'toxicity_score': features.get('averageToxicity', 0.0),
                'comment_count': features.get('totalComments', 0)
            }
            
            self.user_comment_patterns[user_id].append(interaction_record)
            
        except Exception as e:
            logger.error(f"Error updating user comment interaction: {e}")
    
    def get_user_comment_preferences(self, user_id: int) -> Dict[str, float]:
        """
        Analyze user's preferences based on comment interaction history.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of preference scores
        """
        interactions = list(self.user_comment_patterns[user_id])
        
        if not interactions:
            return {
                'positive_content_preference': 0.5,
                'toxicity_tolerance': 0.5,
                'engagement_preference': 0.5
            }
        
        # Analyze positive interactions
        positive_interactions = [i for i in interactions if i['interaction_type'] in ['like', 'save', 'share']]
        negative_interactions = [i for i in interactions if i['interaction_type'] in ['not_interested', 'skip']]
        
        if positive_interactions:
            avg_positive_sentiment = np.mean([i['positive_sentiment'] for i in positive_interactions])
            avg_toxicity_in_liked = np.mean([i['toxicity_score'] for i in positive_interactions])
            avg_comments_in_liked = np.mean([i['comment_count'] for i in positive_interactions])
        else:
            avg_positive_sentiment = 0.5
            avg_toxicity_in_liked = 0.5
            avg_comments_in_liked = 0.0
        
        return {
            'positive_content_preference': avg_positive_sentiment,
            'toxicity_tolerance': avg_toxicity_in_liked,
            'engagement_preference': min(1.0, avg_comments_in_liked / 50.0)  # Normalize
        }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            'cached_analyses': len(self.comment_cache),
            'users_tracked': len(self.user_comment_patterns),
            'total_comment_interactions': sum(len(patterns) for patterns in self.user_comment_patterns.values()),
            'feature_dimension': self.feature_dim,
            'api_base_url': self.api_base_url,
            'data_source': 'Spring API (comment analysis stored after service updates)'
        }