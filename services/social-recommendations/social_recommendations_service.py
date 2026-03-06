#!/usr/bin/env python3
"""
Social Recommendations Service
Dedicated microservice for social-enhanced recommendations with following user vectors
Port: 8081
"""

import logging
import os
import time
import numpy as np
import requests
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, request, jsonify
from flask.cli import load_dotenv
import threading
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("social-recommendations-service")

# Import JWT utilities
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared'))
    from auth.JwtTokenUtil import extract_jwt_token, create_auth_headers, get_token_or_fallback
except ImportError:
    logger.warning("JWT utilities not available, falling back to environment tokens")
    extract_jwt_token = None
    create_auth_headers = None
    get_token_or_fallback = None

# Initialize Flask app
app = Flask(__name__)

class InteractionType(Enum):
    LIKE = "like"
    SAVE = "save"
    NOT_INTERESTED = "not_interested"
    COMMENT_POSITIVE = "comment_positive"
    COMMENT_NEGATIVE = "comment_negative"
    VIEW_TIME_HIGH = "view_time_high"
    VIEW_TIME_LOW = "view_time_low"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"

@dataclass
class SocialSignal:
    user_id: int
    post_id: int
    signal_type: InteractionType
    strength: float
    timestamp: float
    metadata: Optional[Dict] = None

@dataclass
class FollowingUserProfile:
    user_id: int
    following_user_id: int
    influence_weight: float
    preference_alignment: float
    interaction_frequency: float
    mutual_connections: int

@dataclass
class SocialRecommendationMetadata:
    social_boost_applied: bool
    following_influence_score: float
    community_influence_score: float
    sentiment_boost_factor: float
    social_signals_count: int
    following_users_involved: List[int]

class SocialRecommendationsService:
    """
    Enhanced social recommendations service with following user vectors integration
    """
    
    def __init__(self):
        """Initialize the social recommendations service"""
        load_dotenv()
        
        # Service configuration
        self.comment_service_url = os.environ.get('COMMENT_ANALYSIS_SERVICE_URL', 'http://localhost:8082')
        self.core_service_url = os.environ.get('CORE_RECOMMENDATIONS_SERVICE_URL', 'http://localhost:5000')
        self.api_base_url = os.environ.get('SPRING_API_URL', 'http://localhost:8080')
        
        # Social weights and parameters
        self.following_weight = float(os.environ.get('FOLLOWING_WEIGHT', '0.4'))
        self.community_weight = float(os.environ.get('COMMUNITY_WEIGHT', '0.3'))
        self.sentiment_weight = float(os.environ.get('SENTIMENT_WEIGHT', '0.3'))
        self.max_following_users = int(os.environ.get('MAX_FOLLOWING_USERS', '50'))
        
        # Caching
        self._social_cache = {}
        self._following_cache = {}
        self._cache_lock = threading.Lock()
        self.cache_ttl = int(os.environ.get('CACHE_TTL', '1800'))  # 30 minutes
        
        # Social interaction tracking
        self.interaction_weights = {
            InteractionType.LIKE: 0.8,
            InteractionType.SAVE: 1.0,
            InteractionType.COMMENT_POSITIVE: 0.6,
            InteractionType.VIEW_TIME_HIGH: 0.4,
            InteractionType.NOT_INTERESTED: -0.9,
            InteractionType.COMMENT_NEGATIVE: -0.5,
            InteractionType.VIEW_TIME_LOW: -0.2,
            InteractionType.FOLLOW: 1.2,
            InteractionType.UNFOLLOW: -1.0
        }
        
        # Store current JWT token for requests
        self.current_jwt_token = None
        
        logger.info(f"Initialized Social Recommendations Service on port 8081")
        logger.info(f"Following weight: {self.following_weight}, Community weight: {self.community_weight}")
    
    def _update_jwt_token(self, jwt_token: str = None):
        """Update JWT token for this request"""
        if jwt_token:
            self.current_jwt_token = jwt_token
        elif get_token_or_fallback and not self.current_jwt_token:
            # Try to get token from request or fallback
            fallback_token = get_token_or_fallback()
            if fallback_token:
                self.current_jwt_token = fallback_token
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API calls"""
        auth_token = self.current_jwt_token or os.environ.get('SERVICE_AUTH_TOKEN', '')
        headers = {}
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
            headers['X-Service-Role'] = 'SERVICE'
        return headers

    def get_social_recommendations(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """
        Get purely social-based recommendations using following user vectors and social signals
        
        Args:
            request_data: Request containing userId, limit, contentType, etc.
            
        Returns:
            Social recommendation results with metadata
        """
        start_time = time.time()
        
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            user_id = request_data.get("userId")
            if not user_id:
                return self._error_response("userId is required")
            
            content_type = request_data.get("contentType", "POSTS").upper()
            limit = min(request_data.get("limit", 20), 100)
            exclude_seen = request_data.get("excludeSeen", True)
            
            logger.info(f"Getting social recommendations for user {user_id}, type: {content_type}")
            
            # Get following users and their influence data
            following_profiles = self._get_following_user_profiles(user_id)
            if not following_profiles:
                return self._error_response("No following users found for social recommendations")
            
            # Get social recommendation candidates from following users
            social_candidates = self._get_social_candidates_from_following(
                user_id, following_profiles, content_type, limit * 3
            )
            
            if not social_candidates:
                return self._error_response("No social candidates found")
            
            # Get sentiment data for candidates
            candidate_post_ids = [c["postId"] for c in social_candidates]
            sentiment_data = self._get_sentiment_data_batch(candidate_post_ids)
            
            # Calculate social scores with following user vectors
            scored_candidates = self._calculate_social_scores_with_following(
                user_id, social_candidates, following_profiles, sentiment_data
            )
            
            # Sort and limit results
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            top_candidates = scored_candidates[:limit]
            
            # Prepare response
            result = {
                "postIds": [c["postId"] for c in top_candidates],
                "scores": [c["score"] for c in top_candidates],
                "socialMetadata": {
                    "followingUsersInvolved": len(following_profiles),
                    "socialCandidatesConsidered": len(social_candidates),
                    "avgFollowingInfluence": np.mean([p.influence_weight for p in following_profiles]),
                    "sentimentBoostApplied": True,
                    "processingTime": time.time() - start_time
                },
                "totalCount": len(top_candidates),
                "contentType": content_type
            }
            
            logger.info(f"Generated {len(top_candidates)} social recommendations for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting social recommendations: {e}", exc_info=True)
            return self._error_response(f"Error getting social recommendations: {str(e)}")

    def enhance_recommendations_with_social(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """
        Enhance existing recommendations with social signals and following user vectors
        
        Args:
            request_data: Contains userId, postIds, baseScores, socialWeight
            
        Returns:
            Enhanced recommendations with social boosting applied
        """
        start_time = time.time()
        
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            user_id = request_data.get("userId")
            post_ids = request_data.get("postIds", [])
            base_scores = request_data.get("baseScores", [])
            social_weight = request_data.get("socialWeight", 0.25)
            
            if not user_id or not post_ids:
                return self._error_response("userId and postIds are required")
            
            if len(base_scores) != len(post_ids):
                base_scores = [1.0] * len(post_ids)
            
            logger.info(f"Enhancing {len(post_ids)} recommendations with social signals for user {user_id}")
            
            # Get following user profiles
            following_profiles = self._get_following_user_profiles(user_id)
            
            # Get social boost factors for each post
            social_boosts = self._calculate_social_boost_factors(
                user_id, post_ids, following_profiles
            )
            
            # Get sentiment boosts
            sentiment_data = self._get_sentiment_data_batch(post_ids)
            sentiment_boosts = self._calculate_sentiment_boosts(post_ids, sentiment_data)
            
            # Apply social enhancement
            enhanced_scores = []
            social_metadata = {
                "socialBoosts": {},
                "sentimentBoosts": {},
                "followingInfluence": len(following_profiles),
                "socialWeight": social_weight
            }
            
            for i, (post_id, base_score) in enumerate(zip(post_ids, base_scores)):
                social_boost = social_boosts.get(post_id, 1.0)
                sentiment_boost = sentiment_boosts.get(post_id, 1.0)
                
                # Combine boosts with weights
                combined_boost = (
                    social_boost * self.following_weight +
                    sentiment_boost * self.sentiment_weight +
                    1.0 * (1.0 - self.following_weight - self.sentiment_weight)
                )
                
                # Apply social weight to determine final enhancement
                enhanced_score = (
                    base_score * (1.0 - social_weight) +
                    base_score * combined_boost * social_weight
                )
                
                enhanced_scores.append(enhanced_score)
                social_metadata["socialBoosts"][post_id] = social_boost
                social_metadata["sentimentBoosts"][post_id] = sentiment_boost
            
            result = {
                "enhancedScores": enhanced_scores,
                "socialMetadata": social_metadata,
                "processingTime": time.time() - start_time
            }
            
            logger.info(f"Enhanced recommendations with social signals for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing recommendations: {e}", exc_info=True)
            return self._error_response(f"Error enhancing recommendations: {str(e)}")

    def update_social_interaction(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """
        Update user social interaction for future recommendation enhancement
        
        Args:
            request_data: Contains userId, postId, interactionType, strength
            
        Returns:
            Success/failure response
        """
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            user_id = request_data.get("userId")
            post_id = request_data.get("postId")
            interaction_type = request_data.get("interactionType")
            strength = request_data.get("strength", 1.0)
            
            if not all([user_id, post_id, interaction_type]):
                return self._error_response("userId, postId, and interactionType are required")
            
            # Map string to enum
            try:
                interaction_enum = InteractionType(interaction_type.lower())
            except ValueError:
                return self._error_response(f"Invalid interaction type: {interaction_type}")
            
            # Create social signal
            signal = SocialSignal(
                user_id=user_id,
                post_id=post_id,
                signal_type=interaction_enum,
                strength=strength,
                timestamp=time.time()
            )
            
            # Store interaction (in production, this would go to a database)
            logger.info(f"Updated social interaction: user {user_id}, post {post_id}, type {interaction_type}")
            
            # Clear relevant caches
            self._invalidate_user_cache(user_id)
            
            return {"success": True, "message": "Social interaction updated successfully"}
            
        except Exception as e:
            logger.error(f"Error updating social interaction: {e}")
            return self._error_response(f"Error updating social interaction: {str(e)}")

    def get_following_influence_data(self, user_id: int) -> Dict[str, Any]:
        """
        Get following user influence data for a specific user
        
        Args:
            user_id: User ID
            
        Returns:
            Following influence data and statistics
        """
        try:
            following_profiles = self._get_following_user_profiles(user_id)
            
            if not following_profiles:
                return {"followingUsers": [], "totalInfluence": 0.0, "averageAlignment": 0.0}
            
            influence_data = []
            total_influence = 0.0
            total_alignment = 0.0
            
            for profile in following_profiles:
                influence_info = {
                    "followingUserId": profile.following_user_id,
                    "influenceWeight": profile.influence_weight,
                    "preferenceAlignment": profile.preference_alignment,
                    "interactionFrequency": profile.interaction_frequency,
                    "mutualConnections": profile.mutual_connections
                }
                influence_data.append(influence_info)
                total_influence += profile.influence_weight
                total_alignment += profile.preference_alignment
            
            result = {
                "followingUsers": influence_data,
                "totalInfluence": total_influence,
                "averageAlignment": total_alignment / len(following_profiles),
                "followingCount": len(following_profiles)
            }
            
            logger.info(f"Retrieved following influence data for user {user_id}: {len(following_profiles)} following users")
            return result
            
        except Exception as e:
            logger.error(f"Error getting following influence data: {e}")
            return {"error": str(e)}

    def _get_following_user_profiles(self, user_id: int) -> List[FollowingUserProfile]:
        """Get following user profiles with influence data and caching"""
        cache_key = f"following_{user_id}"
        
        with self._cache_lock:
            if cache_key in self._following_cache:
                cached_data, timestamp = self._following_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data
        
        try:
            url = f"{self.api_base_url}/api/social/users/{user_id}/following/influence"
            headers = self._get_auth_headers()
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                profiles = []
                
                for following_data in data.get("followingUsers", []):
                    profile = FollowingUserProfile(
                        user_id=user_id,
                        following_user_id=following_data.get("userId"),
                        influence_weight=following_data.get("influenceWeight", 0.5),
                        preference_alignment=following_data.get("preferenceAlignment", 0.5),
                        interaction_frequency=following_data.get("interactionFrequency", 0.0),
                        mutual_connections=following_data.get("mutualConnections", 0)
                    )
                    profiles.append(profile)
                
                # Limit to max following users for performance
                profiles = profiles[:self.max_following_users]
                
                # Cache the result
                with self._cache_lock:
                    self._following_cache[cache_key] = (profiles, time.time())
                
                logger.info(f"Retrieved {len(profiles)} following user profiles for user {user_id}")
                return profiles
            
        except Exception as e:
            logger.warning(f"Failed to get following user profiles for {user_id}: {e}")
        
        return []

    def _get_social_candidates_from_following(self, user_id: int, following_profiles: List[FollowingUserProfile],
                                            content_type: str, limit: int) -> List[Dict]:
        """Get recommendation candidates based on following users' interactions"""
        try:
            candidates = []
            
            for profile in following_profiles:
                # Get recent interactions from following user
                url = f"{self.api_base_url}/api/social/users/{profile.following_user_id}/recent-interactions"
                params = {
                    "contentType": content_type.upper(),
                    "limit": limit // len(following_profiles) + 5,
                    "interactionTypes": ["LIKE", "SAVE", "COMMENT_POSITIVE"]
                }
                
                headers = self._get_auth_headers()
                response = requests.get(url, params=params, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    interactions = response.json().get("interactions", [])
                    
                    for interaction in interactions:
                        candidate = {
                            "postId": interaction.get("postId"),
                            "sourceUserId": profile.following_user_id,
                            "interactionType": interaction.get("interactionType"),
                            "interactionStrength": interaction.get("strength", 1.0),
                            "followingInfluence": profile.influence_weight,
                            "preferenceAlignment": profile.preference_alignment,
                            "timestamp": interaction.get("timestamp", time.time())
                        }
                        candidates.append(candidate)
            
            # Remove duplicates and sort by influence
            seen_posts = set()
            unique_candidates = []
            
            for candidate in sorted(candidates, key=lambda x: x["followingInfluence"], reverse=True):
                if candidate["postId"] not in seen_posts:
                    seen_posts.add(candidate["postId"])
                    unique_candidates.append(candidate)
            
            return unique_candidates[:limit]
            
        except Exception as e:
            logger.error(f"Error getting social candidates from following: {e}")
            return []

    def _calculate_social_scores_with_following(self, user_id: int, candidates: List[Dict],
                                              following_profiles: List[FollowingUserProfile],
                                              sentiment_data: Dict[int, Dict]) -> List[Dict]:
        """Calculate social scores using following user vectors and interactions"""
        scored_candidates = []
        
        # Create following user lookup
        following_lookup = {p.following_user_id: p for p in following_profiles}
        
        for candidate in candidates:
            post_id = candidate["postId"]
            source_user_id = candidate["sourceUserId"]
            
            # Base score from interaction type and strength
            interaction_weight = self.interaction_weights.get(
                InteractionType(candidate["interactionType"].lower()), 0.5
            )
            base_score = interaction_weight * candidate["interactionStrength"]
            
            # Following user influence boost
            following_profile = following_lookup.get(source_user_id)
            if following_profile:
                following_boost = (
                    following_profile.influence_weight * 0.4 +
                    following_profile.preference_alignment * 0.6
                )
            else:
                following_boost = 0.5
            
            # Sentiment boost
            sentiment = sentiment_data.get(post_id, {})
            sentiment_boost = self._calculate_sentiment_boost_single(sentiment)
            
            # Community boost (simplified - based on multiple following users liking same content)
            community_boost = 1.0  # Would be calculated based on multiple following users
            
            # Calculate final score
            final_score = (
                base_score * 0.3 +
                following_boost * self.following_weight +
                sentiment_boost * self.sentiment_weight +
                community_boost * self.community_weight
            )
            
            scored_candidate = {
                "postId": post_id,
                "score": final_score,
                "sourceUserId": source_user_id,
                "followingInfluence": following_boost,
                "sentimentBoost": sentiment_boost,
                "communityBoost": community_boost,
                "baseInteractionScore": base_score
            }
            
            scored_candidates.append(scored_candidate)
        
        return scored_candidates

    def _calculate_social_boost_factors(self, user_id: int, post_ids: List[int],
                                      following_profiles: List[FollowingUserProfile]) -> Dict[int, float]:
        """Calculate social boost factors for posts based on following users"""
        boosts = {}
        
        for post_id in post_ids:
            boost_factor = 1.0
            
            # Check if any following users have interacted with this post
            for profile in following_profiles:
                interaction_boost = self._get_following_user_interaction_boost(
                    profile.following_user_id, post_id
                )
                
                if interaction_boost > 1.0:
                    # Weight by following user's influence
                    weighted_boost = 1.0 + (interaction_boost - 1.0) * profile.influence_weight
                    boost_factor = max(boost_factor, weighted_boost)
            
            # Clamp boost factor
            boosts[post_id] = min(max(boost_factor, 0.5), 2.0)
        
        return boosts

    def _get_following_user_interaction_boost(self, following_user_id: int, post_id: int) -> float:
        """Get interaction boost factor for a specific following user and post"""
        try:
            url = f"{self.api_base_url}/api/social/users/{following_user_id}/posts/{post_id}/interaction"
            headers = self._get_auth_headers()
            response = requests.get(url, headers=headers, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                interaction_type = data.get("interactionType")
                strength = data.get("strength", 1.0)
                
                if interaction_type:
                    base_weight = self.interaction_weights.get(
                        InteractionType(interaction_type.lower()), 0.0
                    )
                    return max(1.0 + base_weight * strength, 0.5)
            
        except Exception as e:
            logger.debug(f"No interaction found for user {following_user_id} and post {post_id}: {e}")
        
        return 1.0

    def _get_sentiment_data_batch(self, post_ids: List[int]) -> Dict[int, Dict]:
        """Get sentiment data for batch of posts from comment analysis service"""
        try:
            url = f"{self.comment_service_url}/api/comments/sentiment/batch-posts"
            payload = {"postIds": post_ids}
            headers = self._get_auth_headers()
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return {item["postId"]: item for item in data.get("sentimentData", [])}
            
        except Exception as e:
            logger.warning(f"Failed to get sentiment data: {e}")
        
        return {}

    def _calculate_sentiment_boosts(self, post_ids: List[int], sentiment_data: Dict[int, Dict]) -> Dict[int, float]:
        """Calculate sentiment-based boost factors"""
        boosts = {}
        
        for post_id in post_ids:
            sentiment = sentiment_data.get(post_id, {})
            boosts[post_id] = self._calculate_sentiment_boost_single(sentiment)
        
        return boosts

    def _calculate_sentiment_boost_single(self, sentiment: Dict) -> float:
        """Calculate sentiment boost for a single post"""
        if not sentiment:
            return 1.0
        
        sentiment_label = sentiment.get("overallSentiment", "NEUTRAL")
        confidence = sentiment.get("confidenceScore", 0.5)
        comment_count = sentiment.get("totalComments", 0)
        
        # Only apply sentiment boost if there are enough comments
        if comment_count < 3:
            return 1.0
        
        if sentiment_label == "POSITIVE" and confidence > 0.7:
            return 1.0 + (confidence * 0.2)  # Up to 20% boost
        elif sentiment_label == "NEGATIVE" and confidence > 0.7:
            return 1.0 - (confidence * 0.15)  # Up to 15% penalty
        
        return 1.0

    def _invalidate_user_cache(self, user_id: int):
        """Invalidate cache entries for a specific user"""
        with self._cache_lock:
            keys_to_remove = [key for key in self._social_cache.keys() 
                            if key.endswith(f"_{user_id}")]
            for key in keys_to_remove:
                del self._social_cache[key]
            
            # Also invalidate following cache
            following_key = f"following_{user_id}"
            if following_key in self._following_cache:
                del self._following_cache[following_key]

    def _error_response(self, message: str) -> Dict:
        """Create an error response"""
        logger.error(message)
        return {
            "error": message,
            "postIds": [],
            "scores": [],
            "totalCount": 0
        }

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        with self._cache_lock:
            return {
                "socialCacheSize": len(self._social_cache),
                "followingCacheSize": len(self._following_cache),
                "cacheTtl": self.cache_ttl,
                "followingWeight": self.following_weight,
                "communityWeight": self.community_weight,
                "sentimentWeight": self.sentiment_weight,
                "maxFollowingUsers": self.max_following_users
            }

# Create service instance
social_service = SocialRecommendationsService()

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "service": "social-recommendations",
            "version": "1.0.0",
            "stats": social_service.get_service_stats()
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/social/recommendations', methods=['POST'])
def get_social_recommendations():
    """Get purely social-based recommendations"""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400
        
        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()
        
        response = social_service.get_social_recommendations(request_data, jwt_token=jwt_token)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in social recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/social/enhance', methods=['POST'])
def enhance_with_social():
    """Enhance existing recommendations with social signals"""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400
        
        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()
        
        response = social_service.enhance_recommendations_with_social(request_data, jwt_token=jwt_token)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in social enhancement endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/social/interactions/update', methods=['POST'])
def update_social_interaction():
    """Update social interaction data"""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400
        
        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()
        
        response = social_service.update_social_interaction(request_data, jwt_token=jwt_token)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in social interaction update endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/social/following/<int:user_id>/influence', methods=['GET'])
def get_following_influence(user_id):
    """Get following user influence data"""
    try:
        response = social_service.get_following_influence_data(user_id)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in following influence endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/social/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    try:
        stats = social_service.get_service_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('SOCIAL_SERVICE_PORT', os.environ.get('PORT', 8081)))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Social Recommendations Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)