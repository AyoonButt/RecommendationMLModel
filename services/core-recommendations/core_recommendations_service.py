#!/usr/bin/env python3
"""
Core Recommendations Service
Pure ML recommendations using Two-Tower model with inter-service communication
Port: 5000
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import redis
import requests
from flask import Flask, request, jsonify
from flask.cli import load_dotenv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared/components'))

from TwoTower import TwoTowerModel, compute_scores
from MockRedis import MockRedis
# RL Integration: Replace MetadataEnhancer with RL-enhanced version
from MetadataEnhancer import MetadataEnhancer

# Add path for shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared'))
try:
    from auth.ServiceTokenManager import get_service_token_manager
    from auth.JwtTokenUtil import extract_jwt_token, create_auth_headers, get_token_or_fallback
except ImportError:
    # Fallback if auth modules not available
    get_service_token_manager = None
    extract_jwt_token = None
    create_auth_headers = None
    get_token_or_fallback = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("core-recommendations-service")

# Initialize Flask app
app = Flask(__name__)

def _error_response(message: str) -> Dict:
    """Create an error response"""
    logger.error(message)
    return {
        "error": message,
        "postIds": [],
        "totalCount": 0
    }

class ServiceClient:
    """HTTP client for inter-service communication"""
    
    def __init__(self, base_url: str, timeout: int = 10, jwt_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.jwt_token = jwt_token
        # Fallback to environment token if no JWT provided
        if not self.jwt_token:
            self.jwt_token = os.environ.get('SERVICE_AUTH_TOKEN', '')
        
        self.headers = {}
        if self.jwt_token:
            self.headers['Authorization'] = f'Bearer {self.jwt_token}'
            self.headers['X-Service-Role'] = 'SERVICE'
    
    def update_token(self, jwt_token: str):
        """Update the JWT token for this client"""
        self.jwt_token = jwt_token
        if self.jwt_token:
            self.headers['Authorization'] = f'Bearer {self.jwt_token}'
            self.headers['X-Service-Role'] = 'SERVICE'
        else:
            self.headers.pop('Authorization', None)
            self.headers.pop('X-Service-Role', None)
    
    def post(self, endpoint: str, data: Dict) -> Optional[Dict]:
        """Make POST request to service"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, headers=self.headers, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Service request failed: {url}, status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Service communication error: {e}")
        
        return None
    
    def get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make GET request to service"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            # Log request details
            logger.info(f"=== API REQUEST ===")
            logger.info(f"Method: GET")
            logger.info(f"URL: {url}")
            logger.info(f"Params: {params}")
            logger.info(f"Headers: {self.headers}")
            
            response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)
            
            # Log response details
            logger.info(f"=== API RESPONSE ===")
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"Response Body: {response_data}")
                return response_data
            else:
                # Log error response body for debugging
                try:
                    error_body = response.text
                    logger.warning(f"Service request failed: {url}, status: {response.status_code}")
                    logger.warning(f"Error response body: {error_body}")
                except:
                    logger.warning(f"Service request failed: {url}, status: {response.status_code} (could not read response body)")
                
        except Exception as e:
            logger.error(f"Service communication error: {e}")
        
        return None

class CoreRecommendationsService:
    """Core ML recommendation service without social dependencies"""
    
    def __init__(self):
        """Initialize the core recommendations service"""
        self.cursor_tracker = {}
        load_dotenv()

        # Service URLs
        self.api_base_url = os.environ.get('SPRING_API_URL', 'http://10.234.49.210:8080')
        self.social_service_url = os.environ.get('SOCIAL_SERVICE_URL', 'http://127.0.0.1:8081')
        self.comment_service_url = os.environ.get('COMMENT_SERVICE_URL', 'http://10.234.49.210:8080')
        
        # Initialize service clients (will be updated with JWT tokens per request)
        self.social_client = ServiceClient(self.social_service_url)
        self.comment_client = ServiceClient(self.comment_service_url)

        # Redis configuration
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        redis_password = os.environ.get('REDIS_PASSWORD', '')
        redis_ssl = os.environ.get('REDIS_SSL', 'False').lower() == 'true'
        redis_timeout = int(os.environ.get('REDIS_TIMEOUT', 10))

        # Connect to Redis/Valkey
        logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
        is_local_dev = os.environ.get('LOCAL_DEV', 'True').lower() == 'true'

        if is_local_dev:
            logger.info("Using MockRedis for local development")
            self.redis_client = MockRedis(decode_responses=True)
        else:
            logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
            try:
                connection_args = {
                    'host': redis_host,
                    'port': redis_port,
                    'decode_responses': True,
                    'ssl': redis_ssl,
                    'socket_timeout': redis_timeout,
                    'socket_connect_timeout': redis_timeout,
                    'socket_keepalive': True,
                    'health_check_interval': 30,
                    'retry_on_timeout': True
                }

                if redis_password:
                    connection_args['password'] = redis_password

                self.redis_client = redis.Redis(**connection_args)
                self.redis_client.ping()
                self.redis_client.client_setname("core-recommendation-service")
                logger.info("Successfully connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                logger.info("Falling back to MockRedis")
                self.redis_client = MockRedis(decode_responses=True)

        # Cache keys for vectors
        self.user_vector_key_prefix = "user_vector:"
        self.post_vector_key_prefix = "post_vector:"
        self.vector_cache_ttl = 3600  # 1 hour

        # Initialize service token manager
        self.token_manager = None
        if get_service_token_manager:
            try:
                self.token_manager = get_service_token_manager("core-recommendations")
                # Try to get token from API
                if self.token_manager.request_service_token(self.api_base_url):
                    logger.info("Successfully obtained service token from API")
                    # Update current JWT token with the new one from API
                    self.current_jwt_token = self.token_manager.get_access_token()
                else:
                    logger.warning("Could not obtain service token from API, falling back to environment variable")
                    # Fall back to environment variable
                    env_token = os.environ.get('SERVICE_AUTH_TOKEN')
                    if env_token:
                        self.current_jwt_token = env_token
                        logger.info("Using service token from environment variable")
                    else:
                        logger.error("No service token available from API or environment")
                logger.info("Service token manager initialized")
            except Exception as e:
                logger.warning(f"Could not initialize service token manager: {e}")
                # Fall back to environment variable
                env_token = os.environ.get('SERVICE_AUTH_TOKEN')
                if env_token:
                    self.current_jwt_token = env_token
                    logger.info("Using service token from environment variable as fallback")

        # Initialize RL-enhanced metadata enhancer
        self.metadata_enhancer = MetadataEnhancer(self.api_base_url, self.redis_client)
        logger.info("RL-Enhanced MetadataEnhancer initialized")
        
        # Store the current JWT token for this request
        # If we have a token from the token manager, update all clients
        if hasattr(self, 'current_jwt_token') and self.current_jwt_token:
            self.social_client.update_token(self.current_jwt_token)
            self.comment_client.update_token(self.current_jwt_token)
            logger.info("Updated all service clients with token from API")

        # Initialize Two-Tower model
        self.two_tower_model = None
        self._load_model()
        
        logger.info(f"Initialized Core Recommendations Service on port 5000")
    
    def _update_jwt_token(self, jwt_token: str = None):
        """Update JWT token for all service clients and metadata enhancer"""
        if jwt_token:
            self.current_jwt_token = jwt_token
            self.social_client.update_token(jwt_token)
            self.comment_client.update_token(jwt_token)
        elif get_token_or_fallback and not self.current_jwt_token:
            # Try to get token from request or fallback
            fallback_token = get_token_or_fallback()
            if fallback_token:
                self.current_jwt_token = fallback_token
                self.social_client.update_token(fallback_token)
                self.comment_client.update_token(fallback_token)

    def _load_model(self):
        """Load the TwoTower model"""
        try:
            logger.info("Loading TwoTower model...")

            model_dir = os.environ.get('MODEL_DIR', './model_checkpoints')
            user_model_path = os.path.join(model_dir, os.environ.get('USER_MODEL', 'user_tower_latest.h5'))
            post_model_path = os.path.join(model_dir, os.environ.get('POST_MODEL', 'post_tower_latest.h5'))

            self.two_tower_model = TwoTowerModel(
                user_feature_dim=int(os.environ.get('USER_FEATURE_DIM', '64')),
                post_feature_dim=int(os.environ.get('POST_FEATURE_DIM', '64')),
                embedding_dim=int(os.environ.get('EMBEDDING_DIM', '32')),
                hidden_dims=[int(dim) for dim in os.environ.get('HIDDEN_DIMS', '128,64').split(',')]
            )

            if os.path.exists(user_model_path) and os.path.exists(post_model_path):
                self.two_tower_model.load_models(user_model_path, post_model_path)
                logger.info(f"Loaded pre-trained models from {user_model_path} and {post_model_path}")
            else:
                logger.warning("Pre-trained models not found, using initialized model")

            logger.info("TwoTower model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.two_tower_model = None

    def get_recommendations(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """
        Process a recommendation request and return recommended posts
        
        Args:
            request_data: Dictionary containing the recommendation request parameters
                - userId: User ID (required)
                - contentType: Type of content ("posts" or "trailers")
                - limit: Number of recommendations (default: 20)
                - enableSocial: Whether to apply social enhancement (default: False)
                - socialWeight: Weight for social signals (default: 0.25)
                
        Returns:
            Dictionary containing recommended posts with scores
        """
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            # Extract essential request parameters
            user_id = request_data.get("userId")
            if not user_id:
                return _error_response("userId is required")

            content_type = request_data.get("contentType", "posts")
            limit = request_data.get("limit", 20)
            enable_social = request_data.get("enableSocial", False)
            social_weight = request_data.get("socialWeight", 0.25)

            logger.info(f"Processing recommendation request for user {user_id}, "
                       f"content type: {content_type}, social: {enable_social}")

            # Get base recommendations
            if content_type == "trailers":
                result = self.get_trailer_recommendations(user_id=user_id, limit=limit, jwt_token=jwt_token)
            else:
                result = self.get_post_recommendations(user_id=user_id, limit=limit, jwt_token=jwt_token)

            # Apply social enhancement if requested
            if enable_social and "error" not in result:
                enhanced_result = self._apply_social_enhancement(
                    user_id, result, social_weight
                )
                if enhanced_result:
                    result.update(enhanced_result)

            # Add metadata to response
            result["contentType"] = content_type
            result["socialEnhancement"] = enable_social
            return result

        except Exception as e:
            logger.error(f"Error processing recommendation request: {str(e)}", exc_info=True)
            return _error_response(f"Error processing recommendation request: {str(e)}")

    def get_post_recommendations(self, user_id: str, limit: int = 20, jwt_token: str = None) -> Dict[str, Any]:
        """Get post recommendations using the Two-Tower model"""
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            start_time = time.time()

            # Get user vector
            user_vector = self._get_user_vector(user_id)
            if user_vector is None:
                return _error_response(f"Could not retrieve vector for user {user_id}")

            # Get cursors for this user and content type
            user_cursors = self.cursor_tracker.get(user_id, {}).get("posts", {})
            cursor = user_cursors.get("cursor")
            high_quality_cursor = user_cursors.get("highQualityCursor")

            # Fetch candidates
            candidate_data = self._fetch_candidate_vectors(
                user_id=user_id,
                limit=limit * 2,  # Get more candidates for better selection
                content_type="POSTS",
                cursor=cursor,
                high_quality_cursor=high_quality_cursor
            )

            if not candidate_data["vectors"]:
                logger.warning(f"No candidate vectors found for user {user_id}")
                return {
                    "postIds": [],
                    "scores": [],
                    "totalCount": 0,
                    "processingTime": time.time() - start_time,
                    "message": "No candidates found"
                }

            # Score candidates using Two-Tower model with metadata enhancement
            scored_posts = self._score_candidates(
                user_id=user_id,
                user_vector=user_vector,
                candidate_data=candidate_data,
                content_type="posts"
            )

            # Take top results
            top_results = scored_posts[:limit]
            final_post_ids = [int(post_id) for post_id, _ in top_results]
            final_scores = [float(score) for _, score in top_results]

            result = {
                "postIds": final_post_ids,
                "scores": final_scores,
                "totalCount": len(final_post_ids),
                "processingTime": time.time() - start_time,
                "hasMore": candidate_data["hasMore"],
                "nextCursor": candidate_data["nextCursor"],
                "nextHighQualityCursor": candidate_data["nextHighQualityCursor"],
                "distribution": candidate_data["distribution"]
            }

            logger.info(f"Generated {len(final_post_ids)} post recommendations for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error getting post recommendations: {str(e)}", exc_info=True)
            return _error_response(f"Error getting post recommendations: {str(e)}")

    def get_trailer_recommendations(self, user_id: str, limit: int = 20, jwt_token: str = None) -> Dict[str, Any]:
        """Get trailer recommendations using the Two-Tower model"""
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            start_time = time.time()

            # Get user vector
            user_vector = self._get_user_vector(user_id)
            if user_vector is None:
                return _error_response(f"Could not retrieve vector for user {user_id}")

            # Get cursors for this user and content type
            user_cursors = self.cursor_tracker.get(user_id, {}).get("trailers", {})
            cursor = user_cursors.get("cursor")
            high_quality_cursor = user_cursors.get("highQualityCursor")

            # Fetch candidates
            candidate_data = self._fetch_candidate_vectors(
                user_id=user_id,
                limit=limit * 2,
                content_type="TRAILERS",
                cursor=cursor,
                high_quality_cursor=high_quality_cursor
            )

            if not candidate_data["vectors"]:
                logger.warning(f"No trailer candidates found for user {user_id}")
                return {
                    "postIds": [],
                    "scores": [],
                    "totalCount": 0,
                    "processingTime": time.time() - start_time,
                    "message": "No trailer candidates found"
                }

            # Score candidates using Two-Tower model with metadata enhancement
            scored_posts = self._score_candidates(
                user_id=user_id,
                user_vector=user_vector,
                candidate_data=candidate_data,
                content_type="trailers"
            )

            # Take top results
            top_results = scored_posts[:limit]
            final_post_ids = [int(post_id) for post_id, _ in top_results]
            final_scores = [float(score) for _, score in top_results]

            result = {
                "postIds": final_post_ids,
                "scores": final_scores,
                "totalCount": len(final_post_ids),
                "processingTime": time.time() - start_time,
                "hasMore": candidate_data["hasMore"],
                "nextCursor": candidate_data["nextCursor"],
                "nextHighQualityCursor": candidate_data["nextHighQualityCursor"],
                "distribution": candidate_data["distribution"]
            }

            logger.info(f"Generated {len(final_post_ids)} trailer recommendations for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error getting trailer recommendations: {str(e)}", exc_info=True)
            return _error_response(f"Error getting trailer recommendations: {str(e)}")

    def _fetch_candidate_vectors(self, user_id: str, limit: int, content_type: str = "POSTS",
                                 cursor: str = None, high_quality_cursor: str = None) -> Dict[str, Any]:
        """Fetch candidate vectors from the Spring API endpoint"""
        try:
            logger.info(f"Fetching candidates for user {user_id} with content_type: {content_type}")

            url = f"{self.api_base_url}/api/internal/ml/users/{user_id}/candidates"
            params = {
                "limit": limit,
                "contentType": content_type.upper(),
                "includeNewHighQuality": True,
                "newContentRatio": 0.3,
                "interactionLookbackDays": 30
            }

            # Add cursors if provided
            if cursor:
                params["cursor"] = cursor
            if high_quality_cursor:
                params["highQualityCursor"] = high_quality_cursor

            headers = {}
            # Use current JWT token or fallback to environment token
            auth_token = self.current_jwt_token or os.environ.get('SERVICE_AUTH_TOKEN', '')
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
                headers['X-Service-Role'] = 'SERVICE'
            
            logger.info(f"=== CANDIDATE API REQUEST ===")
            logger.info(f"URL: {url}")
            logger.info(f"Params: {params}")
            logger.info(f"Headers: {dict((k, 'Bearer ***' if k == 'Authorization' else v) for k, v in headers.items())}")
            logger.info(f"Timeout: 15s")
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            logger.info(f"=== CANDIDATE API RESPONSE ===")
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response Headers: {dict(response.headers)}")
            logger.info(f"Response Size: {len(response.content)} bytes")

            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"=== SUCCESSFUL RESPONSE DATA ===")
                logger.info(f"Response structure keys: {list(response_data.keys())}")

                # Extract vectors and convert to proper format
                vectors = response_data.get("vectors", {})
                logger.info(f"Raw vectors count: {len(vectors)}")
                converted_vectors = {}
                conversion_errors = 0
                
                for post_id_str, vector_list in vectors.items():
                    try:
                        post_id = int(post_id_str)
                        vector_array = np.array(vector_list, dtype=np.float32)
                        converted_vectors[post_id] = vector_array
                    except (ValueError, TypeError) as e:
                        conversion_errors += 1
                        logger.warning(f"Error converting vector for post {post_id_str}: {e}")
                        continue

                if conversion_errors > 0:
                    logger.warning(f"Total vector conversion errors: {conversion_errors}")

                # Extract candidate result information
                candidate_result = response_data.get("candidateResult", {})
                logger.info(f"Candidate result keys: {list(candidate_result.keys())}")

                result = {
                    "vectors": converted_vectors,
                    "nextCursor": candidate_result.get("nextCursor"),
                    "nextHighQualityCursor": candidate_result.get("nextHighQualityCursor"),
                    "hasMore": candidate_result.get("hasMore", False),
                    "hasMoreHighQuality": candidate_result.get("hasMoreHighQuality", False),
                    "distribution": candidate_result.get("distribution", {}),
                    "candidates": candidate_result.get("candidates", [])
                }

                # Update cursor tracker
                if user_id not in self.cursor_tracker:
                    self.cursor_tracker[user_id] = {}

                self.cursor_tracker[user_id][content_type.lower()] = {
                    "cursor": result["nextCursor"],
                    "highQualityCursor": result["nextHighQualityCursor"]
                }

                logger.info(f"Successfully fetched {len(converted_vectors)} candidates for user {user_id}")
                logger.info(f"Cursor info - Next: {result['nextCursor']}, HighQuality: {result['nextHighQualityCursor']}")
                logger.info(f"Pagination - HasMore: {result['hasMore']}, HasMoreHighQuality: {result['hasMoreHighQuality']}")
                return result

            elif response.status_code == 204:  # No content
                logger.info(f"No candidates available for user {user_id} (204 No Content)")
                return {"vectors": {}, "nextCursor": None, "hasMore": False}
            else:
                logger.warning(f"API returned status {response.status_code} for candidate vectors")
                logger.warning(f"Error response content: {response.text}")
                if response.status_code == 500:
                    logger.error(f"Internal server error accessing candidates for user {user_id}")
                    logger.error(f"Request details - URL: {url}, Params: {params}")
                return {"vectors": {}, "nextCursor": None, "hasMore": False}

        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error fetching candidate vectors for user {user_id}: {str(e)}")
            logger.error(f"Request timed out after 15 seconds - URL: {url}")
            return {"vectors": {}, "nextCursor": None, "hasMore": False}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching candidate vectors for user {user_id}: {str(e)}")
            logger.error(f"Failed to connect to API endpoint: {url}")
            return {"vectors": {}, "nextCursor": None, "hasMore": False}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching candidate vectors for user {user_id}: {str(e)}")
            logger.error(f"Request details - URL: {url}, Params: {params}")
            return {"vectors": {}, "nextCursor": None, "hasMore": False}
        except Exception as e:
            logger.error(f"Unexpected error fetching candidate vectors for user {user_id}: {str(e)}", exc_info=True)
            logger.error(f"Request context - URL: {url}, Content Type: {content_type}")
            return {"vectors": {}, "nextCursor": None, "hasMore": False}

    def _get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Get user vector from Redis or API"""
        # Try to get from Redis first
        redis_key = f"{self.user_vector_key_prefix}{user_id}"
        cached_vector = self.redis_client.get(redis_key)

        if cached_vector:
            vector = np.frombuffer(cached_vector, dtype=np.float32)
            return vector

        # Get from API if not in Redis
        try:
            url = f"{self.api_base_url}/api/internal/ml/users/{user_id}/vector"
            headers = {}
            # Use current JWT token or fallback to environment token
            auth_token = self.current_jwt_token or os.environ.get('SERVICE_AUTH_TOKEN', '')
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
                headers['X-Service-Role'] = 'SERVICE'
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                vector_data = response.json()
                vector = np.array(vector_data, dtype=np.float32)

                # Cache in Redis
                self.redis_client.setex(redis_key, self.vector_cache_ttl, vector.tobytes())
                return vector

        except Exception as e:
            logger.warning(f"Error retrieving user vector from API: {e}")

        # Return default vector if can't retrieve
        logger.info(f"Using default vector for user {user_id}")
        feature_dim = int(os.environ.get('USER_FEATURE_DIM', '64'))
        default_vector = np.random.rand(feature_dim).astype(np.float32)

        # Cache default vector
        self.redis_client.setex(redis_key, self.vector_cache_ttl, default_vector.tobytes())
        return default_vector

    def _score_candidates(self, user_id: str, user_vector: np.ndarray,
                          candidate_data: Dict[str, Any], content_type: str) -> List[tuple]:
        """Score candidate posts using the Two-Tower model with metadata enhancement"""
        if not self.two_tower_model:
            logger.warning("TwoTower model not available, using fallback scoring")
            post_ids = list(candidate_data["vectors"].keys())
            return [(post_id, np.random.random()) for post_id in post_ids]

        try:
            vectors = candidate_data["vectors"]
            candidates = candidate_data.get("candidates", [])

            if not vectors:
                return []

            # Prepare data for scoring
            post_ids = list(vectors.keys())
            post_vectors = list(vectors.values())
            post_vectors_array = np.array(post_vectors)

            # Reshape user vector for batch processing
            user_batch = np.expand_dims(user_vector, axis=0)

            # Calculate base scores using Two-Tower model
            base_scores = compute_scores(user_batch, post_vectors_array, content_type)

            # Apply metadata enhancement
            enhanced_scores = self.metadata_enhancer.enhance_scores(
                user_id=user_id,
                post_ids=post_ids,
                base_scores=base_scores[0],
                candidates=candidates,
                content_type=content_type
            )

            # Combine post IDs with enhanced scores
            scored_posts = list(zip(post_ids, enhanced_scores))

            # Sort by score (descending)
            scored_posts.sort(key=lambda x: x[1], reverse=True)

            return scored_posts

        except Exception as e:
            logger.error(f"Error scoring candidates: {e}", exc_info=True)
            # Fallback to random scores
            post_ids = list(candidate_data["vectors"].keys())
            return [(post_id, np.random.random()) for post_id in post_ids]

    def _apply_social_enhancement(self, user_id: str, base_result: Dict[str, Any], 
                                social_weight: float) -> Optional[Dict[str, Any]]:
        """Apply social enhancement to base recommendations via social service"""
        try:
            if "error" in base_result or not base_result.get("postIds"):
                return None

            # Prepare request for social service
            social_request = {
                "userId": user_id,
                "postIds": base_result["postIds"],
                "baseScores": base_result["scores"],
                "socialWeight": social_weight
            }

            # Call social service for enhancement
            enhanced_data = self.social_client.post("/social/enhance", social_request)
            
            if enhanced_data and "error" not in enhanced_data:
                logger.info(f"Applied social enhancement for user {user_id}")
                return {
                    "scores": enhanced_data.get("enhancedScores", base_result["scores"]),
                    "socialMetadata": enhanced_data.get("socialMetadata", {}),
                    "socialEnhancementApplied": True
                }
            else:
                logger.warning(f"Social enhancement failed for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error applying social enhancement: {e}")

        return {"socialEnhancementApplied": False}

    def get_social_recommendations(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """Get purely social recommendations via social service"""
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            user_id = request_data.get("userId")
            if not user_id:
                return _error_response("userId is required")

            logger.info(f"Getting social recommendations for user {user_id}")

            # Forward request to social service
            social_data = self.social_client.post("/social/recommendations", request_data)
            
            if social_data and "error" not in social_data:
                social_data["source"] = "social_service"
                return social_data
            else:
                return _error_response("Social recommendations service unavailable")

        except Exception as e:
            logger.error(f"Error getting social recommendations: {e}", exc_info=True)
            return _error_response(f"Error getting social recommendations: {str(e)}")

    def process_user_interaction(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """
        Process user interaction feedback for RL learning.
        
        Args:
            request_data: Dictionary containing:
                - userId: User ID (required)
                - postId: Post ID that was interacted with (required)
                - interactionType: Type of interaction (required)
                - additionalContext: Optional additional context
                
        Returns:
            Success/error response
        """
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            user_id = request_data.get("userId")
            post_id = request_data.get("postId")
            interaction_type = request_data.get("interactionType")
            
            if not all([user_id, post_id, interaction_type]):
                return {"error": "userId, postId, and interactionType are required"}
            
            # Process through RL-enhanced metadata enhancer
            additional_context = request_data.get("additionalContext", {})
            additional_context.update({
                'timestamp': time.time(),
                'service': 'core-recommendations'
            })
            
            self.metadata_enhancer.process_user_interaction(
                user_id=str(user_id),
                post_id=int(post_id),
                interaction_type=interaction_type,
                additional_context=additional_context
            )
            
            logger.debug(f"Processed RL interaction: user {user_id}, post {post_id}, type {interaction_type}")
            
            return {
                "success": True,
                "message": f"Processed {interaction_type} interaction for user {user_id}"
            }
            
        except Exception as e:
            logger.error(f"Error processing user interaction: {e}", exc_info=True)
            return {"error": f"Error processing interaction: {str(e)}"}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics including RL stats"""
        try:
            base_stats = {
                "modelLoaded": self.two_tower_model is not None,
                "cacheStatus": {
                    "redisConnected": self.redis_client.ping()
                },
                "serviceConnections": {
                    "socialService": self.social_service_url,
                    "commentService": self.comment_service_url
                },
                "cursorsTracked": len(self.cursor_tracker),
                "version": os.environ.get("SERVICE_VERSION", "1.0.0")
            }
            
            # Add RL statistics
            try:
                rl_stats = self.metadata_enhancer.get_stats()
                base_stats["rl_enhancement"] = rl_stats.get("rl_enhancement", {})
                base_stats["metadata_enhancement"] = {
                    "cache_size": rl_stats.get("cache_size", 0),
                    "boost_factors": rl_stats.get("boost_factors", {})
                }
            except Exception as e:
                logger.warning(f"Error getting RL stats: {e}")
                base_stats["rl_enhancement"] = {"error": str(e)}
            
            return base_stats
            
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {"error": str(e)}

# Create service instance
core_service = CoreRecommendationsService()

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "service": "core-recommendations",
            "version": "1.0.0",
            "stats": core_service.get_service_stats()
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint for getting recommendations"""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400

        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()
        
        response = core_service.get_recommendations(request_data, jwt_token=jwt_token)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "postIds": [], "totalCount": 0}), 500

@app.route('/recommendations/posts', methods=['POST'])
def get_post_recommendations():
    """API endpoint for getting post recommendations"""
    try:
        request_data = request.json or {}
        user_id = request_data.get("userId")
        limit = request_data.get("limit", 20)
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()
        
        response = core_service.get_post_recommendations(user_id, limit, jwt_token=jwt_token)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in post recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "postIds": [], "totalCount": 0}), 500

@app.route('/recommendations/trailers', methods=['POST'])
def get_trailer_recommendations():
    """API endpoint for getting trailer recommendations"""
    try:
        request_data = request.json or {}
        user_id = request_data.get("userId")
        limit = request_data.get("limit", 20)
        
        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()
        
        response = core_service.get_trailer_recommendations(user_id, limit, jwt_token=jwt_token)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in trailer recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "postIds": [], "totalCount": 0}), 500

@app.route('/recommendations/social', methods=['POST'])
def get_social_recommendations():
    """API endpoint for getting social recommendations (proxied to social service)"""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400

        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()
        
        response = core_service.get_social_recommendations(request_data, jwt_token=jwt_token)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in social recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "postIds": [], "totalCount": 0}), 500

@app.route('/interactions', methods=['POST'])
def process_interaction():
    """API endpoint for processing user interactions (RL learning)"""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400

        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()
        
        response = core_service.process_user_interaction(request_data, jwt_token=jwt_token)
        
        if "error" in response:
            return jsonify(response), 400
        else:
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error in interaction endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """API endpoint for getting service statistics"""
    try:
        stats = core_service.get_service_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information"""
    return jsonify({
        "service": "Core Recommendations Service (RL-Enhanced)",
        "version": "1.1.0",
        "description": "ML recommendations with RL-enhanced metadata boosting",
        "endpoints": {
            "health": "/health",
            "recommendations": "/recommendations",
            "post_recommendations": "/recommendations/posts",
            "trailer_recommendations": "/recommendations/trailers",
            "social_recommendations": "/recommendations/social",
            "interactions": "/interactions",
            "stats": "/stats"
        },
        "new_features": {
            "rl_enhancement": "Adaptive boost factors based on user interactions",
            "interaction_processing": "Real-time learning from user feedback"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('CORE_SERVICE_PORT', os.environ.get('PORT', 5000)))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting Core Recommendations Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)