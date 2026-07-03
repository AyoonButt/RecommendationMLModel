#!/usr/bin/env python3
"""
Core Recommendations Service
Pure ML recommendations using Two-Tower model with inter-service communication
Port: 5000
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from CandidatePoolCache import CandidatePoolCache

# RL Integration: Import RL-enhanced version
try:
    from RLEnhancedMetadataEnhancer import RLEnhancedMetadataEnhancer, create_rl_enhanced_metadata_enhancer
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Fallback to basic MetadataEnhancer if RL not available
from MetadataEnhancer import MetadataEnhancer

# Recommendation Explainer - separate module for generating explanations
from RecommendationExplainer import RecommendationExplainer, create_explainer

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

# Shared executor for parallel I/O (user vector + candidate fetch, mark_shown fire-and-forget)
_executor = ThreadPoolExecutor(max_workers=8)

# Initialize Flask app
app = Flask(__name__)

def _error_response(message: str) -> Dict:
    """Create an error response"""
    logger.error(message)
    return {
        "error": message,
        "postIds": [],
        "scores": [],
        "explanations": [],
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
        url = f"{self.base_url}{endpoint}"
        try:
            logger.debug(f"=== SERVICE POST REQUEST ===")
            logger.debug(f"URL: {url}")
            logger.debug(f"Payload keys: {list(data.keys()) if data else 'None'}")
            logger.debug(f"Timeout: {self.timeout}s")

            response = requests.post(url, json=data, headers=self.headers, timeout=self.timeout)

            if response.status_code == 200:
                logger.debug(f"Service POST successful: {url}")
                return response.json()
            else:
                logger.warning(f"=== SERVICE POST FAILED ===")
                logger.warning(f"URL: {url}")
                logger.warning(f"Status code: {response.status_code}")
                try:
                    error_body = response.text[:500]  # Limit error body length
                    logger.warning(f"Response body: {error_body}")
                except:
                    logger.warning("Could not read response body")

                if response.status_code == 401 or response.status_code == 403:
                    logger.error(f"Authentication/Authorization failed for service POST to {endpoint}")
                elif response.status_code == 404:
                    logger.warning(f"Endpoint not found: {url}")
                elif response.status_code >= 500:
                    logger.error(f"Server error from service at {url}")

        except requests.exceptions.Timeout as e:
            logger.error(f"=== SERVICE POST TIMEOUT ===")
            logger.error(f"Request to {url} timed out after {self.timeout}s")
            logger.error(f"Timeout details: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"=== SERVICE POST CONNECTION ERROR ===")
            logger.error(f"Failed to connect to service at {url}")
            logger.error(f"Connection error details: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"=== SERVICE POST REQUEST ERROR ===")
            logger.error(f"Request to {url} failed")
            logger.error(f"Request error details: {str(e)}")
        except Exception as e:
            logger.error(f"=== SERVICE POST UNEXPECTED ERROR ===")
            logger.error(f"Unexpected error during POST to {url}: {e}", exc_info=True)

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
        self._cursor_lock = threading.RLock()
        self.current_jwt_token = None
        load_dotenv()

        # Service URLs
        self.api_base_url = os.environ.get('SPRING_API_URL', 'http://10.234.49.210:8080')
        self.social_service_url = os.environ.get('SOCIAL_SERVICE_URL', 'http://127.0.0.1:8081')
        self.comment_service_url = os.environ.get('COMMENT_SERVICE_URL', 'http://10.234.49.210:8080')
        
        # Initialize service clients (will be updated with JWT tokens per request)
        self.social_client = ServiceClient(self.social_service_url)
        self.comment_client = ServiceClient(self.comment_service_url)

        # Use SERVICE_AUTH_TOKEN from environment for authentication
        logger.info("=== SERVICE TOKEN INITIALIZATION ===")
        self.current_jwt_token = os.environ.get('SERVICE_AUTH_TOKEN')
        if self.current_jwt_token:
            token_preview = self.current_jwt_token[:20] + "..." if len(self.current_jwt_token) > 20 else self.current_jwt_token
            logger.info(f"SERVICE_AUTH_TOKEN loaded from environment")
            logger.info(f"Token preview: {token_preview}")
            logger.info(f"Token length: {len(self.current_jwt_token)} characters")
            # Basic JWT validation (check if it looks like a JWT)
            if self.current_jwt_token.count('.') == 2:
                logger.info("Token format: Valid JWT structure (header.payload.signature)")
            else:
                logger.warning(f"Token format: Does NOT look like a JWT (expected 2 dots, found {self.current_jwt_token.count('.')})")
        else:
            logger.error("=== TOKEN INITIALIZATION FAILED ===")
            logger.error("SERVICE_AUTH_TOKEN is NOT set in environment")
            logger.error("The service will NOT be able to authenticate API requests!")
            logger.error("Please set SERVICE_AUTH_TOKEN in your .env file or environment")

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

        # Candidate pool cache
        self.pool_enabled = os.environ.get('POOL_ENABLED', 'true').lower() == 'true'
        pool_ttl = int(os.environ.get('POOL_TTL', 86400))
        self.pool_cache = CandidatePoolCache(self.redis_client, pool_ttl=pool_ttl)
        logger.info(f"Candidate pool cache initialized (enabled={self.pool_enabled}, ttl={pool_ttl}s, cap=200)")

        # Initialize service token manager
        logger.info("=== SERVICE TOKEN MANAGER INITIALIZATION ===")
        self.token_manager = None
        token_before_manager = self.current_jwt_token
        logger.info(f"Token state before manager init: {'SET' if token_before_manager else 'NOT SET'}")

        if get_service_token_manager:
            logger.info("ServiceTokenManager is available, attempting API token request...")
            try:
                self.token_manager = get_service_token_manager("core-recommendations")
                # Try to get token from API
                if self.token_manager.request_service_token(self.api_base_url):
                    logger.info("Successfully obtained service token from API")
                    # Update current JWT token with the new one from API
                    self.current_jwt_token = self.token_manager.get_access_token()
                    logger.info(f"Updated current_jwt_token with API token (length: {len(self.current_jwt_token) if self.current_jwt_token else 0})")
                else:
                    logger.warning("Could not obtain service token from API, falling back to environment variable")
                    # Fall back to environment variable
                    env_token = os.environ.get('SERVICE_AUTH_TOKEN')
                    if env_token:
                        self.current_jwt_token = env_token
                        logger.info(f"Using service token from environment variable (length: {len(env_token)})")
                    else:
                        logger.error("=== CRITICAL: NO SERVICE TOKEN AVAILABLE ===")
                        logger.error("API token request failed AND SERVICE_AUTH_TOKEN not set in environment")
                logger.info("Service token manager initialized")
            except Exception as e:
                logger.warning(f"Could not initialize service token manager: {e}")
                # Fall back to environment variable
                env_token = os.environ.get('SERVICE_AUTH_TOKEN')
                if env_token:
                    self.current_jwt_token = env_token
                    logger.info(f"Using service token from environment variable as fallback (length: {len(env_token)})")
                else:
                    logger.error("=== CRITICAL: NO SERVICE TOKEN AVAILABLE ===")
                    logger.error(f"Token manager init failed: {e}")
                    logger.error("SERVICE_AUTH_TOKEN also not set in environment")
        else:
            logger.info("ServiceTokenManager not available, relying on environment token only")

        # Final token status check
        logger.info("=== FINAL TOKEN STATUS ===")
        if self.current_jwt_token:
            logger.info(f"Service token is SET (length: {len(self.current_jwt_token)} chars)")
        else:
            logger.error("Service token is NOT SET - API calls requiring auth will FAIL!")

        # Initialize metadata enhancer (RL-enhanced if available)
        # RL disabled by default until performance is optimized
        self.rl_enabled = os.environ.get('RL_ENABLED', 'true').lower() == 'true'

        if RL_AVAILABLE and self.rl_enabled:
            try:
                # RL configuration
                rl_config = {
                    'integration': {
                        'enabled': True,
                        'learning_mode': os.environ.get('RL_LEARNING_MODE', 'online'),
                        'a_b_test_ratio': float(os.environ.get('RL_AB_TEST_RATIO', '0.5')),
                        'safety_threshold': 0.1,
                        'warmup_interactions': int(os.environ.get('RL_WARMUP_INTERACTIONS', '10'))
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
                        'user_dim': int(os.environ.get('EMBEDDING_DIM', '32')),
                        'post_dim': int(os.environ.get('EMBEDDING_DIM', '32'))
                    },
                    'bandit': {
                        'action_space': {'total_actions': 20, 'action_encoding_dim': 10},
                        'exploration': {
                            'strategy': os.environ.get('RL_EXPLORATION_STRATEGY', 'epsilon_greedy'),
                            'epsilon': float(os.environ.get('RL_EPSILON', '0.1'))
                        }
                    },
                    'performance': {
                        'max_workers': 4,
                        'max_processing_time': 0.1,
                        'fallback_enabled': True
                    },
                    'api_base_url': self.api_base_url
                }

                self.metadata_enhancer = create_rl_enhanced_metadata_enhancer(
                    api_base_url=self.api_base_url,
                    redis_client=self.redis_client,
                    rl_config=rl_config
                )
                logger.info("RL-Enhanced MetadataEnhancer initialized with RL agent active")
            except Exception as e:
                logger.error(f"Failed to initialize RL components: {e}", exc_info=True)
                logger.warning("Falling back to basic MetadataEnhancer")
                self.metadata_enhancer = MetadataEnhancer(self.api_base_url, self.redis_client)
                self.rl_enabled = False
        else:
            self.metadata_enhancer = MetadataEnhancer(self.api_base_url, self.redis_client)
            if not RL_AVAILABLE:
                logger.warning("RL components not available, using basic MetadataEnhancer")
            else:
                logger.info("RL disabled via environment, using basic MetadataEnhancer")
        
        # Store the current JWT token for this request
        # If we have a token from the token manager, update all clients
        if hasattr(self, 'current_jwt_token') and self.current_jwt_token:
            self.social_client.update_token(self.current_jwt_token)
            self.comment_client.update_token(self.current_jwt_token)
            # Also update the metadata enhancer with the JWT token
            if hasattr(self.metadata_enhancer, 'set_jwt_token'):
                self.metadata_enhancer.set_jwt_token(self.current_jwt_token)
            logger.info("Updated all service clients and metadata enhancer with token from API")

        # Initialize Two-Tower model
        self.two_tower_model = None
        self._load_model()

        # Initialize Recommendation Explainer (shares metadata cache with enhancer)
        self.explainer = create_explainer(
            api_base_url=self.api_base_url,
            metadata_cache=self.metadata_enhancer.memory_cache
        )
        logger.info("RecommendationExplainer initialized")

        # Final startup summary
        logger.info("=" * 60)
        logger.info("=== CORE RECOMMENDATIONS SERVICE STARTUP SUMMARY ===")
        logger.info("=" * 60)
        logger.info(f"API Base URL: {self.api_base_url}")
        logger.info(f"Auth Token Status: {'CONFIGURED' if self.current_jwt_token else 'NOT CONFIGURED - API AUTH WILL FAIL!'}")
        if self.current_jwt_token:
            logger.info(f"  Token Length: {len(self.current_jwt_token)} chars")
            logger.info(f"  Token Format: {'Valid JWT' if self.current_jwt_token.count('.') == 2 else 'INVALID (not a JWT)'}")
        logger.info(f"RL Enabled: {self.rl_enabled}")
        logger.info(f"Pool Enabled: {self.pool_enabled}")
        logger.info("=" * 60)

        logger.info(f"Initialized Core Recommendations Service on port 5000")

    def _login_to_api(self):
        """Login to Spring API and get JWT token"""
        try:
            username = os.environ.get('SERVICE_USERNAME', 'ml-service')
            password = os.environ.get('SERVICE_PASSWORD', '')

            if not password:
                logger.warning("SERVICE_PASSWORD not set, skipping service login")
                return

            response = requests.post(
                f"{self.api_base_url}/api/auth/service-login",
                json={"username": username, "password": password},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self.current_jwt_token = data.get('accessToken')
                logger.info("Successfully authenticated with Spring API")
            else:
                logger.error(f"Service login failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to login to API: {e}")

    def _update_jwt_token(self, jwt_token: str = None):
        """Update JWT token for all service clients and metadata enhancer"""
        if jwt_token:
            self.current_jwt_token = jwt_token
            self.social_client.update_token(jwt_token)
            self.comment_client.update_token(jwt_token)
            # Also update the metadata enhancer
            if hasattr(self.metadata_enhancer, 'set_jwt_token'):
                self.metadata_enhancer.set_jwt_token(jwt_token)
        elif get_token_or_fallback and not self.current_jwt_token:
            # Try to get token from request or fallback
            fallback_token = get_token_or_fallback()
            if fallback_token:
                self.current_jwt_token = fallback_token
                self.social_client.update_token(fallback_token)
                self.comment_client.update_token(fallback_token)
                # Also update the metadata enhancer
                if hasattr(self.metadata_enhancer, 'set_jwt_token'):
                    self.metadata_enhancer.set_jwt_token(fallback_token)

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
                - includeExplanations: Whether to include explanations (default: True)

        Returns:
            Dictionary containing recommended posts with scores and explanations
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
            include_explanations = request_data.get("includeExplanations", True)

            logger.info(f"Processing recommendation request for user {user_id}, "
                       f"content type: {content_type}, social: {enable_social}")

            # Get base recommendations
            if content_type == "trailers":
                result = self.get_trailer_recommendations(
                    user_id=user_id, limit=limit, jwt_token=jwt_token,
                    include_explanations=include_explanations
                )
            else:
                result = self.get_post_recommendations(
                    user_id=user_id, limit=limit, jwt_token=jwt_token,
                    include_explanations=include_explanations
                )

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

    def get_post_recommendations(self, user_id: str, limit: int = 20,
                                  jwt_token: str = None, include_explanations: bool = True) -> Dict[str, Any]:
        """Get post recommendations using the Two-Tower model"""
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)

            start_time = time.time()
            logger.info(f"=== POST RECOMMENDATIONS REQUEST for user {user_id} ===")
            logger.info(f"Request params - limit: {limit}, include_explanations: {include_explanations}")

            # Fetch user vector and pool candidates in parallel (they are independent)
            logger.info(f"Fetching user vector and pool candidates in parallel for user {user_id}...")
            fetch_limit = limit * 2  # Get more candidates for better selection
            future_user_vec = _executor.submit(self._get_user_vector, user_id)
            future_pool = _executor.submit(
                self.pool_cache.pull_candidates, user_id, "posts"
            ) if self.pool_enabled else None

            user_vector = future_user_vec.result()
            if user_vector is None:
                logger.error(f"Failed to retrieve user vector for user {user_id} - aborting request")
                return _error_response(f"Could not retrieve vector for user {user_id}")
            logger.info(f"User vector retrieved successfully. Shape: {user_vector.shape}")

            # Get cursors — Redis-persisted state survives restarts; fall back to in-memory
            persisted = self.pool_cache.load_cursor(user_id, "posts")
            with self._cursor_lock:
                mem = self.cursor_tracker.get(user_id, {}).get("posts", {})
            cursor = mem.get("cursor") or persisted.get("cursor")
            high_quality_cursor = mem.get("highQualityCursor") or persisted.get("highQualityCursor")
            logger.info(f"Pagination state - cursor: {cursor}, highQualityCursor: {high_quality_cursor}")

            pool_vectors = {}
            if future_pool is not None:
                try:
                    pool_vectors = future_pool.result()
                    logger.info(f"Pool pull: {len(pool_vectors)} unseen candidates available for user {user_id}")
                except Exception as pool_err:
                    logger.warning(f"Pool pull failed (non-fatal): {pool_err}")

            # Always fetch fresh candidates from the API so interaction data (e.g. new
            # likes) is re-validated on every request — the pool alone can go stale for
            # up to its TTL and must never be served without this check. Paginates for
            # more if filtering leaves too few, rather than re-showing seen content.
            logger.info(f"Fetching {fetch_limit} candidates from API (pool has {len(pool_vectors)})...")
            candidate_data = self._fetch_fresh_candidates(
                user_id=user_id,
                fetch_limit=fetch_limit,
                content_type="POSTS",
                cursor=cursor,
                high_quality_cursor=high_quality_cursor,
                pool_content_type="posts"
            )
            fresh_vectors = candidate_data["vectors"]
            # Merge pool candidates into fresh fetch (fresh wins on collision)
            if pool_vectors:
                merged = dict(pool_vectors)
                merged.update(fresh_vectors)
                candidate_data["vectors"] = merged
                logger.info(f"Merged {len(pool_vectors)} pool + {len(fresh_vectors)} fresh = {len(merged)} total candidates")

            candidate_count = len(candidate_data.get("vectors", {}))
            if not candidate_data["vectors"]:
                logger.warning(f"=== NO CANDIDATES FOUND for user {user_id} ===")
                logger.warning(f"Candidate fetch returned empty result. Distribution: {candidate_data.get('distribution', {})}")
                logger.warning(f"Cursor state at failure - cursor: {cursor}, highQualityCursor: {high_quality_cursor}")
                return {
                    "postIds": [],
                    "scores": [],
                    "explanations": [],
                    "totalCount": 0,
                    "processingTime": time.time() - start_time,
                    "message": "No candidates found"
                }

            logger.info(f"Fetched {candidate_count} candidates successfully")
            logger.info(f"Candidate distribution: {candidate_data.get('distribution', {})}")

            # Score candidates using Two-Tower model with metadata enhancement
            scoring_result = self._score_candidates(
                user_id=user_id,
                user_vector=user_vector,
                candidate_data=candidate_data,
                content_type="posts"
            )

            # Take top results
            total_scored = len(scoring_result["scored_posts"])
            top_results = scoring_result["scored_posts"][:limit]
            final_post_ids = [int(post_id) for post_id, _ in top_results]
            final_scores = [float(score) for _, score in top_results]
            logger.info(f"Selected top {len(final_post_ids)} from {total_scored} scored candidates")

            # Mark recommended posts as shown (fire-and-forget — doesn't affect response)
            if self.pool_enabled and final_post_ids:
                _executor.submit(self.pool_cache.mark_shown, user_id, "posts", final_post_ids)

            # Generate explanations
            explanations = []
            if include_explanations:
                logger.info(f"Generating explanations for {len(final_post_ids)} recommendations...")
                explanations = self._generate_explanations(
                    user_id=user_id,
                    post_ids=final_post_ids,
                    base_scores=scoring_result["base_scores"],
                    enhanced_scores=scoring_result["enhanced_scores"],
                    rl_actions=scoring_result.get("rl_actions", []),
                    content_type="posts"
                )
                logger.info(f"Generated {len(explanations)} explanations")
            else:
                logger.info("Skipping explanation generation (disabled)")

            processing_time = time.time() - start_time
            result = {
                "postIds": final_post_ids,
                "scores": final_scores,
                "explanations": explanations,
                "totalCount": len(final_post_ids),
                "processingTime": processing_time,
                "hasMore": candidate_data["hasMore"],
                "nextCursor": candidate_data["nextCursor"],
                "nextHighQualityCursor": candidate_data["nextHighQualityCursor"],
                "distribution": candidate_data["distribution"]
            }

            logger.info(f"=== POST RECOMMENDATIONS COMPLETE for user {user_id} ===")
            logger.info(f"Response summary - posts: {len(final_post_ids)}, processing_time: {processing_time:.3f}s")
            logger.info(f"Score range: [{min(final_scores) if final_scores else 0:.4f}, {max(final_scores) if final_scores else 0:.4f}]")
            logger.info(f"Pagination - hasMore: {candidate_data['hasMore']}, nextCursor: {candidate_data['nextCursor']}")
            return result

        except Exception as e:
            logger.error(f"Error getting post recommendations: {str(e)}", exc_info=True)
            return _error_response(f"Error getting post recommendations: {str(e)}")

    def get_trailer_recommendations(self, user_id: str, limit: int = 20,
                                    jwt_token: str = None, include_explanations: bool = True) -> Dict[str, Any]:
        """Get trailer recommendations using the Two-Tower model"""
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)

            start_time = time.time()
            logger.info(f"=== TRAILER RECOMMENDATIONS REQUEST for user {user_id} ===")
            logger.info(f"Request params - limit: {limit}, include_explanations: {include_explanations}")

            # Fetch user vector and pool candidates in parallel (they are independent)
            logger.info(f"Fetching user vector and pool candidates in parallel for user {user_id}...")
            fetch_limit = limit * 2
            future_user_vec = _executor.submit(self._get_user_vector, user_id)
            future_pool = _executor.submit(
                self.pool_cache.pull_candidates, user_id, "trailers"
            ) if self.pool_enabled else None

            user_vector = future_user_vec.result()
            if user_vector is None:
                logger.error(f"Failed to retrieve user vector for user {user_id} - aborting request")
                return _error_response(f"Could not retrieve vector for user {user_id}")
            logger.info(f"User vector retrieved successfully. Shape: {user_vector.shape}")

            # Get cursors — Redis-persisted state survives restarts; fall back to in-memory
            persisted = self.pool_cache.load_cursor(user_id, "trailers")
            with self._cursor_lock:
                mem = self.cursor_tracker.get(user_id, {}).get("trailers", {})
            cursor = mem.get("cursor") or persisted.get("cursor")
            high_quality_cursor = mem.get("highQualityCursor") or persisted.get("highQualityCursor")
            logger.info(f"Pagination state - cursor: {cursor}, highQualityCursor: {high_quality_cursor}")

            pool_vectors = {}
            if future_pool is not None:
                try:
                    pool_vectors = future_pool.result()
                    logger.info(f"Pool pull: {len(pool_vectors)} unseen trailer candidates available for user {user_id}")
                except Exception as pool_err:
                    logger.warning(f"Pool pull failed (non-fatal): {pool_err}")

            # Always fetch fresh candidates from the API so interaction data (e.g. new
            # likes) is re-validated on every request — the pool alone can go stale for
            # up to its TTL and must never be served without this check.
            logger.info(f"Fetching {fetch_limit} trailer candidates from API (pool has {len(pool_vectors)})...")
            candidate_data = self._fetch_fresh_candidates(
                user_id=user_id,
                fetch_limit=fetch_limit,
                content_type="TRAILERS",
                cursor=cursor,
                high_quality_cursor=high_quality_cursor,
                pool_content_type="trailers"
            )
            fresh_vectors = candidate_data["vectors"]
            # Merge pool candidates into fresh fetch (fresh wins on collision)
            if pool_vectors:
                merged = dict(pool_vectors)
                merged.update(fresh_vectors)
                candidate_data["vectors"] = merged
                logger.info(f"Merged {len(pool_vectors)} pool + {len(fresh_vectors)} fresh = {len(merged)} total trailer candidates")

            candidate_count = len(candidate_data.get("vectors", {}))
            if not candidate_data["vectors"]:
                logger.warning(f"=== NO TRAILER CANDIDATES FOUND for user {user_id} ===")
                logger.warning(f"Candidate fetch returned empty result. Distribution: {candidate_data.get('distribution', {})}")
                logger.warning(f"Cursor state at failure - cursor: {cursor}, highQualityCursor: {high_quality_cursor}")
                return {
                    "postIds": [],
                    "scores": [],
                    "explanations": [],
                    "totalCount": 0,
                    "processingTime": time.time() - start_time,
                    "message": "No trailer candidates found"
                }

            logger.info(f"Fetched {candidate_count} trailer candidates successfully")
            logger.info(f"Candidate distribution: {candidate_data.get('distribution', {})}")

            # Score candidates using Two-Tower model with metadata enhancement
            scoring_result = self._score_candidates(
                user_id=user_id,
                user_vector=user_vector,
                candidate_data=candidate_data,
                content_type="trailers"
            )

            # Take top results
            total_scored = len(scoring_result["scored_posts"])
            top_results = scoring_result["scored_posts"][:limit]
            final_post_ids = [int(post_id) for post_id, _ in top_results]
            final_scores = [float(score) for _, score in top_results]
            logger.info(f"Selected top {len(final_post_ids)} from {total_scored} scored trailer candidates")

            # Mark recommended trailers as shown (fire-and-forget — doesn't affect response)
            if self.pool_enabled and final_post_ids:
                _executor.submit(self.pool_cache.mark_shown, user_id, "trailers", final_post_ids)

            # Generate explanations
            explanations = []
            if include_explanations:
                logger.info(f"Generating explanations for {len(final_post_ids)} trailer recommendations...")
                explanations = self._generate_explanations(
                    user_id=user_id,
                    post_ids=final_post_ids,
                    base_scores=scoring_result["base_scores"],
                    enhanced_scores=scoring_result["enhanced_scores"],
                    rl_actions=scoring_result.get("rl_actions", []),
                    content_type="trailers"
                )
                logger.info(f"Generated {len(explanations)} explanations")
            else:
                logger.info("Skipping explanation generation (disabled)")

            processing_time = time.time() - start_time
            result = {
                "postIds": final_post_ids,
                "scores": final_scores,
                "explanations": explanations,
                "totalCount": len(final_post_ids),
                "processingTime": processing_time,
                "hasMore": candidate_data["hasMore"],
                "nextCursor": candidate_data["nextCursor"],
                "nextHighQualityCursor": candidate_data["nextHighQualityCursor"],
                "distribution": candidate_data["distribution"]
            }

            logger.info(f"=== TRAILER RECOMMENDATIONS COMPLETE for user {user_id} ===")
            logger.info(f"Response summary - trailers: {len(final_post_ids)}, processing_time: {processing_time:.3f}s")
            logger.info(f"Score range: [{min(final_scores) if final_scores else 0:.4f}, {max(final_scores) if final_scores else 0:.4f}]")
            logger.info(f"Pagination - hasMore: {candidate_data['hasMore']}, nextCursor: {candidate_data['nextCursor']}")
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
            logger.info(f"=== TOKEN STATUS FOR CANDIDATE REQUEST ===")
            logger.info(f"self.current_jwt_token: {'SET (' + str(len(self.current_jwt_token)) + ' chars)' if self.current_jwt_token else 'NOT SET (None)'}")
            env_token = os.environ.get('SERVICE_AUTH_TOKEN', '')
            logger.info(f"SERVICE_AUTH_TOKEN env var: {'SET (' + str(len(env_token)) + ' chars)' if env_token else 'NOT SET (empty)'}")

            auth_token = self.current_jwt_token or env_token
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
                headers['X-Service-Role'] = 'SERVICE'
                token_preview = auth_token[:30] + "..." if len(auth_token) > 30 else auth_token
                logger.info(f"Auth token applied: {token_preview}")
                logger.info(f"Auth token length: {len(auth_token)} characters")
                # Check if it looks like a valid JWT
                if auth_token.count('.') != 2:
                    logger.warning(f"Token does NOT appear to be a valid JWT (expected 2 dots, found {auth_token.count('.')})")
            else:
                logger.error("=== NO AUTH TOKEN AVAILABLE FOR CANDIDATE REQUEST ===")
                logger.error("Both self.current_jwt_token and SERVICE_AUTH_TOKEN are empty!")
                logger.error("This request will likely fail with 401/403/500 error")

            logger.info(f"=== CANDIDATE API REQUEST ===")
            logger.info(f"URL: {url}")
            logger.info(f"Params: {params}")
            logger.info(f"Headers: {dict((k, 'Bearer ***' if k == 'Authorization' else v) for k, v in headers.items())}")
            logger.info(f"Timeout: (connect=10s, read=60s)")

            response = requests.get(url, params=params, headers=headers, timeout=(10, 60))
            
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

                # Update cursor tracker (memory + Redis so it survives restarts)
                with self._cursor_lock:
                    if user_id not in self.cursor_tracker:
                        self.cursor_tracker[user_id] = {}
                    self.cursor_tracker[user_id][content_type.lower()] = {
                        "cursor": result["nextCursor"],
                        "highQualityCursor": result["nextHighQualityCursor"]
                    }
                self.pool_cache.save_cursor(
                    user_id, content_type.lower(),
                    result["nextCursor"], result["nextHighQualityCursor"]
                )

                # Insert fresh candidates into the pool cache
                if self.pool_enabled:
                    try:
                        pool_size = self.pool_cache.insert_candidates(
                            user_id=user_id,
                            content_type=content_type.lower(),
                            new_vectors=converted_vectors
                        )
                        logger.info(f"Pool cache updated: {pool_size} total entries for user {user_id} [{content_type.lower()}]")
                    except Exception as pool_err:
                        logger.warning(f"Pool cache insert failed (non-fatal): {pool_err}")

                logger.info(f"Successfully fetched {len(converted_vectors)} candidates for user {user_id}")
                logger.info(f"Cursor info - Next: {result['nextCursor']}, HighQuality: {result['nextHighQualityCursor']}")
                logger.info(f"Pagination - HasMore: {result['hasMore']}, HasMoreHighQuality: {result['hasMoreHighQuality']}")
                return result

            elif response.status_code == 204:  # No content
                logger.info(f"No candidates available for user {user_id} (204 No Content)")
                return {"vectors": {}, "nextCursor": None, "hasMore": False, "distribution": {}}
            else:
                logger.warning(f"=== CANDIDATE FETCH FAILED for user {user_id} ===")
                logger.warning(f"API returned status {response.status_code} for candidate vectors")
                logger.warning(f"Request URL: {url}")
                logger.warning(f"Request params: {params}")
                logger.warning(f"Auth token present: {bool(auth_token)}")
                try:
                    error_body = response.text[:500] if response.text else "(empty response body)"
                    logger.warning(f"Error response content: {error_body}")
                except:
                    logger.warning("Could not read error response body")

                if response.status_code == 401:
                    logger.error(f"=== AUTHENTICATION FAILED (401) ===")
                    logger.error(f"Token may be expired, invalid, or malformed")
                    logger.error(f"Token present: {bool(auth_token)}")
                    logger.error(f"Token length: {len(auth_token) if auth_token else 0} chars")
                    logger.error(f"Headers sent: Authorization={'present' if 'Authorization' in headers else 'MISSING'}, X-Service-Role={'present' if 'X-Service-Role' in headers else 'MISSING'}")
                    if auth_token:
                        logger.error(f"Token preview: {auth_token[:50]}...")
                        logger.error(f"Token looks like JWT: {auth_token.count('.') == 2}")
                    logger.error("ACTION: Check if SERVICE_AUTH_TOKEN is a valid, non-expired JWT")
                elif response.status_code == 403:
                    logger.error(f"=== AUTHORIZATION FAILED (403) ===")
                    logger.error(f"Token present: {bool(auth_token)}, Service role header: {headers.get('X-Service-Role', 'MISSING')}")
                    logger.error(f"Token is valid but lacks required permissions")
                    logger.error(f"This may indicate: 1) Token lacks SERVICE role, 2) User {user_id} not accessible, 3) Endpoint requires admin permissions")
                    logger.error("ACTION: Ensure token has SERVICE role and appropriate permissions")
                elif response.status_code == 404:
                    logger.warning(f"Candidate endpoint not found (404) - check API base URL: {self.api_base_url}")
                elif response.status_code >= 500:
                    logger.error(f"=== SERVER ERROR ({response.status_code}) ===")
                    logger.error(f"Backend service error accessing candidates for user {user_id}")
                    logger.error(f"Token was {'present' if auth_token else 'NOT present (this may be the cause!)'}")
                    if not auth_token:
                        logger.error("=== LIKELY CAUSE: NO AUTH TOKEN ===")
                        logger.error("The 500 error may be caused by missing authentication")
                        logger.error("Spring API may throw internal error when trying to process unauthenticated request")
                        logger.error("ACTION: Set SERVICE_AUTH_TOKEN environment variable")
                    else:
                        logger.error(f"Token was present (length: {len(auth_token)}), so this is likely a backend issue")
                        logger.error("ACTION: Check Spring service logs for the actual error")

                return {"vectors": {}, "nextCursor": None, "hasMore": False, "distribution": {}}

        except requests.exceptions.Timeout as e:
            logger.error(f"=== CANDIDATE FETCH TIMEOUT for user {user_id} ===")
            logger.error(f"Request timed out (connect=10s, read=60s) - URL: {url}")
            logger.error(f"Timeout details: {str(e)}")
            return {"vectors": {}, "nextCursor": None, "hasMore": False, "distribution": {}}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"=== CANDIDATE FETCH CONNECTION ERROR for user {user_id} ===")
            logger.error(f"Failed to connect to API endpoint: {url}")
            logger.error(f"Connection error details: {str(e)}")
            logger.error(f"Check that Spring API is running at {self.api_base_url}")
            return {"vectors": {}, "nextCursor": None, "hasMore": False, "distribution": {}}
        except requests.exceptions.RequestException as e:
            logger.error(f"=== CANDIDATE FETCH REQUEST ERROR for user {user_id} ===")
            logger.error(f"Request failed - URL: {url}, Params: {params}")
            logger.error(f"Request error details: {str(e)}")
            return {"vectors": {}, "nextCursor": None, "hasMore": False, "distribution": {}}
        except Exception as e:
            logger.error(f"=== CANDIDATE FETCH UNEXPECTED ERROR for user {user_id} ===")
            logger.error(f"Request context - URL: {url}, Content Type: {content_type}")
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return {"vectors": {}, "nextCursor": None, "hasMore": False, "distribution": {}}

    def _fetch_fresh_candidates(self, user_id: str, fetch_limit: int, content_type: str,
                                cursor: Optional[str], high_quality_cursor: Optional[str],
                                pool_content_type: str, max_extra_fetches: int = 2) -> Dict[str, Any]:
        """
        Fetch candidates from the API, filtering out already-shown posts. If
        filtering leaves too few fresh candidates and the API has more to offer
        (hasMore), paginate for additional pages rather than re-showing already-
        seen content - bounded by max_extra_fetches so a user who has seen nearly
        everything doesn't cause unbounded latency/API load on a single request.

        Args:
            user_id: User ID
            fetch_limit: Desired candidate count
            content_type: "POSTS" or "TRAILERS" (API param)
            cursor: Pagination cursor to start from
            high_quality_cursor: High-quality pagination cursor to start from
            pool_content_type: "posts" or "trailers" (pool_cache key)
            max_extra_fetches: Cap on additional pagination requests

        Returns:
            candidate_data dict with "vectors" containing only fresh candidates,
            and pagination fields (nextCursor etc.) reflecting the last fetch made
        """
        candidate_data = self._fetch_candidate_vectors(
            user_id=user_id, limit=fetch_limit, content_type=content_type,
            cursor=cursor, high_quality_cursor=high_quality_cursor
        )
        fresh_vectors = self.pool_cache.filter_shown(user_id, pool_content_type, candidate_data["vectors"])
        filtered_count = len(candidate_data["vectors"]) - len(fresh_vectors)
        if filtered_count:
            logger.info(f"Filtered {filtered_count} already-shown {pool_content_type} from fresh API fetch")
        candidate_data["vectors"] = fresh_vectors

        extra_fetches = 0
        while (len(candidate_data["vectors"]) < fetch_limit and candidate_data.get("hasMore")
               and extra_fetches < max_extra_fetches):
            extra_fetches += 1
            logger.info(f"Only {len(candidate_data['vectors'])}/{fetch_limit} fresh {pool_content_type} "
                       f"candidates after filtering - fetching more "
                       f"(attempt {extra_fetches}/{max_extra_fetches})")
            next_batch = self._fetch_candidate_vectors(
                user_id=user_id, limit=fetch_limit, content_type=content_type,
                cursor=candidate_data.get("nextCursor"),
                high_quality_cursor=candidate_data.get("nextHighQualityCursor")
            )
            next_fresh = self.pool_cache.filter_shown(user_id, pool_content_type, next_batch["vectors"])
            merged_vectors = dict(candidate_data["vectors"])
            merged_vectors.update(next_fresh)
            # Carry forward pagination state from the latest fetch; accumulate vectors
            candidate_data = {**next_batch, "vectors": merged_vectors}

        return candidate_data

    def _get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Get user vector from Redis or API"""
        logger.debug(f"Fetching user vector for user {user_id}")

        # Try to get from Redis first
        redis_key = f"{self.user_vector_key_prefix}{user_id}"
        try:
            cached_vector = self.redis_client.get(redis_key)
            if cached_vector:
                vector = np.frombuffer(cached_vector, dtype=np.float32)
                logger.debug(f"User vector cache hit for user {user_id}. Vector shape: {vector.shape}")
                return vector
            logger.debug(f"User vector cache miss for user {user_id}")
        except Exception as redis_err:
            logger.warning(f"Redis error while fetching user vector for {user_id}: {redis_err}")

        # Get from API if not in Redis
        url = f"{self.api_base_url}/api/internal/ml/users/{user_id}/vector"
        try:
            headers = {}
            # Use current JWT token or fallback to environment token
            auth_token = self.current_jwt_token or os.environ.get('SERVICE_AUTH_TOKEN', '')
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
                headers['X-Service-Role'] = 'SERVICE'

            logger.debug(f"Fetching user vector from API: {url}")
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                vector_data = response.json()
                vector = np.array(vector_data, dtype=np.float32)
                logger.info(f"Successfully fetched user vector from API for user {user_id}. Shape: {vector.shape}")

                # Cache in Redis
                try:
                    self.redis_client.setex(redis_key, self.vector_cache_ttl, vector.tobytes())
                    logger.debug(f"Cached user vector for user {user_id} with TTL {self.vector_cache_ttl}s")
                except Exception as cache_err:
                    logger.warning(f"Failed to cache user vector for user {user_id}: {cache_err}")
                return vector
            else:
                logger.warning(f"=== USER VECTOR FETCH FAILED for user {user_id} ===")
                logger.warning(f"API returned status {response.status_code}")
                logger.warning(f"Request URL: {url}")
                try:
                    error_body = response.text[:500]  # Limit error body length
                    logger.warning(f"Response body: {error_body}")
                except:
                    pass
                if response.status_code == 404:
                    logger.warning(f"User {user_id} not found in API - may be new user or invalid ID")
                elif response.status_code == 401 or response.status_code == 403:
                    logger.error(f"Authentication/Authorization failed for user vector request. Token present: {bool(auth_token)}")
                elif response.status_code >= 500:
                    logger.error(f"Server error from API while fetching user vector for user {user_id}")

        except requests.exceptions.Timeout as e:
            logger.error(f"=== USER VECTOR FETCH TIMEOUT for user {user_id} ===")
            logger.error(f"Request timed out after 5s - URL: {url}")
            logger.error(f"Timeout details: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"=== USER VECTOR FETCH CONNECTION ERROR for user {user_id} ===")
            logger.error(f"Failed to connect to API endpoint: {url}")
            logger.error(f"Connection error details: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"=== USER VECTOR FETCH REQUEST ERROR for user {user_id} ===")
            logger.error(f"Request failed - URL: {url}")
            logger.error(f"Request error details: {str(e)}")
        except Exception as e:
            logger.error(f"=== USER VECTOR FETCH UNEXPECTED ERROR for user {user_id} ===")
            logger.error(f"Unexpected error retrieving user vector from API: {e}", exc_info=True)

        # Return default vector if can't retrieve
        logger.warning(f"Using default (random) vector for user {user_id} - API fetch failed")
        feature_dim = int(os.environ.get('USER_FEATURE_DIM', '64'))
        default_vector = np.random.rand(feature_dim).astype(np.float32)
        logger.info(f"Generated default vector with dimension {feature_dim} for user {user_id}")

        # Cache default vector
        try:
            self.redis_client.setex(redis_key, self.vector_cache_ttl, default_vector.tobytes())
            logger.debug(f"Cached default vector for user {user_id}")
        except Exception as cache_err:
            logger.warning(f"Failed to cache default vector for user {user_id}: {cache_err}")
        return default_vector

    def _score_candidates(self, user_id: str, user_vector: np.ndarray,
                          candidate_data: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """
        Score candidate posts using the Two-Tower model with metadata enhancement.

        TMDB ToS Compliant Pipeline:
        1. Two-Tower ML scoring (behavioral embeddings)
        2. Eligibility filtering (TMDB boolean pass/fail)
        3. Behavioral boost (app engagement data only)
        4. Diversity reranking (TMDB genre for ordering only)

        Returns:
            Dict containing:
                - scored_posts: List of (post_id, final_score) tuples sorted by score
                - base_scores: Dict mapping post_id to base ML score
                - enhanced_scores: Dict mapping post_id to final enhanced score
                - rl_actions: List of RL actions taken (if RL active)
                - filtered_posts: List of post IDs filtered out (TMDB ToS compliant)
                - diversity_applied: Boolean indicating diversity reranking was applied
        """
        logger.info(f"=== CANDIDATE SCORING START for user {user_id} ===")
        logger.info(f"Content type: {content_type}")
        logger.info(f"Input candidate count: {len(candidate_data.get('vectors', {}))}")
        logger.info(f"User vector shape: {user_vector.shape if user_vector is not None else 'None'}")

        result = {
            "scored_posts": [],
            "base_scores": {},
            "enhanced_scores": {},
            "rl_actions": [],
            "filtered_posts": [],
            "diversity_applied": False
        }

        if not self.two_tower_model:
            logger.warning("TwoTower model not available, using fallback scoring")
            logger.warning(f"Fallback: Assigning random scores to {len(candidate_data.get('vectors', {}))} candidates")
            post_ids = list(candidate_data["vectors"].keys())
            for post_id in post_ids:
                score = float(np.random.random())
                result["base_scores"][post_id] = score
                result["enhanced_scores"][post_id] = score
            result["scored_posts"] = [(pid, result["enhanced_scores"][pid]) for pid in post_ids]
            logger.info(f"Fallback scoring complete: {len(result['scored_posts'])} candidates scored")
            return result

        try:
            vectors = candidate_data["vectors"]
            candidates = candidate_data.get("candidates", [])
            logger.info(f"Processing {len(vectors)} candidate vectors with {len(candidates)} candidate metadata entries")

            if not vectors:
                logger.warning(f"No candidate vectors to score for user {user_id}")
                return result

            # Prepare data for scoring
            post_ids = list(vectors.keys())
            post_vectors = list(vectors.values())
            post_vectors_array = np.array(post_vectors)
            logger.info(f"Prepared {len(post_ids)} candidates for Two-Tower scoring")
            logger.debug(f"Candidate post IDs: {post_ids[:10]}{'...' if len(post_ids) > 10 else ''}")
            logger.debug(f"Post vectors array shape: {post_vectors_array.shape}")

            # Reshape user vector for batch processing
            user_batch = np.expand_dims(user_vector, axis=0)
            logger.debug(f"User batch shape: {user_batch.shape}")

            # Calculate base scores using Two-Tower model
            logger.info(f"Computing Two-Tower base scores for {len(post_ids)} candidates...")
            base_scores_array = compute_scores(user_batch, post_vectors_array, content_type)
            logger.info(f"Two-Tower scoring complete. Score range: [{base_scores_array.min():.4f}, {base_scores_array.max():.4f}]")
            logger.info(f"Mean base score: {base_scores_array.mean():.4f}, Std: {base_scores_array.std():.4f}")

            # Store base scores
            for i, post_id in enumerate(post_ids):
                result["base_scores"][post_id] = float(base_scores_array[0][i])

            # Apply COMPLIANT metadata enhancement (eligibility filter + behavioral boost)
            logger.info(f"Applying metadata enhancement to {len(post_ids)} candidates...")
            enhanced_scores = self.metadata_enhancer.enhance_scores(
                user_id=user_id,
                post_ids=post_ids,
                base_scores=base_scores_array[0],
                candidates=candidates,
                content_type=content_type
            )
            logger.info(f"Metadata enhancement complete. Enhanced score range: [{enhanced_scores.min():.4f}, {enhanced_scores.max():.4f}]")
            logger.info(f"Mean enhanced score: {enhanced_scores.mean():.4f}, Std: {enhanced_scores.std():.4f}")

            # Store enhanced scores
            for i, post_id in enumerate(post_ids):
                result["enhanced_scores"][post_id] = float(enhanced_scores[i])

            # Track filtered posts (score = 0 after eligibility filtering)
            result["filtered_posts"] = self.metadata_enhancer.get_filtered_posts(
                post_ids, enhanced_scores
            )
            filtered_count = len(result["filtered_posts"])
            if filtered_count > 0:
                logger.info(f"Eligibility filtering removed {filtered_count} candidates (score=0)")
                logger.debug(f"Filtered post IDs: {result['filtered_posts'][:10]}{'...' if filtered_count > 10 else ''}")
            else:
                logger.info(f"Eligibility filtering: All {len(post_ids)} candidates passed")

            # Get RL actions if available
            if hasattr(self.metadata_enhancer, 'rl_manager'):
                try:
                    pending = getattr(self.metadata_enhancer, '_pending_actions', {})
                    user_pending = pending.get(str(user_id), {})
                    if user_pending and 'action' in user_pending:
                        result["rl_actions"] = [user_pending['action'].to_dict()]
                        logger.info(f"RL action applied for user {user_id}: {result['rl_actions']}")
                    else:
                        logger.debug(f"No pending RL actions for user {user_id}")
                except Exception as e:
                    logger.debug(f"Could not retrieve RL actions: {e}")
            else:
                logger.debug("RL manager not available - skipping RL action retrieval")

            # Apply diversity reranking (TMDB genre for ordering only - ToS compliant)
            logger.info(f"Applying diversity reranking to {len(post_ids)} candidates...")
            reranked_ids, reranked_scores = self.metadata_enhancer.apply_diversity_reranking(
                post_ids, enhanced_scores
            )

            if len(reranked_ids) > 0:
                result["diversity_applied"] = True
                # Use reranked results
                result["scored_posts"] = list(zip(reranked_ids, reranked_scores.tolist()))
                logger.info(f"Diversity reranking applied: {len(reranked_ids)} candidates reordered")
            else:
                # Fallback to score-sorted order
                scored_posts = list(zip(post_ids, enhanced_scores))
                scored_posts.sort(key=lambda x: x[1], reverse=True)
                result["scored_posts"] = scored_posts
                logger.info(f"Diversity reranking not applied - using score-sorted order for {len(scored_posts)} candidates")

            # Log final scoring summary
            remaining_candidates = len(result["scored_posts"])
            logger.info(f"=== CANDIDATE SCORING COMPLETE for user {user_id} ===")
            logger.info(f"Scoring pipeline summary:")
            logger.info(f"  - Input candidates: {len(post_ids)}")
            logger.info(f"  - Filtered out: {filtered_count}")
            logger.info(f"  - Final candidates: {remaining_candidates}")
            logger.info(f"  - Diversity applied: {result['diversity_applied']}")
            if remaining_candidates > 0:
                top_5_posts = result["scored_posts"][:5]
                logger.info(f"  - Top 5 scored posts: {[(pid, f'{score:.4f}') for pid, score in top_5_posts]}")

            return result

        except Exception as e:
            logger.error(f"=== CANDIDATE SCORING FAILED for user {user_id} ===")
            logger.error(f"Error scoring candidates: {e}", exc_info=True)
            logger.error(f"Failure context - Content type: {content_type}, Candidate count: {len(candidate_data.get('vectors', {}))}")
            # Fallback to random scores
            post_ids = list(candidate_data["vectors"].keys())
            logger.warning(f"Fallback: Assigning random scores to {len(post_ids)} candidates after scoring failure")
            for post_id in post_ids:
                score = float(np.random.random())
                result["base_scores"][post_id] = score
                result["enhanced_scores"][post_id] = score
            result["scored_posts"] = [(pid, result["enhanced_scores"][pid]) for pid in post_ids]
            logger.info(f"Fallback scoring complete: {len(result['scored_posts'])} candidates scored with random values")
            return result

    def _generate_explanations(self, user_id: str, post_ids: List[int],
                               base_scores: Dict[int, float], enhanced_scores: Dict[int, float],
                               rl_actions: List[Dict] = None, content_type: str = "posts") -> List[Dict[str, Any]]:
        """
        Generate explanations for why each post was recommended.

        This is a separate step that analyzes the scoring results and metadata
        to produce human-readable explanations.

        Args:
            user_id: User ID
            post_ids: List of recommended post IDs (in ranked order)
            base_scores: Dict mapping post_id to base ML score
            enhanced_scores: Dict mapping post_id to final enhanced score
            rl_actions: List of RL actions that were applied
            content_type: Type of content (posts/trailers)

        Returns:
            List of explanation dictionaries for each post
        """
        try:
            # Get base and final scores as lists in the same order as post_ids
            base_score_list = [base_scores.get(pid, 0.0) for pid in post_ids]
            final_score_list = [enhanced_scores.get(pid, 0.0) for pid in post_ids]

            # Get user metadata from cache
            user_metadata = self.metadata_enhancer._get_cached_metadata(f"user:{user_id}")
            logger.debug(f"Explainer user_metadata keys: {list(user_metadata.keys()) if user_metadata else 'None'}")

            # Get post metadata from cache
            post_metadata_map = {}
            for post_id in post_ids:
                post_meta = self.metadata_enhancer._get_cached_metadata(f"post:{post_id}")
                if post_meta:
                    post_metadata_map[post_id] = post_meta
            logger.debug(f"Explainer post_metadata_map: {len(post_metadata_map)} posts have metadata")

              # Fetch behaviorally inferred avoidance signals so the explainer can
            # surface why a post was downweighted, not just why it was boosted
            avoided_genre_counts = {}
            avoided_person_counts = {}
            if getattr(self.metadata_enhancer, 'avoided_signal_penalty_enabled', False):
                avoided_genre_counts = self.metadata_enhancer.avoided_signal_tracker.get_counts(
                    str(user_id), 'genre'
                )
                avoided_person_counts = self.metadata_enhancer.avoided_signal_tracker.get_counts(
                    str(user_id), 'person'
                )

            # Call the explainer
            explanation_objects = self.explainer.explain_recommendations(
                user_id=str(user_id),
                post_ids=post_ids,
                base_scores=base_score_list,
                final_scores=final_score_list,
                user_metadata=user_metadata,
                post_metadata_map=post_metadata_map,
                rl_actions=rl_actions,
                content_type=content_type,
                avoided_genre_counts=avoided_genre_counts,
                avoided_person_counts=avoided_person_counts
            )

            # Format for API response
            return self.explainer.format_response(explanation_objects, compact=False)

        except Exception as e:
            logger.warning(f"Error generating explanations: {e}")
            # Return minimal explanations on error
            return [
                {
                    "postId": pid,
                    "reason": "Recommended based on your preferences",
                    "score": enhanced_scores.get(pid, 0.0)
                }
                for pid in post_ids
            ]

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

            # Check if RL-enhanced metadata enhancer is available
            if not hasattr(self.metadata_enhancer, 'process_user_interaction'):
                logger.info(f"RL not active, interaction logged but not processed: user {user_id}, post {post_id}, type {interaction_type}")
                return {
                    "success": True,
                    "message": f"Interaction logged (RL not active): {interaction_type} for user {user_id}",
                    "rl_processed": False
                }

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

            logger.info(f"Processed RL interaction: user {user_id}, post {post_id}, type {interaction_type}")

            return {
                "success": True,
                "message": f"Processed {interaction_type} interaction for user {user_id}",
                "rl_processed": True
            }

        except Exception as e:
            logger.error(f"Error processing user interaction: {e}", exc_info=True)
            return {"error": f"Error processing interaction: {str(e)}"}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics including RL stats"""
        try:
            with self._cursor_lock:
                cursors_tracked_count = len(self.cursor_tracker)

            base_stats = {
                "modelLoaded": self.two_tower_model is not None,
                "cacheStatus": {
                    "redisConnected": self.redis_client.ping()
                },
                "serviceConnections": {
                    "socialService": self.social_service_url,
                    "commentService": self.comment_service_url,
                    "apiBaseUrl": self.api_base_url
                },
                "cursorsTracked": cursors_tracked_count,
                "version": os.environ.get("SERVICE_VERSION", "1.1.0")
            }

            # Candidate pool stats
            with self._cursor_lock:
                cursors_tracked = len(self.cursor_tracker)
                cursor_user_sample = list(self.cursor_tracker.keys())[:10]
            base_stats["candidatePool"] = {
                "enabled": self.pool_enabled,
                "cap": 200,
                "usersTracked": cursors_tracked,
                "poolSizes": {
                    uid: {
                        ct: self.pool_cache.get_pool_size(uid, ct)
                        for ct in ["posts", "trailers"]
                    }
                    for uid in cursor_user_sample
                }
            }

            # Check RL status
            is_rl_enhancer = hasattr(self.metadata_enhancer, 'get_rl_stats')
            base_stats["rl_active"] = is_rl_enhancer and self.rl_enabled

            # Add RL statistics
            try:
                if is_rl_enhancer:
                    rl_stats = self.metadata_enhancer.get_rl_stats()
                    base_stats["rl_enhancement"] = rl_stats.get("rl_enhancement", {})
                    base_stats["metadata_enhancement"] = {
                        "cache_size": rl_stats.get("cache_size", 0),
                        "boost_factors": rl_stats.get("boost_factors", {}),
                        "redis_available": rl_stats.get("redis_available", False)
                    }
                else:
                    # Basic metadata enhancer stats
                    enhancement_stats = self.metadata_enhancer.get_enhancement_stats()
                    base_stats["rl_enhancement"] = {
                        "enabled": False,
                        "mode": "metadata_only",
                        "message": "RL agent not active, using basic metadata enhancement"
                    }
                    base_stats["metadata_enhancement"] = {
                        "cache_size": enhancement_stats.get("cache_size", 0),
                        "boost_factors": enhancement_stats.get("boost_factors", {}),
                        "redis_available": enhancement_stats.get("redis_available", False)
                    }
            except Exception as e:
                logger.warning(f"Error getting enhancement stats: {e}")
                base_stats["rl_enhancement"] = {"error": str(e)}
                base_stats["metadata_enhancement"] = {"error": str(e)}

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
        include_explanations = request_data.get("includeExplanations", True)

        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()

        response = core_service.get_post_recommendations(
            user_id, limit,
            jwt_token=jwt_token,
            include_explanations=include_explanations
        )
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
        include_explanations = request_data.get("includeExplanations", True)

        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        # Extract JWT token from request headers
        jwt_token = None
        if extract_jwt_token:
            jwt_token = extract_jwt_token()

        response = core_service.get_trailer_recommendations(
            user_id, limit,
            jwt_token=jwt_token,
            include_explanations=include_explanations
        )
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


@app.route('/rl/enable', methods=['POST'])
def enable_rl():
    """Enable RL processing"""
    try:
        if not RL_AVAILABLE:
            return jsonify({"error": "RL components not available"}), 400

        if hasattr(core_service.metadata_enhancer, 'rl_manager'):
            core_service.metadata_enhancer.rl_manager.enable_rl()
            core_service.rl_enabled = True
            return jsonify({"success": True, "message": "RL processing enabled"})
        else:
            return jsonify({"error": "RL manager not initialized"}), 400
    except Exception as e:
        logger.error(f"Error enabling RL: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/rl/disable', methods=['POST'])
def disable_rl():
    """Disable RL processing"""
    try:
        if hasattr(core_service.metadata_enhancer, 'rl_manager'):
            core_service.metadata_enhancer.rl_manager.disable_rl()
            core_service.rl_enabled = False
            return jsonify({"success": True, "message": "RL processing disabled"})
        else:
            return jsonify({"success": True, "message": "RL was not active"})
    except Exception as e:
        logger.error(f"Error disabling RL: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/rl/status', methods=['GET'])
def get_rl_status():
    """Get detailed RL status"""
    try:
        status = {
            "rl_components_available": RL_AVAILABLE,
            "rl_enabled_in_config": core_service.rl_enabled if hasattr(core_service, 'rl_enabled') else False,
            "rl_enhancer_active": hasattr(core_service.metadata_enhancer, 'rl_manager'),
        }

        if hasattr(core_service.metadata_enhancer, 'rl_manager'):
            rl_manager = core_service.metadata_enhancer.rl_manager
            status["rl_manager_status"] = {
                "is_enabled": rl_manager.is_enabled,
                "learning_mode": rl_manager.learning_mode,
                "a_b_test_ratio": rl_manager.a_b_test_ratio,
                "integration_stats": rl_manager.get_integration_stats()
            }

        if hasattr(core_service.metadata_enhancer, 'get_rl_stats'):
            status["rl_stats"] = core_service.metadata_enhancer.get_rl_stats()

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting RL status: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information"""
    rl_status = {
        "available": RL_AVAILABLE,
        "enabled": core_service.rl_enabled if hasattr(core_service, 'rl_enabled') else False,
        "active": hasattr(core_service.metadata_enhancer, 'get_rl_stats')
    }

    return jsonify({
        "service": "Core Recommendations Service",
        "version": "1.1.0",
        "description": "ML recommendations with RL-enhanced metadata boosting",
        "rl_status": rl_status,
        "endpoints": {
            "health": "/health",
            "recommendations": "/recommendations",
            "post_recommendations": "/recommendations/posts",
            "trailer_recommendations": "/recommendations/trailers",
            "social_recommendations": "/recommendations/social",
            "interactions": "/interactions (RL feedback)",
            "stats": "/stats"
        },
        "features": {
            "two_tower_model": "User-Post embedding similarity scoring",
            "metadata_enhancement": "Language, genre, cast/crew, recency boosting",
            "rl_enhancement": "Adaptive boost factors based on user interactions" if rl_status["active"] else "Disabled",
            "social_enhancement": "Optional social signal boosting via social service"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('CORE_SERVICE_PORT', os.environ.get('PORT', 5000)))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting Core Recommendations Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)