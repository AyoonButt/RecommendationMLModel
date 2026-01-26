import logging
import os
import time
from typing import List, Dict, Any, Optional

import numpy as np
import redis
import requests
from flask import Flask, request, jsonify
from flask.cli import load_dotenv

from ContentFilter import create_content_filter
from MetadataEnhancer import MetadataEnhancer
from MockRedis import MockRedis
from SocialSignalProcessor import create_social_signal_processor
from TwoTower import TwoTowerModel, compute_scores, compute_socially_enhanced_scores, compute_dual_preference_scores, \
    compute_filtered_scores

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ml-recommendation-service")

# Initialize Flask app
app = Flask(__name__)


def _error_response(message: str) -> Dict:
    """
    Create an error response.

    Args:
        message: Error message

    Returns:
        Error response dictionary
    """
    logger.error(message)
    return {
        "error": message,
        "postIds": [],
        "totalCount": 0
    }


class MLRecommendationService:
    """Service that receives recommendation requests and returns results using TwoTower model."""

    def __init__(self):
        """Initialize the recommendation service with the TwoTower model."""
        self.cursor_tracker = {}
        load_dotenv()

        # Spring API base URL
        self.api_base_url = os.environ.get('SPRING_API_URL', 'http://localhost:8080')

        # AWS ElastiCache Serverless Valkey configuration
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        redis_password = os.environ.get('REDIS_PASSWORD', '')
        redis_ssl = os.environ.get('REDIS_SSL', 'False').lower() == 'true'
        redis_timeout = int(os.environ.get('REDIS_TIMEOUT', 10))

        # Connect to Redis/Valkey
        logger.info(f"Connecting to AWS ElastiCache Serverless Valkey at {redis_host}:{redis_port}")

        is_local_dev = os.environ.get('LOCAL_DEV', 'True').lower() == 'true'

        if is_local_dev:
            logger.info("Using MockRedis for local development")
            self.redis_client = MockRedis(decode_responses=True)
        else:
            logger.info(f"Connecting to Redis/Valkey at {redis_host}:{redis_port}")
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
                self.redis_client.client_setname("ml-recommendation-service")
                logger.info("Successfully connected to Redis/Valkey")
            except Exception as e:
                logger.error(f"Failed to connect to Redis/Valkey: {e}")
                logger.info("Falling back to MockRedis")
                self.redis_client = MockRedis(decode_responses=True)

        # Cache keys for vectors
        self.user_vector_key_prefix = "user_vector:"
        self.post_vector_key_prefix = "post_vector:"
        self.vector_cache_ttl = 3600  # 1 hour

        # Initialize metadata enhancer
        self.metadata_enhancer = MetadataEnhancer(self.api_base_url, self.redis_client)

        # Initialize social signal processor
        social_config = {
            "api_base_url": f"{self.api_base_url}/api",
            "bert_service_url": os.environ.get('BERT_SERVICE_URL', 'http://localhost:8080'),
            "cache_size": int(os.environ.get('SOCIAL_CACHE_SIZE', '1000')),
            "social_weight": float(os.environ.get('SOCIAL_WEIGHT', '0.25'))
        }
        self.social_processor = create_social_signal_processor(social_config)
        logger.info(f"Initialized social signal processor with weight {social_config['social_weight']}")

        # Initialize content filter
        filter_config = {
            "api_base_url": f"{self.api_base_url}/api",
            "toxicity_service_url": os.environ.get('TOXICITY_SERVICE_URL', 'http://localhost:8081'),
            "cache_size": int(os.environ.get('FILTER_CACHE_SIZE', '2000'))
        }
        self.content_filter = create_content_filter(filter_config)
        logger.info(f"Initialized content filter with cache size {filter_config['cache_size']}")

        self.two_tower_model = None
        self._load_model()

    def _load_model(self):
        """Load the TwoTower model."""
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

    def get_recommendations(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a recommendation request and return recommended posts.

        Args:
            request_data: Dictionary containing the recommendation request parameters
                - userId: User ID (required)
                - contentType: Type of content ("posts" or "trailers")
                - limit: Number of recommendations (default: 20)

        Returns:
            Dictionary containing:
                - postIds: List of recommended post IDs
                - scores: List of recommendation scores
                - totalCount: Total count of recommendations
        """
        try:
            # Extract essential request parameters
            user_id = request_data.get("userId")
            if not user_id:
                return _error_response("userId is required")

            content_type = request_data.get("contentType", "posts")
            limit = request_data.get("limit", 20)

            logger.info(f"Processing recommendation request for user {user_id}, content type: {content_type}")

            # Use the appropriate specialized function based on content type
            if content_type == "trailers":
                result = self.get_trailer_recommendations(user_id=user_id, limit=limit)
            else:
                result = self.get_post_recommendations(user_id=user_id, limit=limit)

            # Add metadata to response
            result["contentType"] = content_type
            return result

        except Exception as e:
            logger.error(f"Error processing recommendation request: {str(e)}", exc_info=True)
            return _error_response(f"Error processing recommendation request: {str(e)}")

    def _fetch_candidate_vectors(self, user_id: str, limit: int, content_type: str = "POSTS",
                                 cursor: str = None, high_quality_cursor: str = None) -> Dict[str, Any]:
        """
        Fetch candidate vectors from the new Spring API endpoint.

        Returns:
            Dictionary containing:
                - vectors: Map of post_id -> vector
                - nextCursor: Next cursor for pagination
                - nextHighQualityCursor: Next cursor for high-quality content
                - hasMore: Boolean indicating if more results available
                - distribution: Information about candidate distribution
        """
        try:
            logger.info(f"Fetching candidates for user {user_id} with content_type: {content_type}")

            url = f"{self.api_base_url}/candidates/{user_id}"
            params = {
                "limit": limit,
                "contentType": content_type.upper(),
                "includeNewHighQuality": True,
                "newContentRatio": 0.3,  # 30% high-quality content
                "interactionLookbackDays": 30
            }

            # Add cursors if provided
            if cursor:
                params["cursor"] = cursor
            if high_quality_cursor:
                params["highQualityCursor"] = high_quality_cursor

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                response_data = response.json()

                # Extract vectors and convert to proper format
                vectors = response_data.get("vectors", {})

                # Convert string keys to integers and lists to numpy arrays
                converted_vectors = {}
                for post_id_str, vector_list in vectors.items():
                    try:
                        post_id = int(post_id_str)
                        vector_array = np.array(vector_list, dtype=np.float32)
                        converted_vectors[post_id] = vector_array
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error converting vector for post {post_id_str}: {e}")
                        continue

                # Extract candidate result information
                candidate_result = response_data.get("candidateResult", {})

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

                logger.info(f"Fetched {len(converted_vectors)} candidates for user {user_id}")
                return result

            elif response.status_code == 204:  # No content
                logger.info(f"No candidates available for user {user_id}")
                return {"vectors": {}, "nextCursor": None, "hasMore": False}
            else:
                logger.warning(f"API returned status {response.status_code} for candidate vectors")
                return {"vectors": {}, "nextCursor": None, "hasMore": False}

        except Exception as e:
            logger.error(f"Error fetching candidate vectors: {str(e)}")
            return {"vectors": {}, "nextCursor": None, "hasMore": False}

    def _get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Get user vector from Redis or API."""
        # Try to get from Redis first
        redis_key = f"{self.user_vector_key_prefix}{user_id}"
        cached_vector = self.redis_client.get(redis_key)

        if cached_vector:
            vector = np.frombuffer(cached_vector, dtype=np.float32)
            return vector

        # Get from API if not in Redis
        try:
            url = f"{self.api_base_url}/api/recommendations/users/{user_id}/vector"
            response = requests.get(url, timeout=5)

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
        """
        Score candidate posts using the Two-Tower model with metadata enhancement.

        Args:
            user_id: User ID
            user_vector: User's feature vector
            candidate_data: Candidate data from API including vectors and candidates
            content_type: Type of content (posts/trailers)

        Returns:
            List of (post_id, score) tuples sorted by score descending
        """
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

    def get_post_recommendations(self, user_id: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get post recommendations using the new candidate fetching structure.
        """
        try:
            start_time = time.time()

            # Get user vector
            user_vector = self._get_user_vector(user_id)
            if user_vector is None:
                return _error_response(f"Could not retrieve vector for user {user_id}")

            # Get cursors for this user and content type
            user_cursors = self.cursor_tracker.get(user_id, {}).get("posts", {})
            cursor = user_cursors.get("cursor")
            high_quality_cursor = user_cursors.get("highQualityCursor")

            # Fetch candidates with enhanced structure
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

    def get_trailer_recommendations(self, user_id: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get trailer recommendations using the new candidate fetching structure.
        """
        try:
            start_time = time.time()

            # Get user vector
            user_vector = self._get_user_vector(user_id)
            if user_vector is None:
                return _error_response(f"Could not retrieve vector for user {user_id}")

            # Get cursors for this user and content type
            user_cursors = self.cursor_tracker.get(user_id, {}).get("trailers", {})
            cursor = user_cursors.get("cursor")
            high_quality_cursor = user_cursors.get("highQualityCursor")

            # Fetch candidates with enhanced structure
            candidate_data = self._fetch_candidate_vectors(
                user_id=user_id,
                limit=limit * 2,  # Get more candidates for better selection
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

    def get_socially_enhanced_recommendations(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a socially enhanced recommendation request using social signals and sentiment analysis.

        Args:
            request_data: Dictionary containing the recommendation request parameters
                - userId: User ID (required)
                - contentType: Type of content ("posts" or "trailers")  
                - limit: Number of recommendations (default: 20)
                - includeSocialSignals: Whether to include social signals (default: True)
                - socialWeight: Weight for social signals 0.0-1.0 (default: 0.25)

        Returns:
            Dictionary containing recommended posts with social enhancement metadata
        """
        start_time = time.time()
        
        try:
            # Extract parameters
            user_id = request_data.get("userId")
            content_type = request_data.get("contentType", "posts").lower()
            limit = min(request_data.get("limit", 20), 100)
            include_social = request_data.get("includeSocialSignals", True)
            social_weight = min(max(request_data.get("socialWeight", 0.25), 0.0), 1.0)
            
            if not user_id:
                return _error_response("userId is required")

            logger.info(f"Getting socially enhanced recommendations for user {user_id}, "
                       f"content: {content_type}, social_weight: {social_weight}")

            # Get user vector
            user_vector = self._get_user_vector(user_id)
            if user_vector is None:
                return _error_response(f"User vector not found for user {user_id}")

            # Fetch candidates
            candidate_data = self._fetch_candidate_vectors(
                user_id=user_id,
                limit=limit * 2,  # Get more candidates for better selection
                content_type=content_type.upper()
            )

            if not candidate_data["vectors"]:
                return _error_response(f"No candidates found for user {user_id}")

            post_ids = list(candidate_data["vectors"].keys())
            post_embeddings = np.array(list(candidate_data["vectors"].values()))
            user_embeddings = user_vector.reshape(1, -1)

            # Apply social enhancement if enabled
            if include_social and self.social_processor:
                # Update social processor weight
                self.social_processor.social_weight = social_weight
                self.social_processor.personal_weight = 1.0 - social_weight
                
                enhanced_scores, social_metadata = compute_socially_enhanced_scores(
                    user_embeddings=user_embeddings,
                    post_embeddings=post_embeddings,
                    user_id=user_id,
                    post_ids=post_ids,
                    social_processor=self.social_processor,
                    content_type=content_type,
                    candidates=candidate_data.get("candidates", [])
                )
                scores = enhanced_scores.flatten()
            else:
                # Use base scoring
                scores = compute_scores(
                    user_embeddings=user_embeddings,
                    post_embeddings=post_embeddings,
                    content_type=content_type,
                    candidates=candidate_data.get("candidates", [])
                ).flatten()
                social_metadata = {"social_enhancement": False}

            # Sort by scores and return top results
            scored_posts = list(zip(post_ids, scores))
            scored_posts.sort(key=lambda x: x[1], reverse=True)
            
            top_results = scored_posts[:limit]
            final_post_ids = [int(post_id) for post_id, _ in top_results]
            final_scores = [float(score) for _, score in top_results]

            result = {
                "postIds": final_post_ids,
                "scores": final_scores,
                "totalCount": len(final_post_ids),
                "processingTime": time.time() - start_time,
                "socialEnhancement": include_social,
                "socialWeight": social_weight,
                "socialMetadata": social_metadata,
                "hasMore": candidate_data["hasMore"],
                "nextCursor": candidate_data["nextCursor"]
            }

            logger.info(f"Generated {len(final_post_ids)} socially enhanced recommendations for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error getting socially enhanced recommendations: {str(e)}", exc_info=True)
            return _error_response(f"Error getting socially enhanced recommendations: {str(e)}")

    def get_dual_preference_recommendations(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process recommendations with dual preference boosting (likes/saves vs not interested).

        Args:
            request_data: Dictionary containing the recommendation request parameters
                - userId: User ID (required)
                - contentType: Type of content ("posts" or "trailers")
                - limit: Number of recommendations (default: 20)
                - positiveInteractions: List of post IDs with positive interactions
                - negativeInteractions: List of post IDs with negative interactions

        Returns:
            Dictionary containing recommendations with dual preference boosting
        """
        start_time = time.time()
        
        try:
            # Extract parameters
            user_id = request_data.get("userId")
            content_type = request_data.get("contentType", "posts").lower()
            limit = min(request_data.get("limit", 20), 100)
            positive_interactions = request_data.get("positiveInteractions", [])
            negative_interactions = request_data.get("negativeInteractions", [])
            
            if not user_id:
                return _error_response("userId is required")

            logger.info(f"Getting dual preference recommendations for user {user_id}, "
                       f"positive: {len(positive_interactions)}, negative: {len(negative_interactions)}")

            # Get user vector
            user_vector = self._get_user_vector(user_id)
            if user_vector is None:
                return _error_response(f"User vector not found for user {user_id}")

            # Fetch candidates
            candidate_data = self._fetch_candidate_vectors(
                user_id=user_id,
                limit=limit * 2,
                content_type=content_type.upper()
            )

            if not candidate_data["vectors"]:
                return _error_response(f"No candidates found for user {user_id}")

            post_ids = list(candidate_data["vectors"].keys())
            post_embeddings = np.array(list(candidate_data["vectors"].values()))
            user_embeddings = user_vector.reshape(1, -1)

            # Apply dual preference boosting
            enhanced_scores = compute_dual_preference_scores(
                user_embeddings=user_embeddings,
                post_embeddings=post_embeddings,
                user_id=user_id,
                post_ids=post_ids,
                positive_interactions=positive_interactions,
                negative_interactions=negative_interactions,
                content_type=content_type
            )
            
            scores = enhanced_scores.flatten()

            # Sort by scores and return top results
            scored_posts = list(zip(post_ids, scores))
            scored_posts.sort(key=lambda x: x[1], reverse=True)
            
            top_results = scored_posts[:limit]
            final_post_ids = [int(post_id) for post_id, _ in top_results]
            final_scores = [float(score) for _, score in top_results]

            result = {
                "postIds": final_post_ids,
                "scores": final_scores,
                "totalCount": len(final_post_ids),
                "processingTime": time.time() - start_time,
                "dualPreferenceApplied": True,
                "positiveBoosts": len([pid for pid in final_post_ids if pid in positive_interactions]),
                "negativePenalties": len([pid for pid in final_post_ids if pid in negative_interactions]),
                "hasMore": candidate_data["hasMore"],
                "nextCursor": candidate_data["nextCursor"]
            }

            logger.info(f"Generated {len(final_post_ids)} dual preference recommendations for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error getting dual preference recommendations: {str(e)}", exc_info=True)
            return _error_response(f"Error getting dual preference recommendations: {str(e)}")

    def get_filtered_recommendations(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process recommendations with comprehensive content filtering.

        Args:
            request_data: Dictionary containing the recommendation request parameters
                - userId: User ID (required)
                - contentType: Type of content ("posts" or "trailers")
                - limit: Number of recommendations (default: 20)
                - enableContentFilter: Whether to apply content filtering (default: True)
                - applySocialBoost: Whether to apply social boosting (default: True)

        Returns:
            Dictionary containing filtered recommendations
        """
        start_time = time.time()
        
        try:
            # Extract parameters
            user_id = request_data.get("userId")
            content_type = request_data.get("contentType", "posts").lower()
            limit = min(request_data.get("limit", 20), 100)
            enable_filter = request_data.get("enableContentFilter", True)
            apply_social = request_data.get("applySocialBoost", True)
            
            if not user_id:
                return _error_response("userId is required")

            logger.info(f"Getting filtered recommendations for user {user_id}, "
                       f"content: {content_type}, filter: {enable_filter}, social: {apply_social}")

            # Get user vector
            user_vector = self._get_user_vector(user_id)
            if user_vector is None:
                return _error_response(f"User vector not found for user {user_id}")

            # Fetch candidates
            candidate_data = self._fetch_candidate_vectors(
                user_id=user_id,
                limit=limit * 2,  # Get more candidates for better selection after filtering
                content_type=content_type.upper()
            )

            if not candidate_data["vectors"]:
                return _error_response(f"No candidates found for user {user_id}")

            post_ids = list(candidate_data["vectors"].keys())
            post_embeddings = np.array(list(candidate_data["vectors"].values()))
            user_embeddings = user_vector.reshape(1, -1)

            # Apply filtering and scoring
            if enable_filter and self.content_filter:
                filtered_scores, filtered_post_ids, filter_metadata = compute_filtered_scores(
                    user_embeddings=user_embeddings,
                    post_embeddings=post_embeddings,
                    user_id=user_id,
                    post_ids=post_ids,
                    content_filter=self.content_filter,
                    content_type=content_type,
                    apply_social_boost=apply_social,
                    social_processor=self.social_processor if apply_social else None
                )
                
                # Re-sort and limit results
                if len(filtered_scores) > 0:
                    scored_posts = list(zip(filtered_post_ids, filtered_scores))
                    scored_posts.sort(key=lambda x: x[1], reverse=True)
                    
                    top_results = scored_posts[:limit]
                    final_post_ids = [int(post_id) for post_id, _ in top_results]
                    final_scores = [float(score) for _, score in top_results]
                else:
                    final_post_ids = []
                    final_scores = []
            else:
                # Use standard scoring without filtering
                if apply_social and self.social_processor:
                    enhanced_scores, social_metadata = compute_socially_enhanced_scores(
                        user_embeddings=user_embeddings,
                        post_embeddings=post_embeddings,
                        user_id=user_id,
                        post_ids=post_ids,
                        social_processor=self.social_processor,
                        content_type=content_type
                    )
                    scores = enhanced_scores.flatten()
                else:
                    scores = compute_scores(
                        user_embeddings=user_embeddings,
                        post_embeddings=post_embeddings,
                        content_type=content_type
                    ).flatten()
                    social_metadata = {"social_enhancement": False}

                # Sort and limit
                scored_posts = list(zip(post_ids, scores))
                scored_posts.sort(key=lambda x: x[1], reverse=True)
                
                top_results = scored_posts[:limit]
                final_post_ids = [int(post_id) for post_id, _ in top_results]
                final_scores = [float(score) for _, score in top_results]
                
                filter_metadata = {"content_filtering_applied": False}

            result = {
                "postIds": final_post_ids,
                "scores": final_scores,
                "totalCount": len(final_post_ids),
                "processingTime": time.time() - start_time,
                "contentFilteringEnabled": enable_filter,
                "socialBoostEnabled": apply_social,
                "filterMetadata": filter_metadata,
                "hasMore": candidate_data["hasMore"],
                "nextCursor": candidate_data["nextCursor"]
            }

            logger.info(f"Generated {len(final_post_ids)} filtered recommendations for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error getting filtered recommendations: {str(e)}", exc_info=True)
            return _error_response(f"Error getting filtered recommendations: {str(e)}")

    def update_social_interaction(self, user_id: int, post_id: int, interaction_type: str, strength: float = 1.0):
        """
        Update user interaction for social signal processing.

        Args:
            user_id: User ID
            post_id: Post ID
            interaction_type: Type of interaction (like, save, not_interested, etc.)
            strength: Interaction strength (0.0 to 1.0)
        """
        try:
            if self.social_processor:
                from SocialSignalProcessor import InteractionType
                
                # Map string to enum
                type_mapping = {
                    "like": InteractionType.LIKE,
                    "save": InteractionType.SAVE,
                    "not_interested": InteractionType.NOT_INTERESTED,
                    "comment_positive": InteractionType.COMMENT_POSITIVE,
                    "comment_negative": InteractionType.COMMENT_NEGATIVE,
                    "view_time_high": InteractionType.VIEW_TIME_HIGH,
                    "view_time_low": InteractionType.VIEW_TIME_LOW
                }
                
                interaction_enum = type_mapping.get(interaction_type.lower())
                if interaction_enum:
                    self.social_processor.update_user_interaction(
                        user_id=user_id,
                        post_id=post_id,
                        interaction_type=interaction_enum,
                        strength=strength
                    )
                    logger.info(f"Updated social interaction: user {user_id}, post {post_id}, type {interaction_type}")
                else:
                    logger.warning(f"Unknown interaction type: {interaction_type}")
        except Exception as e:
            logger.error(f"Error updating social interaction: {e}")

    def get_social_stats(self) -> Dict[str, Any]:
        """Get social signal processor statistics."""
        try:
            if self.social_processor:
                return self.social_processor.get_cache_stats()
            return {"error": "Social processor not initialized"}
        except Exception as e:
            logger.error(f"Error getting social stats: {e}")
            return {"error": str(e)}

    def get_filter_stats(self) -> Dict[str, Any]:
        """Get content filter statistics."""
        try:
            if self.content_filter:
                return self.content_filter.get_filter_stats()
            return {"error": "Content filter not initialized"}
        except Exception as e:
            logger.error(f"Error getting filter stats: {e}")
            return {"error": str(e)}

    def clear_filter_cache(self):
        """Clear content filter caches."""
        try:
            if self.content_filter:
                self.content_filter.clear_caches()
                logger.info("Cleared content filter caches")
        except Exception as e:
            logger.error(f"Error clearing filter cache: {e}")


# Create service instance
ml_service = MLRecommendationService()


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint for getting recommendations."""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400

        response = ml_service.get_recommendations(request_data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "postIds": [], "totalCount": 0}), 500


@app.route('/recommendations/social', methods=['POST'])
def get_socially_enhanced_recommendations():
    """API endpoint for getting socially enhanced recommendations."""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400

        response = ml_service.get_socially_enhanced_recommendations(request_data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in social recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "postIds": [], "totalCount": 0}), 500


@app.route('/recommendations/dual-preference', methods=['POST'])
def get_dual_preference_recommendations():
    """API endpoint for getting dual preference recommendations."""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400

        response = ml_service.get_dual_preference_recommendations(request_data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in dual preference recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "postIds": [], "totalCount": 0}), 500


@app.route('/interactions/update', methods=['POST'])
def update_social_interaction():
    """API endpoint for updating social interactions."""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400

        user_id = request_data.get("userId")
        post_id = request_data.get("postId")
        interaction_type = request_data.get("interactionType")
        strength = request_data.get("strength", 1.0)

        if not all([user_id, post_id, interaction_type]):
            return jsonify({"error": "userId, postId, and interactionType are required"}), 400

        ml_service.update_social_interaction(user_id, post_id, interaction_type, strength)
        return jsonify({"success": True, "message": "Interaction updated successfully"})
    except Exception as e:
        logger.error(f"Error in update interaction endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/recommendations/filtered', methods=['POST'])
def get_filtered_recommendations():
    """API endpoint for getting filtered recommendations."""
    try:
        request_data = request.json
        if not request_data:
            return jsonify({"error": "No request data provided"}), 400

        response = ml_service.get_filtered_recommendations(request_data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in filtered recommendations endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "postIds": [], "totalCount": 0}), 500


@app.route('/social/stats', methods=['GET'])
def get_social_stats():
    """API endpoint for getting social statistics."""
    try:
        stats = ml_service.get_social_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in social stats endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/filter/stats', methods=['GET'])
def get_filter_stats():
    """API endpoint for getting content filter statistics."""
    try:
        stats = ml_service.get_filter_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in filter stats endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/filter/cache/clear', methods=['POST'])
def clear_filter_cache():
    """API endpoint for clearing content filter cache."""
    try:
        ml_service.clear_filter_cache()
        return jsonify({"success": True, "message": "Filter cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error in clear filter cache endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        social_stats = ml_service.get_social_stats()
        filter_stats = ml_service.get_filter_stats()
        return jsonify({
            "status": "healthy",
            "modelLoaded": ml_service.two_tower_model is not None,
            "cacheStatus": {
                "redisConnected": ml_service.redis_client.ping()
            },
            "socialProcessor": {
                "enabled": ml_service.social_processor is not None,
                "stats": social_stats
            },
            "contentFilter": {
                "enabled": ml_service.content_filter is not None,
                "stats": filter_stats
            },
            "version": os.environ.get("SERVICE_VERSION", "1.0.0")
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting ML Recommendation Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)