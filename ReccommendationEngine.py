from typing import List, Dict, Any, Optional
import numpy as np
import time
import logging
import os
import requests
import redis
from collections import OrderedDict
from flask import Flask, request, jsonify
from flask.cli import load_dotenv

from MockRedis import MockRedis
from TwoTower import TwoTowerModel, compute_scores
from MetadataEnhancer import MetadataEnhancer

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


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "modelLoaded": ml_service.two_tower_model is not None,
        "cacheStatus": {
            "redisConnected": ml_service.redis_client.ping()
        },
        "version": os.environ.get("SERVICE_VERSION", "1.0.0")
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting ML Recommendation Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)