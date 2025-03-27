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
from OffsetTracking import RecommendationOffsetTracker
from TwoTower import TwoTowerModel, compute_scores

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
        load_dotenv()

        # Spring API base URL
        self.api_base_url = os.environ.get('SPRING_API_URL', 'http://localhost:8080')

        # AWS ElastiCache Serverless Valkey configuration
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        redis_password = os.environ.get('REDIS_PASSWORD', '')  # Empty string for no password
        redis_ssl = os.environ.get('REDIS_SSL', 'False').lower() == 'true'
        redis_timeout = int(os.environ.get('REDIS_TIMEOUT', 10))

        # Connect to AWS ElastiCache Serverless Valkey
        logger.info(f"Connecting to AWS ElastiCache Serverless Valkey at {redis_host}:{redis_port}")



        is_local_dev = os.environ.get('LOCAL_DEV', 'True').lower() == 'true'

        if is_local_dev:
            logger.info("Using MockRedis for local development")
            self.redis_client = MockRedis(decode_responses=True)
        else:
            # Try to connect to the real Redis/Valkey
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

                # Only add password if it's set
                if redis_password:
                    connection_args['password'] = redis_password

                self.redis_client = redis.Redis(**connection_args)

                # Test the connection
                self.redis_client.ping()

                # Set client name
                self.redis_client.client_setname("ml-recommendation-service")
                logger.info("Successfully connected to Redis/Valkey")
            except Exception as e:
                logger.error(f"Failed to connect to Redis/Valkey: {e}")
                logger.info("Falling back to MockRedis")
                self.redis_client = MockRedis(decode_responses=True)

        self.offset_tracker = RecommendationOffsetTracker()

        # Cache keys for vectors
        self.user_vector_key_prefix = "user_vector:"
        self.post_vector_key_prefix = "post_vector:"
        self.vector_cache_ttl = 3600  # 1 hour

        # New cache for candidate vectors per user
        self.candidate_cache = {}
        self.MAX_CACHE_ENTRIES = int(os.environ.get('MAX_CACHE_ENTRIES', '1000'))

        self.metadata_cache = {}
        self.metadata_cache_ttl = 3600  # 1 hour in seconds

        self.two_tower_model = None
        self._load_model()

    def _load_model(self):
        """Load the TwoTower model."""
        try:
            logger.info("Loading TwoTower model...")

            # Get model paths from environment variables
            model_dir = os.environ.get('MODEL_DIR', './model_checkpoints')
            user_model_path = os.path.join(model_dir, os.environ.get('USER_MODEL', 'user_tower_latest.h5'))
            post_model_path = os.path.join(model_dir, os.environ.get('POST_MODEL', 'post_tower_latest.h5'))

            # Initialize TwoTower model
            self.two_tower_model = TwoTowerModel(
                user_feature_dim=int(os.environ.get('USER_FEATURE_DIM', '64')),
                post_feature_dim=int(os.environ.get('POST_FEATURE_DIM', '64')),
                embedding_dim=int(os.environ.get('EMBEDDING_DIM', '32')),
                hidden_dims=[int(dim) for dim in os.environ.get('HIDDEN_DIMS', '128,64').split(',')]
            )

            # Load pre-trained models if available
            if os.path.exists(user_model_path) and os.path.exists(post_model_path):
                self.two_tower_model.load_models(user_model_path, post_model_path)
                logger.info(f"Loaded pre-trained models from {user_model_path} and {post_model_path}")
            else:
                logger.warning("Pre-trained models not found, using initialized model")

            logger.info("TwoTower model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.two_tower_model = None

    def _ensure_user_cache(self, user_id: str):
        """Ensure cache exists for user and is within size limits."""
        if user_id not in self.candidate_cache:
            self.candidate_cache[user_id] = OrderedDict()

        # Check if cache exceeds maximum size and trim if needed
        if len(self.candidate_cache[user_id]) > self.MAX_CACHE_ENTRIES:
            # Remove oldest entries (FIFO)
            excess = len(self.candidate_cache[user_id]) - self.MAX_CACHE_ENTRIES
            for _ in range(excess):
                self.candidate_cache[user_id].popitem(last=False)

            logger.info(f"Trimmed {excess} entries from cache for user {user_id}")

    def get_recommendations(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a recommendation request and return recommended posts.

        Args:
            request_data: Dictionary containing the recommendation request parameters
                - userId: User ID (required)
                - contentType: Type of content to prioritize ("posts" or "trailers")
                - page: Page number (default: 0)
                - pageSize: Page size (default: 20)
                - loosenFiltering: Whether to loosen filtering criteria (default: False)
                - loosenAttempt: Current loosening attempt level (default: 0)

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
            page_size = request_data.get("pageSize", 20)
            loosen_filtering = request_data.get("loosenFiltering", False)
            loosen_attempt = request_data.get("loosenAttempt", 0)

            logger.info(
                f"Processing recommendation request for user {user_id}, content type: {content_type}")

            # Use the appropriate specialized function based on content type
            if content_type == "trailers":
                result = self.get_top_trailer_recommendations(
                    user_id=user_id,
                    limit=page_size,
                    loosen_filtering=loosen_filtering
                )
            else:
                result = self.get_top_post_recommendations(
                    user_id=user_id,
                    limit=page_size,
                    loosen_filtering=loosen_filtering
                )

            # Add any missing fields from the original response format
            if "loosenFiltering" not in result:
                result["loosenFiltering"] = loosen_filtering
            if "loosenAttempt" not in result:
                result["loosenAttempt"] = loosen_attempt
            if "contentType" not in result:
                result["contentType"] = content_type

            return result

        except Exception as e:
            logger.error(f"Error processing recommendation request: {str(e)}", exc_info=True)
            return _error_response(f"Error processing recommendation request: {str(e)}")

    def _fetch_candidate_vectors(self, user_id, limit, loosen_filtering=False, loosen_attempt=0, content_type="posts"):
        """Fetch candidate vectors from the API with managed offsets for variety."""
        try:
            # Get next offset from tracker
            offset = self.offset_tracker.get_next_offset(user_id, page_size=limit)

            logger.info(
                f"Fetching candidates for user {user_id} with offset {offset} (cycle: {self.offset_tracker.user_offsets[user_id]['current_cycle']})")

            url = f"{self.api_base_url}/api/recommendations/{user_id}/candidates"
            params = {
                "limit": limit,
                "loosenFiltering": loosen_filtering,
                "loosenAttempt": loosen_attempt,
                "offset": offset,
                "contentType": content_type
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                candidates = response.json()
                logger.info(f"Fetched {len(candidates)} candidates with offset {offset}")
                return candidates
            else:
                logger.warning(f"API returned status {response.status_code} for candidate vectors")
                return {}

        except Exception as e:
            logger.error(f"Error fetching candidate vectors: {str(e)}")
            return {}

    def _get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get user vector from Redis or API.
        """
        # Try to get from Redis first
        redis_key = f"{self.user_vector_key_prefix}{user_id}"
        cached_vector = self.redis_client.get(redis_key)

        if cached_vector:
            # Deserialize and return
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
                self.redis_client.setex(
                    redis_key,
                    self.vector_cache_ttl,
                    vector.tobytes()
                )

                return vector

        except Exception as e:
            logger.warning(f"Error retrieving user vector from API: {e}")

        # If vector can't be retrieved, return a default one
        logger.info(f"Using default vector for user {user_id}")
        feature_dim = int(os.environ.get('USER_FEATURE_DIM', '64'))
        default_vector = np.random.rand(feature_dim).astype(np.float32)

        # Cache default vector
        self.redis_client.setex(
            redis_key,
            self.vector_cache_ttl,
            default_vector.tobytes()
        )

        return default_vector

    def _get_post_vector(self, post_id: int) -> np.ndarray:
        """
        Get post vector from Redis or API.
        """
        # Try to get from Redis first
        redis_key = f"{self.post_vector_key_prefix}{post_id}"
        cached_vector = self.redis_client.get(redis_key)

        if cached_vector:
            # Deserialize and return
            vector = np.frombuffer(cached_vector, dtype=np.float32)
            return vector

        # Get from API if not in Redis
        try:
            url = f"{self.api_base_url}/api/recommendations/posts/{post_id}/vector"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                vector_data = response.json()
                vector = np.array(vector_data, dtype=np.float32)

                # Cache in Redis
                self.redis_client.setex(
                    redis_key,
                    self.vector_cache_ttl,
                    vector.tobytes()
                )

                return vector

        except Exception as e:
            logger.warning(f"Error retrieving post vector from API: {e}")

        # If vector can't be retrieved, return a default one
        logger.info(f"Using default vector for post {post_id}")
        feature_dim = int(os.environ.get('POST_FEATURE_DIM', '64'))
        default_vector = np.random.rand(feature_dim).astype(np.float32)

        # Cache default vector
        self.redis_client.setex(
            redis_key,
            self.vector_cache_ttl,
            default_vector.tobytes()
        )

        return default_vector

    def _score_candidates_with_vectors(
            self,
            user_id: str,
            user_vector: np.ndarray,
            post_ids: List[int],
            post_vectors: List[np.ndarray],
            content_type: str = "posts"
    ) -> List[tuple]:
        """
        Score candidate posts using provided vectors with metadata boosting.
        """
        if not self.two_tower_model:
            logger.warning("TwoTower model not available, using fallback scoring")
            return [(post_id, np.random.random()) for post_id in post_ids]

        try:
            # Convert post vectors to numpy array
            post_vectors_array = np.array(post_vectors)

            # Reshape user vector
            user_batch = np.expand_dims(user_vector, axis=0)

            # Calculate base scores using compute_scores function
            base_scores = compute_scores(user_batch, post_vectors_array, content_type)

            # Apply metadata boosting if available
            try:
                # Get user metadata (with caching)
                user_metadata = self._get_cached_metadata(f"user:{user_id}")

                # Extract language preferences if available
                user_languages = {}
                if user_metadata and 'languageWeights' in user_metadata:
                    weights = user_metadata.get('languageWeights', {}).get('weights', {})
                    if isinstance(weights, dict):
                        user_languages = weights

                # Only apply boosting if we have language preferences
                if user_languages:
                    boosted_scores = []

                    for i, post_id in enumerate(post_ids):
                        score = base_scores[0][i]

                        # Get post metadata (with caching)
                        post_metadata = self._get_cached_metadata(f"post:{post_id}")

                        if post_metadata:
                            # Get post language
                            post_language = post_metadata.get('categoricalFeatures', {}).get('language')

                            # Apply language boost if there's a match
                            if post_language and post_language in user_languages:
                                language_weight = user_languages.get(post_language, 0)
                                # Boost by up to 30% based on language preference weight
                                boost_factor = 1.0 + (language_weight * 0.3)
                                score *= boost_factor

                        boosted_scores.append(score)

                    # Return boosted scores
                    return list(zip(post_ids, boosted_scores))

            except Exception as e:
                logger.warning(f"Error applying metadata boosting: {e}")

            # Fallback to base scores if boosting fails
            return list(zip(post_ids, base_scores[0]))

        except Exception as e:
            logger.error(f"Error scoring posts: {e}", exc_info=True)
            return [(post_id, np.random.random()) for post_id in post_ids]

    def _get_cached_metadata(self, key):
        """Get metadata from cache or API with caching."""
        # Check if in cache and not expired
        now = time.time()
        if key in self.metadata_cache:
            entry = self.metadata_cache[key]
            if now - entry['timestamp'] < self.metadata_cache_ttl:
                return entry['data']

        # Not in cache or expired, fetch from API
        try:
            # Parse the key to determine what to fetch
            parts = key.split(':')
            if len(parts) != 2:
                return None

            entity_type, entity_id = parts

            if entity_type == 'user':
                url = f"{self.api_base_url}/api/recommendations/users/{entity_id}/metadata"
            elif entity_type == 'post':
                url = f"{self.api_base_url}/api/recommendations/posts/{entity_id}/metadata"
            else:
                return None

            # Fetch from API
            response = requests.get(url, timeout=3)

            if response.status_code == 200:
                data = response.json()

                # Cache the result
                self.metadata_cache[key] = {
                    'data': data,
                    'timestamp': now
                }

                # Cleanup cache if it gets too large (simple approach)
                if len(self.metadata_cache) > 10000:  # Arbitrary limit
                    # Remove oldest 20% of entries
                    sorted_keys = sorted(self.metadata_cache.keys(),
                                         key=lambda k: self.metadata_cache[k]['timestamp'])
                    for old_key in sorted_keys[:int(len(sorted_keys) * 0.2)]:
                        del self.metadata_cache[old_key]

                return data

        except Exception as e:
            logger.warning(f"Error fetching metadata from API for {key}: {e}")

        return None

    def get_top_post_recommendations(self, user_id: str, limit: int = 20, loosen_filtering: bool = False) -> Dict[
        str, Any]:
        """
        Get top post recommendations by scoring candidates and calling the Spring API.
        Will automatically try to loosen filtering if results are insufficient.

        Args:
            user_id: User ID
            limit: Maximum number of recommendations to return
            loosen_filtering: Whether to loosen filtering criteria (default: False)

        Returns:
            Dictionary containing recommended post IDs, scores, and metadata
        """
        try:
            start_time = time.time()
            loosen_attempt = 0
            min_recommendations = max(1, int(limit * 0.5))  # At least 50% of requested limit
            min_score_threshold = 0.2  # Minimum acceptable average score

            # Initialize variables to store the best results
            best_post_ids = []
            best_scores = []
            best_scored_posts = []  # Store the original scored posts for final attempt

            # Flag to track if we ever got any candidates
            got_any_candidates = False

            # Maximum number of loosening attempts before reverting to unfiltered results
            MAX_LOOSEN_ATTEMPTS = 4

            # Try up to MAX_LOOSEN_ATTEMPTS levels of loosening
            while loosen_attempt < MAX_LOOSEN_ATTEMPTS:
                # Get user vector
                user_vector = self._get_user_vector(user_id)
                if user_vector is None:
                    return _error_response(f"Could not retrieve vector for user {user_id}")

                # Reset offset tracker if we're loosening filters
                if loosen_attempt > 0:
                    try:
                        self.offset_tracker.reset_offset(user_id)
                    except Exception as e:
                        # Fallback if reset_offset fails
                        print(f"WARNING: Error resetting offset for user {user_id}: {str(e)}")

                # Fetch candidate vectors with offset tracking
                candidates = self._fetch_candidate_vectors(
                    user_id,
                    100,  # Get more candidates than needed to allow for filtering
                    loosen_filtering or loosen_attempt > 0,
                    loosen_attempt,
                    "posts"  # Specifically for posts
                )

                if not candidates:
                    # Log with safe logger
                    try:
                        logger.warning(f"No candidate vectors found for user {user_id} (attempt {loosen_attempt})")
                    except:
                        print(f"WARNING: No candidate vectors found for user {user_id} (attempt {loosen_attempt})")

                    loosen_attempt += 1
                    continue

                # Mark that we found candidates at least once
                got_any_candidates = True

                # Score candidates
                post_ids = list(candidates.keys())
                post_vectors = [vector for vector in candidates.values()]

                scored_posts = self._score_candidates_with_vectors(
                    user_id,
                    user_vector,
                    post_ids,
                    post_vectors,
                    "posts"
                )

                # Store the original scored posts for this attempt
                if not best_scored_posts or len(scored_posts) > len(best_scored_posts):
                    best_scored_posts = scored_posts.copy()

                # Convert to list of scored posts for the API
                scored_posts_list = [{"postId": str(post_id), "score": float(score)} for post_id, score in scored_posts]

                # If this is our final attempt, skip filtering and return top results directly
                if loosen_attempt == MAX_LOOSEN_ATTEMPTS - 1:
                    try:
                        logger.info(
                            f"Final attempt reached for user {user_id}. Skipping filtering and returning top results directly.")
                    except:
                        print(
                            f"INFO: Final attempt reached for user {user_id}. Skipping filtering and returning top results directly.")

                    # Sort by score and take top results
                    sorted_scores = sorted(scored_posts, key=lambda x: x[1], reverse=True)
                    top_results = sorted_scores[:limit]

                    # Process the results
                    current_post_ids = [int(post_id) for post_id, _ in top_results]
                    current_scores = [float(score) for _, score in top_results]

                    result = {
                        "postIds": current_post_ids,
                        "scores": current_scores,
                        "totalCount": len(current_post_ids),
                        "processingTime": time.time() - start_time,
                        "loosenFiltering": True,
                        "loosenAttempt": loosen_attempt,
                        "unfiltered": True
                    }

                    try:
                        logger.info(
                            f"Final attempt: Returning top {len(current_post_ids)} unfiltered recommendations for user {user_id}")
                    except:
                        print(
                            f"INFO: Final attempt: Returning top {len(current_post_ids)} unfiltered recommendations for user {user_id}")

                    return result

                # Call existing Spring API for filtered recommendations
                url = f"{self.api_base_url}/api/recommendations/posts/filter"
                payload = {
                    "userId": int(user_id),
                    "scoredPosts": scored_posts_list,
                    "limit": limit
                }

                try:
                    logger.info(
                        f"Requesting filtered post recommendations for user {user_id} with {len(scored_posts_list)} scored posts (attempt {loosen_attempt})")
                except:
                    print(
                        f"INFO: Requesting filtered post recommendations for user {user_id} with {len(scored_posts_list)} scored posts (attempt {loosen_attempt})")

                response = requests.post(url, json=payload, timeout=10)

                if response.status_code != 200:
                    try:
                        logger.error(f"API returned status {response.status_code} for filtered recommendations")
                    except:
                        print(f"ERROR: API returned status {response.status_code} for filtered recommendations")

                    loosen_attempt += 1
                    continue

                # Process API response - expecting a list of ScoredPost objects
                scored_posts_response = response.json()

                # Extract just the postIds and scores from the ScoredPost objects
                current_post_ids = [int(item["postId"]) for item in scored_posts_response]
                current_scores = [float(item["score"]) for item in scored_posts_response]

                # Store these as the best results if they're better than what we have
                if len(current_post_ids) > len(best_post_ids):
                    best_post_ids = current_post_ids
                    best_scores = current_scores

                # Check if we have enough recommendations of sufficient quality
                if len(current_post_ids) >= min_recommendations and (
                        not current_scores or sum(current_scores) / len(current_scores) >= min_score_threshold):
                    # Create the final response
                    result = {
                        "postIds": current_post_ids,
                        "scores": current_scores,
                        "totalCount": len(current_post_ids),
                        "processingTime": time.time() - start_time,
                        "loosenFiltering": loosen_filtering or loosen_attempt > 0,
                        "loosenAttempt": loosen_attempt
                    }

                    try:
                        logger.info(
                            f"Got {len(current_post_ids)} filtered post recommendations for user {user_id} (attempt {loosen_attempt})")
                    except:
                        print(
                            f"INFO: Got {len(current_post_ids)} filtered post recommendations for user {user_id} (attempt {loosen_attempt})")

                    return result
                else:
                    avg_score = sum(current_scores) / len(current_scores) if current_scores else 0
                    try:
                        logger.info(
                            f"Insufficient recommendations ({len(current_post_ids)}/{limit}) or low scores (avg: {avg_score:.3f}). Trying with looser filters.")
                    except:
                        print(
                            f"INFO: Insufficient recommendations ({len(current_post_ids)}/{limit}) or low scores (avg: {avg_score:.3f}). Trying with looser filters.")

                    loosen_attempt += 1

            # If we reached here, we've exhausted all attempts but should have already returned in the final attempt
            # This is a fallback in case the final attempt logic failed for some reason

            # Handle case where we never got any candidates across all attempts
            if not got_any_candidates:
                try:
                    logger.warning(f"No candidates found for user {user_id} after {loosen_attempt} loosening attempts")
                except:
                    print(f"WARNING: No candidates found for user {user_id} after {loosen_attempt} loosening attempts")

                return {
                    "postIds": [],
                    "scores": [],
                    "totalCount": 0,
                    "processingTime": time.time() - start_time,
                    "loosenFiltering": True,
                    "loosenAttempt": loosen_attempt - 1,
                    "message": "No recommendation candidates found for this user"
                }

            # We should have candidate data but filtering eliminated everything
            # Use the best scored posts we saved and return top results directly
            if best_scored_posts:
                sorted_scores = sorted(best_scored_posts, key=lambda x: x[1], reverse=True)
                top_results = sorted_scores[:limit]

                # Process the results
                current_post_ids = [int(post_id) for post_id, _ in top_results]
                current_scores = [float(score) for _, score in top_results]

                try:
                    logger.warning(
                        f"No satisfactory filtered results. Returning {len(current_post_ids)} unfiltered results as fallback.")
                except:
                    print(
                        f"WARNING: No satisfactory filtered results. Returning {len(current_post_ids)} unfiltered results as fallback.")

                return {
                    "postIds": current_post_ids,
                    "scores": current_scores,
                    "totalCount": len(current_post_ids),
                    "processingTime": time.time() - start_time,
                    "loosenFiltering": True,
                    "loosenAttempt": loosen_attempt - 1,
                    "unfiltered": True
                }

            # Return the best filtered results we found (if any)
            try:
                logger.warning(f"No satisfactory recommendations found for user {user_id} after filtering")
            except:
                print(f"WARNING: No satisfactory recommendations found for user {user_id} after filtering")

            return {
                "postIds": best_post_ids,
                "scores": best_scores,
                "totalCount": len(best_post_ids),
                "processingTime": time.time() - start_time,
                "loosenFiltering": True,
                "loosenAttempt": loosen_attempt - 1
            }

        except Exception as e:
            # Safe error logging
            try:
                logger.error(f"Error getting filtered post recommendations: {str(e)}", exc_info=True)
            except:
                print(f"ERROR: Error getting filtered post recommendations: {str(e)}")

            return _error_response(f"Error getting filtered post recommendations: {str(e)}")

    def get_top_trailer_recommendations(self, user_id: str, limit: int = 20, loosen_filtering: bool = False) -> Dict[
        str, Any]:
        """
        Get top trailer recommendations by scoring candidates and calling the Spring API.
        Will automatically try to loosen filtering if results are insufficient.

        Args:
            user_id: User ID
            limit: Maximum number of recommendations to return
            loosen_filtering: Whether to loosen filtering criteria (default: False)

        Returns:
            Dictionary containing recommended trailer IDs, scores, and metadata
        """
        try:
            start_time = time.time()
            loosen_attempt = 0
            min_recommendations = max(1, int(limit * 0.5))  # At least 50% of requested limit
            min_score_threshold = 0.2  # Minimum acceptable average score

            # Initialize variables to store the best results
            best_post_ids = []
            best_scores = []
            best_scored_posts = []  # Store the original scored posts for final attempt

            # Flag to track if we ever got any candidates
            got_any_candidates = False

            # Maximum number of loosening attempts before reverting to unfiltered results
            MAX_LOOSEN_ATTEMPTS = 3

            # Try up to MAX_LOOSEN_ATTEMPTS levels of loosening
            while loosen_attempt < MAX_LOOSEN_ATTEMPTS:
                # Get user vector
                user_vector = self._get_user_vector(user_id)
                if user_vector is None:
                    return _error_response(f"Could not retrieve vector for user {user_id}")

                # Reset offset tracker if we're loosening filters
                if loosen_attempt > 0:
                    try:
                        self.offset_tracker.reset_offset(user_id)
                    except Exception as e:
                        # Fallback if reset_offset fails
                        print(f"WARNING: Error resetting offset for user {user_id}: {str(e)}")

                # Fetch candidate vectors with offset tracking
                candidates = self._fetch_candidate_vectors(
                    user_id,
                    100,  # Get more candidates than needed to allow for filtering
                    loosen_filtering or loosen_attempt > 0,
                    loosen_attempt,
                    "trailers"  # Specifically for trailers
                )

                if not candidates:
                    # Log with safe logger
                    try:
                        logger.warning(f"No trailer candidates found for user {user_id} (attempt {loosen_attempt})")
                    except:
                        print(f"WARNING: No trailer candidates found for user {user_id} (attempt {loosen_attempt})")

                    loosen_attempt += 1
                    continue

                # Mark that we found candidates at least once
                got_any_candidates = True

                # Score candidates
                post_ids = list(candidates.keys())
                post_vectors = [vector for vector in candidates.values()]

                scored_posts = self._score_candidates_with_vectors(
                    user_id,
                    user_vector,
                    post_ids,
                    post_vectors,
                    "trailers"
                )

                # Store the original scored posts for this attempt
                if not best_scored_posts or len(scored_posts) > len(best_scored_posts):
                    best_scored_posts = scored_posts.copy()

                # Convert to list of scored posts for the API
                scored_posts_list = [{"postId": str(post_id), "score": float(score)} for post_id, score in scored_posts]

                # If this is our final attempt, skip filtering and return top results directly
                if loosen_attempt == MAX_LOOSEN_ATTEMPTS - 1:
                    try:
                        logger.info(
                            f"Final attempt reached for user {user_id}. Skipping filtering and returning top results directly.")
                    except:
                        print(
                            f"INFO: Final attempt reached for user {user_id}. Skipping filtering and returning top results directly.")

                    # Sort by score and take top results
                    sorted_scores = sorted(scored_posts, key=lambda x: x[1], reverse=True)
                    top_results = sorted_scores[:limit]

                    # Process the results
                    current_post_ids = [int(post_id) for post_id, _ in top_results]
                    current_scores = [float(score) for _, score in top_results]

                    result = {
                        "postIds": current_post_ids,
                        "scores": current_scores,
                        "totalCount": len(current_post_ids),
                        "processingTime": time.time() - start_time,
                        "loosenFiltering": True,
                        "loosenAttempt": loosen_attempt,
                        "unfiltered": True
                    }

                    try:
                        logger.info(
                            f"Final attempt: Returning top {len(current_post_ids)} unfiltered recommendations for user {user_id}")
                    except:
                        print(
                            f"INFO: Final attempt: Returning top {len(current_post_ids)} unfiltered recommendations for user {user_id}")

                    return result

                # Call existing Spring API for filtered recommendations
                url = f"{self.api_base_url}/api/recommendations/trailers/filter"
                payload = {
                    "userId": int(user_id),
                    "scoredPosts": scored_posts_list,
                    "limit": limit
                }

                try:
                    logger.info(
                        f"Requesting filtered trailer recommendations for user {user_id} with {len(scored_posts_list)} scored posts (attempt {loosen_attempt})")
                except:
                    print(
                        f"INFO: Requesting filtered trailer recommendations for user {user_id} with {len(scored_posts_list)} scored posts (attempt {loosen_attempt})")

                response = requests.post(url, json=payload, timeout=10)

                if response.status_code != 200:
                    try:
                        logger.error(f"API returned status {response.status_code} for filtered trailer recommendations")
                    except:
                        print(f"ERROR: API returned status {response.status_code} for filtered trailer recommendations")

                    loosen_attempt += 1
                    continue

                # Process API response - expecting a list of ScoredPost objects
                scored_posts_response = response.json()

                # Extract just the postIds and scores from the ScoredPost objects
                current_post_ids = [int(item["postId"]) for item in scored_posts_response]
                current_scores = [float(item["score"]) for item in scored_posts_response]

                # Store these as the best results if they're better than what we have
                if len(current_post_ids) > len(best_post_ids):
                    best_post_ids = current_post_ids
                    best_scores = current_scores

                # Check if we have enough recommendations of sufficient quality
                if len(current_post_ids) >= min_recommendations and (
                        not current_scores or sum(current_scores) / len(current_scores) >= min_score_threshold):
                    # Create the final response
                    result = {
                        "postIds": current_post_ids,
                        "scores": current_scores,
                        "totalCount": len(current_post_ids),
                        "processingTime": time.time() - start_time,
                        "loosenFiltering": loosen_filtering or loosen_attempt > 0,
                        "loosenAttempt": loosen_attempt
                    }

                    try:
                        logger.info(
                            f"Got {len(current_post_ids)} filtered trailer recommendations for user {user_id} (attempt {loosen_attempt})")
                    except:
                        print(
                            f"INFO: Got {len(current_post_ids)} filtered trailer recommendations for user {user_id} (attempt {loosen_attempt})")

                    return result
                else:
                    avg_score = sum(current_scores) / len(current_scores) if current_scores else 0
                    try:
                        logger.info(
                            f"Insufficient trailer recommendations ({len(current_post_ids)}/{limit}) or low scores (avg: {avg_score:.3f}). Trying with looser filters.")
                    except:
                        print(
                            f"INFO: Insufficient trailer recommendations ({len(current_post_ids)}/{limit}) or low scores (avg: {avg_score:.3f}). Trying with looser filters.")

                    loosen_attempt += 1

            # If we reached here, we've exhausted all attempts but should have already returned in the final attempt
            # This is a fallback in case the final attempt logic failed for some reason

            # Handle case where we never got any candidates across all attempts
            if not got_any_candidates:
                try:
                    logger.warning(f"No candidates found for user {user_id} after {loosen_attempt} loosening attempts")
                except:
                    print(f"WARNING: No candidates found for user {user_id} after {loosen_attempt} loosening attempts")

                return {
                    "postIds": [],
                    "scores": [],
                    "totalCount": 0,
                    "processingTime": time.time() - start_time,
                    "loosenFiltering": True,
                    "loosenAttempt": loosen_attempt - 1,
                    "message": "No recommendation candidates found for this user"
                }

            # We should have candidate data but filtering eliminated everything
            # Use the best scored posts we saved and return top results directly
            if best_scored_posts:
                sorted_scores = sorted(best_scored_posts, key=lambda x: x[1], reverse=True)
                top_results = sorted_scores[:limit]

                # Process the results
                current_post_ids = [int(post_id) for post_id, _ in top_results]
                current_scores = [float(score) for _, score in top_results]

                try:
                    logger.warning(
                        f"No satisfactory filtered results. Returning {len(current_post_ids)} unfiltered results as fallback.")
                except:
                    print(
                        f"WARNING: No satisfactory filtered results. Returning {len(current_post_ids)} unfiltered results as fallback.")

                return {
                    "postIds": current_post_ids,
                    "scores": current_scores,
                    "totalCount": len(current_post_ids),
                    "processingTime": time.time() - start_time,
                    "loosenFiltering": True,
                    "loosenAttempt": loosen_attempt - 1,
                    "unfiltered": True
                }

            # Return the best filtered results we found (if any)
            try:
                logger.warning(f"No satisfactory recommendations found for user {user_id} after filtering")
            except:
                print(f"WARNING: No satisfactory recommendations found for user {user_id} after filtering")

            return {
                "postIds": best_post_ids,
                "scores": best_scores,
                "totalCount": len(best_post_ids),
                "processingTime": time.time() - start_time,
                "loosenFiltering": True,
                "loosenAttempt": loosen_attempt - 1
            }

        except Exception as e:
            # Safe error logging
            try:
                logger.error(f"Error getting filtered trailer recommendations: {str(e)}", exc_info=True)
            except:
                print(f"ERROR: Error getting filtered trailer recommendations: {str(e)}")

            return _error_response(f"Error getting filtered trailer recommendations: {str(e)}")
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
            "userCount": len(ml_service.candidate_cache),
            "maxCacheSize": ml_service.MAX_CACHE_ENTRIES,
            "redisConnected": ml_service.redis_client.ping()
        },
        "version": os.environ.get("SERVICE_VERSION", "1.0.0")
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting ML Recommendation Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)