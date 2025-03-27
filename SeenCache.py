import json
import requests
import logging
from datetime import datetime
from typing import Set, List, Union
import redis
import os
from redis.retry import Retry

# Configure logging
logger = logging.getLogger("user-seen-cache")


class UserSeenCache:
    def __init__(self, api_base_url, redis_client=None, api_auth=None):
        """Initialize the cache with Redis connection.

        Args:
            api_base_url: Base URL for the API that provides user history
            redis_client: An existing Redis client instance (preferred)
            api_auth: Authentication tuple for API requests (if needed)
        """
        # Use provided Redis client or create a new one using environment variables
        if redis_client:
            self.redis_client = redis_client
        else:
            # Get Redis configuration from environment
            redis_host = os.environ.get('REDIS_HOST', 'localhost')
            redis_port = int(os.environ.get('REDIS_PORT', 6379))
            redis_password = os.environ.get('REDIS_PASSWORD', None)
            redis_ssl = os.environ.get('REDIS_SSL', 'False').lower() == 'true'
            redis_timeout = int(os.environ.get('REDIS_TIMEOUT', 5))

            # Configure retry strategy
            retry_strategy = Retry(
                backoff=Retry.EXPONENTIAL_BACKOFF,
                retries=3,
                retry_on_timeout=True
            )

            # Connect to Redis/Valkey with TLS if enabled
            logger.info(f"Connecting to Valkey cache at {redis_host}:{redis_port} (SSL: {redis_ssl})")

            connection_args = {
                'host': redis_host,
                'port': redis_port,
                'decode_responses': True,  # Auto-decode bytes to strings
                'ssl': redis_ssl,
                'ssl_cert_reqs': None if redis_ssl else None,  # Disable certificate validation if needed
                'socket_timeout': redis_timeout,
                'socket_connect_timeout': redis_timeout,
                'health_check_interval': 30,
                'retry': retry_strategy
            }

            # Only add password if it's set and not empty
            if redis_password:
                connection_args['password'] = redis_password

            self.redis_client = redis.Redis(**connection_args)

            # Tag this client for monitoring
            self.redis_client.client_setname("user-seen-cache-client")

            try:
                if self.redis_client.ping():
                    logger.info("Successfully connected to Valkey")
                else:
                    logger.warning("Connected to Valkey but ping returned False")
            except Exception as e:
                logger.error(f"Failed to connect to Valkey: {e}")

        # API configuration
        self.api_base_url = api_base_url
        self.api_auth = api_auth

        # Cache TTL (how long to consider cached data valid) in seconds
        self.CACHE_TTL_SECONDS = int(os.environ.get('CACHE_TTL_SECONDS', 300))  # Default: 5 minutes

        # Key prefixes for Redis
        self.key_prefix = {
            'posts': "user_seen_posts:",
            'trailers': "user_seen_trailers:"
        }

    def get_recently_seen_content(self, user_id: int, content_type: str = 'posts') -> Set[int]:
        """Get recently seen content for a user from cache or database.

        Args:
            user_id: User ID
            content_type: Type of content ('posts' or 'trailers')

        Returns:
            Set of post IDs the user has recently seen
        """
        if content_type not in self.key_prefix:
            logger.warning(f"Invalid content_type: {content_type}, defaulting to 'posts'")
            content_type = 'posts'

        # Create Redis key for this user and content type
        key = f"{self.key_prefix[content_type]}{user_id}"

        try:
            # Try to get data from Redis
            cache_data = self.redis_client.get(key)

            if cache_data:
                # Parse the JSON data
                try:
                    data = json.loads(cache_data)
                    timestamp = datetime.fromisoformat(data['timestamp'])

                    # Check if cache is still valid
                    if (datetime.now() - timestamp).total_seconds() < self.CACHE_TTL_SECONDS:
                        logger.debug(f"Cache hit for {user_id} ({content_type}): {len(data['seen_ids'])} items")
                        return set(data['seen_ids'])
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Error parsing cache data: {e}")
        except redis.RedisError as e:
            logger.error(f"Redis error when getting cache: {e}")

        # Cache miss or expired, fetch from database via API
        logger.info(f"Cache miss for {user_id} ({content_type}), fetching from API")
        if content_type == 'posts':
            seen_ids = self._fetch_seen_posts(user_id)
        else:  # trailers
            seen_ids = self._fetch_seen_trailers(user_id)

        # Update cache
        try:
            self._update_cache(user_id, seen_ids, content_type)
        except redis.RedisError as e:
            logger.error(f"Redis error when updating cache: {e}")

        return seen_ids

    def add_seen_content(self, user_id: int, post_ids: List[int], content_type: str = 'posts'):
        """Update cache when new content is seen.

        Args:
            user_id: User ID
            post_ids: List of post IDs that were seen
            content_type: Type of content ('posts' or 'trailers')
        """
        if not post_ids:
            return

        if content_type not in self.key_prefix:
            logger.warning(f"Invalid content_type: {content_type}, defaulting to 'posts'")
            content_type = 'posts'

        key = f"{self.key_prefix[content_type]}{user_id}"

        try:
            # Try to get existing data
            cache_data = self.redis_client.get(key)

            if cache_data:
                # Update existing data
                try:
                    data = json.loads(cache_data)
                    existing_ids = set(data['seen_ids'])
                    updated_ids = list(existing_ids.union(set(post_ids)))

                    self._update_cache(user_id, updated_ids, content_type)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing existing cache data: {e}")
                    # Create new cache entry as fallback
                    self._update_cache(user_id, post_ids, content_type)
            else:
                # Create new cache entry
                self._update_cache(user_id, post_ids, content_type)
        except redis.RedisError as e:
            logger.error(f"Redis error in add_seen_content: {e}")

    def _update_cache(self, user_id: int, content_ids: Union[List[int], Set[int]], content_type: str = 'posts'):
        """Helper to update the Redis cache with new data.

        Args:
            user_id: User ID
            content_ids: List or set of content IDs
            content_type: Type of content ('posts' or 'trailers')
        """
        key = f"{self.key_prefix[content_type]}{user_id}"

        data = {
            'seen_ids': list(content_ids),  # Convert set to list for JSON
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Store in Redis with expiration (3x the TTL to allow for cleanup)
            cache_ttl = 3 * self.CACHE_TTL_SECONDS
            self.redis_client.setex(
                key,
                cache_ttl,
                json.dumps(data)
            )
            logger.debug(
                f"Updated cache for user {user_id} ({content_type}): {len(content_ids)} items, TTL: {cache_ttl}s")
        except redis.RedisError as e:
            logger.error(f"Redis error when updating cache: {e}")

    def _fetch_seen_posts(self, user_id: int) -> Set[int]:
        """Fetch recently seen posts from the API."""
        try:
            url = f"{self.api_base_url}/api/interactions/user/{user_id}/recent-views"
            response = requests.get(url, auth=self.api_auth, timeout=5)

            if response.status_code == 200:
                post_ids = response.json()
                logger.info(f"Fetched {len(post_ids)} recently seen posts for user {user_id}")
                return set(post_ids)
            else:
                logger.warning(f"API returned status {response.status_code} for seen posts")

            return set()
        except Exception as e:
            logger.error(f"Error fetching seen posts: {e}")
            return set()

    def _fetch_seen_trailers(self, user_id: int) -> Set[int]:
        """Fetch recently seen trailers from the API."""
        try:
            url = f"{self.api_base_url}/api/trailer-interactions/user/{user_id}/recent-views"
            response = requests.get(url, auth=self.api_auth, timeout=5)

            if response.status_code == 200:
                post_ids = response.json()
                logger.info(f"Fetched {len(post_ids)} recently seen trailers for user {user_id}")
                return set(post_ids)
            else:
                logger.warning(f"API returned status {response.status_code} for seen trailers")

            return set()
        except Exception as e:
            logger.error(f"Error fetching seen trailers: {e}")
            return set()

    def clear_cache(self, user_id: int = None, content_type: str = None):
        """
        Clear cache entries for a specific user or content type.
        If user_id is None, clears for all users.
        If content_type is None, clears for all content types.

        Args:
            user_id: Optional user ID to clear
            content_type: Optional content type to clear
        """
        try:
            if user_id is not None:
                # Clear for specific user
                if content_type:
                    if content_type in self.key_prefix:
                        key = f"{self.key_prefix[content_type]}{user_id}"
                        self.redis_client.delete(key)
                        logger.info(f"Cleared cache for user {user_id}, content type {content_type}")
                else:
                    # Clear all content types for this user
                    for type_key in self.key_prefix.values():
                        key = f"{type_key}{user_id}"
                        self.redis_client.delete(key)
                    logger.info(f"Cleared all content types for user {user_id}")
            else:
                # Clear for all users of a specific content type or all content types
                if content_type:
                    if content_type in self.key_prefix:
                        pattern = f"{self.key_prefix[content_type]}*"
                        keys = self.redis_client.keys(pattern)
                        if keys:
                            self.redis_client.delete(*keys)
                            logger.info(f"Cleared all user caches for content type {content_type}")
                else:
                    # Clear everything
                    for type_key in self.key_prefix.values():
                        pattern = f"{type_key}*"
                        keys = self.redis_client.keys(pattern)
                        if keys:
                            self.redis_client.delete(*keys)
                    logger.info("Cleared all user seen caches")
        except redis.RedisError as e:
            logger.error(f"Redis error when clearing cache: {e}")