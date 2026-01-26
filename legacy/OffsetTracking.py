import random
import time
import logging

# Configure logger
logger = logging.getLogger(__name__)


class RecommendationOffsetTracker:
    """Tracks offsets for user recommendations to prioritize fresh content while ensuring variety."""

    def __init__(self):
        # Main storage for user offset data
        self.user_offsets = {}

        # Settings for freshness vs. variety balance
        self.fresh_content_probability = 0.3  # 30% chance to get fresh content
        self.max_inactive_days = 30
        self.last_cleanup = time.time()
        self.cleanup_interval = 86400  # 1 day in seconds

    def get_next_offset(self, user_id: str, page_size: int = 20) -> int:
        """
        Get the next offset for recommendations with freshness bias.
        Has a chance to return offset 0 to favor fresh content.

        Args:
            user_id: The user identifier
            page_size: Size of each page of recommendations

        Returns:
            The next offset to use for this user
        """
        # Check if cleanup is needed
        self._maybe_cleanup()

        # Initialize user data if not exists
        if user_id not in self.user_offsets:
            self.user_offsets[user_id] = {
                'current_cycle': 0,
                'available_offsets': [],
                'last_total_count': 0,
                'last_access': time.time(),
                'freshness_counter': 0  # Track how often we've shown fresh content
            }

        # Update last access time
        self.user_offsets[user_id]['last_access'] = time.time()

        # Check if we should prioritize fresh content (offset 0)
        # We use freshness_counter to avoid showing fresh content too frequently to the same user
        user_data = self.user_offsets[user_id]

        # Decide whether to show fresh content based on probability and counter
        show_fresh = random.random() < self.fresh_content_probability and user_data['freshness_counter'] <= 0

        if show_fresh:
            # Reset freshness counter - wait a few requests before showing fresh content again
            user_data['freshness_counter'] = 3  # Show fresh content at most once every 3 requests
            try:
                logger.info(f"Prioritizing fresh content for user {user_id} (offset 0)")
            except:
                print(f"INFO: Prioritizing fresh content for user {user_id} (offset 0)")
            return 0
        else:
            # Decrement freshness counter if positive
            if user_data['freshness_counter'] > 0:
                user_data['freshness_counter'] -= 1

            # If available offsets is empty, refresh
            if not user_data['available_offsets']:
                self._refresh_offsets(user_id, page_size)

            # Get next offset
            next_offset = user_data['available_offsets'].pop(0)
            return next_offset

    def _refresh_offsets(self, user_id: str, page_size: int):
        """Refresh the list of available offsets for a user, prioritizing variety."""
        user_data = self.user_offsets[user_id]

        # Get current total count of available items
        current_total = self._get_total_candidate_count(user_id)

        # Check if data size changed significantly
        significant_change = False
        if user_data['last_total_count'] > 0:
            change_ratio = abs(current_total - user_data['last_total_count']) / user_data['last_total_count']
            significant_change = change_ratio > 0.2  # 20% change

        # Reset cycle if significant change in data size
        if significant_change:
            user_data['current_cycle'] = 0

        # Increment cycle and update total count
        user_data['current_cycle'] += 1
        user_data['last_total_count'] = current_total

        # Calculate max valid offset
        max_offset = max(0, ((current_total - 1) // page_size) * page_size)

        # Skip offset 0 since we handle it separately with the freshness bias
        all_offsets = list(range(page_size, max_offset + 1, page_size))

        # First cycle is sequential, subsequent cycles are randomized
        if user_data['current_cycle'] > 1:
            random.shuffle(all_offsets)

        user_data['available_offsets'] = all_offsets

    def _get_total_candidate_count(self, user_id: str) -> int:
        """
        Get the total count of recommendation candidates for a user.
        In a real implementation, this would query your recommendation service.
        """
        # Replace with actual logic to get candidate count
        try:
            # Simulated response - replace with actual API call
            count = 500  # Default value
            # count = recommendation_service.get_candidate_count(user_id)
            return count
        except Exception as e:
            # If unable to get count, use last known count
            return self.user_offsets[user_id].get('last_total_count', 100)

    def _maybe_cleanup(self):
        """Periodically clean up data for inactive users."""
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_inactive_users()
            self.last_cleanup = now

    def _cleanup_inactive_users(self):
        """Remove data for users who haven't been active for a while."""
        now = time.time()
        inactive_threshold = now - (self.max_inactive_days * 86400)

        users_to_remove = []
        for user_id, data in self.user_offsets.items():
            if data['last_access'] < inactive_threshold:
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del self.user_offsets[user_id]

    def reset_offset(self, user_id: str):
        """Reset the offset tracking for a specific user."""
        if user_id in self.user_offsets:
            # Reset available offsets to start from the beginning
            self.user_offsets[user_id]['available_offsets'] = []
            # Reset freshness counter to allow fresh content on next request
            self.user_offsets[user_id]['freshness_counter'] = 0

            try:
                logger.info(f"Reset offset tracking for user {user_id}")
            except:
                print(f"INFO: Reset offset tracking for user {user_id}")