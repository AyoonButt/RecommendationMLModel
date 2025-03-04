class InteractionTracker:
    def __init__(self, db_connection):
        self.db = db_connection
        self.interaction_weights = {
            'save': 5.0,
            'like': 3.0,
            'comment_positive': 2.5,
            'comment_neutral': 1.0,
            'comment_negative': -1.0,
            'view_complete': 2.0,
            'trailer_watch': 1.0,
            'click': 0.5
        }

    def record_interaction(self, user_id, post_id, interaction_type, metadata=None):
        # Store interaction in database
        self.db.execute("""
            INSERT INTO user_post_interactions 
            (user_id, post_id, interaction_type, metadata, timestamp) 
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [user_id, post_id, interaction_type, json.dumps(metadata)])

        # Update user profile
        self._update_user_profile(user_id, post_id, interaction_type)

    def get_interaction_score(self, user_id, post_id):
        # Get all interactions between user and post
        interactions = self.db.query("""
            SELECT interaction_type 
            FROM user_post_interactions 
            WHERE user_id = ? AND post_id = ?
        """, [user_id, post_id])

        # Calculate weighted score
        score = sum(self.interaction_weights.get(i['interaction_type'], 0) for i in interactions)
        return score

    def _update_user_profile(self, user_id, post_id, interaction_type):
# Update user profile based on interaction
# ...