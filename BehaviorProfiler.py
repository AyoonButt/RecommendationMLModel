class BehaviorProfiler:
    def __init__(self, db_connection):
        self.db = db_connection

    def get_user_profile(self, user_id):
        # Get user's interaction history
        interactions = self.db.query("""
            SELECT interaction_type, COUNT(*) as count
            FROM user_post_interactions
            WHERE user_id = ?
            GROUP BY interaction_type
        """, [user_id])

        # Calculate behavioral metrics
        total_interactions = sum(i['count'] for i in interactions)

        profile = {}
        interaction_counts = {i['interaction_type']: i['count'] for i in interactions}

        # Calculate profile dimensions
        profile['commenter'] = interaction_counts.get('comment', 0) / total_interactions if total_interactions else 0
        profile['saver'] = interaction_counts.get('save', 0) / total_interactions if total_interactions else 0
        profile['trailer_watcher'] = interaction_counts.get('trailer_watch',
                                                            0) / total_interactions if total_interactions else 0
        # Additional dimensions...

        # Determine dominant profile
        profile['dominant_type'] = max(profile, key=profile.get)

        return profile