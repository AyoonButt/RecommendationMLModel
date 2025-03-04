class PopularityRanker:
    def __init__(self, db_connection):
        self.db = db_connection

    def get_recommendations(self, user_id, limit=10):
        # Get user's genre preferences
        genres = self.db.query("SELECT genre_id FROM user_games WHERE user_id = ?", [user_id])

        # Get popular posts in those genres
        if genres:
            recommended_posts = self.db.query("""
                SELECT p.post_id, p.title, p.popularity 
                FROM posts p
                JOIN post_games pg ON p.post_id = pg.post_id
                WHERE pg.genre_id IN (?)
                ORDER BY p.popularity DESC, p.vote_count DESC
                LIMIT ?
            """, [genres, limit])
        else:
            # Fallback to overall popularity
            recommended_posts = self.db.query("""
                SELECT post_id, title, popularity 
                FROM posts
                ORDER BY popularity DESC, vote_count DESC
                LIMIT ?
            """, [limit])

        return recommended_posts