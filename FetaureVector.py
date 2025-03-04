class ContentFeatureExtractor:
    def __init__(self, db_connection):
        self.db = db_connection

    def extract_post_features(self, post_id):
        # Basic post information
        post_info = self.db.query("SELECT * FROM posts WHERE post_id = ?", [post_id])

        # Genre information
        genres = self.db.query("SELECT genre_id FROM post_games WHERE post_id = ?", [post_id])

        # Cast/crew information (focus on prominent roles)
        cast = self.db.query("""
            SELECT person_id, character 
            FROM cast_members 
            WHERE post_id = ? 
            ORDER BY order_index
            LIMIT 5
        """, [post_id])

        crew = self.db.query("""
            SELECT person_id, job 
            FROM crew 
            WHERE post_id = ? AND job IN ('director', 'writer')
        """, [post_id])

        # Combine features
        features = {
            'basic_info': post_info,
            'genres': genres,
            'cast': cast,
            'crew': crew
        }

        # Convert to vector representation
        return self._vectorize_features(features)

    def _vectorize_features(self, features):
        # Feature engineering logic
        # ...
        return feature_vector


```