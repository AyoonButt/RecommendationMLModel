import tensorflow as tf


class TwoTowerModel:
    def __init__(self, user_feature_dim, post_feature_dim, embedding_dim=32):
        self.user_feature_dim = user_feature_dim
        self.post_feature_dim = post_feature_dim
        self.embedding_dim = embedding_dim

        # Build user tower
        self.user_model = self._build_user_tower()

        # Build post tower
        self.post_model = self._build_post_tower()

    def _build_user_tower(self):
        user_input = tf.keras.layers.Input(shape=(self.user_feature_dim,), name='user_features')
        x = tf.keras.layers.Dense(128, activation='relu')(user_input)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(self.embedding_dim)(x)
        user_embedding = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1)
        )(x)

        return tf.keras.Model(inputs=user_input, outputs=user_embedding)

    def _build_post_tower(self):
        post_input = tf.keras.layers.Input(shape=(self.post_feature_dim,), name='post_features')
        x = tf.keras.layers.Dense(128, activation='relu')(post_input)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(self.embedding_dim)(x)
        post_embedding = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1)
        )(x)

        return tf.keras.Model(inputs=post_input, outputs=post_embedding)

    def compute_score(self, user_features, post_features):
        user_embedding = self.user_model(user_features)
        post_embedding = self.post_model(post_features)

        # Compute dot product similarity
        return tf.reduce_sum(user_embedding * post_embedding, axis=1)

    def train(self, user_features, post_features, labels, epochs=10, batch_size=64):

    # Training implementation
    # ...

    def get_user_embedding(self, user_features):
        return self.user_model.predict(user_features)

    def get_post_embedding(self, post_features):
        return self.post_model.predict(post_features)