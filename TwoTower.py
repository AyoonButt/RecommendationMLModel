import tensorflow as tf
import numpy as np
import os
from typing import List, Dict
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("two-tower-model")


def compute_scores(user_embeddings: np.ndarray, post_embeddings: np.ndarray, content_type: str = "posts") -> np.ndarray:
    """
    Compute similarity scores between user and post embeddings.

    Args:
        user_embeddings: User embedding vectors of shape (batch_size, embedding_dim)
        post_embeddings: Post embedding vectors of shape (num_posts, embedding_dim)
        content_type: Type of content being scored ("posts" or "trailers")

    Returns:
        Scores of shape (batch_size, num_posts)
    """
    # Ensure vectors are normalized
    user_norm = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    post_norm = np.linalg.norm(post_embeddings, axis=1, keepdims=True)

    user_embeddings_normalized = user_embeddings / np.maximum(user_norm, 1e-8)
    post_embeddings_normalized = post_embeddings / np.maximum(post_norm, 1e-8)

    # Compute dot product (cosine similarity since vectors are normalized)
    scores = np.matmul(user_embeddings_normalized, post_embeddings_normalized.T)

    # Ensure scores are in [0, 1] range
    scores = 0.5 * (scores + 1.0)

    # Apply content-type specific boost if needed
    if content_type == "trailers":
        # Get boost factor from environment variable or use default
        boost_factor = float(os.environ.get('TRAILER_BOOST_FACTOR', '1.15'))
        # Boost trailer scores to prioritize them
        scores = np.minimum(scores * boost_factor, 1.0)  # Cap at 1.0

    return scores


class TwoTowerModel:
    """
    Two-Tower neural network model for recommendation system.

    This model consists of two separate neural networks (towers):
    1. User Tower: Processes user features
    2. Post Tower: Processes post features

    The similarity between user and post embeddings is calculated
    using dot product to generate recommendation scores.
    """

    def __init__(
            self,
            user_feature_dim: int = 64,
            post_feature_dim: int = 64,
            embedding_dim: int = 32,
            hidden_dims: List[int] = [128, 64],
            dropout_rate: float = 0.2,
            l2_regularization: float = 1e-5,
            learning_rate: float = 0.001,
            model_dir: str = "./model_checkpoints",
    ):
        """
        Initialize the Two-Tower model.

        Args:
            user_feature_dim: Dimension of user feature input
            post_feature_dim: Dimension of post feature input
            embedding_dim: Dimension of final embedding vectors
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            l2_regularization: L2 regularization factor
            learning_rate: Learning rate for optimizer
            model_dir: Directory to save model checkpoints
        """
        self.user_feature_dim = user_feature_dim
        self.post_feature_dim = post_feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_regularization
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.user_model = None
        self.post_model = None
        self.full_model = None

        # Create directories for model if they don't exist
        os.makedirs(model_dir, exist_ok=True)

        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the Two-Tower model architecture."""
        # Initialize regularizer
        regularizer = tf.keras.regularizers.l2(self.l2_reg)

        # User Tower
        user_input = tf.keras.layers.Input(shape=(self.user_feature_dim,), name='user_features')
        user_x = user_input

        for i, dim in enumerate(self.hidden_dims):
            user_x = tf.keras.layers.Dense(
                dim,
                activation=None,
                kernel_regularizer=regularizer,
                name=f'user_dense_{i}'
            )(user_x)
            user_x = tf.keras.layers.BatchNormalization(name=f'user_bn_{i}')(user_x)
            user_x = tf.keras.layers.ReLU(name=f'user_relu_{i}')(user_x)
            user_x = tf.keras.layers.Dropout(self.dropout_rate, name=f'user_dropout_{i}')(user_x)

        user_embedding = tf.keras.layers.Dense(
            self.embedding_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name='user_embedding'
        )(user_x)

        # L2 normalization for cosine similarity alignment
        user_embedding = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1),
            name='user_embedding_normalized'
        )(user_embedding)

        # Post Tower
        post_input = tf.keras.layers.Input(shape=(self.post_feature_dim,), name='post_features')
        post_x = post_input

        for i, dim in enumerate(self.hidden_dims):
            post_x = tf.keras.layers.Dense(
                dim,
                activation=None,
                kernel_regularizer=regularizer,
                name=f'post_dense_{i}'
            )(post_x)
            post_x = tf.keras.layers.BatchNormalization(name=f'post_bn_{i}')(post_x)
            post_x = tf.keras.layers.ReLU(name=f'post_relu_{i}')(post_x)
            post_x = tf.keras.layers.Dropout(self.dropout_rate, name=f'post_dropout_{i}')(post_x)

        post_embedding = tf.keras.layers.Dense(
            self.embedding_dim,
            activation=None,
            kernel_regularizer=regularizer,
            name='post_embedding'
        )(post_x)

        # L2 normalization for cosine similarity alignment
        post_embedding = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1),
            name='post_embedding_normalized'
        )(post_embedding)

        # Create separate models for each tower
        self.user_model = tf.keras.Model(inputs=user_input, outputs=user_embedding, name='user_tower')
        self.post_model = tf.keras.Model(inputs=post_input, outputs=post_embedding, name='post_tower')

        # Combined model for training
        # Dot product for similarity scoring
        dot_product = tf.keras.layers.Dot(axes=1, normalize=False, name='dot_product')([
            user_embedding, post_embedding
        ])

        # Sigmoid activation to get scores between 0-1
        output = tf.keras.layers.Activation('sigmoid', name='prediction')(dot_product)

        # Create full model
        self.full_model = tf.keras.Model(
            inputs=[user_input, post_input],
            outputs=output,
            name='two_tower_model'
        )

        # Compile with binary cross-entropy loss
        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        logger.info("Two-Tower model built successfully")

    def train(
            self,
            user_features: np.ndarray,
            post_features: np.ndarray,
            labels: np.ndarray,
            validation_split: float = 0.2,
            batch_size: int = 256,
            epochs: int = 10,
            callbacks: List = None
    ):
        """
        Train the Two-Tower model.

        Args:
            user_features: User feature vectors
            post_features: Post feature vectors
            labels: Binary labels (1 for interaction, 0 for no interaction)
            validation_split: Fraction of data to use for validation
            batch_size: Training batch size
            epochs: Number of training epochs
            callbacks: List of Keras callbacks

        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, "model_{epoch:02d}_{val_loss:.4f}.h5"),
                    monitor='val_loss',
                    save_best_only=True
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.model_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
                    update_freq='epoch'
                )
            ]

        logger.info(f"Starting training with {len(user_features)} samples")

        history = self.full_model.fit(
            [user_features, post_features],
            labels,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed")
        return history

    def save_models(self, prefix: str = ""):
        """
        Save the user and post tower models.

        Args:
            prefix: Prefix for model filenames
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_model_path = os.path.join(self.model_dir, f"{prefix}user_tower_{timestamp}.h5")
        post_model_path = os.path.join(self.model_dir, f"{prefix}post_tower_{timestamp}.h5")

        self.user_model.save(user_model_path)
        self.post_model.save(post_model_path)

        logger.info(f"Models saved: {user_model_path}, {post_model_path}")

    def load_models(self, user_model_path: str, post_model_path: str):
        """
        Load pre-trained user and post tower models.

        Args:
            user_model_path: Path to saved user tower model
            post_model_path: Path to saved post tower model
        """
        self.user_model = tf.keras.models.load_model(user_model_path)
        self.post_model = tf.keras.models.load_model(post_model_path)

        logger.info("Models loaded successfully")

    def get_user_embedding(self, user_features: np.ndarray) -> np.ndarray:
        """
        Generate user embeddings.

        Args:
            user_features: User feature vectors

        Returns:
            User embeddings
        """
        if self.user_model is None:
            raise ValueError("User model not initialized. Make sure to load or build the model first.")
        return self.user_model.predict(user_features)

    def get_post_embedding(self, post_features: np.ndarray) -> np.ndarray:
        """
        Generate post embeddings.

        Args:
            post_features: Post feature vectors

        Returns:
            Post embeddings
        """
        if self.post_model is None:
            raise ValueError("Post model not initialized. Make sure to load or build the model first.")
        return self.post_model.predict(post_features)

    def predict_scores(
            self,
            user_features: np.ndarray,
            post_features: np.ndarray,
            content_type: str = "posts"
    ) -> np.ndarray:
        """
        Predict recommendation scores.

        Args:
            user_features: User feature vectors
            post_features: Post feature vectors
            content_type: Type of content ("posts" or "trailers")

        Returns:
            Recommendation scores
        """
        # Get embeddings
        user_embeddings = self.get_user_embedding(user_features)
        post_embeddings = self.get_post_embedding(post_features)

        # Compute scores with content type consideration
        return compute_scores(user_embeddings, post_embeddings, content_type)

    def evaluate(
            self,
            user_features: np.ndarray,
            post_features: np.ndarray,
            labels: np.ndarray,
            batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            user_features: User feature vectors
            post_features: Post feature vectors
            labels: Ground truth labels
            batch_size: Evaluation batch size

        Returns:
            Dictionary of evaluation metrics
        """
        if self.full_model is None:
            raise ValueError("Model not initialized. Make sure to load or build the model first.")

        metrics = self.full_model.evaluate(
            [user_features, post_features],
            labels,
            batch_size=batch_size,
            verbose=1
        )

        results = dict(zip(self.full_model.metrics_names, metrics))
        logger.info(f"Evaluation results: {results}")

        return results


# Helper classes for JSON serialization of numpy arrays
class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy arrays.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__numpy_array__': True,
                'data': obj.tolist()
            }
        return json.JSONEncoder.default(self, obj)


def decode_numpy(obj):
    """
    Custom JSON decoder for numpy arrays.
    """
    if '__numpy_array__' in obj:
        return np.array(obj['data'], dtype=np.float32)
    return obj