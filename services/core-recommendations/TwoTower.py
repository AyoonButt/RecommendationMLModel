import tensorflow as tf
import numpy as np
import os
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

# Import social signal processor and content filter
from SocialSignalProcessor import SocialSignalProcessor, InteractionType
from ContentFilter import ContentFilter, create_content_filter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("two-tower-model")


def compute_scores(user_embeddings: np.ndarray, post_embeddings: np.ndarray,
                   content_type: str = "posts", candidates: List[Dict] = None) -> np.ndarray:
    """
    Compute similarity scores between user and post embeddings with candidate metadata enhancement.

    Args:
        user_embeddings: User embedding vectors of shape (batch_size, embedding_dim)
        post_embeddings: Post embedding vectors of shape (num_posts, embedding_dim)
        content_type: Type of content being scored ("posts" or "trailers")
        candidates: List of candidate metadata for score enhancement

    Returns:
        Scores of shape (batch_size, num_posts)
    """
    # Ensure vectors are normalized
    user_norm = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    post_norm = np.linalg.norm(post_embeddings, axis=1, keepdims=True)

    user_embeddings_normalized = user_embeddings / np.maximum(user_norm, 1e-8)
    post_embeddings_normalized = post_embeddings / np.maximum(post_norm, 1e-8)

    # Compute dot product (cosine similarity since vectors are normalized)
    base_scores = np.matmul(user_embeddings_normalized, post_embeddings_normalized.T)

    # Ensure scores are in [0, 1] range
    base_scores = 0.5 * (base_scores + 1.0)

    # Apply candidate-based score enhancements
    if candidates:
        enhanced_scores = _apply_candidate_boosts(base_scores, candidates, content_type)
        return enhanced_scores

    return base_scores


def compute_socially_enhanced_scores(user_embeddings: np.ndarray, 
                                   post_embeddings: np.ndarray,
                                   user_id: int,
                                   post_ids: List[int],
                                   social_processor: SocialSignalProcessor,
                                   content_type: str = "posts",
                                   candidates: List[Dict] = None) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute socially enhanced similarity scores using social signals and sentiment analysis.

    Args:
        user_embeddings: User embedding vectors of shape (batch_size, embedding_dim)
        post_embeddings: Post embedding vectors of shape (num_posts, embedding_dim)
        user_id: User ID for social signal processing
        post_ids: List of post IDs corresponding to post embeddings
        social_processor: Social signal processor instance
        content_type: Type of content being scored ("posts" or "trailers")
        candidates: List of candidate metadata for score enhancement

    Returns:
        Tuple of (enhanced_scores, metadata)
    """
    try:
        # Get base scores first
        base_scores = compute_scores(user_embeddings, post_embeddings, content_type, candidates)
        
        # Apply social enhancement if user_id is provided
        if user_id and social_processor:
            logger.info(f"Applying social enhancement for user {user_id}")
            
            # Use social processor to enhance scores
            enhanced_scores, social_metadata = social_processor.process_social_boost(
                user_id=user_id,
                post_ids=post_ids,
                user_embedding=user_embeddings[0] if user_embeddings.shape[0] > 0 else user_embeddings,
                post_embeddings=post_embeddings
            )
            
            # Ensure scores are properly shaped
            if enhanced_scores.ndim == 1:
                enhanced_scores = enhanced_scores.reshape(1, -1)
            
            return enhanced_scores, social_metadata
        else:
            logger.info("No social enhancement applied - using base scores")
            return base_scores, {"social_enhancement": False}
            
    except Exception as e:
        logger.error(f"Error in socially enhanced scoring: {e}")
        # Fallback to base scores
        base_scores = compute_scores(user_embeddings, post_embeddings, content_type, candidates)
        return base_scores, {"error": str(e), "fallback": True}


def compute_dual_preference_scores(user_embeddings: np.ndarray,
                                 post_embeddings: np.ndarray,
                                 user_id: int,
                                 post_ids: List[int],
                                 positive_interactions: List[int] = None,
                                 negative_interactions: List[int] = None,
                                 content_type: str = "posts") -> np.ndarray:
    """
    Compute scores with dual preference boosting (likes/saves vs not interested).

    Args:
        user_embeddings: User embedding vectors
        post_embeddings: Post embedding vectors  
        user_id: User ID
        post_ids: List of post IDs
        positive_interactions: List of post IDs with positive interactions
        negative_interactions: List of post IDs with negative interactions
        content_type: Type of content

    Returns:
        Dual preference enhanced scores
    """
    try:
        # Get base scores
        base_scores = compute_scores(user_embeddings, post_embeddings, content_type)
        
        if base_scores.ndim == 1:
            base_scores = base_scores.reshape(1, -1)
        
        enhanced_scores = base_scores.copy()
        
        # Apply positive boosts
        if positive_interactions:
            positive_set = set(positive_interactions)
            for i, post_id in enumerate(post_ids):
                if post_id in positive_set and i < enhanced_scores.shape[1]:
                    # Boost score for positive interactions
                    enhanced_scores[0, i] *= 1.2  # 20% boost
        
        # Apply negative penalties
        if negative_interactions:
            negative_set = set(negative_interactions)
            for i, post_id in enumerate(post_ids):
                if post_id in negative_set and i < enhanced_scores.shape[1]:
                    # Penalize score for negative interactions
                    enhanced_scores[0, i] *= 0.7  # 30% penalty
        
        # Clamp scores to valid range
        enhanced_scores = np.clip(enhanced_scores, 0.0, 1.0)
        
        logger.info(f"Applied dual preference boosting for user {user_id}")
        return enhanced_scores
        
    except Exception as e:
        logger.error(f"Error in dual preference scoring: {e}")
        return compute_scores(user_embeddings, post_embeddings, content_type)


def compute_filtered_scores(user_embeddings: np.ndarray,
                          post_embeddings: np.ndarray,
                          user_id: int,
                          post_ids: List[int],
                          content_filter: ContentFilter,
                          content_type: str = "posts",
                          apply_social_boost: bool = True,
                          social_processor: SocialSignalProcessor = None) -> tuple[np.ndarray, List[int], Dict[str, Any]]:
    """
    Compute recommendation scores with content filtering applied.
    
    Args:
        user_embeddings: User embedding vectors
        post_embeddings: Post embedding vectors
        user_id: User ID
        post_ids: List of post IDs
        content_filter: Content filter instance
        content_type: Type of content
        apply_social_boost: Whether to apply social boosting
        social_processor: Social signal processor instance
        
    Returns:
        Tuple of (filtered_scores, filtered_post_ids, filter_metadata)
    """
    try:
        # Get base scores
        if apply_social_boost and social_processor:
            base_scores, social_metadata = compute_socially_enhanced_scores(
                user_embeddings, post_embeddings, user_id, post_ids, 
                social_processor, content_type
            )
            if base_scores.ndim > 1:
                base_scores = base_scores.flatten()
        else:
            base_scores = compute_scores(user_embeddings, post_embeddings, content_type)
            if base_scores.ndim > 1:
                base_scores = base_scores.flatten()
            social_metadata = {"social_enhancement": False}
        
        # Apply content filtering
        filtered_post_ids, filtered_scores, filter_metadata = content_filter.filter_recommendations(
            user_id=user_id,
            post_ids=post_ids,
            scores=base_scores,
            apply_user_preferences=True
        )
        
        # Combine metadata
        combined_metadata = {
            **social_metadata,
            **filter_metadata,
            "content_filtering_applied": True
        }
        
        logger.info(f"Applied content filtering: {len(post_ids)} -> {len(filtered_post_ids)} posts for user {user_id}")
        return filtered_scores, filtered_post_ids, combined_metadata
        
    except Exception as e:
        logger.error(f"Error in filtered scoring: {e}")
        # Return original data if filtering fails
        base_scores = compute_scores(user_embeddings, post_embeddings, content_type)
        if base_scores.ndim > 1:
            base_scores = base_scores.flatten()
        return base_scores, post_ids, {"error": str(e), "content_filtering_applied": False}


def _apply_candidate_boosts(scores: np.ndarray, candidates: List[Dict], content_type: str) -> np.ndarray:
    """
    Apply score boosts based on candidate metadata.

    Args:
        scores: Base similarity scores
        candidates: Candidate metadata list
        content_type: Type of content

    Returns:
        Enhanced scores with candidate-based boosts
    """
    enhanced_scores = scores.copy()

    # Create candidate lookup by postId for efficient access
    candidate_lookup = {candidate.get('postId'): candidate for candidate in candidates}

    for i in range(scores.shape[1]):  # For each post
        if i < len(candidates):
            candidate = candidates[i]
            candidate_type = candidate.get('candidateType', '')
            source = candidate.get('source', '')
            original_score = candidate.get('score', 1.0)

            # Apply boosts based on candidate type
            boost_factor = _get_candidate_boost_factor(candidate_type, source, content_type, original_score)

            # Apply the boost to all users (assuming single user batch)
            for j in range(scores.shape[0]):
                enhanced_scores[j, i] *= boost_factor

    # Ensure scores stay in [0, 1] range after boosting
    enhanced_scores = np.clip(enhanced_scores, 0.0, 1.0)

    return enhanced_scores


def _get_candidate_boost_factor(candidate_type: str, source: str, content_type: str, original_score: float) -> float:
    """
    Calculate boost factor based on candidate metadata.

    Args:
        candidate_type: Type of candidate
        source: Source of candidate
        content_type: Content type (posts/trailers)
        original_score: Original candidate score

    Returns:
        Boost factor to multiply with base score
    """
    boost_factor = 1.0

    # Boost based on candidate type
    type_boosts = {
        'NEW_HIGH_QUALITY': 1.15,
        'NEW_HIGH_QUALITY_TRAILERS': 1.15,
        'CURSOR_BASED': 1.05,
        'CURSOR_BASED_TRAILERS': 1.05,
        'STRATIFIED_PRIMARY': 1.10,
        'CROSS_LANGUAGE': 0.95,  # Slight penalty for cross-language
        'EMERGENCY_FALLBACK': 0.85,  # Penalty for emergency fallback
        'EMERGENCY_FALLBACK_TRAILERS': 0.85
    }

    boost_factor *= type_boosts.get(candidate_type, 1.0)

    # Boost based on source
    source_boosts = {
        'PRIMARY': 1.05,
        'NEW_HIGH_QUALITY': 1.10,
        'FALLBACK': 0.95
    }

    boost_factor *= source_boosts.get(source, 1.0)

    # Boost based on original candidate score (higher scored candidates get slight boost)
    if original_score > 1.2:
        boost_factor *= 1.08  # High quality candidates
    elif original_score > 1.0:
        boost_factor *= 1.03  # Medium quality candidates
    elif original_score < 0.8:
        boost_factor *= 0.95  # Lower quality candidates get slight penalty

    # Content-type specific adjustments
    if content_type == "trailers":
        if "TRAILER" in candidate_type:
            boost_factor *= 1.05  # Extra boost for trailer-specific candidates

    return boost_factor


class TwoTowerModel:
    """
    Enhanced Two-Tower neural network model for recommendation system with candidate awareness.

    This model consists of two separate neural networks (towers):
    1. User Tower: Processes user features
    2. Post Tower: Processes post features

    The similarity between user and post embeddings is calculated
    using dot product to generate recommendation scores, enhanced with
    candidate metadata for improved recommendations.
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
        self.candidate_aware_model = None

        # Create directories for model if they don't exist
        os.makedirs(model_dir, exist_ok=True)

        # Build the model
        self._build_model()

    def _build_model(self):
        """Build the Two-Tower model architecture with candidate awareness."""
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

        # Build candidate-aware model with additional inputs
        self._build_candidate_aware_model(user_embedding, post_embedding)

        # Compile with binary cross-entropy loss
        self.full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        logger.info("Two-Tower model built successfully")

    def _build_candidate_aware_model(self, user_embedding, post_embedding):
        """
        Build candidate-aware model that incorporates candidate metadata.

        Args:
            user_embedding: User embedding layer
            post_embedding: Post embedding layer
        """
        # Additional inputs for candidate metadata
        candidate_type_input = tf.keras.layers.Input(shape=(1,), name='candidate_type_encoded')
        candidate_source_input = tf.keras.layers.Input(shape=(1,), name='candidate_source_encoded')
        candidate_score_input = tf.keras.layers.Input(shape=(1,), name='candidate_score')

        # Process candidate features
        candidate_features = tf.keras.layers.concatenate([
            candidate_type_input,
            candidate_source_input,
            candidate_score_input
        ], name='candidate_features')

        # Small network to process candidate features
        candidate_processed = tf.keras.layers.Dense(16, activation='relu', name='candidate_dense')(candidate_features)
        candidate_processed = tf.keras.layers.Dense(8, activation='relu', name='candidate_dense_2')(candidate_processed)
        candidate_boost = tf.keras.layers.Dense(1, activation='sigmoid', name='candidate_boost')(candidate_processed)

        # Base similarity score
        dot_product = tf.keras.layers.Dot(axes=1, normalize=False, name='base_similarity')([
            user_embedding, post_embedding
        ])
        base_score = tf.keras.layers.Activation('sigmoid', name='base_score')(dot_product)

        # Apply candidate boost
        enhanced_score = tf.keras.layers.Multiply(name='enhanced_score')([base_score, candidate_boost])

        # Final activation to ensure [0,1] range
        final_score = tf.keras.layers.Activation('sigmoid', name='final_prediction')(enhanced_score)

        # Create candidate-aware model
        self.candidate_aware_model = tf.keras.Model(
            inputs=[
                self.user_model.input,
                self.post_model.input,
                candidate_type_input,
                candidate_source_input,
                candidate_score_input
            ],
            outputs=final_score,
            name='candidate_aware_model'
        )

    def train_with_candidates(
            self,
            user_features: np.ndarray,
            post_features: np.ndarray,
            candidate_types: np.ndarray,
            candidate_sources: np.ndarray,
            candidate_scores: np.ndarray,
            labels: np.ndarray,
            validation_split: float = 0.2,
            batch_size: int = 256,
            epochs: int = 10,
            callbacks: List = None
    ):
        """
        Train the candidate-aware Two-Tower model.

        Args:
            user_features: User feature vectors
            post_features: Post feature vectors
            candidate_types: Encoded candidate types
            candidate_sources: Encoded candidate sources
            candidate_scores: Candidate scores
            labels: Binary labels (1 for interaction, 0 for no interaction)
            validation_split: Fraction of data to use for validation
            batch_size: Training batch size
            epochs: Number of training epochs
            callbacks: List of Keras callbacks

        Returns:
            Training history
        """
        if self.candidate_aware_model is None:
            logger.warning("Candidate-aware model not available, falling back to standard training")
            return self.train(user_features, post_features, labels, validation_split, batch_size, epochs, callbacks)

        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, "candidate_aware_model_{epoch:02d}_{val_loss:.4f}.h5"),
                    monitor='val_loss',
                    save_best_only=True
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.model_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
                    update_freq='epoch'
                )
            ]

        # Compile candidate-aware model
        self.candidate_aware_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        logger.info(f"Starting candidate-aware training with {len(user_features)} samples")

        history = self.candidate_aware_model.fit(
            [user_features, post_features, candidate_types, candidate_sources, candidate_scores],
            labels,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Candidate-aware training completed")
        return history

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
        Train the standard Two-Tower model.

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

    def predict_scores_with_candidates(
            self,
            user_features: np.ndarray,
            post_features: np.ndarray,
            candidates: List[Dict],
            content_type: str = "posts"
    ) -> np.ndarray:
        """
        Predict recommendation scores using candidate metadata.

        Args:
            user_features: User feature vectors
            post_features: Post feature vectors
            candidates: List of candidate metadata
            content_type: Type of content ("posts" or "trailers")

        Returns:
            Enhanced recommendation scores
        """
        # Get base embeddings
        user_embeddings = self.get_user_embedding(user_features)
        post_embeddings = self.get_post_embedding(post_features)

        # Compute enhanced scores with candidate metadata
        return compute_scores(user_embeddings, post_embeddings, content_type, candidates)

    def predict_scores(
            self,
            user_features: np.ndarray,
            post_features: np.ndarray,
            content_type: str = "posts"
    ) -> np.ndarray:
        """
        Predict recommendation scores (standard method).

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

        # Compute scores without candidate enhancement
        return compute_scores(user_embeddings, post_embeddings, content_type)

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

        # Save candidate-aware model if available
        if self.candidate_aware_model:
            candidate_model_path = os.path.join(self.model_dir, f"{prefix}candidate_aware_{timestamp}.h5")
            self.candidate_aware_model.save(candidate_model_path)
            logger.info(f"Models saved: {user_model_path}, {post_model_path}, {candidate_model_path}")
        else:
            logger.info(f"Models saved: {user_model_path}, {post_model_path}")

    def load_models(self, user_model_path: str, post_model_path: str, candidate_model_path: str = None):
        """
        Load pre-trained user and post tower models.

        Args:
            user_model_path: Path to saved user tower model
            post_model_path: Path to saved post tower model
            candidate_model_path: Optional path to candidate-aware model
        """
        self.user_model = tf.keras.models.load_model(user_model_path)
        self.post_model = tf.keras.models.load_model(post_model_path)

        if candidate_model_path and os.path.exists(candidate_model_path):
            self.candidate_aware_model = tf.keras.models.load_model(candidate_model_path)
            logger.info("Candidate-aware model loaded successfully")

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