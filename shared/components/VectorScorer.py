"""NumPy-based vector similarity scoring for recommendations.

The deployed recommendation path compares the 64-dimensional user and post
vectors built by the Spring API. No neural model or checkpoint is required.

Scoring uses raw cosine similarity (range [-1, 1]) with optional confidence
blending and negative-centroid penalty. See docs/IDEAL_VECTOR_SETUP_PLAN.md.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("vector-scorer")


def compute_scores(user_embeddings: np.ndarray, post_embeddings: np.ndarray,
                   content_type: str = "posts", candidates: Optional[List[Dict]] = None) -> np.ndarray:
    """Compute raw cosine similarity scores with optional candidate boosts.

    Args:
        user_embeddings: User vectors of shape (batch_size, embedding_dim).
        post_embeddings: Post vectors of shape (num_posts, embedding_dim).
        content_type: Content type ("posts" or "trailers").
        candidates: Optional candidate metadata used for score boosts.

    Returns:
        Scores of shape (batch_size, num_posts) as raw cosine similarity
        in the [-1, 1] range. Wider spread than the old [0.5, 1.0] mapping
        gives heuristics and RL less dominance over ranking.
    """
    user_embeddings = np.asarray(user_embeddings, dtype=np.float64)
    post_embeddings = np.asarray(post_embeddings, dtype=np.float64)

    user_norm = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    post_norm = np.linalg.norm(post_embeddings, axis=1, keepdims=True)

    user_embeddings_normalized = user_embeddings / np.maximum(user_norm, 1e-8)
    post_embeddings_normalized = post_embeddings / np.maximum(post_norm, 1e-8)

    base_scores = np.matmul(user_embeddings_normalized, post_embeddings_normalized.T)

    if candidates:
        return _apply_candidate_boosts(base_scores, candidates, content_type)

    return base_scores


def score_with_confidence(cosine_scores: np.ndarray,
                          popularity_scores: np.ndarray,
                          interaction_count: int,
                          k: int = 10) -> np.ndarray:
    """Blend cosine scores with popularity based on user interaction count.

    confidence(n) = n / (n + k).  At n=0 the result is pure popularity;
    at n=50 (k=10) it is 83% personalised.  Graceful cold-start degradation.

    Args:
        cosine_scores: Raw cosine scores from compute_scores.
        popularity_scores: Pre-computed popularity scores in [0, 1],
            same shape as cosine_scores.
        interaction_count: Total positive interactions for this user.
        k: Half-saturation constant (interaction count at 50% confidence).

    Returns:
        Blended scores, same shape as cosine_scores.
    """
    confidence = interaction_count / (interaction_count + k)
    return confidence * cosine_scores + (1.0 - confidence) * popularity_scores


def score_with_negatives(cosine_scores: np.ndarray,
                         post_vectors: np.ndarray,
                         negative_centroid: Optional[np.ndarray],
                         lambda_val: float = 0.3) -> np.ndarray:
    """Penalise posts that are similar to negatively-rated content.

    The penalty is continuous and recoverable — unlike a hard avoid-genre
    filter, new positive interactions shift the user vector away from the
    negative centroid and the penalty fades naturally.

    Args:
        cosine_scores: Raw cosine scores from compute_scores, shape (num_posts,).
        post_vectors: Post vectors of shape (num_posts, embedding_dim).
        negative_centroid: Average vector of disliked posts, shape (embedding_dim,).
            None or zero vector disables the penalty.
        lambda_val: Penalty strength (0 disables).

    Returns:
        Adjusted scores, same shape as cosine_scores.
    """
    if negative_centroid is None or lambda_val == 0.0:
        return cosine_scores

    negative_centroid = np.asarray(negative_centroid, dtype=np.float64)
    neg_norm = np.linalg.norm(negative_centroid)
    if neg_norm < 1e-8:
        return cosine_scores

    neg_unit = negative_centroid / neg_norm
    post_norms = np.linalg.norm(post_vectors, axis=1, keepdims=True)
    post_unit = post_vectors / np.maximum(post_norms, 1e-8)
    neg_similarity = post_unit @ neg_unit  # (num_posts,)

    return cosine_scores - lambda_val * neg_similarity


def compute_socially_enhanced_scores(user_embeddings: np.ndarray,
                                     post_embeddings: np.ndarray,
                                     user_id: int,
                                     post_ids: List[int],
                                     social_processor: Any,
                                     content_type: str = "posts",
                                     candidates: Optional[List[Dict]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute cosine scores with social-signal enhancement."""
    try:
        base_scores = compute_scores(user_embeddings, post_embeddings, content_type, candidates)

        if user_id and social_processor:
            logger.info("Applying social enhancement for user %s", user_id)
            enhanced_scores, social_metadata = social_processor.process_social_boost(
                user_id=user_id,
                post_ids=post_ids,
                user_embedding=user_embeddings[0] if user_embeddings.shape[0] > 0 else user_embeddings,
                post_embeddings=post_embeddings
            )

            if enhanced_scores.ndim == 1:
                enhanced_scores = enhanced_scores.reshape(1, -1)

            return enhanced_scores, social_metadata

        logger.info("No social enhancement applied - using base scores")
        return base_scores, {"social_enhancement": False}

    except Exception as e:
        logger.error("Error in socially enhanced scoring: %s", e)
        base_scores = compute_scores(user_embeddings, post_embeddings, content_type, candidates)
        return base_scores, {"error": str(e), "fallback": True}


def compute_dual_preference_scores(user_embeddings: np.ndarray,
                                   post_embeddings: np.ndarray,
                                   user_id: int,
                                   post_ids: List[int],
                                   positive_interactions: Optional[List[int]] = None,
                                   negative_interactions: Optional[List[int]] = None,
                                   content_type: str = "posts") -> np.ndarray:
    """Compute cosine scores with positive/negative interaction adjustments."""
    try:
        base_scores = compute_scores(user_embeddings, post_embeddings, content_type)

        if base_scores.ndim == 1:
            base_scores = base_scores.reshape(1, -1)

        enhanced_scores = base_scores.copy()

        if positive_interactions:
            positive_set = set(positive_interactions)
            for i, post_id in enumerate(post_ids):
                if post_id in positive_set and i < enhanced_scores.shape[1]:
                    enhanced_scores[0, i] *= 1.2

        if negative_interactions:
            negative_set = set(negative_interactions)
            for i, post_id in enumerate(post_ids):
                if post_id in negative_set and i < enhanced_scores.shape[1]:
                    enhanced_scores[0, i] *= 0.7

        enhanced_scores = np.clip(enhanced_scores, -1.0, 1.0)

        logger.info("Applied dual preference boosting for user %s", user_id)
        return enhanced_scores

    except Exception as e:
        logger.error("Error in dual preference scoring: %s", e)
        return compute_scores(user_embeddings, post_embeddings, content_type)


def compute_filtered_scores(user_embeddings: np.ndarray,
                            post_embeddings: np.ndarray,
                            user_id: int,
                            post_ids: List[int],
                            content_filter: Any,
                            content_type: str = "posts",
                            apply_social_boost: bool = True,
                            social_processor: Any = None) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """Compute scores and apply content filtering."""
    try:
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

        filtered_post_ids, filtered_scores, filter_metadata = content_filter.filter_recommendations(
            user_id=user_id,
            post_ids=post_ids,
            scores=base_scores,
            apply_user_preferences=True
        )

        combined_metadata = {
            **social_metadata,
            **filter_metadata,
            "content_filtering_applied": True
        }

        logger.info(
            "Applied content filtering: %s -> %s posts for user %s",
            len(post_ids), len(filtered_post_ids), user_id
        )
        return filtered_scores, filtered_post_ids, combined_metadata

    except Exception as e:
        logger.error("Error in filtered scoring: %s", e)
        base_scores = compute_scores(user_embeddings, post_embeddings, content_type)
        if base_scores.ndim > 1:
            base_scores = base_scores.flatten()
        return base_scores, post_ids, {"error": str(e), "content_filtering_applied": False}


def _apply_candidate_boosts(scores: np.ndarray, candidates: List[Dict], content_type: str) -> np.ndarray:
    """Apply score boosts based on candidate metadata."""
    enhanced_scores = scores.copy()

    for i in range(scores.shape[1]):
        if i < len(candidates):
            candidate = candidates[i]
            candidate_type = candidate.get('candidateType', '')
            source = candidate.get('source', '')
            original_score = candidate.get('score', 1.0)
            boost_factor = _get_candidate_boost_factor(
                candidate_type, source, content_type, original_score
            )

            for j in range(scores.shape[0]):
                enhanced_scores[j, i] *= boost_factor

    return np.clip(enhanced_scores, -1.0, 1.0)


def _get_candidate_boost_factor(candidate_type: str, source: str,
                                content_type: str, original_score: float) -> float:
    """Calculate a boost factor from candidate metadata."""
    boost_factor = 1.0

    type_boosts = {
        'NEW_HIGH_QUALITY': 1.15,
        'NEW_HIGH_QUALITY_TRAILERS': 1.15,
        'CURSOR_BASED': 1.05,
        'CURSOR_BASED_TRAILERS': 1.05,
        'STRATIFIED_PRIMARY': 1.10,
        'CROSS_LANGUAGE': 0.95,
        'EMERGENCY_FALLBACK': 0.85,
        'EMERGENCY_FALLBACK_TRAILERS': 0.85
    }

    boost_factor *= type_boosts.get(candidate_type, 1.0)

    source_boosts = {
        'PRIMARY': 1.05,
        'NEW_HIGH_QUALITY': 1.10,
        'FALLBACK': 0.95
    }

    boost_factor *= source_boosts.get(source, 1.0)

    if original_score > 1.2:
        boost_factor *= 1.08
    elif original_score > 1.0:
        boost_factor *= 1.03
    elif original_score < 0.8:
        boost_factor *= 0.95

    if content_type == "trailers" and "TRAILER" in candidate_type:
        boost_factor *= 1.05

    return boost_factor
