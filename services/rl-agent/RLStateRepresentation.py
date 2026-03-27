import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import json

logger = logging.getLogger("rl-state-representation")

@dataclass
class RLState:
    """Comprehensive state representation for RL agent."""
    # Core embeddings (from existing Two-Tower system)
    user_embedding: np.ndarray  # 32D user embedding
    post_embedding: np.ndarray  # 32D post embedding
    
    # Temporal context
    temporal_features: np.ndarray  # Time-based features
    session_features: np.ndarray   # Session-level features
    sequence_features: np.ndarray  # Sequential interaction features
    
    # User context
    user_preference_vector: np.ndarray  # Derived from metadata
    user_satisfaction_score: float
    user_exploration_tendency: float
    
    # Content context
    content_features: np.ndarray  # Content metadata features
    content_appeal_score: float   # From cast/crew appeal
    content_recency_score: float  # From recency boosting
    
    # Social context
    social_signals: np.ndarray    # Social influence features
    
    # Interaction context
    interaction_history_vector: np.ndarray  # Recent interaction patterns
    
    # Full state vector (concatenation of all features)
    state_vector: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            'user_embedding': self.user_embedding.tolist(),
            'post_embedding': self.post_embedding.tolist(),
            'temporal_features': self.temporal_features.tolist(),
            'session_features': self.session_features.tolist(),
            'sequence_features': self.sequence_features.tolist(),
            'user_preference_vector': self.user_preference_vector.tolist(),
            'user_satisfaction_score': self.user_satisfaction_score,
            'user_exploration_tendency': self.user_exploration_tendency,
            'content_features': self.content_features.tolist(),
            'content_appeal_score': self.content_appeal_score,
            'content_recency_score': self.content_recency_score,
            'social_signals': self.social_signals.tolist(),
            'interaction_history_vector': self.interaction_history_vector.tolist(),
            'state_vector': self.state_vector.tolist()
        }


class RLStateBuilder:
    """
    Builds comprehensive state representations for RL agent.
    Integrates with existing Two-Tower embeddings and adds temporal/contextual features.
    """
    
    def __init__(self, user_embedding_dim: int = 32, post_embedding_dim: int = 32,
                 api_base_url: str = "http://localhost:8080"):
        """
        Initialize the state builder.
        
        Args:
            user_embedding_dim: Dimension of user embeddings
            post_embedding_dim: Dimension of post embeddings
            api_base_url: URL for Spring API (where comment analysis is stored)
        """
        self.user_embedding_dim = user_embedding_dim
        self.post_embedding_dim = post_embedding_dim
        
        # Feature dimensions
        self.temporal_dim = 8
        self.session_dim = 6
        self.sequence_dim = 10
        self.user_preference_dim = 16
        self.content_features_dim = 12
        self.social_signals_dim = 8
        self.interaction_history_dim = 20
        self.comment_analysis_dim = 13  # New: comment analysis features
        
        # Total state dimension (updated with comment features)
        self.state_dim = (
            user_embedding_dim + post_embedding_dim + 
            self.temporal_dim + self.session_dim + self.sequence_dim +
            self.user_preference_dim + self.content_features_dim + 
            self.social_signals_dim + self.interaction_history_dim +
            self.comment_analysis_dim +  # New comment analysis dimension
            3  # scalar features: satisfaction, exploration, appeal, recency
        )
        
        # Track user interaction patterns for feature engineering
        self.user_interaction_patterns: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.user_session_data: Dict[int, Dict] = defaultdict(dict)
        
        # Initialize comment analysis integration
        from RLCommentAnalysisIntegration import RLCommentAnalysisIntegrator
        self.comment_integrator = RLCommentAnalysisIntegrator(api_base_url)
        
        logger.info(f"RL State Builder initialized with state dimension: {self.state_dim}")
        logger.info("Comment analysis integration enabled (fetching from API)")
    
    def build_state(self, user_id: int, post_id: int, context: Dict[str, Any]) -> RLState:
        """
        Build comprehensive state representation.
        
        Args:
            user_id: User ID
            post_id: Post ID
            context: Context information including embeddings, metadata, etc.
            
        Returns:
            RLState object with all features
        """
        # Get core embeddings (from existing Two-Tower system)
        user_embedding = self._get_user_embedding(user_id, context)
        post_embedding = self._get_post_embedding(post_id, context)
        
        # Build temporal features
        temporal_features = self._build_temporal_features(user_id, context)
        
        # Build session features
        session_features = self._build_session_features(user_id, context)
        
        # Build sequence features
        sequence_features = self._build_sequence_features(user_id, context)
        
        # Build user preference vector
        user_preference_vector = self._build_user_preference_vector(user_id, context)
        
        # Build content features
        content_features = self._build_content_features(post_id, context)
        
        # Build social signals
        social_signals = self._build_social_signals(user_id, post_id, context)
        
        # Build interaction history vector
        interaction_history_vector = self._build_interaction_history_vector(user_id, context)
        
        # Build comment analysis features
        comment_analysis_features = self._build_comment_analysis_features(post_id, user_id, context)
        
        # Extract scalar features
        user_satisfaction_score = context.get('user_satisfaction_score', 0.5)
        user_exploration_tendency = self._calculate_exploration_tendency(user_id, context)
        content_appeal_score = context.get('content_appeal_score', 0.5)
        content_recency_score = context.get('content_recency_score', 1.0)
        
        # Concatenate all features into state vector
        state_vector = np.concatenate([
            user_embedding,
            post_embedding,
            temporal_features,
            session_features,
            sequence_features,
            user_preference_vector,
            content_features,
            social_signals,
            interaction_history_vector,
            comment_analysis_features,  # New comment analysis features
            [user_satisfaction_score, user_exploration_tendency, content_appeal_score]
        ])
        
        # Normalize state vector
        state_vector = self._normalize_state_vector(state_vector)
        
        return RLState(
            user_embedding=user_embedding,
            post_embedding=post_embedding,
            temporal_features=temporal_features,
            session_features=session_features,
            sequence_features=sequence_features,
            user_preference_vector=user_preference_vector,
            user_satisfaction_score=user_satisfaction_score,
            user_exploration_tendency=user_exploration_tendency,
            content_features=content_features,
            content_appeal_score=content_appeal_score,
            content_recency_score=content_recency_score,
            social_signals=social_signals,
            interaction_history_vector=interaction_history_vector,
            state_vector=state_vector
        )
    
    def _get_user_embedding(self, user_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Get user embedding from existing Two-Tower system."""
        # This would integrate with your existing user embedding system
        user_embedding = context.get('user_embedding')
        if user_embedding is not None:
            return np.array(user_embedding, dtype=np.float32)
        
        # Fallback: create placeholder embedding
        logger.warning(f"No user embedding found for user {user_id}, using random embedding")
        return np.random.normal(0, 0.1, self.user_embedding_dim).astype(np.float32)
    
    def _get_post_embedding(self, post_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Get post embedding from existing Two-Tower system."""
        # This would integrate with your existing post embedding system
        post_embedding = context.get('post_embedding')
        if post_embedding is not None:
            return np.array(post_embedding, dtype=np.float32)
        
        # Fallback: create placeholder embedding
        logger.warning(f"No post embedding found for post {post_id}, using random embedding")
        return np.random.normal(0, 0.1, self.post_embedding_dim).astype(np.float32)
    
    def _build_temporal_features(self, user_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Build temporal context features."""
        current_time = time.time()
        
        # Time of day features (cyclical encoding)
        hour = time.localtime(current_time).tm_hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day of week features (cyclical encoding)
        day = time.localtime(current_time).tm_wday
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        
        # Time since last interaction
        last_interaction_time = context.get('last_interaction_time', current_time)
        time_since_last = min(3600, current_time - last_interaction_time) / 3600  # Cap at 1 hour, normalize
        
        # Session duration
        session_start_time = context.get('session_start_time', current_time)
        session_duration = min(7200, current_time - session_start_time) / 7200  # Cap at 2 hours, normalize
        
        # Weekend indicator
        is_weekend = 1.0 if day >= 5 else 0.0
        
        return np.array([
            hour_sin, hour_cos, day_sin, day_cos,
            time_since_last, session_duration, is_weekend, 0.0  # 8 dimensions
        ], dtype=np.float32)
    
    def _build_session_features(self, user_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Build session-level features."""
        session_id = context.get('session_id', 'unknown')
        
        # Update session data
        if session_id not in self.user_session_data[user_id]:
            self.user_session_data[user_id][session_id] = {
                'start_time': time.time(),
                'interaction_count': 0,
                'positive_interactions': 0,
                'negative_interactions': 0,
                'unique_posts': set(),
                'content_types': set()
            }
        
        session_data = self.user_session_data[user_id][session_id]
        
        # Session length (normalized)
        session_length = session_data['interaction_count'] / 50.0  # Normalize by typical session length
        
        # Interaction rate
        session_duration = time.time() - session_data['start_time']
        interaction_rate = session_data['interaction_count'] / max(1, session_duration / 60)  # interactions per minute
        interaction_rate = min(1.0, interaction_rate / 10.0)  # Normalize
        
        # Positive interaction ratio
        total_feedback = session_data['positive_interactions'] + session_data['negative_interactions']
        positive_ratio = session_data['positive_interactions'] / max(1, total_feedback)
        
        # Content diversity in session
        content_diversity = len(session_data['unique_posts']) / max(1, session_data['interaction_count'])
        
        # Content type diversity
        type_diversity = len(session_data['content_types']) / max(1, len(session_data['content_types']))
        
        return np.array([
            session_length, interaction_rate, positive_ratio,
            content_diversity, type_diversity, 0.0  # 6 dimensions
        ], dtype=np.float32)
    
    def _build_sequence_features(self, user_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Build sequential interaction features."""
        interaction_history = list(self.user_interaction_patterns[user_id])
        
        if len(interaction_history) < 2:
            return np.zeros(self.sequence_dim, dtype=np.float32)
        
        # Recent interaction pattern (last 5 interactions)
        recent_interactions = interaction_history[-5:]
        
        # Count interaction types
        interaction_type_counts = defaultdict(int)
        for interaction in recent_interactions:
            interaction_type_counts[interaction.get('type', 'unknown')] += 1
        
        # Normalize counts
        total_recent = len(recent_interactions)
        like_ratio = interaction_type_counts['like'] / total_recent
        save_ratio = interaction_type_counts['save'] / total_recent
        skip_ratio = interaction_type_counts['skip'] / total_recent
        not_interested_ratio = interaction_type_counts['not_interested'] / total_recent
        
        # Temporal patterns
        if len(recent_interactions) >= 2:
            time_diffs = []
            for i in range(1, len(recent_interactions)):
                time_diff = recent_interactions[i]['timestamp'] - recent_interactions[i-1]['timestamp']
                time_diffs.append(min(300, time_diff))  # Cap at 5 minutes
            
            avg_time_between = np.mean(time_diffs) / 300.0  # Normalize
            time_consistency = 1.0 - (np.std(time_diffs) / np.mean(time_diffs)) if np.mean(time_diffs) > 0 else 0.0
        else:
            avg_time_between = 0.0
            time_consistency = 0.0
        
        # Trend features
        recent_scores = [interaction.get('reward', 0.0) for interaction in recent_interactions]
        if len(recent_scores) >= 3:
            score_trend = recent_scores[-1] - recent_scores[0]  # Simple trend
            score_variance = np.var(recent_scores)
        else:
            score_trend = 0.0
            score_variance = 0.0
        
        return np.array([
            like_ratio, save_ratio, skip_ratio, not_interested_ratio,
            avg_time_between, time_consistency, score_trend, score_variance,
            0.0, 0.0  # 10 dimensions
        ], dtype=np.float32)
    
    def _build_user_preference_vector(self, user_id: int, context: Dict[str, Any]) -> np.ndarray:
        """
        Build user preference vector from BEHAVIORAL data only.

        TMDB ToS Compliant: This method derives all features from user interactions
        within the app, NOT from TMDB metadata like interestWeights or castCrewPreferences.

        Behavioral features (16D total):
        - 8D: Genre preferences from interaction patterns (likes/saves vs skips)
        - 4D: Language preferences from interaction history
        - 2D: Engagement patterns (depth, positive_ratio)
        - 2D: Recency features (recent_rate, velocity)
        """
        interaction_history = list(self.user_interaction_patterns[user_id])

        # 8D: Genre preferences from behavioral interaction patterns
        genre_scores = self._calc_behavioral_genre_prefs(interaction_history)

        # 4D: Language preferences from interactions
        language_scores = self._calc_behavioral_language_prefs(interaction_history)

        # 2D: Engagement patterns (depth, positive_ratio)
        engagement_patterns = self._calc_engagement_patterns(interaction_history)

        # 2D: Recency features (recent_rate, velocity)
        recency_features = self._calc_recency_features(interaction_history)

        preference_vector = np.concatenate([
            genre_scores, language_scores, engagement_patterns, recency_features
        ]).astype(np.float32)

        # Pad or truncate to desired dimension
        if len(preference_vector) > self.user_preference_dim:
            preference_vector = preference_vector[:self.user_preference_dim]
        else:
            padding = np.zeros(self.user_preference_dim - len(preference_vector))
            preference_vector = np.concatenate([preference_vector, padding])

        return preference_vector

    def _calc_behavioral_genre_prefs(self, history: List[Dict]) -> np.ndarray:
        """
        Calculate genre preferences from behavioral data (likes/saves vs skips).

        TMDB ToS Compliant: Uses interaction types, not TMDB metadata.
        """
        genre_engagement = defaultdict(lambda: {'pos': 0, 'neg': 0, 'total': 0})

        for interaction in history:
            # Get genres from the interaction record (stored at interaction time)
            for genre in interaction.get('genres', []):
                genre_engagement[genre]['total'] += 1
                if interaction.get('type') in ['like', 'save', 'share']:
                    genre_engagement[genre]['pos'] += 1
                elif interaction.get('type') in ['not_interested', 'skip']:
                    genre_engagement[genre]['neg'] += 1

        # Score = (pos - neg) / total, normalized to [0, 1]
        prefs = {}
        for g, c in genre_engagement.items():
            if c['total'] > 0:
                prefs[g] = (c['pos'] - c['neg']) / max(1, c['total']) / 2 + 0.5

        # Get top 8 genres by preference score
        top_8 = sorted(prefs.items(), key=lambda x: x[1], reverse=True)[:8]
        result = [p for _, p in top_8] + [0.5] * (8 - len(top_8))
        return np.array(result, dtype=np.float32)

    def _calc_behavioral_language_prefs(self, history: List[Dict]) -> np.ndarray:
        """
        Calculate language preferences from behavioral data.

        TMDB ToS Compliant: Uses interaction language data, not TMDB languageWeights.
        """
        lang_engagement = defaultdict(lambda: {'pos': 0, 'total': 0})

        for interaction in history:
            lang = interaction.get('language', 'en')
            lang_engagement[lang]['total'] += 1
            if interaction.get('type') in ['like', 'save', 'share', 'more_info']:
                lang_engagement[lang]['pos'] += 1

        # Calculate preference scores
        prefs = {}
        for lang, counts in lang_engagement.items():
            if counts['total'] > 0:
                prefs[lang] = counts['pos'] / counts['total']

        # Get top 4 languages
        top_4 = sorted(prefs.items(), key=lambda x: x[1], reverse=True)[:4]
        result = [p for _, p in top_4] + [0.0] * (4 - len(top_4))
        return np.array(result, dtype=np.float32)

    def _calc_engagement_patterns(self, history: List[Dict]) -> np.ndarray:
        """
        Calculate engagement patterns from behavioral data.

        Returns 2D: [engagement_depth, positive_ratio]
        """
        if not history:
            return np.array([0.5, 0.5], dtype=np.float32)

        # Engagement depth: variety of interaction types
        interaction_types = set(h.get('type', 'unknown') for h in history)
        depth = min(1.0, len(interaction_types) / 5.0)  # Normalize by expected variety

        # Positive ratio: positive vs negative interactions
        positive = sum(1 for h in history if h.get('type') in ['like', 'save', 'share', 'more_info'])
        negative = sum(1 for h in history if h.get('type') in ['not_interested', 'skip'])
        total = positive + negative
        positive_ratio = positive / total if total > 0 else 0.5

        return np.array([depth, positive_ratio], dtype=np.float32)

    def _calc_recency_features(self, history: List[Dict]) -> np.ndarray:
        """
        Calculate recency features from behavioral data.

        Returns 2D: [recent_rate, velocity]
        """
        if len(history) < 2:
            return np.array([0.5, 0.5], dtype=np.float32)

        current_time = time.time()

        # Recent rate: proportion of interactions in last hour
        recent_count = sum(1 for h in history
                         if current_time - h.get('timestamp', 0) < 3600)
        recent_rate = min(1.0, recent_count / max(1, len(history)))

        # Velocity: interaction frequency trend
        # Compare recent interactions to older interactions
        recent_interactions = [h for h in history
                             if current_time - h.get('timestamp', 0) < 3600]
        older_interactions = [h for h in history
                             if current_time - h.get('timestamp', 0) >= 3600]

        if len(recent_interactions) > 0 and len(older_interactions) > 0:
            # Calculate time between interactions for each group
            velocity = min(1.0, len(recent_interactions) / len(older_interactions))
        else:
            velocity = 0.5

        return np.array([recent_rate, velocity], dtype=np.float32)
    
    def _build_content_features(self, post_id: int, context: Dict[str, Any]) -> np.ndarray:
        """
        Build content-specific features from BEHAVIORAL data only.

        TMDB ToS Compliant: This method derives features from app behavioral data,
        NOT from TMDB metadata like voteAverage, popularity, genreWeights.

        Behavioral features (12D total):
        - 4D: Comment-based features (sentiment from app comment analysis)
        - 4D: Interaction rates (clicks, like_rate, save_rate, share_rate)
        - 4D: Engagement velocity (recent activity trends)
        """
        post_metadata = context.get('post_metadata', {})

        # 4D: Comment-based features (sentiment from app's comment analysis)
        comment_features = self._get_comment_features(post_id, post_metadata)

        # 4D: Interaction rates (app behavioral data, NOT TMDB)
        interaction_features = self._get_interaction_rates(post_metadata)

        # 4D: Engagement velocity (recent activity trends from app)
        velocity_features = self._get_engagement_velocity(post_metadata)

        content_features = np.concatenate([
            comment_features, interaction_features, velocity_features
        ]).astype(np.float32)

        # Ensure correct dimension
        if len(content_features) > self.content_features_dim:
            content_features = content_features[:self.content_features_dim]
        else:
            padding = np.zeros(self.content_features_dim - len(content_features))
            content_features = np.concatenate([content_features, padding])

        return content_features

    def _get_comment_features(self, post_id: int, post_metadata: Dict) -> np.ndarray:
        """
        Get comment-based features from app's comment analysis.

        Returns 4D: [sentiment_score, comment_activity, controversy_score, helpfulness]
        """
        # Comment analysis from app (NOT TMDB)
        comment_analysis = post_metadata.get('commentAnalysis', {})

        if comment_analysis:
            sentiment = comment_analysis.get('averageSentiment', 0.5)
            activity = min(1.0, comment_analysis.get('commentCount', 0) / 100.0)
            controversy = comment_analysis.get('controversyScore', 0.0)
            helpfulness = comment_analysis.get('helpfulnessScore', 0.5)
        else:
            sentiment = 0.5
            activity = 0.0
            controversy = 0.0
            helpfulness = 0.5

        return np.array([sentiment, activity, controversy, helpfulness], dtype=np.float32)

    def _get_interaction_rates(self, post_metadata: Dict) -> np.ndarray:
        """
        Get interaction rates from app behavioral data.

        Returns 4D: [click_rate, like_rate, save_rate, share_rate]

        TMDB ToS Compliant: All metrics from app interactions, NOT TMDB.
        """
        # Info button clicks (app engagement metric)
        info_clicks = post_metadata.get('infoButtonClicks', {})
        click_count = info_clicks.get('count', 0) if isinstance(info_clicks, dict) else 0
        clicks = min(1.0, click_count / 100.0)

        # View count for calculating rates
        views = max(1, post_metadata.get('viewCount', 1))

        # Interaction rates from app data
        like_rate = min(1.0, post_metadata.get('likeCount', 0) / views)
        save_rate = min(1.0, post_metadata.get('saveCount', 0) / views)
        share_rate = min(1.0, post_metadata.get('shareCount', 0) / views)

        return np.array([clicks, like_rate, save_rate, share_rate], dtype=np.float32)

    def _get_engagement_velocity(self, post_metadata: Dict) -> np.ndarray:
        """
        Get engagement velocity features (recent activity trends).

        Returns 4D: [is_trending, hourly_velocity, daily_velocity, velocity_trend]

        TMDB ToS Compliant: All from app behavioral data.

        Supports two velocity data formats:
        1. Legacy: recentEngagement with likesLast24h, savesLast24h, etc.
        2. New: engagementVelocity with isTrending, hourlyVelocity, dailyVelocity
        """
        # Try new velocity data format first (from /posts/{postId}/engagement-velocity endpoint)
        engagement_velocity = post_metadata.get('engagementVelocity', {})

        if engagement_velocity:
            # New format with explicit trending and velocity data
            is_trending = 1.0 if engagement_velocity.get('isTrending', False) else 0.0
            hourly_velocity = min(1.0, engagement_velocity.get('hourlyVelocity', 0.0) / 10.0)
            daily_velocity = min(1.0, engagement_velocity.get('dailyVelocity', 0.0) / 100.0)

            # Velocity trend: compare hourly vs daily
            hourly = engagement_velocity.get('hourlyVelocity', 0.0)
            daily = engagement_velocity.get('dailyVelocity', 0.0)
            if daily > 0:
                velocity_trend = min(1.0, hourly / (daily / 24))  # How today's trend compares to daily avg
            else:
                velocity_trend = 0.5

            return np.array([is_trending, hourly_velocity, daily_velocity, velocity_trend],
                           dtype=np.float32)

        # Fallback to legacy format
        recent_engagement = post_metadata.get('recentEngagement', {})

        if recent_engagement:
            # Derive trending indicator from engagement trends
            likes_24h = recent_engagement.get('likesLast24h', 0)
            saves_24h = recent_engagement.get('savesLast24h', 0)
            comments_24h = recent_engagement.get('commentsLast24h', 0)

            # Estimated trending based on engagement volume
            is_trending = 1.0 if (likes_24h + saves_24h + comments_24h) > 50 else 0.0

            like_velocity = min(1.0, likes_24h / 50.0)
            save_velocity = min(1.0, saves_24h / 20.0)
            comment_velocity = min(1.0, comments_24h / 30.0)
            trend_score = recent_engagement.get('trendScore', 0.5)

            # Combine into 4D: is_trending, avg_velocity, max_velocity, trend_score
            avg_velocity = (like_velocity + save_velocity + comment_velocity) / 3.0
            max_velocity = max(like_velocity, save_velocity, comment_velocity)

            return np.array([is_trending, avg_velocity, max_velocity, trend_score],
                           dtype=np.float32)
        else:
            return np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    
    def _build_social_signals(self, user_id: int, post_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Build social influence features."""
        social_data = context.get('social_signals', {})
        
        # Social influence score
        social_influence = social_data.get('social_influence_score', 0.0)
        
        # Following user interactions with this content
        following_interactions = social_data.get('following_interactions', {})
        following_likes = following_interactions.get('likes', 0) / max(1, following_interactions.get('total', 1))
        following_saves = following_interactions.get('saves', 0) / max(1, following_interactions.get('total', 1))
        
        # User's social activity level
        user_social_activity = social_data.get('user_social_activity', 0.0)
        
        # Content social popularity
        content_social_score = social_data.get('content_social_score', 0.0)
        
        # Network effects
        network_size = min(1.0, social_data.get('user_network_size', 0) / 100.0)  # Normalize
        network_activity = social_data.get('network_activity_score', 0.0)
        
        return np.array([
            social_influence, following_likes, following_saves, user_social_activity,
            content_social_score, network_size, network_activity, 0.0
        ], dtype=np.float32)
    
    def _build_interaction_history_vector(self, user_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Build vector representing user's interaction history patterns."""
        interaction_history = list(self.user_interaction_patterns[user_id])
        
        if not interaction_history:
            return np.zeros(self.interaction_history_dim, dtype=np.float32)
        
        # Aggregate statistics over different time windows
        now = time.time()
        
        # Last hour, last day, last week
        windows = [3600, 86400, 604800]  # 1 hour, 1 day, 1 week in seconds
        window_stats = []
        
        for window in windows:
            window_interactions = [
                interaction for interaction in interaction_history
                if now - interaction['timestamp'] <= window
            ]
            
            if window_interactions:
                # Interaction count (normalized)
                count = len(window_interactions) / 50.0  # Normalize by expected max
                
                # Average reward
                avg_reward = np.mean([interaction.get('reward', 0.0) for interaction in window_interactions])
                avg_reward = (avg_reward + 1.0) / 2.0  # Normalize to 0-1
                
                # Interaction diversity
                interaction_types = set(interaction['type'] for interaction in window_interactions)
                diversity = len(interaction_types) / max(1, len(window_interactions))
                
                window_stats.extend([count, avg_reward, diversity])
            else:
                window_stats.extend([0.0, 0.5, 0.0])  # No interactions in this window
        
        # Overall statistics
        if interaction_history:
            total_interactions = len(interaction_history)
            avg_session_length = total_interactions / max(1, len(set(interaction.get('session_id', 'unknown') 
                                                                   for interaction in interaction_history)))
            
            # User consistency (how consistent are their interaction patterns)
            recent_rewards = [interaction.get('reward', 0.0) for interaction in interaction_history[-10:]]
            consistency = 1.0 - np.std(recent_rewards) if len(recent_rewards) > 1 else 1.0
            
            # Exploration tendency
            recent_positions = [interaction.get('position', 0) for interaction in interaction_history[-20:]]
            avg_position = np.mean(recent_positions) if recent_positions else 0.0
            exploration_score = min(1.0, avg_position / 10.0)  # Normalize
            
            global_stats = [
                min(1.0, total_interactions / 1000.0),  # Total interactions (normalized)
                min(1.0, avg_session_length / 20.0),    # Avg session length (normalized)
                consistency,
                exploration_score,
                0.0  # Padding
            ]
        else:
            global_stats = [0.0, 0.0, 0.5, 0.0, 0.0]
        
        history_vector = np.array(window_stats + global_stats, dtype=np.float32)
        
        # Ensure correct dimension
        if len(history_vector) > self.interaction_history_dim:
            history_vector = history_vector[:self.interaction_history_dim]
        else:
            padding = np.zeros(self.interaction_history_dim - len(history_vector))
            history_vector = np.concatenate([history_vector, padding])
        
        return history_vector
    
    def _build_comment_analysis_features(self, post_id: int, user_id: int, context: Dict[str, Any]) -> np.ndarray:
        """Build comment analysis features for RL state."""
        try:
            # Get comment features from integrator
            comment_features = self.comment_integrator.get_comment_features_for_state(post_id, user_id)
            
            # Ensure correct dimension
            if len(comment_features) != self.comment_analysis_dim:
                logger.warning(f"Comment features dimension mismatch: expected {self.comment_analysis_dim}, got {len(comment_features)}")
                # Pad or truncate as needed
                if len(comment_features) > self.comment_analysis_dim:
                    comment_features = comment_features[:self.comment_analysis_dim]
                else:
                    padding = np.zeros(self.comment_analysis_dim - len(comment_features))
                    comment_features = np.concatenate([comment_features, padding])
            
            return comment_features
            
        except Exception as e:
            logger.error(f"Error building comment analysis features: {e}")
            # Return default features
            return np.zeros(self.comment_analysis_dim, dtype=np.float32)
    
    def _calculate_exploration_tendency(self, user_id: int, context: Dict[str, Any]) -> float:
        """Calculate user's tendency to explore (vs exploit)."""
        interaction_history = list(self.user_interaction_patterns[user_id])
        
        if len(interaction_history) < 5:
            return 0.5  # Neutral for new users
        
        # Look at ranking positions of content user interacted with
        recent_positions = []
        for interaction in interaction_history[-20:]:  # Last 20 interactions
            position = interaction.get('position', 0)
            if position > 0:  # Valid position
                recent_positions.append(position)
        
        if not recent_positions:
            return 0.5
        
        # Calculate exploration score based on positions
        avg_position = np.mean(recent_positions)
        exploration_score = min(1.0, avg_position / 15.0)  # Normalize by reasonable max position
        
        # Adjust for variety in positions (consistent exploration vs random clicking)
        position_variety = np.std(recent_positions) / max(1, np.mean(recent_positions))
        exploration_score *= min(1.0, position_variety + 0.5)  # Boost for variety
        
        return min(1.0, exploration_score)
    
    def _encode_categorical(self, value: str, categories: List[str]) -> List[float]:
        """One-hot encode categorical variable."""
        encoding = [0.0] * len(categories)
        if value in categories:
            encoding[categories.index(value)] = 1.0
        else:
            # Unknown category
            encoding[-1] = 1.0 if 'other' in categories else 0.0
        return encoding
    
    def _normalize_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        """Normalize state vector to reasonable range."""
        # Clip extreme values
        state_vector = np.clip(state_vector, -5.0, 5.0)
        
        # Apply tanh normalization to keep values in reasonable range
        state_vector = np.tanh(state_vector / 2.0)
        
        return state_vector
    
    def update_user_interaction(self, user_id: int, interaction_data: Dict[str, Any]):
        """Update user interaction patterns for feature engineering."""
        self.user_interaction_patterns[user_id].append({
            'timestamp': time.time(),
            'type': interaction_data.get('interaction_type', 'unknown'),
            'post_id': interaction_data.get('post_id'),
            'reward': interaction_data.get('reward', 0.0),
            'position': interaction_data.get('position', 0),
            'session_id': interaction_data.get('session_id', 'unknown')
        })
    
    def get_state_dimension(self) -> int:
        """Get the total dimension of the state vector."""
        return self.state_dim
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about state representation."""
        return {
            'total_dimension': self.state_dim,
            'component_dimensions': {
                'user_embedding': self.user_embedding_dim,
                'post_embedding': self.post_embedding_dim,
                'temporal_features': self.temporal_dim,
                'session_features': self.session_dim,
                'sequence_features': self.sequence_dim,
                'user_preference_vector': self.user_preference_dim,
                'content_features': self.content_features_dim,
                'social_signals': self.social_signals_dim,
                'interaction_history': self.interaction_history_dim,
                'scalar_features': 3
            },
            'users_tracked': len(self.user_interaction_patterns)
        }