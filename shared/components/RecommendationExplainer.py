"""
RecommendationExplainer - Separate module for explaining recommendation decisions.

This module analyzes scored recommendations and generates human-readable
explanations for why each post was recommended. It runs as a separate step
after the scoring/enhancement flow.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("recommendation-explainer")


@dataclass
class RecommendationReason:
    """A single reason contributing to a recommendation."""
    factor: str
    weight: float
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "weight": round(self.weight, 3),
            "description": self.description,
            "details": self.details
        }


@dataclass
class PostExplanation:
    """Complete explanation for a single post recommendation."""
    post_id: int
    base_score: float
    final_score: float
    rank: int
    reasons: List[RecommendationReason] = field(default_factory=list)
    primary_reason: str = ""

    def add_reason(self, factor: str, weight: float, description: str,
                   details: Dict[str, Any] = None):
        self.reasons.append(RecommendationReason(
            factor=factor,
            weight=weight,
            description=description,
            details=details or {}
        ))

    def get_primary_reason(self) -> str:
        """Get the most significant reason for this recommendation."""
        if self.primary_reason:
            return self.primary_reason

        if not self.reasons:
            return "Matches your viewing preferences"

        # Find highest weighted reason
        top_reason = max(self.reasons, key=lambda r: r.weight)
        return top_reason.description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "postId": self.post_id,
            "rank": self.rank,
            "baseScore": round(self.base_score, 4),
            "finalScore": round(self.final_score, 4),
            "scoreBoost": round(self.final_score - self.base_score, 4),
            "primaryReason": self.get_primary_reason(),
            "factors": [r.to_dict() for r in self.reasons]
        }

    def to_compact(self) -> Dict[str, Any]:
        """Compact format with just key info."""
        return {
            "postId": self.post_id,
            "reason": self.get_primary_reason(),
            "confidence": round(self.final_score, 2)
        }


class RecommendationExplainer:
    """
    Analyzes scored recommendations and generates explanations.

    This is a separate step that runs after MetadataEnhancer scoring.
    It examines user preferences and post metadata to determine
    which factors contributed to each recommendation.
    """

    def __init__(self, api_base_url: str, metadata_cache: Dict = None):
        """
        Initialize the explainer.

        Args:
            api_base_url: Base URL for API calls (if needed)
            metadata_cache: Shared cache from MetadataEnhancer (optional)
        """
        self.api_base_url = api_base_url
        self.metadata_cache = metadata_cache or {}

        # Boost factor thresholds for explanation
        self.boost_thresholds = {
            'language': 0.3,
            'genre': 0.2,
            'cast_crew': 0.2,
            'popularity': 0.15,
            'recency': 0.25,
            'demographic': 0.1,
            'engagement': 0.1
        }

    def explain_recommendations(
        self,
        user_id: str,
        post_ids: List[int],
        base_scores: List[float],
        final_scores: List[float],
        user_metadata: Dict[str, Any] = None,
        post_metadata_map: Dict[int, Dict] = None,
        rl_actions: List[Dict] = None,
        content_type: str = "posts"
    ) -> List[PostExplanation]:
        """
        Generate explanations for a list of recommendations.

        Args:
            user_id: The user receiving recommendations
            post_ids: List of recommended post IDs (in ranked order)
            base_scores: Base scores from Two-Tower model
            final_scores: Final scores after all enhancements
            user_metadata: User preferences and profile data
            post_metadata_map: Metadata for each post {post_id: metadata}
            rl_actions: RL actions that were applied (if any)
            content_type: Type of content (posts/trailers)

        Returns:
            List of PostExplanation objects
        """
        explanations = []

        user_metadata = user_metadata or {}
        post_metadata_map = post_metadata_map or {}

        for rank, post_id in enumerate(post_ids):
            base_score = base_scores[rank] if rank < len(base_scores) else 0.0
            final_score = final_scores[rank] if rank < len(final_scores) else 0.0
            post_metadata = post_metadata_map.get(post_id, {})

            explanation = PostExplanation(
                post_id=post_id,
                base_score=float(base_score),
                final_score=float(final_score),
                rank=rank + 1
            )

            # Analyze each factor
            self._analyze_base_score(explanation, base_score)
            self._analyze_language_match(explanation, user_metadata, post_metadata)
            self._analyze_genre_match(explanation, user_metadata, post_metadata)
            self._analyze_cast_crew_match(explanation, user_metadata, post_metadata)
            self._analyze_popularity(explanation, post_metadata)
            self._analyze_recency(explanation, post_metadata)
            self._analyze_demographic_match(explanation, user_metadata, post_metadata)
            self._analyze_engagement(explanation, post_metadata)

            # Analyze RL adjustments if present
            if rl_actions:
                self._analyze_rl_adjustments(explanation, rl_actions)

            explanations.append(explanation)

        return explanations

    def _analyze_base_score(self, explanation: PostExplanation, base_score: float):
        """Analyze the base ML model score."""
        if base_score > 0.7:
            explanation.add_reason(
                factor="ml_similarity",
                weight=base_score,
                description="Strong match based on your viewing history",
                details={"model": "two_tower", "similarity": base_score}
            )
        elif base_score > 0.5:
            explanation.add_reason(
                factor="ml_similarity",
                weight=base_score,
                description="Good match for your preferences",
                details={"model": "two_tower", "similarity": base_score}
            )
        else:
            explanation.add_reason(
                factor="ml_similarity",
                weight=base_score,
                description="Potential new interest for you",
                details={"model": "two_tower", "similarity": base_score}
            )

    def _analyze_language_match(self, explanation: PostExplanation,
                                user_metadata: Dict, post_metadata: Dict):
        """Check if post language matches user preferences."""
        user_lang_prefs = user_metadata.get('languageWeights', {}).get('weights', {})
        if not user_lang_prefs:
            return

        post_features = post_metadata.get('categoricalFeatures', {})
        post_language = post_features.get('language', 'en')

        if post_language in user_lang_prefs:
            weight = user_lang_prefs[post_language]
            if weight > 0.3:
                explanation.add_reason(
                    factor="language",
                    weight=weight * self.boost_thresholds['language'],
                    description=f"In your preferred language ({post_language.upper()})",
                    details={"language": post_language, "preference_weight": weight}
                )

    def _analyze_genre_match(self, explanation: PostExplanation,
                            user_metadata: Dict, post_metadata: Dict):
        """Check genre alignment with user preferences."""
        user_interests = user_metadata.get('interestWeights', {})
        post_genres = post_metadata.get('genreWeights', {})

        if not user_interests or not post_genres:
            return

        # Find matching genres
        matching_genres = []
        total_alignment = 0.0

        for genre, post_weight in post_genres.items():
            if genre in user_interests:
                user_pref = user_interests[genre]
                if user_pref > 0.3:
                    matching_genres.append(genre)
                    total_alignment += user_pref * post_weight

        if matching_genres:
            primary_genre = matching_genres[0]
            explanation.add_reason(
                factor="genre",
                weight=min(total_alignment, 1.0) * self.boost_thresholds['genre'],
                description=f"Matches your interest in {primary_genre.title()}",
                details={
                    "matching_genres": matching_genres[:3],
                    "alignment_score": total_alignment
                }
            )

    def _analyze_cast_crew_match(self, explanation: PostExplanation,
                                 user_metadata: Dict, post_metadata: Dict):
        """Check if cast/crew matches user preferences."""
        cast_crew_prefs = user_metadata.get('castCrewPreferences', {})
        if not cast_crew_prefs or cast_crew_prefs.get('error'):
            return

        cast_prefs = cast_crew_prefs.get('castPreferences', {})
        crew_prefs = cast_crew_prefs.get('crewPreferences', {})

        post_cast = post_metadata.get('cast', [])
        post_crew = post_metadata.get('crew', {})

        matched_people = []

        # Check cast
        for cast_member in post_cast[:5]:  # Top 5 cast
            person_id = str(cast_member.get('id', ''))
            person_name = cast_member.get('name', '')
            if person_id in cast_prefs:
                matched_people.append(person_name)

        # Check key crew (directors)
        directors = post_crew.get('director', []) or post_crew.get('Director', [])
        for director in directors[:2]:
            person_id = str(director.get('id', ''))
            person_name = director.get('name', '')
            if person_id in crew_prefs:
                matched_people.append(f"{person_name} (Director)")

        if matched_people:
            explanation.add_reason(
                factor="cast_crew",
                weight=self.boost_thresholds['cast_crew'],
                description=f"Features {matched_people[0]}",
                details={"matched_people": matched_people[:3]}
            )

    def _analyze_popularity(self, explanation: PostExplanation, post_metadata: Dict):
        """Check if post is highly rated."""
        vote_average = post_metadata.get('voteAverage', 0)

        if isinstance(vote_average, (int, float)) and vote_average > 7.5:
            explanation.add_reason(
                factor="popularity",
                weight=self.boost_thresholds['popularity'],
                description=f"Highly rated ({vote_average:.1f}/10)",
                details={"vote_average": vote_average}
            )

    def _analyze_recency(self, explanation: PostExplanation, post_metadata: Dict):
        """Check if post has recency boost."""
        recency_boost = post_metadata.get('recencyBoost', 1.0)

        if isinstance(recency_boost, (int, float)) and recency_boost > 1.1:
            explanation.add_reason(
                factor="recency",
                weight=self.boost_thresholds['recency'],
                description="Recently released",
                details={"recency_boost": recency_boost}
            )

    def _analyze_demographic_match(self, explanation: PostExplanation,
                                   user_metadata: Dict, post_metadata: Dict):
        """Check if post is popular in user's region."""
        user_features = user_metadata.get('categoricalFeatures', {})
        user_region = user_features.get('region')

        if not user_region:
            return

        region_weights = post_metadata.get('regionWeights', {})
        if user_region in region_weights:
            region_pop = region_weights[user_region]
            if region_pop > 0.5:
                explanation.add_reason(
                    factor="demographic",
                    weight=region_pop * self.boost_thresholds['demographic'],
                    description=f"Popular in your region",
                    details={"region": user_region, "popularity": region_pop}
                )

    def _analyze_engagement(self, explanation: PostExplanation, post_metadata: Dict):
        """Check engagement metrics."""
        info_clicks = post_metadata.get('infoButtonClicks', {})
        if isinstance(info_clicks, dict):
            click_count = info_clicks.get('count', 0)
            if click_count > 50:
                explanation.add_reason(
                    factor="engagement",
                    weight=self.boost_thresholds['engagement'],
                    description="Trending with high engagement",
                    details={"engagement_count": click_count}
                )

    def _analyze_rl_adjustments(self, explanation: PostExplanation,
                                rl_actions: List[Dict]):
        """Analyze RL-based adjustments."""
        for action in rl_actions:
            action_type = action.get('action_type', '')

            if action_type == 'boost_adjustment':
                params = action.get('parameters', {})
                adjustments = []

                if params.get('recency_boost_adjustment', 0) > 0:
                    adjustments.append("recency")
                if params.get('genre_boost_adjustment', 0) > 0:
                    adjustments.append("genre")
                if params.get('cast_crew_boost_adjustment', 0) > 0:
                    adjustments.append("cast/crew")

                if adjustments:
                    explanation.add_reason(
                        factor="personalization",
                        weight=0.1,
                        description="Personalized based on your recent activity",
                        details={"adjusted_factors": adjustments}
                    )

            elif action_type == 'exploration':
                explanation.add_reason(
                    factor="discovery",
                    weight=0.05,
                    description="Suggested to help you discover new content",
                    details={"exploration": True}
                )

    def format_response(self, explanations: List[PostExplanation],
                       compact: bool = False) -> List[Dict[str, Any]]:
        """
        Format explanations for API response.

        Args:
            explanations: List of PostExplanation objects
            compact: If True, return compact format

        Returns:
            List of explanation dictionaries
        """
        if compact:
            return [exp.to_compact() for exp in explanations]
        return [exp.to_dict() for exp in explanations]


def create_explainer(api_base_url: str, metadata_cache: Dict = None) -> RecommendationExplainer:
    """Factory function to create a RecommendationExplainer."""
    return RecommendationExplainer(api_base_url, metadata_cache)
