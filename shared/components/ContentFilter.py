#!/usr/bin/env python3
"""
Content Filter for ML Recommendation Model
Handles explicit filtering, content safety, and user preference filtering
"""

import numpy as np
import requests
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import time
from functools import lru_cache
import re

# Configure logging
logger = logging.getLogger(__name__)

class FilterType(Enum):
    EXPLICIT_CONTENT = "explicit_content"
    TOXICITY = "toxicity"
    NSFW = "nsfw"
    HATE_SPEECH = "hate_speech"
    SPAM = "spam"
    MISINFORMATION = "misinformation"
    COPYRIGHT = "copyright"
    AGE_INAPPROPRIATE = "age_inappropriate"

class FilterAction(Enum):
    BLOCK = "block"
    WARN = "warn"
    DOWNRANK = "downrank"
    FLAG = "flag"

@dataclass
class FilterRule:
    """Represents a content filtering rule"""
    filter_type: FilterType
    action: FilterAction
    threshold: float
    user_configurable: bool = True
    reason: str = ""

@dataclass
class ContentAnalysis:
    """Results of content analysis"""
    post_id: int
    toxicity_score: float
    explicit_score: float
    spam_score: float
    hate_speech_score: float
    nsfw_score: float
    category_tags: List[str]
    language: str
    age_rating: str
    violations: List[FilterRule]

@dataclass
class UserFilterPreferences:
    """User's content filtering preferences"""
    user_id: int
    nsfw_filter: bool = True
    toxicity_threshold: float = 0.7
    explicit_content_filter: bool = True
    hate_speech_filter: bool = True
    spam_filter: bool = True
    age_appropriate_only: bool = False
    blocked_categories: Set[str] = None
    blocked_languages: Set[str] = None
    custom_keywords: Set[str] = None

class ContentFilter:
    """
    Comprehensive content filtering system for recommendations
    """
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8080/api",
                 toxicity_service_url: str = "http://localhost:8081",
                 cache_size: int = 2000):
        """
        Initialize the content filter
        
        Args:
            api_base_url: Base URL for the Kotlin API
            toxicity_service_url: URL for toxicity detection service
            cache_size: Size of the LRU cache for analysis results
        """
        self.api_base_url = api_base_url
        self.toxicity_service_url = toxicity_service_url
        self.cache_size = cache_size
        
        # Caching
        self._analysis_cache = {}
        self._preference_cache = {}
        
        # Default filter rules
        self.default_rules = [
            FilterRule(FilterType.EXPLICIT_CONTENT, FilterAction.BLOCK, 0.8, True, "Explicit content detected"),
            FilterRule(FilterType.TOXICITY, FilterAction.DOWNRANK, 0.7, True, "Toxic content detected"),
            FilterRule(FilterType.NSFW, FilterAction.WARN, 0.6, True, "NSFW content detected"),
            FilterRule(FilterType.HATE_SPEECH, FilterAction.BLOCK, 0.8, True, "Hate speech detected"),
            FilterRule(FilterType.SPAM, FilterAction.BLOCK, 0.9, False, "Spam content detected"),
            FilterRule(FilterType.AGE_INAPPROPRIATE, FilterAction.BLOCK, 0.7, True, "Age inappropriate content")
        ]
        
        # Content categories to filter
        self.sensitive_categories = {
            "adult", "violence", "drugs", "gambling", "weapons", 
            "hate", "extremism", "self_harm", "illegal"
        }
        
        # Explicit keywords for basic filtering
        self.explicit_keywords = self._load_explicit_keywords()
        
        logger.info(f"Initialized ContentFilter with {len(self.default_rules)} default rules")

    def filter_recommendations(self, 
                             user_id: int, 
                             post_ids: List[int],
                             scores: np.ndarray,
                             apply_user_preferences: bool = True) -> Tuple[List[int], np.ndarray, Dict[str, Any]]:
        """
        Filter recommendations based on content analysis and user preferences
        
        Args:
            user_id: User ID
            post_ids: List of post IDs to filter
            scores: Recommendation scores
            apply_user_preferences: Whether to apply user-specific preferences
            
        Returns:
            Tuple of (filtered_post_ids, filtered_scores, filter_metadata)
        """
        try:
            start_time = time.time()
            
            # Get user filter preferences
            user_prefs = None
            if apply_user_preferences:
                user_prefs = self._get_user_filter_preferences(user_id)
            
            # Analyze content for all posts
            content_analyses = self._analyze_posts_content(post_ids)
            
            # Apply filtering logic
            filtered_results = []
            filter_metadata = {
                "total_posts": len(post_ids),
                "blocked_posts": 0,
                "downranked_posts": 0,
                "warned_posts": 0,
                "filter_reasons": {},
                "processing_time": 0
            }
            
            for i, post_id in enumerate(post_ids):
                analysis = content_analyses.get(post_id)
                score = scores[i] if i < len(scores) else 0.0
                
                if analysis:
                    # Check if post should be filtered
                    filter_decision = self._evaluate_filter_decision(analysis, user_prefs)
                    
                    if filter_decision["action"] == FilterAction.BLOCK:
                        filter_metadata["blocked_posts"] += 1
                        filter_metadata["filter_reasons"][post_id] = filter_decision["reasons"]
                        continue
                    elif filter_decision["action"] == FilterAction.DOWNRANK:
                        score *= 0.3  # Significantly reduce score
                        filter_metadata["downranked_posts"] += 1
                    elif filter_decision["action"] == FilterAction.WARN:
                        score *= 0.8  # Slightly reduce score
                        filter_metadata["warned_posts"] += 1
                
                filtered_results.append((post_id, score))
            
            # Extract filtered data
            filtered_post_ids = [post_id for post_id, _ in filtered_results]
            filtered_scores = np.array([score for _, score in filtered_results])
            
            filter_metadata["filtered_posts"] = len(filtered_post_ids)
            filter_metadata["processing_time"] = time.time() - start_time
            
            logger.info(f"Filtered {len(post_ids)} posts to {len(filtered_post_ids)} for user {user_id}")
            return filtered_post_ids, filtered_scores, filter_metadata
            
        except Exception as e:
            logger.error(f"Error in content filtering: {e}")
            # Return original data if filtering fails
            return post_ids, scores, {"error": str(e)}

    def _analyze_posts_content(self, post_ids: List[int]) -> Dict[int, ContentAnalysis]:
        """Analyze content for multiple posts with caching"""
        results = {}
        uncached_posts = []
        
        # Check cache first
        for post_id in post_ids:
            cache_key = f"analysis_{post_id}"
            if cache_key in self._analysis_cache:
                cached_data, timestamp = self._analysis_cache[cache_key]
                if time.time() - timestamp < 3600:  # 1 hour cache
                    results[post_id] = cached_data
                else:
                    uncached_posts.append(post_id)
            else:
                uncached_posts.append(post_id)
        
        # Analyze uncached posts
        if uncached_posts:
            batch_analyses = self._batch_analyze_content(uncached_posts)
            
            for post_id, analysis in batch_analyses.items():
                results[post_id] = analysis
                
                # Cache the result
                cache_key = f"analysis_{post_id}"
                if len(self._analysis_cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = min(self._analysis_cache.keys(), 
                                   key=lambda k: self._analysis_cache[k][1])
                    del self._analysis_cache[oldest_key]
                
                self._analysis_cache[cache_key] = (analysis, time.time())
        
        return results

    def _batch_analyze_content(self, post_ids: List[int]) -> Dict[int, ContentAnalysis]:
        """Perform batch content analysis"""
        results = {}
        
        try:
            # Get post content from API
            post_contents = self._fetch_post_contents(post_ids)
            
            # Perform toxicity analysis
            toxicity_results = self._analyze_toxicity_batch(post_contents)
            
            # Perform explicit content detection
            explicit_results = self._analyze_explicit_content_batch(post_contents)
            
            # Get post metadata (categories, language, etc.)
            metadata_results = self._fetch_post_metadata(post_ids)
            
            # Combine results
            for post_id in post_ids:
                content = post_contents.get(post_id, "")
                toxicity = toxicity_results.get(post_id, {})
                explicit = explicit_results.get(post_id, {})
                metadata = metadata_results.get(post_id, {})
                
                analysis = ContentAnalysis(
                    post_id=post_id,
                    toxicity_score=toxicity.get("toxicity_score", 0.0),
                    explicit_score=explicit.get("explicit_score", 0.0),
                    spam_score=self._calculate_spam_score(content),
                    hate_speech_score=toxicity.get("hate_speech_score", 0.0),
                    nsfw_score=explicit.get("nsfw_score", 0.0),
                    category_tags=metadata.get("categories", []),
                    language=metadata.get("language", "unknown"),
                    age_rating=metadata.get("age_rating", "unrated"),
                    violations=[]
                )
                
                # Check for violations
                analysis.violations = self._check_content_violations(analysis)
                results[post_id] = analysis
                
        except Exception as e:
            logger.error(f"Error in batch content analysis: {e}")
            # Return empty analyses for failed posts
            for post_id in post_ids:
                results[post_id] = self._create_default_analysis(post_id)
        
        return results

    def _fetch_post_contents(self, post_ids: List[int]) -> Dict[int, str]:
        """Fetch post content from API"""
        results = {}
        
        try:
            for post_id in post_ids:
                url = f"{self.api_base_url}/posts/{post_id}/content"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    # Combine title, description, and other text content
                    content_parts = []
                    if data.get("title"):
                        content_parts.append(data["title"])
                    if data.get("description"):
                        content_parts.append(data["description"])
                    if data.get("text_content"):
                        content_parts.append(data["text_content"])
                    
                    results[post_id] = " ".join(content_parts)
                else:
                    results[post_id] = ""
                    
        except Exception as e:
            logger.warning(f"Error fetching post contents: {e}")
        
        return results

    def _analyze_toxicity_batch(self, post_contents: Dict[int, str]) -> Dict[int, Dict]:
        """Analyze toxicity for batch of posts"""
        results = {}
        
        try:
            # Prepare batch request
            texts = [content for content in post_contents.values()]
            post_id_mapping = list(post_contents.keys())
            
            if not texts:
                return results
            
            # Call toxicity service (if available)
            url = f"{self.toxicity_service_url}/analyze/batch"
            payload = {"texts": texts}
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                toxicity_data = response.json()
                
                for i, result in enumerate(toxicity_data.get("results", [])):
                    if i < len(post_id_mapping):
                        post_id = post_id_mapping[i]
                        results[post_id] = {
                            "toxicity_score": result.get("toxicity", 0.0),
                            "hate_speech_score": result.get("severe_toxicity", 0.0)
                        }
            
        except Exception as e:
            logger.warning(f"Toxicity analysis failed, using fallback: {e}")
            # Fallback to keyword-based detection
            for post_id, content in post_contents.items():
                results[post_id] = self._fallback_toxicity_analysis(content)
        
        return results

    def _analyze_explicit_content_batch(self, post_contents: Dict[int, str]) -> Dict[int, Dict]:
        """Analyze explicit content for batch of posts"""
        results = {}
        
        try:
            for post_id, content in post_contents.items():
                explicit_score = self._calculate_explicit_score(content)
                nsfw_score = self._calculate_nsfw_score(content)
                
                results[post_id] = {
                    "explicit_score": explicit_score,
                    "nsfw_score": nsfw_score
                }
                
        except Exception as e:
            logger.warning(f"Explicit content analysis failed: {e}")
        
        return results

    def _fetch_post_metadata(self, post_ids: List[int]) -> Dict[int, Dict]:
        """Fetch post metadata including categories and language"""
        results = {}
        
        try:
            for post_id in post_ids:
                url = f"{self.api_base_url}/posts/{post_id}/metadata"
                response = requests.get(url, timeout=3)
                
                if response.status_code == 200:
                    results[post_id] = response.json()
                else:
                    results[post_id] = {}
                    
        except Exception as e:
            logger.warning(f"Error fetching post metadata: {e}")
        
        return results

    def _calculate_spam_score(self, content: str) -> float:
        """Calculate spam score based on content analysis"""
        if not content:
            return 0.0
        
        spam_indicators = 0
        content_lower = content.lower()
        
        # Check for spam patterns
        spam_patterns = [
            r'click here', r'buy now', r'limited time', r'act fast',
            r'free money', r'get rich quick', r'make money fast',
            r'[!]{3,}', r'[A-Z]{10,}', r'\$+\d+', r'www\.'
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, content_lower):
                spam_indicators += 1
        
        # Check for excessive capitalization
        if len(content) > 20:
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if caps_ratio > 0.3:
                spam_indicators += 1
        
        # Check for repeated phrases
        words = content_lower.split()
        if len(words) > 10:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.5:
                spam_indicators += 1
        
        return min(1.0, spam_indicators / 5.0)

    def _calculate_explicit_score(self, content: str) -> float:
        """Calculate explicit content score"""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        explicit_count = 0
        
        for keyword in self.explicit_keywords:
            if keyword in content_lower:
                explicit_count += 1
        
        # Normalize by content length and keyword frequency
        if len(content) > 0:
            score = min(1.0, explicit_count / max(len(content.split()) / 50, 1))
        else:
            score = 0.0
        
        return score

    def _calculate_nsfw_score(self, content: str) -> float:
        """Calculate NSFW score"""
        if not content:
            return 0.0
        
        nsfw_keywords = ['nsfw', 'adult', 'mature', '18+', 'explicit']
        content_lower = content.lower()
        
        nsfw_count = sum(1 for keyword in nsfw_keywords if keyword in content_lower)
        return min(1.0, nsfw_count / 3.0)

    def _fallback_toxicity_analysis(self, content: str) -> Dict:
        """Fallback toxicity analysis using keyword matching"""
        if not content:
            return {"toxicity_score": 0.0, "hate_speech_score": 0.0}
        
        toxic_keywords = [
            'hate', 'kill', 'die', 'stupid', 'idiot', 'moron',
            'racist', 'sexist', 'bigot', 'nazi', 'terrorist'
        ]
        
        content_lower = content.lower()
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in content_lower)
        
        toxicity_score = min(1.0, toxic_count / 3.0)
        hate_speech_score = min(1.0, toxic_count / 5.0)
        
        return {
            "toxicity_score": toxicity_score,
            "hate_speech_score": hate_speech_score
        }

    def _check_content_violations(self, analysis: ContentAnalysis) -> List[FilterRule]:
        """Check for content violations against filter rules"""
        violations = []
        
        for rule in self.default_rules:
            violated = False
            
            if rule.filter_type == FilterType.EXPLICIT_CONTENT and analysis.explicit_score >= rule.threshold:
                violated = True
            elif rule.filter_type == FilterType.TOXICITY and analysis.toxicity_score >= rule.threshold:
                violated = True
            elif rule.filter_type == FilterType.NSFW and analysis.nsfw_score >= rule.threshold:
                violated = True
            elif rule.filter_type == FilterType.HATE_SPEECH and analysis.hate_speech_score >= rule.threshold:
                violated = True
            elif rule.filter_type == FilterType.SPAM and analysis.spam_score >= rule.threshold:
                violated = True
            elif rule.filter_type == FilterType.AGE_INAPPROPRIATE:
                if analysis.age_rating in ['18+', 'mature'] and rule.threshold < 0.8:
                    violated = True
            
            if violated:
                violations.append(rule)
        
        return violations

    def _evaluate_filter_decision(self, analysis: ContentAnalysis, user_prefs: Optional[UserFilterPreferences]) -> Dict:
        """Evaluate filtering decision based on analysis and user preferences"""
        decision = {
            "action": FilterAction.FLAG,  # Default action
            "reasons": [],
            "confidence": 0.0
        }
        
        # Check for blocking conditions
        blocking_violations = [v for v in analysis.violations if v.action == FilterAction.BLOCK]
        if blocking_violations:
            decision["action"] = FilterAction.BLOCK
            decision["reasons"] = [v.reason for v in blocking_violations]
            decision["confidence"] = max(v.threshold for v in blocking_violations)
            return decision
        
        # Apply user preferences if available
        if user_prefs:
            if user_prefs.nsfw_filter and analysis.nsfw_score > 0.5:
                decision["action"] = FilterAction.BLOCK
                decision["reasons"].append("User NSFW filter enabled")
                return decision
            
            if user_prefs.explicit_content_filter and analysis.explicit_score > 0.6:
                decision["action"] = FilterAction.BLOCK
                decision["reasons"].append("User explicit content filter enabled")
                return decision
            
            if analysis.toxicity_score > user_prefs.toxicity_threshold:
                decision["action"] = FilterAction.DOWNRANK
                decision["reasons"].append(f"Toxicity above user threshold ({user_prefs.toxicity_threshold})")
                return decision
            
            # Check blocked categories
            if user_prefs.blocked_categories:
                blocked_cats = user_prefs.blocked_categories.intersection(set(analysis.category_tags))
                if blocked_cats:
                    decision["action"] = FilterAction.BLOCK
                    decision["reasons"].append(f"Blocked categories: {blocked_cats}")
                    return decision
        
        # Check for downranking conditions
        downrank_violations = [v for v in analysis.violations if v.action == FilterAction.DOWNRANK]
        if downrank_violations:
            decision["action"] = FilterAction.DOWNRANK
            decision["reasons"] = [v.reason for v in downrank_violations]
            return decision
        
        # Check for warning conditions
        warn_violations = [v for v in analysis.violations if v.action == FilterAction.WARN]
        if warn_violations:
            decision["action"] = FilterAction.WARN
            decision["reasons"] = [v.reason for v in warn_violations]
            return decision
        
        return decision

    def _get_user_filter_preferences(self, user_id: int) -> Optional[UserFilterPreferences]:
        """Get user's content filtering preferences with caching"""
        cache_key = f"prefs_{user_id}"
        
        if cache_key in self._preference_cache:
            cached_data, timestamp = self._preference_cache[cache_key]
            if time.time() - timestamp < 1800:  # 30 minute cache
                return cached_data
        
        try:
            url = f"{self.api_base_url}/users/{user_id}/filter-preferences"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                prefs = UserFilterPreferences(
                    user_id=user_id,
                    nsfw_filter=data.get("nsfwFilter", True),
                    toxicity_threshold=data.get("toxicityThreshold", 0.7),
                    explicit_content_filter=data.get("explicitContentFilter", True),
                    hate_speech_filter=data.get("hateSpeechFilter", True),
                    spam_filter=data.get("spamFilter", True),
                    age_appropriate_only=data.get("ageAppropriateOnly", False),
                    blocked_categories=set(data.get("blockedCategories", [])),
                    blocked_languages=set(data.get("blockedLanguages", [])),
                    custom_keywords=set(data.get("customKeywords", []))
                )
                
                # Cache the result
                self._preference_cache[cache_key] = (prefs, time.time())
                return prefs
            
        except Exception as e:
            logger.warning(f"Failed to get user filter preferences for {user_id}: {e}")
        
        return None

    def _create_default_analysis(self, post_id: int) -> ContentAnalysis:
        """Create default content analysis for failed posts"""
        return ContentAnalysis(
            post_id=post_id,
            toxicity_score=0.0,
            explicit_score=0.0,
            spam_score=0.0,
            hate_speech_score=0.0,
            nsfw_score=0.0,
            category_tags=[],
            language="unknown",
            age_rating="unrated",
            violations=[]
        )

    def _load_explicit_keywords(self) -> Set[str]:
        """Load explicit keywords for content filtering"""
        # Basic set of explicit keywords (you would load this from a file or database)
        return {
            'explicit', 'adult', 'sexual', 'pornographic', 'nude', 'naked',
            'sex', 'xxx', 'porn', 'erotic', 'intimate', 'mature'
        }

    def update_filter_rules(self, rules: List[FilterRule]):
        """Update the filter rules"""
        self.default_rules = rules
        logger.info(f"Updated filter rules: {len(rules)} rules active")

    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        return {
            "cache_size": len(self._analysis_cache),
            "preference_cache_size": len(self._preference_cache),
            "active_rules": len(self.default_rules),
            "sensitive_categories": len(self.sensitive_categories)
        }

    def clear_caches(self):
        """Clear all caches"""
        self._analysis_cache.clear()
        self._preference_cache.clear()
        logger.info("Cleared content filter caches")


# Utility functions
def create_content_filter(config: Dict[str, Any]) -> ContentFilter:
    """Factory function to create content filter"""
    return ContentFilter(
        api_base_url=config.get("api_base_url", "http://localhost:8080/api"),
        toxicity_service_url=config.get("toxicity_service_url", "http://localhost:8081"),
        cache_size=config.get("cache_size", 2000)
    )