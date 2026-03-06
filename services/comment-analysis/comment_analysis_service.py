#!/usr/bin/env python3
"""
Comment Analysis Service
Enhanced microservice for sentiment analysis, toxicity detection, and comment processing
Port: 8082
"""

import logging
import os
import time
import requests
import threading
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask.cli import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("comment-analysis-service")

# Initialize Flask app
app = Flask(__name__)

# Import JWT utilities
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared'))
    from auth.JwtTokenUtil import extract_jwt_token, create_auth_headers, get_token_or_fallback
except ImportError:
    logger.warning("JWT utilities not available, falling back to environment tokens")
    extract_jwt_token = None
    create_auth_headers = None
    get_token_or_fallback = None

class SentimentLabel(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

@dataclass
class SentimentResult:
    post_id: int
    overall_sentiment: SentimentLabel
    confidence_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    total_comments: int
    sentiment_distribution: Dict[str, int]
    toxicity_score: float
    processing_time: float

@dataclass
class CommentAnalysis:
    comment_id: int
    post_id: int
    sentiment: SentimentLabel
    confidence: float
    toxicity_score: float
    hate_speech_score: float
    spam_score: float
    text_length: int
    language: str

class CommentAnalysisService:
    """
    Enhanced comment analysis service with BERT sentiment analysis and toxicity detection
    """
    
    def __init__(self):
        """Initialize the comment analysis service"""
        load_dotenv()
        
        # Service configuration
        self.api_base_url = os.environ.get('SPRING_API_URL', 'http://localhost:8080')
        
        # Model configuration
        self.model_name = os.getenv('BERT_MODEL', 'nlptown/bert-base-multilingual-uncased-sentiment')
        self.max_length = 512
        self.batch_size = 8
        self.cache_size = int(os.environ.get('SENTIMENT_CACHE_SIZE', '2000'))
        
        # Caching
        self._sentiment_cache = {}
        self._comment_cache = {}
        self._cache_lock = threading.Lock()
        self.cache_ttl = int(os.environ.get('CACHE_TTL', '7200'))  # 2 hours
        
        # Initialize BERT model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.classifier = None
        
        self._load_bert_model()
        
        # Toxicity detection patterns (simplified - in production use proper toxicity model)
        self.toxic_keywords = {
            'hate', 'kill', 'die', 'stupid', 'idiot', 'moron', 'racist', 'sexist', 
            'bigot', 'nazi', 'terrorist', 'loser', 'pathetic', 'disgusting'
        }
        
        self.spam_patterns = [
            r'click here', r'buy now', r'limited time', r'act fast',
            r'free money', r'get rich quick', r'make money fast',
            r'[!]{3,}', r'[A-Z]{10,}', r'\$+\d+', r'www\.'
        ]
        
        logger.info(f"Initialized Comment Analysis Service on port 8082")
        logger.info(f"Using device: {self.device}, Model: {self.model_name}")
        
        # Store current JWT token for requests
        self.current_jwt_token = None
    
    def _update_jwt_token(self, jwt_token: str = None):
        """Update JWT token for this request"""
        if jwt_token:
            self.current_jwt_token = jwt_token
        elif get_token_or_fallback and not self.current_jwt_token:
            # Try to get token from request or fallback
            fallback_token = get_token_or_fallback()
            if fallback_token:
                self.current_jwt_token = fallback_token
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API calls"""
        auth_token = self.current_jwt_token or os.environ.get('SERVICE_AUTH_TOKEN', '')
        headers = {}
        if auth_token:
            headers['Authorization'] = f'Bearer {auth_token}'
            headers['X-Service-Role'] = 'SERVICE'
        return headers

    def _load_bert_model(self):
        """Load BERT model for sentiment analysis"""
        try:
            logger.info(f"Loading BERT model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("BERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            self.classifier = None

    def analyze_post_sentiment(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment for all comments on a specific post
        
        Args:
            request_data: Contains postId and optional parameters
            
        Returns:
            Comprehensive sentiment analysis for the post
        """
        start_time = time.time()
        
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            post_id = request_data.get("postId")
            if not post_id:
                return self._error_response("postId is required")
            
            include_individual = request_data.get("includeIndividualComments", False)
            min_comments = request_data.get("minComments", 1)
            
            logger.info(f"Analyzing sentiment for post {post_id}")
            
            # Check cache first
            cache_key = f"post_sentiment_{post_id}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Fetch comments for the post
            comments = self._fetch_post_comments(post_id)
            
            if len(comments) < min_comments:
                return self._create_default_sentiment_result(post_id, 0, start_time)
            
            # Analyze individual comments
            comment_analyses = self._analyze_comments_batch(comments)
            
            # Aggregate sentiment analysis
            aggregated_result = self._aggregate_comment_sentiment(
                post_id, comment_analyses, start_time
            )
            
            # Add individual comment analyses if requested
            if include_individual:
                aggregated_result["individualComments"] = [
                    self._comment_analysis_to_dict(analysis) for analysis in comment_analyses
                ]
            
            # Cache the result
            self._cache_result(cache_key, aggregated_result)
            
            logger.info(f"Analyzed sentiment for post {post_id}: {len(comments)} comments, "
                       f"overall: {aggregated_result['overallSentiment']}")
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Error analyzing post sentiment: {e}", exc_info=True)
            return self._error_response(f"Error analyzing post sentiment: {str(e)}")

    def analyze_comments_batch(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment for a batch of comment texts
        
        Args:
            request_data: Contains texts array and optional parameters
            
        Returns:
            Batch sentiment analysis results
        """
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            texts = request_data.get("texts", [])
            if not texts:
                return self._error_response("texts array is required")
            
            if len(texts) > 100:
                return self._error_response("Maximum 100 texts allowed per batch")
            
            logger.info(f"Analyzing sentiment for {len(texts)} text samples")
            
            # Analyze each text
            results = []
            for i, text in enumerate(texts):
                if text and len(text.strip()) >= 3:
                    analysis = self._analyze_single_text(text, comment_id=i)
                    results.append(self._comment_analysis_to_dict(analysis))
                else:
                    results.append(self._create_default_comment_analysis(i))
            
            return {"results": results, "totalAnalyzed": len(results)}
            
        except Exception as e:
            logger.error(f"Error in batch comment analysis: {e}", exc_info=True)
            return self._error_response(f"Error in batch analysis: {str(e)}")

    def get_post_sentiment_summary(self, post_id: int, jwt_token: str = None) -> Dict[str, Any]:
        """
        Get cached sentiment summary for a post or analyze if not cached
        
        Args:
            post_id: Post ID
            
        Returns:
            Post sentiment summary
        """
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            # Check cache first
            cache_key = f"post_sentiment_{post_id}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Analyze if not cached
            return self.analyze_post_sentiment({"postId": post_id}, jwt_token=jwt_token)
            
        except Exception as e:
            logger.error(f"Error getting post sentiment summary: {e}")
            return self._error_response(f"Error getting sentiment summary: {str(e)}")

    def analyze_batch_posts_sentiment(self, request_data: Dict[str, Any], jwt_token: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment for multiple posts in batch
        
        Args:
            request_data: Contains postIds array
            
        Returns:
            Batch post sentiment analysis
        """
        try:
            # Update JWT token for this request
            self._update_jwt_token(jwt_token)
            
            post_ids = request_data.get("postIds", [])
            if not post_ids:
                return self._error_response("postIds array is required")
            
            if len(post_ids) > 50:
                return self._error_response("Maximum 50 posts allowed per batch")
            
            logger.info(f"Analyzing sentiment for {len(post_ids)} posts")
            
            results = []
            for post_id in post_ids:
                sentiment_data = self.get_post_sentiment_summary(post_id, jwt_token=jwt_token)
                if "error" not in sentiment_data:
                    results.append(sentiment_data)
            
            return {"sentimentData": results, "totalPosts": len(results)}
            
        except Exception as e:
            logger.error(f"Error in batch post sentiment analysis: {e}", exc_info=True)
            return self._error_response(f"Error in batch post analysis: {str(e)}")

    def _fetch_post_comments(self, post_id: int) -> List[Dict]:
        """Fetch comments for a specific post"""
        try:
            url = f"{self.api_base_url}/api/posts/{post_id}/comments"
            params = {"includeReplies": True, "limit": 500}  # Analyze up to 500 comments
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("comments", [])
            else:
                logger.warning(f"Failed to fetch comments for post {post_id}: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Error fetching comments for post {post_id}: {e}")
        
        return []

    def _analyze_comments_batch(self, comments: List[Dict]) -> List[CommentAnalysis]:
        """Analyze sentiment for a batch of comments"""
        analyses = []
        
        for comment in comments:
            comment_id = comment.get("id", 0)
            post_id = comment.get("postId", 0)
            text = comment.get("text", "")
            
            if text and len(text.strip()) >= 3:
                analysis = self._analyze_single_text(
                    text, comment_id=comment_id, post_id=post_id
                )
                analyses.append(analysis)
        
        return analyses

    def _analyze_single_text(self, text: str, comment_id: int = 0, post_id: int = 0) -> CommentAnalysis:
        """Analyze sentiment for a single text using BERT"""
        try:
            if not self.classifier:
                return self._create_default_comment_analysis(comment_id, post_id)
            
            # Truncate text if too long
            text = text[:self.max_length]
            
            # Get BERT sentiment prediction
            results = self.classifier(text)
            
            # Process results (assuming 3-class sentiment model)
            scores = {result['label'].lower(): result['score'] for result in results[0]}
            
            # Map to our sentiment labels
            positive_score = scores.get('positive', scores.get('pos', 0.0))
            negative_score = scores.get('negative', scores.get('neg', 0.0))
            neutral_score = scores.get('neutral', 1.0 - positive_score - negative_score)
            
            # Determine overall sentiment
            max_score = max(positive_score, negative_score, neutral_score)
            if max_score == positive_score:
                sentiment = SentimentLabel.POSITIVE
            elif max_score == negative_score:
                sentiment = SentimentLabel.NEGATIVE
            else:
                sentiment = SentimentLabel.NEUTRAL
            
            # Calculate toxicity and other scores
            toxicity_score = self._calculate_toxicity_score(text)
            hate_speech_score = self._calculate_hate_speech_score(text)
            spam_score = self._calculate_spam_score(text)
            language = self._detect_language(text)
            
            return CommentAnalysis(
                comment_id=comment_id,
                post_id=post_id,
                sentiment=sentiment,
                confidence=float(max_score),
                toxicity_score=toxicity_score,
                hate_speech_score=hate_speech_score,
                spam_score=spam_score,
                text_length=len(text),
                language=language
            )
            
        except Exception as e:
            logger.error(f"Error analyzing single text: {e}")
            return self._create_default_comment_analysis(comment_id, post_id)

    def _aggregate_comment_sentiment(self, post_id: int, analyses: List[CommentAnalysis], 
                                   start_time: float) -> Dict[str, Any]:
        """Aggregate individual comment analyses into post-level sentiment"""
        if not analyses:
            return self._create_default_sentiment_result(post_id, 0, start_time)
        
        # Count sentiments
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        total_confidence = 0.0
        total_toxicity = 0.0
        
        for analysis in analyses:
            sentiment_counts[analysis.sentiment.value.lower()] += 1
            total_confidence += analysis.confidence
            total_toxicity += analysis.toxicity_score
        
        total_comments = len(analyses)
        
        # Calculate percentages
        positive_pct = sentiment_counts["positive"] / total_comments
        negative_pct = sentiment_counts["negative"] / total_comments
        neutral_pct = sentiment_counts["neutral"] / total_comments
        
        # Determine overall sentiment
        if positive_pct > negative_pct and positive_pct > neutral_pct:
            overall_sentiment = SentimentLabel.POSITIVE
            confidence = positive_pct
        elif negative_pct > positive_pct and negative_pct > neutral_pct:
            overall_sentiment = SentimentLabel.NEGATIVE
            confidence = negative_pct
        else:
            overall_sentiment = SentimentLabel.NEUTRAL
            confidence = neutral_pct
        
        return {
            "postId": post_id,
            "overallSentiment": overall_sentiment.value,
            "confidenceScore": float(confidence),
            "positiveScore": float(positive_pct),
            "negativeScore": float(negative_pct),
            "neutralScore": float(neutral_pct),
            "totalComments": total_comments,
            "sentimentDistribution": sentiment_counts,
            "averageToxicity": float(total_toxicity / total_comments),
            "processingTime": time.time() - start_time
        }

    def _calculate_toxicity_score(self, text: str) -> float:
        """Calculate toxicity score based on keyword matching"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        toxic_count = sum(1 for keyword in self.toxic_keywords if keyword in text_lower)
        
        # Normalize by text length
        words = text_lower.split()
        if len(words) > 0:
            return min(1.0, toxic_count / max(len(words) / 10, 1))
        
        return 0.0

    def _calculate_hate_speech_score(self, text: str) -> float:
        """Calculate hate speech score"""
        hate_keywords = {'racist', 'sexist', 'bigot', 'nazi', 'terrorist', 'hate'}
        text_lower = text.lower()
        
        hate_count = sum(1 for keyword in hate_keywords if keyword in text_lower)
        return min(1.0, hate_count / 3.0)

    def _calculate_spam_score(self, text: str) -> float:
        """Calculate spam score based on patterns"""
        if not text:
            return 0.0
        
        import re
        spam_indicators = 0
        text_lower = text.lower()
        
        for pattern in self.spam_patterns:
            if re.search(pattern, text_lower):
                spam_indicators += 1
        
        # Check for excessive capitalization
        if len(text) > 20:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.3:
                spam_indicators += 1
        
        return min(1.0, spam_indicators / 5.0)

    def _detect_language(self, text: str) -> str:
        """Simple language detection (placeholder - use proper language detection in production)"""
        # This is a simplified implementation
        # In production, use a proper language detection library
        return "unknown"

    def _comment_analysis_to_dict(self, analysis: CommentAnalysis) -> Dict:
        """Convert CommentAnalysis to dictionary"""
        return {
            "commentId": analysis.comment_id,
            "postId": analysis.post_id,
            "sentiment": analysis.sentiment.value,
            "confidence": analysis.confidence,
            "toxicityScore": analysis.toxicity_score,
            "hateSpeechScore": analysis.hate_speech_score,
            "spamScore": analysis.spam_score,
            "textLength": analysis.text_length,
            "language": analysis.language
        }

    def _create_default_sentiment_result(self, post_id: int, comment_count: int, start_time: float) -> Dict:
        """Create default sentiment result for posts with no/few comments"""
        return {
            "postId": post_id,
            "overallSentiment": SentimentLabel.NEUTRAL.value,
            "confidenceScore": 0.5,
            "positiveScore": 0.33,
            "negativeScore": 0.33,
            "neutralScore": 0.34,
            "totalComments": comment_count,
            "sentimentDistribution": {"positive": 0, "negative": 0, "neutral": comment_count},
            "averageToxicity": 0.0,
            "processingTime": time.time() - start_time
        }

    def _create_default_comment_analysis(self, comment_id: int, post_id: int = 0) -> CommentAnalysis:
        """Create default comment analysis"""
        return CommentAnalysis(
            comment_id=comment_id,
            post_id=post_id,
            sentiment=SentimentLabel.NEUTRAL,
            confidence=0.5,
            toxicity_score=0.0,
            hate_speech_score=0.0,
            spam_score=0.0,
            text_length=0,
            language="unknown"
        )

    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result if valid"""
        with self._cache_lock:
            if cache_key in self._sentiment_cache:
                cached_data, timestamp = self._sentiment_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data
        return None

    def _cache_result(self, cache_key: str, result: Dict):
        """Cache analysis result"""
        with self._cache_lock:
            if len(self._sentiment_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = min(self._sentiment_cache.keys(), 
                               key=lambda k: self._sentiment_cache[k][1])
                del self._sentiment_cache[oldest_key]
            
            self._sentiment_cache[cache_key] = (result, time.time())

    def _error_response(self, message: str) -> Dict:
        """Create error response"""
        logger.error(message)
        return {"error": message}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        with self._cache_lock:
            return {
                "sentimentCacheSize": len(self._sentiment_cache),
                "commentCacheSize": len(self._comment_cache),
                "cacheTtl": self.cache_ttl,
                "modelLoaded": self.classifier is not None,
                "device": str(self.device),
                "modelName": self.model_name
            }

    def clear_caches(self):
        """Clear all caches"""
        with self._cache_lock:
            self._sentiment_cache.clear()
            self._comment_cache.clear()
        logger.info("Cleared all caches")

    def update_comment_analysis(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update comment with analysis results via API call.
        
        Args:
            request_data: Dictionary containing:
                - commentId: Comment ID to update (required)
                - postId: Post ID (required)
                - analysisData: Analysis results to store (required)
                - updateFields: Fields to update (optional)
                
        Returns:
            Success/error response
        """
        try:
            comment_id = request_data.get("commentId")
            post_id = request_data.get("postId") 
            analysis_data = request_data.get("analysisData")
            
            if not all([comment_id, post_id, analysis_data]):
                return self._error_response("commentId, postId, and analysisData are required")
            
            # Prepare update payload for Spring API
            update_payload = {
                "commentId": comment_id,
                "sentimentAnalysis": {
                    "sentiment": analysis_data.get("sentiment", "NEUTRAL"),
                    "confidence": analysis_data.get("confidence", 0.5),
                    "toxicityScore": analysis_data.get("toxicityScore", 0.0),
                    "hateSpeechScore": analysis_data.get("hateSpeechScore", 0.0),
                    "spamScore": analysis_data.get("spamScore", 0.0),
                    "analysisTimestamp": time.time(),
                    "modelVersion": "bert-sentiment-v1.0"
                }
            }
            
            # Add optional metadata
            if "textLength" in analysis_data:
                update_payload["sentimentAnalysis"]["textLength"] = analysis_data["textLength"]
            if "language" in analysis_data:
                update_payload["sentimentAnalysis"]["language"] = analysis_data["language"]
            
            # Call Spring API to update comment
            response = self._update_comment_via_api(comment_id, update_payload)
            
            if response:
                logger.info(f"Updated comment {comment_id} analysis successfully")
                return {
                    "success": True,
                    "commentId": comment_id,
                    "message": "Comment analysis updated successfully"
                }
            else:
                return self._error_response(f"Failed to update comment {comment_id}")
                
        except Exception as e:
            logger.error(f"Error updating comment analysis: {e}", exc_info=True)
            return self._error_response(f"Error updating comment analysis: {str(e)}")

    def batch_update_comments(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update multiple comments with analysis results.
        
        Args:
            request_data: Dictionary containing:
                - updates: List of comment updates
                
        Returns:
            Batch update results
        """
        try:
            updates = request_data.get("updates", [])
            if not updates:
                return self._error_response("updates array is required")
            
            if len(updates) > 100:
                return self._error_response("Maximum 100 comment updates allowed per batch")
            
            results = []
            successful_updates = 0
            
            for update in updates:
                result = self.update_comment_analysis(update)
                results.append(result)
                
                if result.get("success"):
                    successful_updates += 1
            
            return {
                "success": True,
                "totalUpdates": len(updates),
                "successfulUpdates": successful_updates,
                "failedUpdates": len(updates) - successful_updates,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch comment update: {e}", exc_info=True)
            return self._error_response(f"Error in batch update: {str(e)}")

    def analyze_and_update_comment(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze comment text and automatically update the comment with results.
        
        Args:
            request_data: Dictionary containing:
                - commentId: Comment ID (required)
                - postId: Post ID (required)  
                - text: Comment text to analyze (required)
                - autoUpdate: Whether to auto-update (default: true)
                
        Returns:
            Analysis results and update status
        """
        try:
            comment_id = request_data.get("commentId")
            post_id = request_data.get("postId")
            text = request_data.get("text")
            auto_update = request_data.get("autoUpdate", True)
            
            if not all([comment_id, post_id, text]):
                return self._error_response("commentId, postId, and text are required")
            
            # Analyze the comment text
            analysis = self._analyze_single_text(text, comment_id, post_id)
            analysis_dict = self._comment_analysis_to_dict(analysis)
            
            response = {
                "commentId": comment_id,
                "postId": post_id,
                "analysis": analysis_dict,
                "updated": False
            }
            
            # Auto-update if requested
            if auto_update:
                update_request = {
                    "commentId": comment_id,
                    "postId": post_id,
                    "analysisData": analysis_dict
                }
                
                update_result = self.update_comment_analysis(update_request)
                response["updated"] = update_result.get("success", False)
                response["updateResult"] = update_result
            
            return response
            
        except Exception as e:
            logger.error(f"Error in analyze and update: {e}", exc_info=True)
            return self._error_response(f"Error in analyze and update: {str(e)}")

    def _update_comment_via_api(self, comment_id: int, update_payload: Dict[str, Any]) -> bool:
        """Update comment via Spring API call."""
        try:
            url = f"{self.api_base_url}/api/comments/{comment_id}/analysis"
            
            response = requests.put(url, json=update_payload, timeout=10)
            
            if response.status_code in [200, 204]:
                return True
            else:
                logger.warning(f"API returned status {response.status_code} for comment update {comment_id}")
                return False
                
        except Exception as e:
            logger.warning(f"Error updating comment {comment_id} via API: {e}")
            return False

# Create service instance
comment_service = CommentAnalysisService()

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "service": "comment-analysis",
            "version": "1.0.0",
            "stats": comment_service.get_service_stats()
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_single_text():
    """Analyze sentiment for a single text (BERT service compatibility)"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        analysis = comment_service._analyze_single_text(text)
        
        # Return in BERT service compatible format
        result = {
            "sentiment": analysis.sentiment.value,
            "confidence": analysis.confidence,
            "scores": [0.33, 0.33, 0.34],  # Placeholder scores array
            "toxicity_score": analysis.toxicity_score
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch_texts():
    """Analyze sentiment for multiple texts (BERT service compatibility)"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400
        
        response = comment_service.analyze_comments_batch(data)
        
        # Convert to BERT service compatible format
        if "results" in response:
            bert_results = []
            for result in response["results"]:
                bert_result = {
                    "sentiment": result.get("sentiment", "NEUTRAL"),
                    "confidence": result.get("confidence", 0.5),
                    "scores": [0.33, 0.33, 0.34],  # Placeholder
                    "toxicity": result.get("toxicityScore", 0.0)
                }
                bert_results.append(bert_result)
            
            return jsonify({"results": bert_results})
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch analyze endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/comments/posts/<int:post_id>/sentiment', methods=['GET'])
def get_post_sentiment(post_id):
    """Get sentiment analysis for a specific post's comments"""
    try:
        response = comment_service.get_post_sentiment_summary(post_id)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in post sentiment endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/comments/sentiment/posts/<int:post_id>', methods=['GET'])
def get_post_sentiment_alt(post_id):
    """Alternative endpoint for post sentiment (compatibility)"""
    return get_post_sentiment(post_id)

@app.route('/comments/sentiment/batch', methods=['POST'])
def analyze_batch_posts():
    """Analyze sentiment for multiple posts"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No request data provided"}), 400
        
        response = comment_service.analyze_batch_posts_sentiment(data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in batch post sentiment endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/comments/posts/<int:post_id>/analyze', methods=['POST'])
def analyze_post_comments(post_id):
    """Comprehensive analysis of post comments with individual comment details"""
    try:
        data = request.get_json() or {}
        data["postId"] = post_id
        data["includeIndividualComments"] = True
        
        response = comment_service.analyze_post_sentiment(data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in post comment analysis endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear analysis cache"""
    try:
        comment_service.clear_caches()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": "Error clearing cache"}), 500

@app.route('/cache/info', methods=['GET'])
def cache_info():
    """Get cache information"""
    try:
        stats = comment_service.get_service_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return jsonify({"error": "Error getting cache info"}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    try:
        stats = comment_service.get_service_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in stats endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service information"""
    return jsonify({
        "service": "Comment Analysis Service",
        "version": "1.0.0",
        "description": "Enhanced sentiment analysis and comment processing",
        "endpoints": {
            "health": "/health",
            "analyze_single": "/analyze",
            "analyze_batch": "/analyze/batch",
            "post_sentiment": "/comments/posts/{postId}/sentiment",
            "post_analysis": "/comments/posts/{postId}/analyze",
            "batch_posts": "/comments/sentiment/batch",
            "cache_clear": "/cache/clear",
            "cache_info": "/cache/info",
            "stats": "/stats"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('COMMENT_SERVICE_PORT', os.environ.get('PORT', 8082)))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Comment Analysis Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)