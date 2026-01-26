# Social Analysis Implementation - Phase 1: BERT Comment Sentiment Analysis

This implementation adds sophisticated comment sentiment analysis using BERT to your recommendation system, enabling social analysis features.

## 🚀 Features Implemented

### 1. BERT-Based Sentiment Analysis
- **Deep Learning Model**: Uses pre-trained BERT model for accurate sentiment analysis
- **Multi-language Support**: Supports multiple languages through multilingual BERT
- **Real-time Analysis**: Analyzes comments as they're posted
- **Batch Processing**: Efficient batch analysis for existing comments
- **Fallback System**: Rule-based fallback when BERT service is unavailable

### 2. Database Schema
- **Comment Sentiment Table**: Detailed sentiment scores for each comment
- **Post Sentiment Analysis**: Aggregated sentiment analysis for posts
- **Performance Optimized**: Proper indexing and database triggers
- **Data Integrity**: Foreign key constraints and validation checks

### 3. API Endpoints
- `GET /api/comments/sentiment/{commentId}` - Get comment sentiment
- `POST /api/comments/sentiment/batch` - Batch analyze comments
- `GET /api/comments/sentiment/posts/{postId}` - Get post sentiment analysis
- `POST /api/comments/sentiment/posts/{postId}/analyze` - Trigger post analysis
- `GET /api/comments/sentiment/statistics` - Get sentiment statistics
- `POST /api/comments/sentiment/process-batch` - Trigger batch processing

### 4. Integration Features
- **Automatic Analysis**: New comments analyzed automatically
- **WebSocket Integration**: Real-time sentiment updates
- **Caching System**: Intelligent caching for performance
- **Error Handling**: Comprehensive error handling and logging

## 📦 Installation & Setup

### 1. Database Setup
```sql
-- Run the migration script
psql -d your_database -f sentiment_analysis_migration.sql
```

### 2. BERT Service Setup
```bash
# Install Python dependencies
cd /path/to/REST-API
pip install -r requirements.txt

# Start BERT service
./start_bert_service.sh
```

### 3. Kotlin Application Configuration
Add to your `application.properties`:
```properties
# BERT Service Configuration
comment.analysis.bert.url=http://localhost:8080/analyze
comment.analysis.cache.enabled=true
comment.analysis.timeout=5000

# Sentiment Analysis Settings
sentiment.analysis.batch-size=100
sentiment.analysis.cache-enabled=true
sentiment.analysis.max-cache-size=10000
```

## 🔧 Configuration Options

### BERT Service Environment Variables
```bash
export BERT_MODEL="nlptown/bert-base-multilingual-uncased-sentiment"
export PORT=8080
export DEBUG=false
```

### Application Properties
```properties
# BERT Service Configuration
comment.analysis.bert.url=http://localhost:8080/analyze
comment.analysis.cache.enabled=true
comment.analysis.timeout=5000
comment.analysis.fallback.enabled=true

# Batch Processing
sentiment.analysis.batch-size=100
sentiment.analysis.retry-attempts=3
sentiment.analysis.retry-delay=1000
```

## 📚 API Usage Examples

### Analyze Single Comment
```bash
curl -X GET "http://localhost:8080/api/comments/sentiment/123"
```

Response:
```json
{
  "commentId": 123,
  "sentimentLabel": "POSITIVE",
  "confidenceScore": 0.89,
  "scores": {
    "positive": 0.89,
    "negative": 0.05,
    "neutral": 0.06
  },
  "analysisModel": "BERT",
  "isValid": true,
  "analyzedAt": "2024-01-15T10:30:00"
}
```

### Batch Analyze Comments
```bash
curl -X POST "http://localhost:8080/api/comments/sentiment/batch" \
  -H "Content-Type: application/json" \
  -d '{"commentIds": [123, 124, 125]}'
```

### Get Post Sentiment Analysis
```bash
curl -X GET "http://localhost:8080/api/comments/sentiment/posts/456"
```

Response:
```json
{
  "postId": 456,
  "totalComments": 150,
  "analyzedComments": 150,
  "averageSentimentScore": 0.23,
  "sentimentDistribution": {
    "positive": 85,
    "negative": 20,
    "neutral": 45
  },
  "overallSentiment": "POSITIVE",
  "confidenceScore": 0.76,
  "sentimentTrend": "IMPROVING",
  "lastAnalyzed": "2024-01-15T10:30:00"
}
```

## 🚀 Performance Features

### Caching System
- **In-Memory Cache**: LRU cache for frequently analyzed texts
- **Cache Statistics**: Monitor cache hit rates and performance
- **Cache Management**: Clear cache via API endpoint

### Batch Processing
- **Async Processing**: Non-blocking batch processing
- **Configurable Batch Size**: Adjust based on system capacity
- **Error Recovery**: Automatic retry with exponential backoff

### Database Optimization
- **Proper Indexing**: Optimized database queries
- **Materialized Views**: Fast sentiment statistics
- **Automatic Cleanup**: Remove stale analysis data

## 🔍 Monitoring & Debugging

### Health Checks
```bash
# Check BERT service health
curl -X GET "http://localhost:8080/health"

# Get cache statistics
curl -X GET "http://localhost:8080/cache/info"
```

### Logging
- **Structured Logging**: JSON-formatted logs for analysis
- **Performance Metrics**: Request timing and throughput
- **Error Tracking**: Detailed error logging and stack traces

### Database Queries
```sql
-- Get sentiment statistics
SELECT * FROM get_sentiment_statistics();

-- View post sentiment overview
SELECT * FROM post_sentiment_overview LIMIT 10;

-- Find posts needing analysis
SELECT post_id FROM posts 
WHERE post_id NOT IN (
  SELECT post_id FROM post_sentiment_analysis 
  WHERE last_analyzed > NOW() - INTERVAL '1 day'
);
```

## 🎯 Integration with Recommendation System

The sentiment analysis integrates with your recommendation system by:

1. **Comment Quality Scoring**: Use sentiment to boost positive content
2. **User Preference Learning**: Understand user sentiment patterns
3. **Content Filtering**: Filter out consistently negative content
4. **Social Signals**: Use comment sentiment as social proof

### Next Steps for Full Social Analysis

This is Phase 1 of the social analysis implementation. Future phases will include:

1. **Phase 2**: Followed user vectors integration
2. **Phase 3**: Dual preference boosting (likes/saves vs not interested)
3. **Phase 4**: Risk assessment and explicit filtering
4. **Phase 5**: Diversification and final ranking enhancements
5. **Phase 6**: Recommendation explanations

## 🛠️ Troubleshooting

### Common Issues

1. **BERT Service Not Starting**
   - Check Python dependencies: `pip install -r requirements.txt`
   - Verify model download: Models are downloaded on first run
   - Check port availability: Default port 8080

2. **High Memory Usage**
   - Reduce batch size in configuration
   - Use CPU-only mode if GPU memory is limited
   - Implement model quantization for production

3. **Slow Response Times**
   - Enable caching: `comment.analysis.cache.enabled=true`
   - Increase timeout: `comment.analysis.timeout=10000`
   - Use GPU acceleration if available

4. **Database Performance**
   - Ensure indexes are created: Run migration script
   - Monitor query performance: Use `EXPLAIN ANALYZE`
   - Consider partitioning for large datasets

### Support

For issues or questions:
1. Check logs in both Kotlin application and BERT service
2. Verify database connectivity and permissions
3. Test BERT service independently using curl
4. Monitor resource usage (CPU, memory, disk)

## 📈 Performance Benchmarks

Expected performance metrics:
- **Single Comment Analysis**: 50-200ms (depending on text length)
- **Batch Processing**: 20-50 comments/second
- **Cache Hit Rate**: 70-90% for typical workloads
- **Memory Usage**: 2-4GB for BERT service (GPU mode)

## 🔐 Security Considerations

1. **Input Validation**: All text inputs are sanitized and length-limited
2. **Rate Limiting**: Consider implementing rate limiting for API endpoints
3. **Authentication**: Ensure proper authentication for admin endpoints
4. **Data Privacy**: Sentiment analysis doesn't store personal information
5. **Resource Limits**: BERT service has built-in memory and processing limits