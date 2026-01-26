# Recommendation System Microservices Architecture

This document describes the 3-microservice architecture for the recommendation system with enhanced social features and following user vectors integration.

## Architecture Overview

The recommendation system has been refactored from a monolithic service into three specialized microservices:

1. **Core Recommendations Service** (Port 5000) - Pure ML recommendations using Two-Tower model
2. **Comment Analysis Service** (Port 8080) - BERT sentiment analysis and text processing  
3. **Social Recommendations Service** (Port 8081) - Social-enhanced recommendations with following user vectors

## Services Detail

### 1. Core Recommendations Service (Port 5000)

**Responsibilities:**
- Pure ML recommendations using Two-Tower model
- Candidate fetching and base scoring
- User/post vector management and caching
- Metadata enhancement
- Orchestrates social enhancement via service calls

**Key Features:**
- Two-Tower neural network for user-post similarity
- Redis caching for user/post vectors
- Metadata-enhanced scoring
- Inter-service communication with social and comment services
- Cursor-based pagination support

**API Endpoints:**
```
POST /recommendations - Get recommendations with optional social enhancement
POST /recommendations/posts - Get post recommendations  
POST /recommendations/trailers - Get trailer recommendations
POST /recommendations/social - Get social recommendations (proxied)
GET /health - Health check
GET /stats - Service statistics
```

**Environment Variables:**
```
PORT=5000
SPRING_API_URL=http://localhost:8080
SOCIAL_SERVICE_URL=http://localhost:8081
COMMENT_SERVICE_URL=http://localhost:8080
REDIS_HOST=localhost
REDIS_PORT=6379
USER_FEATURE_DIM=64
POST_FEATURE_DIM=64
EMBEDDING_DIM=32
MODEL_DIR=./model_checkpoints
```

### 2. Comment Analysis Service (Port 8080)

**Responsibilities:**
- BERT-based sentiment analysis for comments and posts
- Text toxicity detection and content analysis
- Batch processing of sentiment analysis requests
- Comment aggregation and sentiment scoring
- Caching of analysis results

**Key Features:**
- BERT multilingual sentiment analysis
- Toxicity and hate speech detection
- Spam detection using pattern matching
- Batch processing capabilities
- LRU caching for performance
- GPU support for BERT inference

**API Endpoints:**
```
POST /analyze - Analyze single text sentiment
POST /analyze/batch - Batch text sentiment analysis
GET /comments/posts/{postId}/sentiment - Get post sentiment summary
POST /comments/posts/{postId}/analyze - Detailed comment analysis
POST /comments/sentiment/batch - Batch post sentiment analysis
POST /cache/clear - Clear analysis cache
GET /health - Health check
GET /stats - Service statistics
```

**Environment Variables:**
```
PORT=8080
BERT_MODEL=nlptown/bert-base-multilingual-uncased-sentiment
SENTIMENT_CACHE_SIZE=2000
CACHE_TTL=7200
SPRING_API_URL=http://localhost:8080
```

### 3. Social Recommendations Service (Port 8081)

**Responsibilities:**
- Social signal processing and boosting
- Following user vectors integration
- Social influence calculation based on followed users
- Community-based recommendation enhancement
- Social interaction tracking and management

**Key Features:**
- Following user influence scoring
- Social boost factor calculation
- Sentiment-enhanced social scoring
- Social interaction tracking (likes, saves, follows)
- Community influence detection
- Dual preference boosting (positive/negative interactions)

**API Endpoints:**
```
POST /social/recommendations - Get purely social recommendations
POST /social/enhance - Enhance recommendations with social signals
POST /social/interactions/update - Update social interaction data
GET /social/following/{userId}/influence - Get following influence data
GET /health - Health check
GET /stats - Service statistics
```

**Environment Variables:**
```
PORT=8081
COMMENT_ANALYSIS_SERVICE_URL=http://localhost:8080
CORE_RECOMMENDATIONS_SERVICE_URL=http://localhost:5000
SPRING_API_URL=http://localhost:8080
FOLLOWING_WEIGHT=0.4
COMMUNITY_WEIGHT=0.3
SENTIMENT_WEIGHT=0.3
MAX_FOLLOWING_USERS=50
CACHE_TTL=1800
```

## Enhanced Social Features

### Following User Vectors Integration

The social service now incorporates following user vectors to enhance recommendations:

1. **Following User Profiles**: Each user's following relationships are analyzed for influence weight, preference alignment, and interaction frequency.

2. **Social Candidate Generation**: Recommendations are generated based on content that followed users have positively interacted with.

3. **Influence Scoring**: Following users are weighted by their influence on the target user based on interaction patterns and mutual connections.

4. **Community Detection**: Users with similar following patterns receive boosted recommendations for content popular within their community.

### Social Signal Processing

Enhanced social signals include:
- **LIKE**: Positive interaction (weight: 0.8)
- **SAVE**: Strong positive signal (weight: 1.0)  
- **COMMENT_POSITIVE**: Engagement signal (weight: 0.6)
- **VIEW_TIME_HIGH**: Attention signal (weight: 0.4)
- **NOT_INTERESTED**: Strong negative signal (weight: -0.9)
- **FOLLOW/UNFOLLOW**: Relationship signals (weight: ±1.2)

## Service Communication

### Request Flow

1. **Standard Recommendations**:
   ```
   Client → Core Service → [Candidate Fetch] → [ML Scoring] → Response
   ```

2. **Social-Enhanced Recommendations**:
   ```
   Client → Core Service → [Base Scoring] → Social Service → [Social Enhancement] → Response
   ```

3. **Social-Only Recommendations**:
   ```
   Client → Core Service → Social Service → Comment Service → [Sentiment Data] → Response
   ```

### Inter-Service Communication

Services communicate via HTTP REST APIs with:
- **Timeout handling**: 10-30 second timeouts depending on service
- **Fallback mechanisms**: Graceful degradation when services are unavailable
- **Error propagation**: Clear error messages between services
- **Circuit breaker pattern**: Prevents cascade failures

## Deployment Options

### Local Development

1. **Using Scripts**:
   ```bash
   # Start all services
   ./start_microservices.sh
   
   # Stop all services
   ./stop_microservices.sh
   ```

2. **Manual Start**:
   ```bash
   # Copy environment variables
   cp .env.microservices .env
   
   # Start Redis
   redis-server
   
   # Start services (in separate terminals)
   python comment_analysis_service.py
   python social_recommendations_service.py  
   python core_recommendations_service.py
   ```

### Docker Deployment

```bash
# Build and start all services
docker-compose -f docker-compose.microservices.yml up --build

# Start specific service
docker-compose -f docker-compose.microservices.yml up core-recommendations

# Scale services
docker-compose -f docker-compose.microservices.yml up --scale social-recommendations=2
```

### Production Deployment

For production, consider:
- **Container orchestration**: Kubernetes or Docker Swarm
- **Load balancing**: Nginx or cloud load balancers
- **Service discovery**: Consul, etcd, or cloud-native solutions
- **Monitoring**: Prometheus, Grafana, and distributed tracing
- **Secrets management**: Vault or cloud secrets services

## Monitoring and Health Checks

### Health Endpoints

Each service provides health checks at `/health`:
- **Core Service**: Checks model loading and Redis connectivity
- **Comment Service**: Checks BERT model availability and cache status
- **Social Service**: Checks service connectivity and cache status

### Metrics

Services expose metrics for:
- Request rates and response times
- Cache hit/miss ratios
- Model inference times
- Inter-service communication latency
- Error rates and types

### Logging

Structured JSON logging includes:
- Request/response tracking
- Service communication events
- Error details and stack traces
- Performance metrics
- User interaction events

## Configuration Management

### Environment Variables

All services use environment variables for configuration:
- **Service URLs**: For inter-service communication
- **Cache settings**: TTL, size limits, Redis configuration
- **Model parameters**: Dimensions, paths, weights
- **Feature flags**: Enable/disable social features

### Secrets

Sensitive configuration:
- Redis passwords
- API keys
- SSL certificates
- Database credentials

## Security Considerations

### Inter-Service Communication

- **Network isolation**: Services run in isolated networks
- **Service authentication**: API keys or mutual TLS
- **Request validation**: Input sanitization and validation
- **Rate limiting**: Prevent abuse and cascade failures

### Data Protection

- **Input sanitization**: Prevent injection attacks
- **Output encoding**: Safe data transmission
- **Sensitive data**: Proper handling of user data
- **Audit logging**: Track all data access and modifications

## Performance Optimization

### Caching Strategy

- **Redis**: User/post vectors, sentiment analysis results
- **Service-level caching**: Following user profiles, social signals
- **TTL management**: Different cache lifetimes based on data type
- **Cache invalidation**: Smart invalidation on user interactions

### Scaling Recommendations

- **Horizontal scaling**: Multiple instances of each service
- **Load balancing**: Distribute requests across instances
- **Database optimization**: Indexed queries, connection pooling
- **Async processing**: Background jobs for heavy computations

## Migration Guide

### From Monolithic to Microservices

1. **Preparation**:
   - Set up Redis instance
   - Prepare environment variables
   - Build Docker images

2. **Gradual Migration**:
   - Start with comment analysis service
   - Add social service while keeping monolith
   - Switch core service last

3. **Testing**:
   - A/B test microservices vs monolith
   - Monitor performance metrics
   - Validate recommendation quality

4. **Rollback Plan**:
   - Keep monolithic service available
   - Database compatibility maintained
   - Quick switch capability

## API Examples

### Get Standard Recommendations

```bash
curl -X POST http://localhost:5000/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "12345",
    "contentType": "posts",
    "limit": 20,
    "enableSocial": false
  }'
```

### Get Social-Enhanced Recommendations

```bash
curl -X POST http://localhost:5000/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "12345", 
    "contentType": "posts",
    "limit": 20,
    "enableSocial": true,
    "socialWeight": 0.3
  }'
```

### Get Social-Only Recommendations

```bash
curl -X POST http://localhost:8081/social/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "12345",
    "contentType": "posts", 
    "limit": 20
  }'
```

### Analyze Comment Sentiment

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a great post! I love it."
  }'
```

### Update Social Interaction

```bash
curl -X POST http://localhost:8081/social/interactions/update \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 12345,
    "postId": 67890,
    "interactionType": "like",
    "strength": 1.0
  }'
```

## Troubleshooting

### Common Issues

1. **Service Communication Failures**:
   - Check service URLs in environment variables
   - Verify services are running and healthy
   - Check network connectivity between containers

2. **BERT Model Loading Issues**:
   - Ensure sufficient memory (4GB+ recommended)
   - Check Transformers library version
   - Verify model cache directory permissions

3. **Redis Connection Issues**:
   - Verify Redis is running
   - Check Redis host/port configuration
   - Test connection with redis-cli

4. **Performance Issues**:
   - Monitor cache hit rates
   - Check service response times
   - Scale bottleneck services

### Debug Commands

```bash
# Check service health
curl http://localhost:5000/health
curl http://localhost:8080/health  
curl http://localhost:8081/health

# Check service stats
curl http://localhost:5000/stats
curl http://localhost:8080/stats
curl http://localhost:8081/stats

# Test Redis connection
redis-cli ping

# Check service logs
tail -f logs/core-recommendations.log
tail -f logs/comment-analysis.log
tail -f logs/social-recommendations.log

# Test service communication
curl -X POST http://localhost:5000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"userId": "test", "limit": 5}'
```

## Contributing

### Development Setup

1. Clone repository and install dependencies
2. Copy `.env.microservices` to `.env`  
3. Start Redis: `redis-server`
4. Start services: `./start_microservices.sh`
5. Run tests: `pytest tests/`

### Code Standards

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Add docstrings for all public methods
- Include error handling and logging
- Write unit tests for new features

### Service Dependencies

- **Core Service**: TwoTower, MetadataEnhancer, MockRedis
- **Comment Service**: Transformers, torch, flask
- **Social Service**: numpy, requests, flask

This architecture provides a scalable, maintainable foundation for the recommendation system with enhanced social features and following user vectors integration.