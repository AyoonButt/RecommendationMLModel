# Technical Specification: Enhanced Two-Tower Recommendation System

## 1. System Overview

The recommendation system utilizes an enhanced Two-Tower neural network architecture with comprehensive social analysis and content filtering capabilities. The system has evolved from basic personalized recommendations to include:

- **Social Signal Processing**: Leveraging followed user vectors and social influence
- **Sentiment Analysis**: BERT-based comment sentiment analysis 
- **Dual Preference Boosting**: Advanced handling of positive/negative user feedback
- **Content Filtering**: Multi-layered content safety and appropriateness filtering
- **Real-time Processing**: Social signals and content analysis in real-time

## 2. Enhanced Data Schema Integration

### 2.1 Core Database Entities

The system integrates with the existing database schema:

- **users**: User profiles and demographics
- **posts**: Content items (movies/shows)
- **user_post_interactions**: Records of user engagement with content
- **user_games/post_games**: Genre associations
- **subscription_providers**: Content providers
- **crew/cast_members**: Personnel involved in content creation

### 2.2 Social Analysis Extensions

New social data structures:

- **user_follows**: Following relationships between users
- **comment_sentiment**: BERT-analyzed sentiment scores for comments
- **social_influence_metrics**: Calculated influence scores between users
- **user_social_vectors**: Social influence embeddings for users
- **content_filter_preferences**: User-specific content filtering settings

### 2.3 Enhanced Vector Representations

- **user_embeddings**: Enhanced with social signals and preference history
- **post_embeddings**: Content characteristics with sentiment metadata
- **social_embeddings**: Followed user influence vectors
- **interaction_weights**: Configurable weights including social and sentiment factors

## 3. Enhanced Two-Tower Model Architecture

### 3.1 Enhanced User Tower

**Inputs:**
- User ID (for embedding lookup)
- Personal features:
  - Demographic data (region, language)
  - Subscription information
  - Genre preferences from user_games
  - Behavioral metrics
- **Social features**:
  - Followed user vectors
  - Social influence scores
  - Social activity level
- **Preference history**:
  - Positive interactions (likes, saves)
  - Negative interactions (not interested)
  - Dual preference weights

**Architecture:**
- Embedding layer for user ID
- Social vector integration layer
- Feature concatenation with enhanced social signals
- Dense layers with batch normalization (128 → 64 → 32 units)
- Output: 32-dimensional enhanced user embedding

### 3.2 Enhanced Post Tower

**Inputs:**
- Post ID (for embedding lookup)
- Content features:
  - Genre associations from post_games
  - Content type and metadata
  - Release information
  - Cast/crew attributes
  - Popularity metrics
- **Sentiment features**:
  - BERT-analyzed comment sentiment
  - Sentiment confidence scores
  - Comment volume and distribution
- **Content safety features**:
  - Toxicity scores
  - Explicit content ratings
  - NSFW classifications
  - Category-based safety tags

**Architecture:**
- Embedding layer for post ID
- Sentiment analysis integration layer
- Feature concatenation with sentiment and safety signals
- Dense layers with batch normalization (128 → 64 → 32 units)
- Output: 32-dimensional enhanced post embedding

### 3.3 Advanced Scoring Mechanisms

Multiple scoring functions available:

1. **Base Scoring**: `compute_scores()`
   - Cosine similarity between user and post embeddings
   - Candidate metadata enhancement

2. **Socially Enhanced Scoring**: `compute_socially_enhanced_scores()`
   - Base scores + social influence boosting
   - Followed user preference alignment
   - Social activity weighting

3. **Dual Preference Scoring**: `compute_dual_preference_scores()`
   - Positive interaction boosting (likes, saves)
   - Negative interaction penalties (not interested)
   - Balanced preference learning

4. **Filtered Scoring**: `compute_filtered_scores()`
   - Content safety filtering
   - User preference-based filtering
   - Comprehensive content analysis

## 4. Social Analysis System

### 4.1 Social Signal Processor

**Core Components:**
- **SocialSignalProcessor Class**: Central hub for social signal processing
- **Social Vector Generation**: Creates influence-weighted user vectors
- **Sentiment Integration**: BERT sentiment analysis integration
- **Dual Preference Boosting**: Advanced positive/negative signal handling

**Key Features:**
- LRU caching for performance optimization
- Configurable social weights (0.0-1.0)
- Real-time social signal updates
- Fallback mechanisms for service failures

### 4.2 BERT Sentiment Analysis Service

**Components:**
- **bert_sentiment_service.py**: Standalone Flask service
- **Multilingual BERT Model**: Support for multiple languages
- **Batch Processing**: Efficient analysis of multiple comments
- **Caching**: LRU cache for repeated sentiment analysis

**API Endpoints:**
- `/analyze` - Single text sentiment analysis
- `/analyze/batch` - Batch sentiment analysis
- `/health` - Service health check
- `/cache/clear` - Clear analysis cache

### 4.3 Social Vector Service Integration

**Features:**
- **Followed User Influence**: Weighted influence from followed users
- **Social Boost Calculation**: Dynamic social signal boosting
- **Preference Alignment**: Alignment scoring between user preferences
- **Social Activity Weighting**: Activity-based influence scaling

## 5. Content Filtering System

### 5.1 Content Filter Architecture

**Core Components:**
- **ContentFilter Class**: Multi-layered content analysis and filtering
- **Toxicity Detection**: Integration with external toxicity services
- **Explicit Content Detection**: Keyword and pattern-based detection
- **User Preference Filtering**: Customizable user-specific filters

**Filter Types:**
- **EXPLICIT_CONTENT**: Adult/sexual content detection
- **TOXICITY**: Toxic language and behavior detection  
- **NSFW**: Not Safe For Work content identification
- **HATE_SPEECH**: Hate speech and discrimination detection
- **SPAM**: Spam and low-quality content filtering
- **AGE_INAPPROPRIATE**: Age-based content appropriateness

### 5.2 Filter Actions

**Action Types:**
- **BLOCK**: Completely remove content from recommendations
- **DOWNRANK**: Significantly reduce recommendation score (30% of original)
- **WARN**: Slightly reduce score (80% of original) 
- **FLAG**: Mark for manual review

### 5.3 User Preference Management

**Configurable Settings:**
- NSFW filter enable/disable
- Toxicity threshold (0.0-1.0)
- Explicit content filtering
- Hate speech filtering
- Spam filtering
- Age-appropriate content only
- Blocked categories and languages
- Custom keyword blocking

## 6. Implementation Phases (Completed)

### 6.1 Phase 1: BERT Comment Analysis ✅

**Implemented Components:**
- BERT sentiment analysis service
- Comment sentiment database integration
- API endpoints for sentiment analysis
- Sentiment caching and optimization

### 6.2 Phase 2: Social Vector Integration ✅

**Implemented Components:**
- Social vector service for followed user influence
- Enhanced user vector service with social signals
- Social influence API endpoints
- Real-time social signal processing

### 6.3 Phase 3: Dual Preference Boosting ✅

**Implemented Components:**
- Python ML model integration
- Social signal processor
- Enhanced TwoTower model with social scoring
- Dual preference boosting algorithms
- Updated recommendation engine API

### 6.4 Phase 4: Content Filtering ✅

**Implemented Components:**
- Comprehensive content filter system
- Multi-layered safety analysis
- User preference-based filtering
- Content filtering API endpoints
- Integration with recommendation scoring

### 6.5 Future Phases

**Phase 5: Risk Assessment** (Planned)
- Content risk scoring
- User behavior risk analysis
- Dynamic risk-based filtering

**Phase 6: Diversification** (Planned)
- Content diversity algorithms
- Category and genre diversification
- Temporal diversity optimization

**Phase 7: Advanced Ranking** (Planned)
- Multi-objective optimization
- Learning-to-rank improvements
- Contextual ranking factors

**Phase 8: Explanation System** (Planned)
- Recommendation explanation generation
- User-friendly reason descriptions
- Transparency and interpretability

## 7. API Endpoints

### 7.1 Core Recommendation Endpoints

- `POST /recommendations` - Basic personalized recommendations
- `POST /recommendations/social` - Socially enhanced recommendations
- `POST /recommendations/dual-preference` - Dual preference boosted recommendations
- `POST /recommendations/filtered` - Fully filtered recommendations

### 7.2 Social Analysis Endpoints

- `POST /interactions/update` - Update user social interactions
- `GET /social/stats` - Social processor statistics
- `GET /social/users/:userId/influence` - User social influence data

### 7.3 Content Filtering Endpoints

- `GET /filter/stats` - Content filter statistics
- `POST /filter/cache/clear` - Clear content filter cache
- `GET /users/:userId/filter-preferences` - User filter preferences

### 7.4 Sentiment Analysis Endpoints

- `POST /analyze` - Single text sentiment analysis
- `POST /analyze/batch` - Batch sentiment analysis
- `GET /cache/info` - Sentiment cache information

### 7.5 Health and Monitoring

- `GET /health` - Comprehensive service health check
- `GET /cache/stats` - Cache performance statistics

## 8. Enhanced Performance Considerations

### 8.1 Caching Strategy

**Multi-level Caching:**
- **Redis/Valkey**: User vectors, post vectors, candidate data
- **LRU Caches**: Social signals, sentiment analysis, content analysis
- **TTL Management**: Configurable cache expiration (1-2 hours)

**Cache Keys:**
- `user_vector:{userId}` - User embedding vectors
- `post_vector:{postId}` - Post embedding vectors
- `social_profile_{userId}` - User social profiles
- `sentiment_{postId}` - Post sentiment analysis
- `analysis_{postId}` - Content safety analysis

### 8.2 Batch Processing

**Optimizations:**
- Batch sentiment analysis (up to 100 texts)
- Batch content filtering
- Parallel API calls for candidate fetching
- Vectorized similarity computations

### 8.3 Fallback Mechanisms

**Robustness:**
- BERT service fallback to keyword-based sentiment
- Toxicity service fallback to pattern matching
- Social processor graceful degradation
- Content filter error handling

## 9. Configuration Management

### 9.1 Environment Variables

**Core Configuration:**
- `SPRING_API_URL` - Kotlin API base URL
- `BERT_SERVICE_URL` - BERT sentiment service URL
- `TOXICITY_SERVICE_URL` - Toxicity detection service URL
- `REDIS_HOST/PORT/PASSWORD` - Redis connection settings

**Feature Toggles:**
- `SOCIAL_WEIGHT` - Social signal weighting (default: 0.25)
- `SOCIAL_CACHE_SIZE` - Social signal cache size (default: 1000)
- `FILTER_CACHE_SIZE` - Content filter cache size (default: 2000)

### 9.2 Model Configuration

**ML Model Settings:**
- `USER_FEATURE_DIM` - User feature vector dimension (default: 64)
- `POST_FEATURE_DIM` - Post feature vector dimension (default: 64)
- `EMBEDDING_DIM` - Final embedding dimension (default: 32)
- `HIDDEN_DIMS` - Neural network layer sizes (default: 128,64)

## 10. Security and Safety

### 10.1 Content Safety

**Multi-layer Protection:**
- Automated toxicity detection
- Explicit content filtering
- Hate speech prevention
- Spam content blocking
- Age-appropriate content enforcement

### 10.2 User Privacy

**Privacy Considerations:**
- Social signal anonymization
- Interaction data encryption
- Configurable privacy settings
- GDPR compliance ready

### 10.3 Service Security

**Security Measures:**
- API endpoint rate limiting
- Input validation and sanitization
- Error message sanitization
- Service health monitoring

## 11. Monitoring and Observability

### 11.1 Metrics Collection

**Performance Metrics:**
- Recommendation response times
- Cache hit rates
- Social signal processing latency
- Content filtering throughput

**Quality Metrics:**
- Sentiment analysis accuracy
- Content filter precision/recall
- Social signal effectiveness
- User engagement improvements

### 11.2 Health Monitoring

**Service Health:**
- Model loading status
- Cache connectivity
- External service availability
- Processing pipeline status

### 11.3 Error Tracking

**Error Monitoring:**
- API endpoint errors
- Model inference failures
- External service timeouts
- Cache eviction rates

## 12. Future Enhancements

### 12.1 Advanced ML Features

**Planned Improvements:**
- Transformer-based user modeling
- Graph neural networks for social signals
- Multi-task learning for engagement prediction
- Reinforcement learning for recommendation optimization

### 12.2 Real-time Processing

**Streaming Enhancements:**
- Real-time social signal updates
- Live sentiment analysis
- Dynamic content filtering
- Instant preference learning

### 12.3 Advanced Analytics

**Analytics Extensions:**
- User journey analysis
- Content performance tracking
- Social influence measurement
- Filter effectiveness analysis

This enhanced specification reflects the current state of the recommendation system with comprehensive social analysis, sentiment processing, dual preference boosting, and multi-layered content filtering capabilities.