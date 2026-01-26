# Recommendation System Microservices

A scalable recommendation system built with microservices architecture, featuring enhanced social features and following user vectors integration.

## 🏗️ Architecture Overview

This repository contains a recommendation system divided into three specialized microservices:

1. **Core Recommendations Service** (Port 5000) - Pure ML recommendations using Two-Tower model
2. **Comment Analysis Service** (Port 8080) - BERT sentiment analysis and text processing
3. **Social Recommendations Service** (Port 8081) - Social-enhanced recommendations with following user vectors

## 📁 Repository Structure

```
RecommendationMLModel/
├── services/                          # Microservices
│   ├── core-recommendations/          # Core ML recommendations
│   │   ├── core_recommendations_service.py
│   │   └── requirements.txt
│   ├── social-recommendations/        # Social-enhanced recommendations
│   │   ├── social_recommendations_service.py
│   │   └── requirements.txt
│   └── comment-analysis/              # Comment sentiment analysis
│       ├── comment_analysis_service.py
│       ├── bert_sentiment_service.py.legacy
│       └── requirements.txt
├── shared/                            # Shared components
│   ├── components/                    # Reusable components
│   │   ├── TwoTower.py               # Two-Tower ML model
│   │   ├── MetadataEnhancer.py       # Metadata enhancement
│   │   ├── MockRedis.py              # Redis mock for testing
│   │   ├── ContentFilter.py          # Content filtering
│   │   └── SocialSignalProcessor.py  # Social signal processing
│   └── utils/                         # Shared utilities
├── deployment/                        # Deployment configurations
│   ├── docker/                        # Docker configurations
│   │   ├── docker-compose.microservices.yml
│   │   ├── Dockerfile.core-recommendations
│   │   ├── Dockerfile.social-recommendations
│   │   ├── Dockerfile.comment-analysis
│   │   └── nginx/                     # Load balancer config
│   └── kubernetes/                    # Kubernetes manifests
│       ├── core-recommendations-deployment.yaml
│       ├── social-recommendations-deployment.yaml
│       └── comment-analysis-deployment.yaml
├── scripts/                           # Utility scripts
│   ├── start_microservices.sh        # Start all services
│   ├── stop_microservices.sh         # Stop all services
│   └── start_bert_service.sh         # Legacy BERT service
├── docs/                              # Documentation
│   ├── MICROSERVICES_README.md       # Detailed microservices guide
│   ├── SENTIMENT_ANALYSIS_README.md  # Sentiment analysis docs
│   └── recommendation-system-spec.md # System specifications
├── legacy/                            # Legacy monolithic code
│   ├── ReccommendationEngine.py      # Original recommendation engine
│   ├── OffsetTracking.py             # Offset tracking
│   └── SeenCache.py                  # Seen content cache
├── model_checkpoints/                 # ML model files
├── logs/                              # Service logs
├── .env.microservices                # Environment configuration
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Redis
- Docker (optional)
- 4GB+ RAM (for BERT model)

### Local Development

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd RecommendationMLModel
   cp .env.microservices .env
   ```

2. **Start Redis**:
   ```bash
   # macOS
   brew services start redis
   
   # Linux
   sudo systemctl start redis
   
   # Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```

3. **Start All Services**:
   ```bash
   chmod +x scripts/start_microservices.sh
   ./scripts/start_microservices.sh
   ```

4. **Verify Services**:
   ```bash
   curl http://localhost:5000/health  # Core Recommendations
   curl http://localhost:8080/health  # Comment Analysis
   curl http://localhost:8081/health  # Social Recommendations
   ```

### Docker Deployment

```bash
cd deployment/docker
docker-compose -f docker-compose.microservices.yml up --build
```

## 📊 Service Details

### Core Recommendations Service (Port 5000)

**Purpose**: Pure ML recommendations using Two-Tower neural network model.

**Key Features**:
- Two-Tower model for user-post similarity
- Candidate fetching and scoring
- Metadata enhancement
- Redis caching for vectors
- Inter-service communication

**API Endpoints**:
- `POST /recommendations` - Get recommendations (with optional social enhancement)
- `POST /recommendations/posts` - Get post recommendations
- `POST /recommendations/trailers` - Get trailer recommendations
- `GET /health` - Health check

### Comment Analysis Service (Port 8080)

**Purpose**: BERT-based sentiment analysis and text processing.

**Key Features**:
- BERT multilingual sentiment analysis
- Toxicity detection
- Batch processing
- Post-level sentiment aggregation
- GPU support for inference

**API Endpoints**:
- `POST /analyze` - Analyze single text
- `POST /analyze/batch` - Batch text analysis
- `GET /comments/posts/{postId}/sentiment` - Post sentiment summary
- `POST /comments/sentiment/batch` - Batch post sentiment

### Social Recommendations Service (Port 8081)

**Purpose**: Social-enhanced recommendations with following user vectors.

**Key Features**:
- Following user vectors integration
- Social influence scoring
- Community-based recommendations
- Social interaction tracking
- Sentiment-enhanced social scoring

**API Endpoints**:
- `POST /social/recommendations` - Pure social recommendations
- `POST /social/enhance` - Enhance recommendations with social signals
- `GET /social/following/{userId}/influence` - Following influence data
- `POST /social/interactions/update` - Update social interactions

## 🔗 Enhanced Social Features

### Following User Vectors Integration

The system now incorporates following user relationships to enhance recommendations:

1. **Following User Profiles**: Analysis of following relationships for influence weight and preference alignment
2. **Social Candidate Generation**: Recommendations based on followed users' interactions
3. **Influence Scoring**: Weighted influence based on interaction patterns
4. **Community Detection**: Content boosting within user communities

### Social Signal Processing

Enhanced social signals include:
- **LIKE** (weight: 0.8) - Positive interaction
- **SAVE** (weight: 1.0) - Strong positive signal
- **FOLLOW** (weight: 1.2) - Relationship signal
- **NOT_INTERESTED** (weight: -0.9) - Strong negative signal

## 🔄 Request Flow Examples

### Standard Recommendations
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

### Social-Enhanced Recommendations
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

### Pure Social Recommendations
```bash
curl -X POST http://localhost:8081/social/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "12345",
    "contentType": "posts",
    "limit": 20
  }'
```

### Comment Sentiment Analysis
```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a great post! I love it."
  }'
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env.microservices`:

```bash
# Service Ports
CORE_SERVICE_PORT=5000
SOCIAL_SERVICE_PORT=8081  
COMMENT_SERVICE_PORT=8080

# Service URLs
SPRING_API_URL=http://localhost:8080
SOCIAL_SERVICE_URL=http://localhost:8081
COMMENT_SERVICE_URL=http://localhost:8080

# Social Features
FOLLOWING_WEIGHT=0.4
COMMUNITY_WEIGHT=0.3
SENTIMENT_WEIGHT=0.3
MAX_FOLLOWING_USERS=50

# ML Model
USER_FEATURE_DIM=64
POST_FEATURE_DIM=64
EMBEDDING_DIM=32
MODEL_DIR=./model_checkpoints

# Cache Configuration
REDIS_HOST=localhost
CACHE_TTL=1800
```

## 🐳 Docker Configuration

The system includes comprehensive Docker support:

- **Individual Dockerfiles** for each service
- **Docker Compose** for orchestration
- **Nginx** load balancer configuration
- **Redis** for caching
- **Health checks** and monitoring

### Build and Deploy
```bash
# Build all services
cd deployment/docker
docker-compose -f docker-compose.microservices.yml build

# Start all services
docker-compose -f docker-compose.microservices.yml up

# Scale specific service
docker-compose -f docker-compose.microservices.yml up --scale social-recommendations=3
```

## ☸️ Kubernetes Deployment

Kubernetes manifests are provided for production deployment:

```bash
# Deploy all services
kubectl apply -f deployment/kubernetes/

# Check deployments
kubectl get deployments
kubectl get services
kubectl get pods
```

## 📈 Monitoring and Health

### Health Checks

Each service provides health endpoints:
- Core: `GET /health` - Checks model loading and Redis connectivity
- Comment: `GET /health` - Checks BERT model and cache status  
- Social: `GET /health` - Checks service connectivity

### Logging

Structured logging includes:
- Request/response tracking
- Service communication events
- Error details and stack traces
- Performance metrics

### Metrics

Services expose metrics for:
- Request rates and response times
- Cache hit/miss ratios
- Model inference times
- Inter-service communication latency

## 🔧 Development

### Adding New Features

1. **Shared Components**: Add reusable code to `shared/components/`
2. **Service-Specific**: Add to appropriate service directory
3. **Dependencies**: Update service `requirements.txt`
4. **Configuration**: Add environment variables to `.env.microservices`
5. **Documentation**: Update relevant docs

### Testing

```bash
# Install dependencies for each service
cd services/core-recommendations && pip install -r requirements.txt
cd ../social-recommendations && pip install -r requirements.txt  
cd ../comment-analysis && pip install -r requirements.txt

# Run tests (implement based on your testing framework)
pytest tests/
```

## 🛠️ Troubleshooting

### Common Issues

1. **Service Communication Failures**:
   - Check service URLs in environment variables
   - Verify all services are running: `./scripts/start_microservices.sh`
   - Test connectivity: `curl http://localhost:PORT/health`

2. **BERT Model Loading**:
   - Ensure 4GB+ RAM available
   - Check model cache directory permissions
   - Verify transformers library version

3. **Redis Connection**:
   - Start Redis: `redis-server`
   - Test connection: `redis-cli ping`
   - Check Redis configuration in `.env`

4. **Import Errors**:
   - Verify shared components path in service files
   - Check Python path configuration
   - Ensure all dependencies installed

### Debug Commands

```bash
# Check service status
curl http://localhost:5000/health
curl http://localhost:8080/health
curl http://localhost:8081/health

# View logs
tail -f logs/core-recommendations.log
tail -f logs/comment-analysis.log
tail -f logs/social-recommendations.log

# Test basic functionality
curl -X POST http://localhost:5000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"userId": "test", "limit": 5}'
```

## 📚 Documentation

- **[Microservices Guide](docs/MICROSERVICES_README.md)** - Detailed architecture documentation
- **[Sentiment Analysis](docs/SENTIMENT_ANALYSIS_README.md)** - Comment analysis details
- **[System Specifications](docs/recommendation-system-spec.md)** - Technical specifications

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test locally
4. Update documentation as needed
5. Submit pull request

### Code Standards

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Add docstrings for public methods
- Include error handling and logging
- Write tests for new features

## 📄 License

[Your License Here]

## 🆘 Support

For issues and questions:
- Check the troubleshooting section above
- Review service logs in `logs/` directory
- Open an issue on GitHub
- Contact the development team

---

This microservices architecture provides a scalable, maintainable foundation for the recommendation system with enhanced social features and following user vectors integration.