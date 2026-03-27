# TMDB API & ML Recommendation System Integration Report

## Executive Summary

This report documents the complete data flow from TMDB API through the ML recommendation system. The architecture uses a **cache-first strategy** where the Android app fetches TMDB data, caches it via the Spring Boot backend, and the ML microservices consume enriched metadata for personalized recommendations.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ANDROID APP                                         │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────────┐                  │
│  │  ViewModels  │───▶│ CacheFirstReq   │───▶│  TMDB Direct   │                  │
│  │  (UI Layer)  │    │    Manager      │    │  (ApiService)  │                  │
│  └──────────────┘    └────────┬────────┘    └───────┬────────┘                  │
│                               │                      │                           │
│                               ▼                      ▼                           │
│                      ┌────────────────┐      api.themoviedb.org                 │
│                      │ BatchCacheMan  │             │                           │
└──────────────────────┼──────────────────────────────┼───────────────────────────┘
                       │                              │
                       ▼                              │
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         SPRING BOOT BACKEND                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        PostgreSQL Cache                                      │ │
│  │  • TMDB responses cached    • User interactions stored                      │ │
│  │  • Content metadata         • Engagement metrics                            │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                            │
│                                      ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                    Internal ML API Endpoints                                 │ │
│  │  GET  /api/internal/ml/users/{id}/candidates                                │ │
│  │  GET  /api/internal/ml/users/{id}/vector                                    │ │
│  │  GET  /api/internal/users/{id}/metadata                                     │ │
│  │  POST /api/internal/posts/metadata/batch                                    │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         ML MICROSERVICES (Python)                                 │
│                                                                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                │
│  │ Core Recommend. │   │ Social Signals  │   │ Comment Analysis│                │
│  │   Port 5000     │   │   Port 8081     │   │   Port 8082     │                │
│  │                 │   │                 │   │                 │                │
│  │ • Two-Tower     │   │ • Following     │   │ • BERT Sentiment│                │
│  │ • Metadata Enh. │   │ • Community     │   │ • Toxicity      │                │
│  │ • RL Agent      │   │ • Sentiment     │   │ • Spam Filter   │                │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘                │
│           │                     │                      │                         │
│           └─────────────────────┼──────────────────────┘                         │
│                                 ▼                                                │
│                    ┌─────────────────────┐                                       │
│                    │   Redis (Valkey)    │                                       │
│                    │   Caching Layer     │                                       │
│                    └─────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. TMDB API Endpoints Used

### Direct API Calls (Android App → TMDB)

| Endpoint | Purpose | Used In |
|----------|---------|---------|
| `GET discover/movie` | Find movies by filters (provider, language, date) | ApiWorker, ApiService |
| `GET discover/tv` | Find TV shows by filters | ApiWorker, ApiService |
| `GET movie/{id}/videos` | Get trailers/videos | PosterFragment, CustomPlayer |
| `GET tv/{id}/videos` | Get TV trailers | PosterFragment, CustomPlayer |
| `GET movie/{id}/credits` | Cast & crew | PosterFragment |
| `GET tv/{id}/aggregate_credits` | TV cast & crew | PosterFragment |
| `GET movie/{id}` | Full movie details | Search, detail views |
| `GET tv/{id}` | Full TV details | Search, detail views |
| `GET search/multi` | Search movies/TV/people | SearchRepositoryImpl |
| `GET person/{id}` | Actor/person details | PersonDetailFragment |
| `GET watch/providers/movie` | Streaming services list | ProvidersManager |

### Cache-First Data Flow

```
1. App requests content (e.g., movie discovery)
2. CacheFirstRequestManager checks backend cache first
3. Cache hit? → Return cached data immediately
4. Cache miss? → Call TMDB API directly
5. BatchCacheManager queues response for backend storage
6. ApiWorker (background) bulk-syncs TMDB data to backend
```

---

## 3. TMDB Data → ML Feature Mapping

### Content Metadata Fields (Post)

| TMDB Field | ML Feature | Data Type | Usage |
|------------|------------|-----------|-------|
| `genres` | `genreWeights` | Dict[str, float] | Genre preference matching |
| `vote_average` | `voteAverage` | Float (0-10) | Quality scoring |
| `popularity` | `popularityScore` | Float | Trending boost |
| `original_language` | `language` | String | Language matching |
| `cast` | `cast` array | List[Dict] | Actor preference matching |
| `crew` | `crew` dict | Dict | Director/writer preferences |
| `release_date` | `recencyBoost` | Float | Freshness factor |
| `runtime` | `categoricalFeatures` | Int | Content type classification |
| `production_countries` | `regionWeights` | Dict[str, float] | Regional popularity |

### User Metadata Fields

| Field | Source | Purpose |
|-------|--------|---------|
| `languageWeights` | User settings + interactions | Preferred content languages |
| `interestWeights` | Derived from TMDB genres watched | Genre preferences |
| `castPreferences` | Interaction history | Favorite actors |
| `crewPreferences` | Interaction history | Favorite directors/writers |
| `categoricalFeatures` | User profile | Region, age group |

---

## 4. ML Model Architecture

### Two-Tower Neural Network

```
┌─────────────────────────────────────────────────────────────────┐
│                    TWO-TOWER MODEL                               │
│                                                                  │
│   USER TOWER                         POST TOWER                  │
│   ─────────                          ──────────                  │
│   Input: 64D                         Input: 64D                  │
│      ↓                                  ↓                        │
│   Dense(128) + ReLU                  Dense(128) + ReLU           │
│   BatchNorm + Dropout(0.2)           BatchNorm + Dropout(0.2)    │
│      ↓                                  ↓                        │
│   Dense(64) + ReLU                   Dense(64) + ReLU            │
│   BatchNorm + Dropout(0.2)           BatchNorm + Dropout(0.2)    │
│      ↓                                  ↓                        │
│   Dense(32) + L2 Norm                Dense(32) + L2 Norm         │
│      ↓                                  ↓                        │
│   User Embedding (32D)               Post Embedding (32D)        │
│      └──────────────┬────────────────────┘                       │
│                     ↓                                            │
│              Dot Product (Cosine Similarity)                     │
│                     ↓                                            │
│              Score Scaling [0, 1]                                │
└─────────────────────────────────────────────────────────────────┘
```

### MetadataEnhancer Boost Factors

| Factor | Max Boost | Condition |
|--------|-----------|-----------|
| Language Match | +30% | User's preferred language matches content |
| Genre Alignment | +20% | Content genres match user interests |
| Popularity | +15% | vote_average > 7.0 |
| Recency | +25% | Recent release date |
| Cast/Crew Appeal | +20% | Features user's favorite actors/directors |
| Regional | +10% | Content popular in user's region |
| Engagement | +10% | High info button clicks |

### Reinforcement Learning State (160D)

```python
state_components = {
    'user_embedding':     32,  # From Two-Tower
    'post_embedding':     32,  # From Two-Tower
    'temporal_features':   8,  # Hour, day, session duration
    'session_features':    6,  # Session metrics
    'sequence_features':  10,  # Interaction patterns
    'user_preferences':   16,  # From interest_weights (TMDB genres)
    'content_features':   12,  # From TMDB metadata
    'social_signals':      8,  # From sentiment analysis
    'interaction_history': 20, # Recent user actions
    'comment_analysis':   13,  # BERT sentiment features
    'scalar_features':     3   # Satisfaction, exploration, appeal
}
# Total: 160 dimensions
```

---

## 5. Feature Engineering Pipeline

### Stage 1: TMDB Data Extraction (Spring Boot)

```
Raw TMDB Response
    ↓
┌─────────────────────────────────────┐
│ Extract & Normalize:                │
│ • genres → weighted vector          │
│ • cast → ordered list with appeal   │
│ • crew → role-based grouping        │
│ • vote_average → [0-10] preserved   │
│ • popularity → log-scaled           │
│ • release_date → recency factor     │
└─────────────────────────────────────┘
    ↓
Enriched Post Metadata (PostgreSQL)
```

### Stage 2: ML Feature Processing (Python)

```python
# Genre Alignment Calculation
genre_alignment = sum(
    user_interest[genre] * post_genre[genre]
    for genre in common_genres
)
boost = genre_alignment * 0.20

# Popularity Boost (only for high-rated content)
if vote_average > 7.0:
    boost = (vote_average - 7.0) / 3.0 * 0.15

# Cast/Crew Appeal
content_appeal = metadata['castCrewAppealScores']['combinedAppealScore']
user_alignment = match_user_preferences(user_cast_prefs, post_cast)
combined = (content_appeal * 0.5) + (user_alignment * 0.5)
```

### Stage 3: RL Reward Shaping

```python
reward_mapping = {
    'like':           +0.6,
    'save':           +1.0,  # Strongest positive signal
    'share':          +0.8,
    'more_info':      +0.3,
    'skip':           -0.2,
    'not_interested': -0.9   # Strongest negative signal
}

shaping_weights = {
    'exploration':  0.15,  # Encourage novel recommendations
    'engagement':   0.20,  # Maximize click-through
    'diversity':    0.10,  # Avoid filter bubbles
    'novelty':      0.10,  # Balance familiar vs new
    'long_term':    0.25   # Optimize future satisfaction
}
```

---

## 6. Service Communication

### Request Flow: Android → Recommendations

```
Android App
    │
    │ GET /api/recommendations/{userId}/{language}?limit=20
    ▼
Spring Boot Backend
    │
    │ 1. Authenticate request
    │ 2. Fetch user metadata
    │ 3. Forward to ML service
    ▼
Core Recommendations Service (Port 5000)
    │
    │ GET /api/internal/ml/users/{id}/candidates
    │     └─ Returns: vectors + candidateResult
    │
    │ POST /api/internal/posts/metadata/batch
    │     └─ Returns: enriched TMDB metadata
    │
    ├─ Two-Tower scoring
    ├─ MetadataEnhancer boosting
    ├─ RL adjustment (optional)
    │
    ▼
Ranked PostDto List → Spring Boot → Android App
```

### Candidate Selection Parameters

```json
{
  "limit": 40,
  "contentType": "POSTS",
  "includeNewHighQuality": true,
  "newContentRatio": 0.3,
  "interactionLookbackDays": 30,
  "cursor": "pagination_token",
  "highQualityCursor": "quality_cursor"
}
```

### Candidate Type Boosts

| Candidate Type | Boost Factor |
|----------------|--------------|
| `NEW_HIGH_QUALITY` | 1.15x |
| `STRATIFIED_PRIMARY` | 1.10x |
| `CURSOR_BASED` | 1.05x |
| `PRIMARY` | 1.00x |
| `CROSS_LANGUAGE` | 0.95x |
| `FALLBACK` | 0.90x |

---

## 7. Social & Comment Analysis Integration

### Social Signal Processing

```
User Social Graph
    │
    ▼
┌─────────────────────────────────────┐
│ Social Recommendations Service      │
│                                     │
│ Weights:                            │
│ • Following influence: 0.4         │
│ • Community trends:    0.3         │
│ • Sentiment signals:   0.3         │
└─────────────────────────────────────┘
    │
    ▼
Social-adjusted scores
```

### Comment Analysis Features (BERT)

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Positive sentiment | 1 | Avg positive score |
| Negative sentiment | 1 | Avg negative score |
| Neutral sentiment | 1 | Avg neutral score |
| Toxicity level | 1 | Detected toxicity [0-1] |
| Comment count | 1 | Normalized engagement |
| Spam score | 1 | Spam probability |
| Controversy | 1 | Variance in sentiment |
| Engagement signals | 6 | Like/reply patterns |
| **Total** | **13** | Fed to RL state |

### Interaction Weights

| Action | Weight | Signal |
|--------|--------|--------|
| Save | 1.0 | Strong positive |
| Share | 0.8 | Strong positive |
| Like | 0.6 | Positive |
| Comment (positive) | 0.6 | Positive |
| More info | 0.3 | Mild positive |
| View time (high) | 0.4 | Positive |
| Skip | -0.2 | Mild negative |
| Comment (negative) | -0.5 | Negative |
| Not interested | -0.9 | Strong negative |

---

## 8. Caching Strategy

### Multi-Layer Cache Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: In-Memory (Python dict)                        │
│ • Capacity: 1000 entries per service                   │
│ • TTL: Process lifetime                                 │
│ • Use: Hot data, session-specific                      │
├─────────────────────────────────────────────────────────┤
│ Layer 2: Redis/Valkey (Distributed)                    │
│ • TTL: 3600s (metadata), 7200s (sentiment)             │
│ • Use: Cross-service, persistence                      │
├─────────────────────────────────────────────────────────┤
│ Layer 3: PostgreSQL (Spring Boot)                      │
│ • TTL: Configurable per content type                   │
│ • Use: TMDB response caching, source of truth          │
└─────────────────────────────────────────────────────────┘
```

### Batch Optimization

```python
# Instead of N individual requests:
for post_id in post_ids:
    metadata = fetch_metadata(post_id)  # N API calls

# Single batch request:
metadata_batch = fetch_metadata_batch(post_ids)  # 1 API call
```

---

## 9. Performance Characteristics

| Metric | Target | Implementation |
|--------|--------|----------------|
| Recommendation latency | < 200ms | Two-Tower + cached metadata |
| RL processing | < 100ms | Async with fallback |
| Cache hit ratio | > 80% | Multi-layer caching |
| Batch efficiency | 20-40 posts/request | Batch metadata API |
| Model cold start | < 30s | Pre-warmed containers |

### Resource Allocation

| Service | Memory | CPU | Replicas |
|---------|--------|-----|----------|
| Core Recommendations | 2-4 GB | 0.5-1.0 | 2 |
| Social Recommendations | 256-512 MB | 0.2-0.4 | 2 |
| Comment Analysis | 2-4 GB | 0.5-1.0 | 1 |
| Redis | 1 GB | - | 1 |

---

## 10. Data Privacy & Compliance

### TMDB Attribution Requirements

Per TMDB Terms of Service:
- Display TMDB logo on content sourced from their API
- Include attribution: "This product uses the TMDB API but is not endorsed or certified by TMDB"
- Do not cache data beyond permitted TTL
- Do not redistribute raw API responses

### User Data Handling

| Data Type | Storage | Retention |
|-----------|---------|-----------|
| User preferences | PostgreSQL | Account lifetime |
| Interaction history | PostgreSQL | 90 days rolling |
| ML embeddings | Redis | 1 hour TTL |
| Session data | Memory | Session only |

---

## 11. Summary: TMDB → ML Component Mapping

| TMDB Field | ML Component | Boost/Feature | Impact |
|------------|--------------|---------------|--------|
| `genres` | MetadataEnhancer | 0-20% boost | Genre matching |
| `vote_average` | MetadataEnhancer | 0-15% boost | Quality signal |
| `popularity` | RLStateBuilder | Feature vector | Trending factor |
| `original_language` | MetadataEnhancer | 0-30% boost | Language pref |
| `cast` | MetadataEnhancer | 0-20% boost | Actor appeal |
| `crew` | MetadataEnhancer | 0-20% boost | Director appeal |
| `release_date` | MetadataEnhancer | 0-25% boost | Recency |
| `runtime` | RLStateBuilder | Feature vector | Content type |
| `overview` | Comment Analysis | Indirect | Context |
| `videos` | Candidate Selection | Type boost | Trailer handling |
| `production_countries` | MetadataEnhancer | 0-10% boost | Regional |

---

## Appendix: API Endpoint Reference

### Spring Boot Internal APIs (Called by ML Services)

```
GET  /api/internal/ml/users/{userId}/candidates
     → Returns candidate posts with embeddings

GET  /api/internal/ml/users/{userId}/vector
     → Returns user embedding vector

GET  /api/internal/users/{userId}/metadata
     → Returns user preferences and weights

GET  /api/internal/posts/{postId}/metadata
     → Returns single post TMDB-enriched metadata

POST /api/internal/posts/metadata/batch
     Body: {"postIds": [1, 2, 3, ...]}
     → Returns batch post metadata
```

### ML Service APIs (Called by Spring Boot)

```
POST /recommendations
     Body: {"user_id": 123, "limit": 20, "content_type": "POSTS"}
     → Returns ranked recommendation list

POST /social/recommendations
     Body: {"user_id": 123, "base_recommendations": [...]}
     → Returns socially-adjusted rankings

POST /analyze/sentiment
     Body: {"post_id": 456, "comments": [...]}
     → Returns sentiment analysis results
```

---

*Report generated: March 2026*
*System Version: 1.0.0*
