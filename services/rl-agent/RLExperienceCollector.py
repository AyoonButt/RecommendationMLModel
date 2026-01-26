import time
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import redis
from datetime import datetime, timedelta

logger = logging.getLogger("rl-experience-collector")

@dataclass
class RLExperience:
    """Single RL experience tuple (state, action, reward, next_state, done)."""
    experience_id: str
    user_id: int
    session_id: str
    timestamp: float
    
    # RL Core Components
    state: Dict[str, Any]  # User/content state representation
    action: Dict[str, Any]  # Action taken (ranking, boost params, etc.)
    reward: float  # Immediate reward from user feedback
    next_state: Optional[Dict[str, Any]]  # State after action
    done: bool  # Whether episode/session ended
    
    # Context Information
    sequence_position: int  # Position in session sequence
    time_since_last_action: float  # Seconds since previous action
    content_type: str  # "posts" or "trailers"
    
    # Metadata for analysis
    post_id: int
    interaction_type: str  # "like", "save", "not_interested", "more_info", etc.
    original_score: float  # ML model's original score
    boosted_score: float  # Score after metadata enhancement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary for storage/transmission."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLExperience':
        """Create experience from dictionary."""
        return cls(**data)


class RLExperienceCollector:
    """
    Collects real-time user interactions and converts them into RL experiences.
    Handles session tracking, temporal context, and reward engineering.
    """
    
    def __init__(self, redis_client=None, experience_buffer_size: int = 10000,
                 session_timeout_minutes: int = 30, reward_config: Dict[str, float] = None):
        """
        Initialize the RL experience collector.
        
        Args:
            redis_client: Redis client for experience storage
            experience_buffer_size: Maximum experiences to keep in memory
            session_timeout_minutes: Session timeout in minutes
            reward_config: Mapping of interaction types to reward values
        """
        self.redis_client = redis_client
        self.experience_buffer_size = experience_buffer_size
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # Default reward configuration
        self.reward_config = reward_config or {
            "like": 0.6,
            "save": 1.0,
            "not_interested": -0.9,
            "more_info": 0.3,
            "comment": 0.4,
            "share": 0.8,
            "skip": -0.2,
            "long_view": 0.5,  # Watched for extended time
            "return_visit": 0.7  # Came back to same content
        }
        
        # In-memory buffers
        self.experience_buffer = deque(maxlen=experience_buffer_size)
        self.active_sessions: Dict[int, Dict] = {}  # user_id -> session info
        self.user_state_cache: Dict[int, Dict] = {}  # user_id -> current state
        
        # Sequence tracking
        self.user_sequences: Dict[int, List[Dict]] = defaultdict(list)
        self.last_action_time: Dict[int, float] = {}
        
        logger.info(f"RL Experience Collector initialized with buffer size {experience_buffer_size}")
    
    def collect_interaction_experience(self, user_id: int, post_id: int, 
                                     interaction_type: str, 
                                     content_type: str = "posts",
                                     additional_context: Dict[str, Any] = None) -> RLExperience:
        """
        Collect a single interaction and convert it to RL experience.
        
        Args:
            user_id: User ID
            post_id: Post ID
            interaction_type: Type of interaction (like, save, not_interested, etc.)
            content_type: Content type (posts, trailers)
            additional_context: Additional context information
            
        Returns:
            RLExperience object
        """
        current_time = time.time()
        session_id = self._get_or_create_session(user_id, current_time)
        
        # Get current state
        current_state = self._get_current_state(user_id, post_id, content_type, additional_context)
        
        # Determine action taken (this could be expanded based on your system)
        action = self._extract_action_from_context(additional_context)
        
        # Calculate reward
        reward = self._calculate_reward(interaction_type, additional_context)
        
        # Get sequence position and timing
        sequence_position = len(self.user_sequences[user_id])
        time_since_last = current_time - self.last_action_time.get(user_id, current_time)
        
        # Check if session/episode is done
        done = self._is_episode_done(interaction_type, time_since_last)
        
        # Create experience
        experience = RLExperience(
            experience_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            timestamp=current_time,
            state=current_state,
            action=action,
            reward=reward,
            next_state=None,  # Will be filled when next interaction happens
            done=done,
            sequence_position=sequence_position,
            time_since_last_action=time_since_last,
            content_type=content_type,
            post_id=post_id,
            interaction_type=interaction_type,
            original_score=additional_context.get('original_score', 0.5),
            boosted_score=additional_context.get('boosted_score', 0.5)
        )
        
        # Update previous experience with next_state
        self._update_previous_experience_next_state(user_id, current_state)
        
        # Store experience
        self._store_experience(experience)
        
        # Update tracking
        self.user_sequences[user_id].append({
            'experience_id': experience.experience_id,
            'timestamp': current_time,
            'interaction_type': interaction_type
        })
        self.last_action_time[user_id] = current_time
        self.user_state_cache[user_id] = current_state
        
        logger.debug(f"Collected RL experience: User {user_id}, Post {post_id}, "
                    f"Action {interaction_type}, Reward {reward:.3f}")
        
        return experience
    
    def _get_or_create_session(self, user_id: int, current_time: float) -> str:
        """Get existing session or create new one for user."""
        # Check if user has active session
        if user_id in self.active_sessions:
            session_info = self.active_sessions[user_id]
            last_activity = datetime.fromtimestamp(session_info['last_activity'])
            
            # Check if session is still active
            if datetime.now() - last_activity < self.session_timeout:
                # Update last activity
                session_info['last_activity'] = current_time
                return session_info['session_id']
        
        # Create new session
        session_id = f"session_{user_id}_{int(current_time)}"
        self.active_sessions[user_id] = {
            'session_id': session_id,
            'start_time': current_time,
            'last_activity': current_time,
            'interaction_count': 0
        }
        
        logger.debug(f"Created new session {session_id} for user {user_id}")
        return session_id
    
    def _get_current_state(self, user_id: int, post_id: int, content_type: str, 
                          additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Construct current state representation for RL.
        This would integrate with your existing embedding system.
        """
        additional_context = additional_context or {}
        
        # Base state components
        state = {
            'user_id': user_id,
            'post_id': post_id,
            'content_type': content_type,
            'timestamp': time.time()
        }
        
        # Add user embeddings (these would come from your existing system)
        state['user_embedding'] = additional_context.get('user_embedding', [0.0] * 32)
        state['post_embedding'] = additional_context.get('post_embedding', [0.0] * 32)
        
        # Add temporal context
        state['session_length'] = len(self.user_sequences.get(user_id, []))
        state['time_since_last_action'] = time.time() - self.last_action_time.get(user_id, time.time())
        
        # Add recent interaction history (last 5 interactions)
        recent_interactions = self.user_sequences.get(user_id, [])[-5:]
        state['recent_interactions'] = recent_interactions
        
        # Add user preferences (from metadata)
        state['user_preferences'] = additional_context.get('user_preferences', {})
        
        # Add post metadata
        state['post_metadata'] = additional_context.get('post_metadata', {})
        
        # Add social signals
        state['social_signals'] = additional_context.get('social_signals', {})
        
        return state
    
    def _extract_action_from_context(self, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract the action taken from context information."""
        additional_context = additional_context or {}
        
        return {
            'recommendation_shown': True,
            'boost_factors': additional_context.get('boost_factors', {}),
            'ranking_position': additional_context.get('ranking_position', -1),
            'candidate_type': additional_context.get('candidate_type', 'unknown'),
            'exploration_factor': additional_context.get('exploration_factor', 0.0)
        }
    
    def _calculate_reward(self, interaction_type: str, additional_context: Dict[str, Any] = None) -> float:
        """Calculate reward based on interaction type and context."""
        additional_context = additional_context or {}
        
        # Base reward from interaction type
        base_reward = self.reward_config.get(interaction_type, 0.0)
        
        # Apply reward shaping based on context
        shaped_reward = base_reward
        
        # Time-based shaping (longer engagement = higher reward)
        engagement_time = additional_context.get('engagement_time_seconds', 0)
        if engagement_time > 30:  # More than 30 seconds
            shaped_reward += 0.1
        elif engagement_time > 120:  # More than 2 minutes
            shaped_reward += 0.2
        
        # Position-based shaping (interacting with lower-ranked items = higher exploration reward)
        position = additional_context.get('ranking_position', -1)
        if position > 5:  # Beyond top 5
            shaped_reward += 0.1
        
        # Diversity bonus (interacting with different types of content)
        if additional_context.get('content_diversity_bonus', False):
            shaped_reward += 0.05
        
        return np.clip(shaped_reward, -1.0, 1.0)  # Keep rewards in reasonable range
    
    def _is_episode_done(self, interaction_type: str, time_since_last: float) -> bool:
        """Determine if this interaction ends an episode."""
        # Episode ends on strong negative feedback or long inactivity
        if interaction_type in ['not_interested', 'block_content']:
            return True
        
        # Episode ends if user has been inactive for too long
        if time_since_last > 1800:  # 30 minutes
            return True
        
        return False
    
    def _update_previous_experience_next_state(self, user_id: int, current_state: Dict[str, Any]):
        """Update the previous experience with the current state as next_state."""
        user_sequence = self.user_sequences.get(user_id, [])
        if not user_sequence:
            return
        
        # Get the last experience ID
        last_experience_id = user_sequence[-1]['experience_id']
        
        # Update in memory buffer
        for i, exp in enumerate(self.experience_buffer):
            if exp.experience_id == last_experience_id:
                self.experience_buffer[i].next_state = current_state
                break
        
        # Update in Redis if available
        if self.redis_client:
            try:
                key = f"rl_experience:{last_experience_id}"
                experience_data = self.redis_client.get(key)
                if experience_data:
                    exp_dict = json.loads(experience_data)
                    exp_dict['next_state'] = current_state
                    self.redis_client.setex(key, 3600, json.dumps(exp_dict))  # 1 hour TTL
            except Exception as e:
                logger.warning(f"Error updating experience in Redis: {e}")
    
    def _store_experience(self, experience: RLExperience):
        """Store experience in memory buffer and Redis."""
        # Add to memory buffer
        self.experience_buffer.append(experience)
        
        # Store in Redis for persistence and sharing across services
        if self.redis_client:
            try:
                key = f"rl_experience:{experience.experience_id}"
                self.redis_client.setex(key, 3600, json.dumps(experience.to_dict()))
                
                # Also add to user's experience list for easy retrieval
                user_key = f"user_experiences:{experience.user_id}"
                self.redis_client.lpush(user_key, experience.experience_id)
                self.redis_client.ltrim(user_key, 0, 999)  # Keep last 1000 experiences
                self.redis_client.expire(user_key, 86400)  # 24 hour TTL
                
            except Exception as e:
                logger.warning(f"Error storing experience in Redis: {e}")
    
    def get_user_experiences(self, user_id: int, limit: int = 100) -> List[RLExperience]:
        """Get recent experiences for a user."""
        experiences = []
        
        # Try Redis first
        if self.redis_client:
            try:
                user_key = f"user_experiences:{user_id}"
                experience_ids = self.redis_client.lrange(user_key, 0, limit - 1)
                
                for exp_id in experience_ids:
                    exp_key = f"rl_experience:{exp_id.decode()}"
                    exp_data = self.redis_client.get(exp_key)
                    if exp_data:
                        experiences.append(RLExperience.from_dict(json.loads(exp_data)))
                
                return experiences
            except Exception as e:
                logger.warning(f"Error retrieving experiences from Redis: {e}")
        
        # Fallback to memory buffer
        user_experiences = [exp for exp in self.experience_buffer if exp.user_id == user_id]
        return sorted(user_experiences, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_recent_experiences(self, limit: int = 1000) -> List[RLExperience]:
        """Get recent experiences across all users for training."""
        # Convert deque to list and sort by timestamp
        experiences = list(self.experience_buffer)
        experiences.sort(key=lambda x: x.timestamp, reverse=True)
        return experiences[:limit]
    
    def get_experience_batch(self, batch_size: int = 32) -> List[RLExperience]:
        """Get a random batch of experiences for training."""
        if len(self.experience_buffer) < batch_size:
            return list(self.experience_buffer)
        
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        return [self.experience_buffer[i] for i in indices]
    
    def clear_old_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_users = []
        
        for user_id, session_info in self.active_sessions.items():
            last_activity = datetime.fromtimestamp(session_info['last_activity'])
            if current_time - last_activity > self.session_timeout:
                expired_users.append(user_id)
        
        for user_id in expired_users:
            del self.active_sessions[user_id]
            # Also clean up sequence tracking
            if user_id in self.user_sequences:
                del self.user_sequences[user_id]
            if user_id in self.last_action_time:
                del self.last_action_time[user_id]
            if user_id in self.user_state_cache:
                del self.user_state_cache[user_id]
        
        if expired_users:
            logger.info(f"Cleaned up {len(expired_users)} expired sessions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            'total_experiences': len(self.experience_buffer),
            'active_sessions': len(self.active_sessions),
            'users_with_sequences': len(self.user_sequences),
            'reward_config': self.reward_config,
            'buffer_utilization': len(self.experience_buffer) / self.experience_buffer_size
        }