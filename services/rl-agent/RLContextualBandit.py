import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABC, abstractmethod
import random
import json

logger = logging.getLogger("rl-contextual-bandit")

@dataclass
class Action:
    """Represents an action taken by the RL agent."""
    action_id: str
    action_type: str  # "boost_adjustment", "ranking_modification", "exploration"
    parameters: Dict[str, Any]
    expected_reward: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'action_type': self.action_type,
            'parameters': self.parameters,
            'expected_reward': self.expected_reward,
            'confidence': self.confidence
        }

@dataclass
class ActionResult:
    """Result of taking an action."""
    action: Action
    actual_reward: float
    user_id: int
    post_id: int
    timestamp: float
    metadata: Dict[str, Any]

class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies."""
    
    @abstractmethod
    def select_action(self, actions: List[Action], context: Dict[str, Any]) -> Action:
        """Select an action based on the exploration strategy."""
        pass

class EpsilonGreedyExploration(ExplorationStrategy):
    """Epsilon-greedy exploration strategy."""
    
    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.step_count = 0
    
    def select_action(self, actions: List[Action], context: Dict[str, Any]) -> Action:
        """Select action using epsilon-greedy strategy."""
        self.step_count += 1
        
        # Decay epsilon over time
        current_epsilon = max(
            self.min_epsilon,
            self.initial_epsilon * (self.decay_rate ** self.step_count)
        )
        
        if random.random() < current_epsilon:
            # Explore: random action
            return random.choice(actions)
        else:
            # Exploit: best action
            return max(actions, key=lambda a: a.expected_reward)

class ThompsonSamplingExploration(ExplorationStrategy):
    """Thompson sampling exploration strategy."""
    
    def __init__(self):
        self.action_stats = defaultdict(lambda: {'alpha': 1.0, 'beta': 1.0})
    
    def select_action(self, actions: List[Action], context: Dict[str, Any]) -> Action:
        """Select action using Thompson sampling."""
        sampled_rewards = []
        
        for action in actions:
            stats = self.action_stats[action.action_id]
            # Sample from Beta distribution
            sampled_reward = np.random.beta(stats['alpha'], stats['beta'])
            sampled_rewards.append(sampled_reward)
        
        # Select action with highest sampled reward
        best_idx = np.argmax(sampled_rewards)
        return actions[best_idx]
    
    def update_action_stats(self, action_id: str, reward: float):
        """Update action statistics for Thompson sampling."""
        stats = self.action_stats[action_id]
        if reward > 0:
            stats['alpha'] += reward
        else:
            stats['beta'] += abs(reward)

class PolicyNetwork(nn.Module):
    """Neural network for action selection policy."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(PolicyNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value."""
        features = state
        
        # Pass through main network layers (except last)
        for layer in self.network[:-1]:
            features = layer(features)
        
        # Get action logits
        action_logits = self.network[-1](features)
        
        # Get state value
        state_value = self.value_head(features)
        
        return action_logits, state_value

class ValueNetwork(nn.Module):
    """Neural network for state-action value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(ValueNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # State encoder
        state_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            state_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.state_encoder = nn.Sequential(*state_layers)
        
        # Combined network for state-action value
        self.value_network = nn.Sequential(
            nn.Linear(prev_dim + action_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q(s,a) value."""
        state_features = self.state_encoder(state)
        combined = torch.cat([state_features, action], dim=-1)
        return self.value_network(combined)

class RLContextualBandit:
    """
    Contextual bandit implementation for RL-based recommendation optimization.
    Integrates with existing Two-Tower system to provide real-time action selection.
    """
    
    def __init__(self, state_dim: int = 147, config: Dict[str, Any] = None):
        """
        Initialize the contextual bandit.
        
        Args:
            state_dim: Dimension of state vector (from RLStateBuilder)
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.config = config or self._get_default_config()
        
        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            action_dim=self.config['action_space']['total_actions'],
            hidden_dims=self.config['model']['policy_hidden_dims']
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim=state_dim,
            action_dim=self.config['action_space']['action_encoding_dim'],
            hidden_dims=self.config['model']['value_hidden_dims']
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config['training']['policy_lr']
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=self.config['training']['value_lr']
        )
        
        # Exploration strategy
        exploration_type = self.config['exploration']['strategy']
        if exploration_type == 'epsilon_greedy':
            self.exploration_strategy = EpsilonGreedyExploration(
                epsilon=self.config['exploration']['epsilon'],
                decay_rate=self.config['exploration']['decay_rate']
            )
        elif exploration_type == 'thompson_sampling':
            self.exploration_strategy = ThompsonSamplingExploration()
        else:
            self.exploration_strategy = EpsilonGreedyExploration()
        
        # Action tracking
        self.action_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.action_performance = defaultdict(lambda: {'rewards': [], 'counts': 0})
        
        # Training buffer
        self.experience_buffer = deque(maxlen=self.config['training']['buffer_size'])
        
        logger.info(f"RL Contextual Bandit initialized with state_dim={state_dim}")
        logger.info(f"Using device: {self.device}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'action_space': {
                'total_actions': 20,  # Number of discrete action types
                'action_encoding_dim': 10,  # Dimension for action encoding
                'boost_actions': 8,   # Actions for metadata boost adjustment
                'ranking_actions': 6,  # Actions for ranking modification
                'exploration_actions': 6  # Actions for exploration control
            },
            'model': {
                'policy_hidden_dims': [256, 128, 64],
                'value_hidden_dims': [256, 128, 64]
            },
            'training': {
                'policy_lr': 0.001,
                'value_lr': 0.002,
                'buffer_size': 10000,
                'batch_size': 64,
                'update_frequency': 10,
                'gamma': 0.95
            },
            'exploration': {
                'strategy': 'epsilon_greedy',
                'epsilon': 0.1,
                'decay_rate': 0.995
            },
            'action_parameters': {
                'boost_adjustment_range': [-0.3, 0.3],  # Range for boost factor adjustments
                'ranking_adjustment_range': [-5, 5],    # Range for ranking position adjustments
                'exploration_probability_range': [0.0, 0.5]  # Range for exploration probability
            }
        }
    
    def select_action(self, state: np.ndarray, user_id: int, context: Dict[str, Any]) -> Action:
        """
        Select action based on current state and context.
        
        Args:
            state: Current state vector
            user_id: User ID
            context: Additional context information
            
        Returns:
            Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities and state value from policy network
        with torch.no_grad():
            action_logits, state_value = self.policy_net(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
        
        # Generate candidate actions
        candidate_actions = self._generate_candidate_actions(
            state, user_id, context, action_probs.cpu().numpy()[0]
        )
        
        # Select action using exploration strategy
        selected_action = self.exploration_strategy.select_action(candidate_actions, context)
        
        # Log action selection
        logger.debug(f"Selected action {selected_action.action_type} for user {user_id}")
        
        return selected_action
    
    def _generate_candidate_actions(self, state: np.ndarray, user_id: int, 
                                  context: Dict[str, Any], action_probs: np.ndarray) -> List[Action]:
        """Generate candidate actions based on current state."""
        actions = []
        action_config = self.config['action_parameters']
        
        # Generate boost adjustment actions
        for i in range(self.config['action_space']['boost_actions']):
            boost_adjustment = np.random.uniform(*action_config['boost_adjustment_range'])
            
            action = Action(
                action_id=f"boost_adj_{i}",
                action_type="boost_adjustment",
                parameters={
                    'recency_boost_adjustment': boost_adjustment * 0.5,
                    'cast_crew_boost_adjustment': boost_adjustment * 0.5,
                    'boost_type': 'metadata'
                },
                expected_reward=float(action_probs[i] if i < len(action_probs) else 0.1),
                confidence=0.8
            )
            actions.append(action)
        
        # Generate ranking modification actions
        start_idx = self.config['action_space']['boost_actions']
        for i in range(self.config['action_space']['ranking_actions']):
            ranking_adjustment = np.random.randint(*action_config['ranking_adjustment_range'])
            
            action = Action(
                action_id=f"rank_mod_{i}",
                action_type="ranking_modification",
                parameters={
                    'position_adjustment': ranking_adjustment,
                    'rerank_scope': min(20, max(5, abs(ranking_adjustment) + 5)),
                    'modification_type': 'position_boost'
                },
                expected_reward=float(action_probs[start_idx + i] if start_idx + i < len(action_probs) else 0.1),
                confidence=0.7
            )
            actions.append(action)
        
        # Generate exploration actions
        start_idx += self.config['action_space']['ranking_actions']
        for i in range(self.config['action_space']['exploration_actions']):
            exploration_prob = np.random.uniform(*action_config['exploration_probability_range'])
            
            action = Action(
                action_id=f"explore_{i}",
                action_type="exploration",
                parameters={
                    'exploration_probability': exploration_prob,
                    'diversity_weight': np.random.uniform(0.1, 0.4),
                    'novelty_weight': np.random.uniform(0.1, 0.3),
                    'exploration_type': 'diversity_boost'
                },
                expected_reward=float(action_probs[start_idx + i] if start_idx + i < len(action_probs) else 0.1),
                confidence=0.6
            )
            actions.append(action)
        
        return actions
    
    def update_action_result(self, action_result: ActionResult):
        """
        Update the bandit with the result of an action.
        
        Args:
            action_result: Result of the action taken
        """
        # Add to experience buffer
        self.experience_buffer.append(action_result)
        
        # Update action performance tracking
        action_id = action_result.action.action_id
        self.action_performance[action_id]['rewards'].append(action_result.actual_reward)
        self.action_performance[action_id]['counts'] += 1
        
        # Update user action history
        self.action_history[action_result.user_id].append({
            'action': action_result.action,
            'reward': action_result.actual_reward,
            'timestamp': action_result.timestamp
        })
        
        # Update exploration strategy if applicable
        if isinstance(self.exploration_strategy, ThompsonSamplingExploration):
            self.exploration_strategy.update_action_stats(action_id, action_result.actual_reward)
        
        # Trigger training if buffer is sufficiently full
        if len(self.experience_buffer) >= self.config['training']['batch_size']:
            if len(self.experience_buffer) % self.config['training']['update_frequency'] == 0:
                self._train_networks()
    
    def _train_networks(self):
        """Train policy and value networks on experience buffer."""
        if len(self.experience_buffer) < self.config['training']['batch_size']:
            return
        
        batch_size = self.config['training']['batch_size']
        gamma = self.config['training']['gamma']
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in batch:
            # Convert action to encoding
            action_encoding = self._encode_action(exp.action)
            
            states.append(exp.metadata.get('state', np.zeros(self.state_dim)))
            actions.append(action_encoding)
            rewards.append(exp.actual_reward)
            next_states.append(exp.metadata.get('next_state', np.zeros(self.state_dim)))
            dones.append(exp.metadata.get('done', False))
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Train value network
        self._train_value_network(states_tensor, actions_tensor, rewards_tensor, 
                                next_states_tensor, dones_tensor, gamma)
        
        # Train policy network
        self._train_policy_network(states_tensor, actions_tensor, rewards_tensor)
        
        logger.debug(f"Trained networks on batch of {batch_size} experiences")
    
    def _train_value_network(self, states: torch.Tensor, actions: torch.Tensor, 
                           rewards: torch.Tensor, next_states: torch.Tensor, 
                           dones: torch.Tensor, gamma: float):
        """Train the value network."""
        # Compute current Q values
        current_q_values = self.value_net(states, actions).squeeze()
        
        # Compute target Q values
        with torch.no_grad():
            # For simplicity, use reward as target (can be extended with next state values)
            target_q_values = rewards
        
        # Compute loss
        value_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
    
    def _train_policy_network(self, states: torch.Tensor, actions: torch.Tensor, 
                            rewards: torch.Tensor):
        """Train the policy network using policy gradient."""
        # Get action probabilities
        action_logits, state_values = self.policy_net(states)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Compute advantages (simplified - using rewards directly)
        advantages = rewards - state_values.squeeze()
        
        # Compute policy loss (REINFORCE with baseline)
        # This is a simplified version - would need proper action selection in practice
        log_probs = F.log_softmax(action_logits, dim=-1)
        policy_loss = -(log_probs.mean(dim=-1) * advantages.detach()).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(state_values.squeeze(), rewards)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
    
    def _encode_action(self, action: Action) -> np.ndarray:
        """Encode action into vector representation."""
        encoding_dim = self.config['action_space']['action_encoding_dim']
        encoding = np.zeros(encoding_dim)
        
        # Simple encoding based on action type and parameters
        if action.action_type == "boost_adjustment":
            encoding[0] = 1.0
            encoding[1] = action.parameters.get('recency_boost_adjustment', 0.0)
            encoding[2] = action.parameters.get('cast_crew_boost_adjustment', 0.0)
        elif action.action_type == "ranking_modification":
            encoding[3] = 1.0
            encoding[4] = action.parameters.get('position_adjustment', 0.0) / 10.0  # Normalize
            encoding[5] = action.parameters.get('rerank_scope', 10) / 20.0  # Normalize
        elif action.action_type == "exploration":
            encoding[6] = 1.0
            encoding[7] = action.parameters.get('exploration_probability', 0.0)
            encoding[8] = action.parameters.get('diversity_weight', 0.0)
            encoding[9] = action.parameters.get('novelty_weight', 0.0)
        
        return encoding
    
    def get_action_recommendations(self, user_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action recommendations for a user based on their history.
        
        Args:
            user_id: User ID
            context: Current context
            
        Returns:
            Action recommendations and insights
        """
        user_history = list(self.action_history[user_id])
        
        if not user_history:
            return {
                'recommendations': [],
                'insights': 'No action history available for this user'
            }
        
        # Analyze user's action performance
        action_performance = defaultdict(list)
        for entry in user_history:
            action_type = entry['action'].action_type
            action_performance[action_type].append(entry['reward'])
        
        # Generate recommendations
        recommendations = []
        for action_type, rewards in action_performance.items():
            avg_reward = np.mean(rewards)
            recommendations.append({
                'action_type': action_type,
                'average_reward': avg_reward,
                'sample_count': len(rewards),
                'confidence': min(1.0, len(rewards) / 20.0)  # Confidence based on sample size
            })
        
        # Sort by average reward
        recommendations.sort(key=lambda x: x['average_reward'], reverse=True)
        
        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'total_actions': len(user_history),
            'insights': f"Best performing action type: {recommendations[0]['action_type']}" if recommendations else "No recommendations available"
        }
    
    def get_bandit_stats(self) -> Dict[str, Any]:
        """Get bandit performance statistics."""
        return {
            'experience_buffer_size': len(self.experience_buffer),
            'users_tracked': len(self.action_history),
            'total_actions': sum(perf['counts'] for perf in self.action_performance.values()),
            'action_performance_summary': {
                action_id: {
                    'count': perf['counts'],
                    'avg_reward': np.mean(perf['rewards']) if perf['rewards'] else 0.0,
                    'std_reward': np.std(perf['rewards']) if perf['rewards'] else 0.0
                }
                for action_id, perf in self.action_performance.items()
            },
            'model_config': self.config,
            'device': str(self.device)
        }
    
    def save_model(self, filepath: str):
        """Save trained models."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config
        }, filepath)
        logger.info(f"Saved RL models to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        logger.info(f"Loaded RL models from {filepath}")