import time
import logging
import asyncio
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import torch
import redis
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from abc import ABC, abstractmethod

from RLExperienceCollector import RLExperienceCollector, RLExperience
from RLRewardEngineer import RLRewardEngineer
from RLContextualBandit import RLContextualBandit, ActionResult
from RLIntegration import RLIntegrationManager

logger = logging.getLogger("rl-online-learning")

@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments."""
    experiment_id: str
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    treatment_ratio: float  # Fraction of users receiving RL treatment
    control_group_size: int
    treatment_group_size: int
    success_metrics: List[str]
    is_active: bool = True

@dataclass
class ExperimentResult:
    """Results from A/B test experiment."""
    experiment_id: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]
    confidence_level: float
    sample_sizes: Dict[str, int]
    duration_days: float
    recommendation: str  # "deploy", "continue", "stop"

@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance metrics."""
    timestamp: datetime
    reward_mean: float
    reward_std: float
    interaction_rate: float
    user_satisfaction: float
    diversity_score: float
    coverage_score: float
    model_version: str
    sample_size: int

class SafetyMonitor:
    """Monitor system performance and implement safety checks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = deque(maxlen=config.get('history_size', 1000))
        self.alerts_sent = deque(maxlen=100)
        self.safety_enabled = True
        
    def check_safety_violations(self, metrics: ModelPerformanceMetrics) -> List[str]:
        """Check for safety violations in current metrics."""
        violations = []
        
        # Check reward degradation
        if len(self.performance_history) >= 10:
            recent_rewards = [m.reward_mean for m in list(self.performance_history)[-10:]]
            baseline_rewards = [m.reward_mean for m in list(self.performance_history)[-50:-10]] if len(self.performance_history) >= 50 else recent_rewards
            
            if baseline_rewards:
                recent_avg = np.mean(recent_rewards)
                baseline_avg = np.mean(baseline_rewards)
                degradation = (baseline_avg - recent_avg) / abs(baseline_avg) if baseline_avg != 0 else 0
                
                if degradation > self.config['reward_degradation_threshold']:
                    violations.append(f"Reward degradation: {degradation:.2%}")
        
        # Check interaction rate drop
        if metrics.interaction_rate < self.config['min_interaction_rate']:
            violations.append(f"Low interaction rate: {metrics.interaction_rate:.3f}")
        
        # Check user satisfaction
        if metrics.user_satisfaction < self.config['min_user_satisfaction']:
            violations.append(f"Low user satisfaction: {metrics.user_satisfaction:.3f}")
        
        # Check model stability
        if metrics.reward_std > self.config['max_reward_variance']:
            violations.append(f"High reward variance: {metrics.reward_std:.3f}")
        
        self.performance_history.append(metrics)
        return violations
    
    def should_trigger_rollback(self, violations: List[str]) -> bool:
        """Determine if violations warrant immediate rollback."""
        critical_violations = [v for v in violations if any(keyword in v.lower() for keyword in ['degradation', 'low interaction'])]
        return len(critical_violations) >= 2

class StreamingModelUpdater:
    """Handles streaming updates to RL models."""
    
    def __init__(self, contextual_bandit: RLContextualBandit, config: Dict[str, Any]):
        self.contextual_bandit = contextual_bandit
        self.config = config
        self.update_queue = queue.Queue(maxsize=config.get('queue_size', 1000))
        self.is_running = False
        self.update_thread = None
        self.batch_size = config.get('batch_size', 32)
        self.update_frequency = config.get('update_frequency_seconds', 60)
        
    def start_streaming_updates(self):
        """Start streaming model updates."""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Started streaming model updates")
    
    def stop_streaming_updates(self):
        """Stop streaming model updates."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Stopped streaming model updates")
    
    def queue_update(self, action_result: ActionResult):
        """Queue an action result for model update."""
        try:
            self.update_queue.put_nowait(action_result)
        except queue.Full:
            logger.warning("Update queue is full, dropping action result")
    
    def _update_loop(self):
        """Main update loop for streaming updates."""
        update_batch = []
        last_update = time.time()
        
        while self.is_running:
            try:
                # Collect updates with timeout
                timeout = max(0.1, self.update_frequency - (time.time() - last_update))
                
                try:
                    action_result = self.update_queue.get(timeout=timeout)
                    update_batch.append(action_result)
                except queue.Empty:
                    pass
                
                # Process batch if ready
                should_update = (
                    len(update_batch) >= self.batch_size or
                    (update_batch and time.time() - last_update >= self.update_frequency)
                )
                
                if should_update:
                    self._process_update_batch(update_batch)
                    update_batch.clear()
                    last_update = time.time()
                    
            except Exception as e:
                logger.error(f"Error in streaming update loop: {e}")
                time.sleep(1)
    
    def _process_update_batch(self, batch: List[ActionResult]):
        """Process a batch of action results."""
        try:
            logger.debug(f"Processing update batch of size {len(batch)}")
            
            # Update contextual bandit with all results in batch
            for action_result in batch:
                self.contextual_bandit.update_action_result(action_result)
            
            # Log update
            avg_reward = np.mean([ar.actual_reward for ar in batch])
            logger.info(f"Updated model with {len(batch)} experiences, avg reward: {avg_reward:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing update batch: {e}")

class ABTestManager:
    """Manages A/B testing experiments for RL deployment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_experiments = {}
        self.experiment_results = {}
        self.user_assignments = {}  # user_id -> experiment_id
        
    def create_experiment(self, experiment_config: ExperimentConfig) -> str:
        """Create new A/B test experiment."""
        experiment_id = experiment_config.experiment_id
        self.active_experiments[experiment_id] = experiment_config
        
        logger.info(f"Created A/B experiment: {experiment_id}")
        logger.info(f"Treatment ratio: {experiment_config.treatment_ratio:.2%}")
        
        return experiment_id
    
    def assign_user_to_experiment(self, user_id: int, experiment_id: str) -> str:
        """Assign user to experiment group."""
        if experiment_id not in self.active_experiments:
            return "control"
        
        experiment = self.active_experiments[experiment_id]
        
        # Consistent assignment based on user ID
        assignment_hash = hash(f"{user_id}_{experiment_id}") % 100
        
        if assignment_hash < experiment.treatment_ratio * 100:
            group = "treatment"
        else:
            group = "control"
        
        self.user_assignments[user_id] = {
            'experiment_id': experiment_id,
            'group': group,
            'assigned_at': datetime.now()
        }
        
        return group
    
    def should_use_rl(self, user_id: int, experiment_id: str = None) -> bool:
        """Determine if user should receive RL treatment."""
        if experiment_id and experiment_id in self.active_experiments:
            group = self.assign_user_to_experiment(user_id, experiment_id)
            return group == "treatment"
        
        # Default behavior if no active experiment
        return user_id % 100 < 20  # 20% default treatment rate
    
    def record_experiment_data(self, user_id: int, metrics: Dict[str, Any]):
        """Record experiment data for analysis."""
        assignment = self.user_assignments.get(user_id)
        if not assignment:
            return
        
        experiment_id = assignment['experiment_id']
        group = assignment['group']
        
        if experiment_id not in self.experiment_results:
            self.experiment_results[experiment_id] = {
                'control': defaultdict(list),
                'treatment': defaultdict(list)
            }
        
        # Record metrics for user's group
        for metric_name, value in metrics.items():
            self.experiment_results[experiment_id][group][metric_name].append(value)
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Analyze A/B test results."""
        if experiment_id not in self.experiment_results:
            raise ValueError(f"No data found for experiment {experiment_id}")
        
        data = self.experiment_results[experiment_id]
        control_data = data['control']
        treatment_data = data['treatment']
        
        # Calculate metrics for each group
        control_metrics = {}
        treatment_metrics = {}
        statistical_significance = {}
        
        for metric_name in control_data.keys():
            if metric_name in treatment_data:
                control_values = control_data[metric_name]
                treatment_values = treatment_data[metric_name]
                
                if control_values and treatment_values:
                    control_metrics[metric_name] = np.mean(control_values)
                    treatment_metrics[metric_name] = np.mean(treatment_values)
                    
                    # Simple statistical test (in production, use proper statistical tests)
                    p_value = self._calculate_p_value(control_values, treatment_values)
                    statistical_significance[metric_name] = p_value
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            control_metrics, treatment_metrics, statistical_significance
        )
        
        return ExperimentResult(
            experiment_id=experiment_id,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            statistical_significance=statistical_significance,
            confidence_level=0.95,
            sample_sizes={
                'control': sum(len(values) for values in control_data.values()),
                'treatment': sum(len(values) for values in treatment_data.values())
            },
            duration_days=(datetime.now() - self.active_experiments[experiment_id].start_time).days,
            recommendation=recommendation
        )
    
    def _calculate_p_value(self, control: List[float], treatment: List[float]) -> float:
        """Calculate p-value for difference in means (simplified)."""
        # This is a simplified implementation
        # In production, use scipy.stats or proper statistical libraries
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        control_std = np.std(control)
        treatment_std = np.std(treatment)
        
        # Simple t-test approximation
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        if pooled_std == 0:
            return 0.5
        
        t_stat = abs(treatment_mean - control_mean) / pooled_std
        # Rough approximation: p_value = 1 / (1 + t_stat)
        return 1 / (1 + t_stat)
    
    def _generate_recommendation(self, control: Dict, treatment: Dict, 
                               significance: Dict) -> str:
        """Generate deployment recommendation based on results."""
        # Simple heuristic-based recommendation
        significant_improvements = 0
        significant_degradations = 0
        
        for metric in control.keys():
            if metric in treatment and metric in significance:
                p_value = significance[metric]
                improvement = (treatment[metric] - control[metric]) / abs(control[metric]) if control[metric] != 0 else 0
                
                if p_value < 0.05:  # Statistically significant
                    if improvement > 0.05:  # 5% improvement
                        significant_improvements += 1
                    elif improvement < -0.05:  # 5% degradation
                        significant_degradations += 1
        
        if significant_degradations > 0:
            return "stop"
        elif significant_improvements >= 2:
            return "deploy"
        else:
            return "continue"

class RLOnlineLearningManager:
    """
    Comprehensive online learning manager for RL-based recommendations.
    Handles streaming updates, A/B testing, safety monitoring, and production deployment.
    """
    
    def __init__(self, rl_integration_manager: RLIntegrationManager, config: Dict[str, Any] = None):
        """
        Initialize the online learning manager.
        
        Args:
            rl_integration_manager: RL integration manager instance
            config: Configuration dictionary
        """
        self.rl_manager = rl_integration_manager
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.safety_monitor = SafetyMonitor(self.config['safety'])
        self.streaming_updater = StreamingModelUpdater(
            self.rl_manager.contextual_bandit,
            self.config['streaming']
        )
        self.ab_test_manager = ABTestManager(self.config['ab_testing'])
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=1000)
        self.system_stats = defaultdict(int)
        
        # Control flags
        self.is_online_learning_enabled = True
        self.current_experiment_id = None
        
        # Model versioning
        self.model_version = "1.0.0"
        self.model_checkpoints = deque(maxlen=10)
        
        logger.info("RL Online Learning Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'safety': {
                'reward_degradation_threshold': 0.15,  # 15% degradation triggers alert
                'min_interaction_rate': 0.05,  # Minimum 5% interaction rate
                'min_user_satisfaction': 0.4,  # Minimum 40% satisfaction
                'max_reward_variance': 0.5,    # Maximum reward standard deviation
                'history_size': 1000
            },
            'streaming': {
                'batch_size': 32,
                'update_frequency_seconds': 60,
                'queue_size': 1000
            },
            'ab_testing': {
                'default_treatment_ratio': 0.2,
                'min_experiment_duration_days': 7,
                'confidence_level': 0.95
            },
            'model_management': {
                'checkpoint_frequency_hours': 6,
                'auto_rollback_enabled': True,
                'performance_window_hours': 24
            }
        }
    
    def start_online_learning(self, experiment_config: ExperimentConfig = None):
        """Start online learning with optional A/B testing."""
        try:
            # Start streaming updates
            self.streaming_updater.start_streaming_updates()
            
            # Setup A/B test if provided
            if experiment_config:
                self.current_experiment_id = self.ab_test_manager.create_experiment(experiment_config)
            
            # Enable RL in integration manager
            self.rl_manager.enable_rl()
            self.rl_manager.set_learning_mode('online')
            
            self.is_online_learning_enabled = True
            self.system_stats['online_learning_starts'] += 1
            
            logger.info("Online learning started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start online learning: {e}")
            raise
    
    def stop_online_learning(self, save_checkpoint: bool = True):
        """Stop online learning and optionally save checkpoint."""
        try:
            # Stop streaming updates
            self.streaming_updater.stop_streaming_updates()
            
            # Save model checkpoint if requested
            if save_checkpoint:
                self._save_model_checkpoint("manual_stop")
            
            self.is_online_learning_enabled = False
            self.system_stats['online_learning_stops'] += 1
            
            logger.info("Online learning stopped")
            
        except Exception as e:
            logger.error(f"Error stopping online learning: {e}")
    
    def process_interaction_feedback(self, user_id: int, post_id: int, 
                                   interaction_type: str, context: Dict[str, Any]):
        """Process user interaction feedback for online learning."""
        try:
            # Check if user is in A/B test
            if self.current_experiment_id:
                use_rl = self.ab_test_manager.should_use_rl(user_id, self.current_experiment_id)
                
                # Record experiment data
                self.ab_test_manager.record_experiment_data(user_id, {
                    'interaction_type': interaction_type,
                    'post_id': post_id,
                    'timestamp': time.time(),
                    'used_rl': use_rl
                })
            
            # Process interaction through RL manager
            self.rl_manager.process_user_interaction(
                user_id, post_id, interaction_type, context
            )
            
            # Update performance metrics
            self._update_performance_metrics(user_id, interaction_type, context)
            
            # Check safety conditions
            self._perform_safety_checks()
            
        except Exception as e:
            logger.error(f"Error processing interaction feedback: {e}")
    
    def _update_performance_metrics(self, user_id: int, interaction_type: str, 
                                  context: Dict[str, Any]):
        """Update system performance metrics."""
        # Calculate reward for this interaction
        reward = self.rl_manager.reward_engineer.calculate_reward(
            interaction_type, user_id, context.get('post_id', 0), context
        ).final_reward
        
        # Get user satisfaction
        user_satisfaction = self.rl_manager.reward_engineer.get_user_satisfaction_score(user_id)
        
        # Create metrics snapshot
        metrics = ModelPerformanceMetrics(
            timestamp=datetime.now(),
            reward_mean=reward,
            reward_std=0.0,  # Will be calculated from history
            interaction_rate=1.0,  # This interaction happened
            user_satisfaction=user_satisfaction,
            diversity_score=0.5,  # Placeholder
            coverage_score=0.5,   # Placeholder
            model_version=self.model_version,
            sample_size=1
        )
        
        self.performance_metrics.append(metrics)
        
        # Calculate aggregate metrics if we have enough history
        if len(self.performance_metrics) >= 10:
            recent_rewards = [m.reward_mean for m in list(self.performance_metrics)[-10:]]
            aggregate_metrics = ModelPerformanceMetrics(
                timestamp=datetime.now(),
                reward_mean=np.mean(recent_rewards),
                reward_std=np.std(recent_rewards),
                interaction_rate=np.mean([m.interaction_rate for m in list(self.performance_metrics)[-10:]]),
                user_satisfaction=np.mean([m.user_satisfaction for m in list(self.performance_metrics)[-10:]]),
                diversity_score=0.5,
                coverage_score=0.5,
                model_version=self.model_version,
                sample_size=10
            )
            
            # Store aggregate metrics for safety monitoring
            self.performance_metrics.append(aggregate_metrics)
    
    def _perform_safety_checks(self):
        """Perform safety checks and take action if needed."""
        if not self.performance_metrics:
            return
        
        # Get latest metrics
        latest_metrics = list(self.performance_metrics)[-1]
        
        # Check for safety violations
        violations = self.safety_monitor.check_safety_violations(latest_metrics)
        
        if violations:
            logger.warning(f"Safety violations detected: {violations}")
            self.system_stats['safety_violations'] += len(violations)
            
            # Check if rollback is needed
            if self.safety_monitor.should_trigger_rollback(violations):
                logger.critical("Critical safety violations detected, triggering rollback")
                self._trigger_emergency_rollback(violations)
        
    def _trigger_emergency_rollback(self, violations: List[str]):
        """Trigger emergency rollback to safe state."""
        try:
            # Stop online learning
            self.stop_online_learning(save_checkpoint=False)
            
            # Disable RL
            self.rl_manager.disable_rl()
            
            # Restore from last known good checkpoint
            if self.model_checkpoints:
                last_checkpoint = self.model_checkpoints[-1]
                logger.info(f"Restoring from checkpoint: {last_checkpoint['timestamp']}")
                # Implementation would restore model state here
            
            self.system_stats['emergency_rollbacks'] += 1
            
            logger.critical(f"Emergency rollback completed. Violations: {violations}")
            
        except Exception as e:
            logger.critical(f"Failed to perform emergency rollback: {e}")
    
    def _save_model_checkpoint(self, checkpoint_type: str):
        """Save model checkpoint."""
        checkpoint_data = {
            'timestamp': datetime.now(),
            'model_version': self.model_version,
            'checkpoint_type': checkpoint_type,
            'performance_metrics': list(self.performance_metrics)[-10:] if self.performance_metrics else [],
            'system_stats': dict(self.system_stats)
        }
        
        self.model_checkpoints.append(checkpoint_data)
        
        # Save to disk/database in production
        checkpoint_filename = f"rl_checkpoint_{int(time.time())}_{checkpoint_type}.json"
        logger.info(f"Saved model checkpoint: {checkpoint_filename}")
    
    def get_experiment_results(self, experiment_id: str = None) -> Optional[ExperimentResult]:
        """Get A/B test experiment results."""
        target_experiment = experiment_id or self.current_experiment_id
        if not target_experiment:
            return None
        
        try:
            return self.ab_test_manager.analyze_experiment(target_experiment)
        except Exception as e:
            logger.error(f"Error analyzing experiment: {e}")
            return None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        health_status = {
            'online_learning_enabled': self.is_online_learning_enabled,
            'current_experiment': self.current_experiment_id,
            'model_version': self.model_version,
            'safety_status': 'healthy',
            'performance_metrics': {},
            'system_stats': dict(self.system_stats)
        }
        
        # Add performance metrics if available
        if self.performance_metrics:
            recent_metrics = list(self.performance_metrics)[-10:]
            health_status['performance_metrics'] = {
                'avg_reward': np.mean([m.reward_mean for m in recent_metrics]),
                'avg_satisfaction': np.mean([m.user_satisfaction for m in recent_metrics]),
                'sample_size': len(recent_metrics),
                'last_update': recent_metrics[-1].timestamp.isoformat()
            }
        
        # Check safety status
        if self.performance_metrics:
            latest_metrics = list(self.performance_metrics)[-1]
            violations = self.safety_monitor.check_safety_violations(latest_metrics)
            if violations:
                health_status['safety_status'] = 'warning'
                health_status['safety_violations'] = violations
        
        return health_status
    
    def create_ab_test(self, name: str, description: str, treatment_ratio: float,
                      duration_days: int = 14) -> str:
        """Create new A/B test experiment."""
        experiment_config = ExperimentConfig(
            experiment_id=f"rl_experiment_{int(time.time())}",
            name=name,
            description=description,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(days=duration_days),
            treatment_ratio=treatment_ratio,
            control_group_size=0,  # Will be calculated dynamically
            treatment_group_size=0,  # Will be calculated dynamically
            success_metrics=['reward', 'user_satisfaction', 'interaction_rate']
        )
        
        return self.ab_test_manager.create_experiment(experiment_config)