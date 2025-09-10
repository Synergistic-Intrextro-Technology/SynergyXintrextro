import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import random
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
import math

logger = logging.getLogger(__name__)

class IntrextroLearningImplementation:
    """
    Expanded implementation of Intrextro Learning for SC2SynergyBot.
    
    This implementation includes:
    - Curiosity-driven exploration
    - Opponent modeling and adaptation
    - Meta-learning for strategy selection
    - Hierarchical skill acquisition
    """
    
    def __init__(self, config=None):
        """
        Initialize the expanded Intrextro Learning system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            "feature_dim": 128,
            "memory_size": 10000,
            "curiosity_weight": 0.5,
            "adaptation_rate": 0.1,
            "meta_learning_rate": 0.01,
            "skill_hierarchy_levels": 3,
            "opponent_model_size": 50,
            "exploration_bonus_scale": 0.1
        }
        
        # Initialize components
        self.curiosity_module = CuriosityModule(
            feature_dim=self.config.get("feature_dim", 128),
            hidden_dim=self.config.get("feature_dim", 128) * 2
        )
        
        self.opponent_modeler = ExpandedOpponentModeler(
            feature_dim=self.config.get("feature_dim", 128),
            memory_size=self.config.get("opponent_model_size", 50)
        )
        
        self.meta_learner = MetaLearner(
            num_strategies=5,
            feature_dim=self.config.get("feature_dim", 128)
        )
        
        self.skill_hierarchy = SkillHierarchy(
            levels=self.config.get("skill_hierarchy_levels", 3),
            feature_dim=self.config.get("feature_dim", 128)
        )
        
        # Experience memory
        self.memory = deque(maxlen=self.config.get("memory_size", 10000))
        
        # Current state
        self.current_state = None
        self.current_strategy = None
        self.current_adaptation = None
        
        # Performance tracking
        self.reward_history = deque(maxlen=100)
        self.novelty_history = deque(maxlen=100)
        self.prediction_errors = deque(maxlen=100)
        
        # Exploration state
        self.visit_counts = defaultdict(int)
        self.state_values = defaultdict(float)
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and generate adaptation.
        
        Args:
            inputs: Dictionary containing game state, reward, and other information
            
        Returns:
            Dictionary with adaptation vector and metrics
        """
        # Extract inputs
        game_state = inputs.get("game_state")
        reward = inputs.get("reward")
        opponent_state = inputs.get("opponent_state")
        strategy = inputs.get("strategy")
        
        # Update reward history if reward is provided
        if reward is not None:
            self.reward_history.append(reward)
        
        # Process game state if provided
        if game_state is not None:
            # Convert to numpy array if needed
            if isinstance(game_state, torch.Tensor):
                game_state = game_state.detach().cpu().numpy()
            
            # Compute curiosity signal
            curiosity_output = self.curiosity_module.process(game_state)
            prediction_error = curiosity_output["prediction_error"]
            self.prediction_errors.append(prediction_error)
            
            # Update opponent model if opponent state is provided
            if opponent_state is not None:
                self.opponent_modeler.update(opponent_state)
            
            # Get opponent adaptation
            opponent_style = self.opponent_modeler.get_dominant_style()
            adaptation_strategy = self.opponent_modeler.get_adaptation_strategy()
            
            # Update meta-learner
            if strategy is not None:
                self.meta_learner.update(game_state, strategy, reward)
            
            # Get recommended strategy from meta-learner
            meta_strategy = self.meta_learner.select_strategy(game_state)
            
            # Update skill hierarchy
            self.skill_hierarchy.update(game_state, reward)
            
            # Compute exploration bonus
            exploration_bonus = self._compute_exploration_bonus(game_state)
            
            # Combine all signals for final adaptation
            intrinsic_motivation = prediction_error + exploration_bonus
            
            # Compute extrinsic motivation from rewards
            extrinsic_motivation = self._compute_extrinsic_motivation()
            
            # Combine intrinsic and extrinsic motivations
            combined_motivation = (
                self.config.get("curiosity_weight", 0.5) * intrinsic_motivation +
                (1 - self.config.get("curiosity_weight", 0.5)) * extrinsic_motivation
            )
            
            # Generate adaptation vector
            adaptation_vector = (
                0.4 * adaptation_strategy +
                0.4 * meta_strategy +
                0.2 * np.random.randn(self.config.get("feature_dim", 128)) * combined_motivation
            )
            
            # Normalize adaptation vector
            norm = np.linalg.norm(adaptation_vector)
            if norm > 0:
                adaptation_vector /= norm
            
            # Store current state and adaptation
            self.current_state = game_state
            self.current_adaptation = adaptation_vector
            self.current_strategy = meta_strategy
            
            # Add to memory
            self.memory.append({
                "state": game_state,
                "adaptation": adaptation_vector,
                "reward": reward,
                "intrinsic_motivation": intrinsic_motivation,
                "extrinsic_motivation": extrinsic_motivation
            })
            
            return {
                "adaptation_vector": adaptation_vector,
                "intrinsic_motivation": intrinsic_motivation,
                "extrinsic_motivation": extrinsic_motivation,
                "prediction_error": prediction_error,
                "exploration_bonus": exploration_bonus,
                "opponent_style": opponent_style,
                "meta_strategy": meta_strategy,
                "skill_levels": self.skill_hierarchy.get_skill_levels()
            }
        
        # Return current adaptation if no game state
        return {
            "adaptation_vector": self.current_adaptation if self.current_adaptation is not None 
                                else np.zeros(self.config.get("feature_dim", 128))
        }
    
    def _compute_extrinsic_motivation(self):
        """Compute extrinsic motivation based on reward history"""
        if not self.reward_history:
            return 0.5
        
        # Use recent reward trend
        recent_rewards = list(self.reward_history)[-10:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Normalize to [0, 1]
        extrinsic_motivation = 1.0 / (1.0 + np.exp(-avg_reward))
        
        return extrinsic_motivation
    
    def _compute_exploration_bonus(self, state):
        """
        Compute exploration bonus using count-based exploration.
        
        Args:
            state: Current state
            
        Returns:
            Exploration bonus
        """
        # Discretize state for counting
        state_key = self._discretize_state(state)
        
        # Update visit count
        self.visit_counts[state_key] += 1
        
        # Compute bonus (1/sqrt(N))
        bonus = self.config.get("exploration_bonus_scale", 0.1) / math.sqrt(self.visit_counts[state_key])
        
        return bonus
    
    def _discretize_state(self, state):
        """
        Discretize continuous state for counting.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Discretized state key
        """
        # Simple discretization: round to 1 decimal place and convert to tuple
        discretized = tuple(np.round(state[:10] * 10) / 10)
        return discretized
    
    def reset(self):
        """Reset the learning system between episodes"""
        # Reset current state
        self.current_state = None
        self.current_strategy = None
        
        # Keep adaptation vector for continuity
        
        # Reset components that need resetting
        self.curiosity_module.reset()
        self.meta_learner.reset_episode()
        
        logger.info("Intrextro learning system reset")

class CuriosityModule:
    """
    Curiosity-driven exploration module.
    
    Implements intrinsic curiosity module (ICM) with forward and inverse models.
    """
    
    def __init__(self, feature_dim=128, hidden_dim=256):
        """
        Initialize curiosity module.
        
        Args:
            feature_dim: Dimension of feature vectors
            hidden_dim: Dimension of hidden layers
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Forward model (predicts next state features)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + feature_dim//4, hidden_dim),  # State + action
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Inverse model (predicts action from states)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # State + next state
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim//4)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.forward_model.parameters()) +
            list(self.inverse_model.parameters()),
            lr=0.001
        )
        
        # Memory for training
        self.memory = deque(maxlen=1000)
        
        # Current state
        self.current_state = None
        self.current_features = None
        self.current_action = None
    
    def process(self, state, action=None):
        """
        Process state and compute curiosity signal.
        
        Args:
            state: Current state
            action: Action taken (optional)
            
        Returns:
            Dictionary with prediction error and features
        """
        # Convert to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Encode state
        with torch.no_grad():
            features = self.encoder(state_tensor)
        
        # Generate random action if none provided
        if action is None:
            action = np.random.randn(self.feature_dim // 4)
        
        action_tensor = torch.FloatTensor(action)
        
        # Predict next state
        with torch.no_grad():
            predicted_next_features = self.forward_model(
                torch.cat([features, action_tensor], dim=0)
            )
        
        # Store for training if we have a previous state
        if self.current_state is not None:
            self.memory.append({
                "state": self.current_state,
                "next_state": state,
                "action": self.current_action if self.current_action is not None else action,
                "features": self.current_features,
                "next_features": features.detach().numpy()
            })
            
            # Train models occasionally
            if len(self.memory) >= 64 and random.random() < 0.1:
                self._train_models()
        
        # Compute prediction error (curiosity)
        if self.current_features is not None:
            prediction_error = torch.nn.functional.mse_loss(
                predicted_next_features, 
                features
            ).item()
        else:
            prediction_error = 0.0
        
        # Update current state
        self.current_state = state
        self.current_features = features.detach().numpy()
        self.current_action = action
        
        return {
            "prediction_error": prediction_error,
            "features": features.detach().numpy()
        }
    
    def _train_models(self):
        """Train forward and inverse models"""
        if len(self.memory) < 64:
            return
        
        # Sample batch
        batch = random.sample(self.memory, 64)
        
        # Prepare tensors
        states = torch.FloatTensor(np.array([item["state"] for item in batch]))
        next_states = torch.FloatTensor(np.array([item["next_state"] for item in batch]))
        actions = torch.FloatTensor(np.array([item["action"] for item in batch]))
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # Encode states
        features = self.encoder(states)
        next_features = self.encoder(next_states)
        
        # Forward model
        predicted_next_features = self.forward_model(
            torch.cat([features, actions], dim=1)
        )
        
        # Inverse model
        predicted_actions = self.inverse_model(
            torch.cat([features, next_features], dim=1)
        )
        
        # Compute losses
        forward_loss = F.mse_loss(predicted_next_features, next_features.detach())
        inverse_loss = F.mse_loss(predicted_actions, actions)
        
        # Combined loss
        loss = 0.8 * forward_loss + 0.2 * inverse_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
    
    def reset(self):
        """Reset curiosity module"""
        self.current_state = None
        self.current_features = None
        self.current_action = None

class ExpandedOpponentModeler:
    """
    Enhanced opponent modeling system.
    
    Tracks opponent behavior and generates adaptation strategies.
    """
    
    def __init__(self, feature_dim=128, memory_size=50):
        """
        Initialize opponent modeler.
        
        Args:
            feature_dim: Dimension of feature vectors
            memory_size: Size of opponent memory
        """
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # Opponent style classification
        self.style_categories = [
            "aggressive_rush", "aggressive_timing", 
            "defensive_turtle", "defensive_counter",
            "economic_greedy", "economic_tech",
            "harass", "air_focused"
        ]
        
        # Style vectors (simplified representation of each style)
        self.style_vectors = {
            "aggressive_rush": np.array([0.9, 0.7, 0.1, 0.1, 0.2, 0.1, 0.3, 0.1]),
            "aggressive_timing": np.array([0.7, 0.8, 0.2, 0.3, 0.3, 0.4, 0.2, 0.3]),
            "defensive_turtle": np.array([0.2, 0.3, 0.9, 0.7, 0.4, 0.3, 0.1, 0.2]),
            "defensive_counter": np.array([0.3, 0.5, 0.7, 0.9, 0.3, 0.4, 0.2, 0.3]),
            "economic_greedy": np.array([0.1, 0.2, 0.4, 0.3, 0.9, 0.6, 0.2, 0.3]),
            "economic_tech": np.array([0.2, 0.3, 0.3, 0.4, 0.6, 0.9, 0.3, 0.5]),
            "harass": np.array([0.5, 0.4, 0.2, 0.3, 0.4, 0.3, 0.9, 0.4]),
            "air_focused": np.array([0.3, 0.4, 0.3, 0.4, 0.4, 0.6, 0.4, 0.9])
        }
        
        # Counter strategies
        self.counter_strategies = {
            "aggressive_rush": np.array([0.2, 0.3, 0.8, 0.7, 0.3, 0.2, 0.1, 0.2]),  # Defend
            "aggressive_timing": np.array([0.3, 0.2, 0.7, 0.8, 0.4, 0.3, 0.2, 0.3]),  # Defend
            "defensive_turtle": np.array([0.3, 0.2, 0.2, 0.3, 0.8, 0.7, 0.3, 0.4]),  # Economy
            "defensive_counter": np.array([0.2, 0.3, 0.3, 0.2, 0.7, 0.8, 0.4, 0.5]),  # Economy/Tech
            "economic_greedy": np.array([0.8, 0.7, 0.2, 0.3, 0.1, 0.2, 0.5, 0.3]),  # Aggression
            "economic_tech": np.array([0.7, 0.6, 0.3, 0.4, 0.2, 0.1, 0.4, 0.7]),  # Aggression/Air
            "harass": np.array([0.3, 0.2, 0.8, 0.7, 0.4, 0.3, 0.2, 0.3]),  # Defend
            "air_focused": np.array([0.4, 0.5, 0.6, 0.7, 0.3, 0.2, 0.3, 0.1])   # Anti-air
        }
        
        # Opponent memory
        self.opponent_observations = deque(maxlen=memory_size)
        
        # Current opponent model
        self.style_probabilities = np.ones(len(self.style_categories)) / len(self.style_categories)
        
        # Adaptation history
        self.adaptation_history = deque(maxlen=10)
        
        # Temporal patterns
        self.temporal_patterns = {}
        for style in self.style_categories:
            self.temporal_patterns[style] = deque(maxlen=5)
    
    def update(self, observation):
        """
        Update opponent model with new observation.
        
        Args:
            observation: Feature vector of opponent behavior
        """
        # Ensure observation is numpy array
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
        
        # Add to memory
        self.opponent_observations.append(observation)
        
        # Update style probabilities
        if len(self.opponent_observations) >= 3:
            self._update_style_probabilities()
            
            # Update temporal patterns
            dominant_style = self.get_dominant_style()
            if dominant_style != "mixed":
                self.temporal_patterns[dominant_style].append(len(self.opponent_observations))
    
    def _update_style_probabilities(self):
        """Update probabilities of opponent playing each style"""
        if not self.opponent_observations:
            return
        
        # Extract recent observations
        recent_obs = list(self.opponent_observations)[-3:]
        
        # Extract features relevant to style classification
        # In a real implementation, this would extract meaningful features
        # Here we just use the first 8 elements as a simplified representation
        style_features = np.mean([obs[:8] if len(obs) >= 8 else np.zeros(8) for obs in recent_obs], axis=0)
        
        # Compute similarity to each style
        similarities = {}
        for style, vector in self.style_vectors.items():
            # Use cosine similarity
            similarity = np.dot(style_features, vector) / (
                np.linalg.norm(style_features) * np.linalg.norm(vector) + 1e-8
            )
            similarities[style] = max(0, similarity)
        
        # Normalize to probabilities
        total_similarity = sum(similarities.values())
        if total_similarity > 0:
            for i, style in enumerate(self.style_categories):
                self.style_probabilities[i] = similarities[style] / total_similarity
    
    def get_dominant_style(self):
        """Get the most likely opponent style"""
        if np.max(self.style_probabilities) < 0.3:
            # No clear dominant style
            return "mixed"
        
        dominant_idx = np.argmax(self.style_probabilities)
        return self.style_categories[dominant_idx]
    
    def get_adaptation_strategy(self):
        """Get adaptation strategy based on opponent model"""
        # Weighted combination of counter strategies
        adaptation = np.zeros(8)
        for i, style in enumerate(self.style_categories):
            adaptation += self.style_probabilities[i] * self.counter_strategies[style]
        
        # Normalize
        norm = np.linalg.norm(adaptation)
        if norm > 0:
            adaptation = adaptation / norm
        
        # Extend to full feature dimension
        full_adaptation = np.zeros(self.feature_dim)
        full_adaptation[:len(adaptation)] = adaptation
        
        # Add some noise for exploration
        full_adaptation += np.random.randn(self.feature_dim) * 0.05
        
        # Store in adaptation history
        self.adaptation_history.append(full_adaptation)
        
        return full_adaptation
    
    def predict_next_style(self):
        """Predict opponent's next style based on temporal patterns"""
        dominant_style = self.get_dominant_style()
        
        if dominant_style == "mixed" or len(self.temporal_patterns[dominant_style]) < 2:
            return dominant_style
        
        # Check if there's a pattern in style transitions
        # This is a simplified implementation
        return dominant_style

class MetaLearner:
    """
    Meta-learning system for strategy selection.
    
    Learns which strategies work best in different game states.
    """
    
    def __init__(self, num_strategies=5, feature_dim=128):
        """
        Initialize meta-learner.
        
        Args:
            num_strategies: Number of high-level strategies
            feature_dim: Dimension of feature vectors
        """
        self.num_strategies = num_strategies
        self.feature_dim = feature_dim
        
        # Strategy definitions (simplified)
        self.strategies = [
            "aggressive",
            "defensive",
            "economic",
            "tech",
            "balanced"
        ]
        
        # Strategy vectors
        self.strategy_vectors = {
            "aggressive": np.array([0.8, 0.2, 0.1, 0.1, 0.5]),
            "defensive": np.array([0.2, 0.8, 0.3, 0.3, 0.5]),
            "economic": np.array([0.1, 0.3, 0.8, 0.2, 0.5]),
            "tech": np.array([0.1, 0.3, 0.2, 0.8, 0.5]),
            "balanced": np.array([0.5, 0.5, 0.5, 0.5, 0.8])
        }
        
        # Value function for each strategy
        self.strategy_values = {}
        for strategy in self.strategies:
            self.strategy_values[strategy] = {}  # State -> value mapping
        
        # Learning rate
        self.learning_rate = 0.1
        
        # Discount factor
        self.gamma = 0.9
        
        # Exploration parameter
        self.epsilon = 0.2
        
        # Current strategy
        self.current_strategy = "balanced"
        self.current_state_key = None
        
        # Episode history
        self.episode_history = []
    
    def select_strategy(self, state):
        """
        Select best strategy for current state.
        
        Args:
            state: Current game state
            
        Returns:
            Strategy vector
        """
        # Discretize state
        state_key = self._discretize_state(state)
        self.current_state_key = state_key
        
        # Epsilon-greedy strategy selection
        if random.random() < self.epsilon:
            # Random strategy
            self.current_strategy = random.choice(self.strategies)
        else:
            # Best strategy based on value function
            best_value = float('-inf')
            best_strategy = "balanced"  # Default
            
            for strategy in self.strategies:
                value = self.strategy_values[strategy].get(state_key, 0.0)
                if value > best_value:
                    best_value = value
                    best_strategy = strategy
            
            self.current_strategy = best_strategy
        
        # Add to episode history
        self.episode_history.append((state_key, self.current_strategy))
        
        # Return strategy vector
        return self.strategy_vectors[self.current_strategy]
    
    def update(self, state, strategy, reward):
        """
        Update value function based on reward.
        
        Args:
            state: Current state
            strategy: Strategy used
            reward: Reward received
        """
        if isinstance(strategy, np.ndarray):
            # Convert strategy vector to name
            strategy = self._vector_to_strategy(strategy)
        
        state_key = self._discretize_state(state)
        
        # Initialize value if not exists
        if state_key not in self.strategy_values[strategy]:
            self.strategy_values[strategy][state_key] = 0.0
        
        # Update value function
        current_value = self.strategy_values[strategy][state_key]
        self.strategy_values[strategy][state_key] = current_value + self.learning_rate * (reward - current_value)
    
    def _discretize_state(self, state):
        """
        Discretize continuous state for value function.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Discretized state key
        """
        # Simple discretization: round to 1 decimal place and convert to tuple
        # Use only first few elements for simplicity
        discretized = tuple(np.round(state[:5] * 10) / 10)
        return discretized
    
    def _vector_to_strategy(self, vector):
        """
        Convert strategy vector to strategy name.
        
        Args:
            vector: Strategy vector
            
        Returns:
            Strategy name
        """
        # Find closest strategy
        best_similarity = -1
        best_strategy = self.strategies[0]
        
        for strategy, strategy_vector in self.strategy_vectors.items():
            # Use cosine similarity
            similarity = np.dot(vector[:len(strategy_vector)], strategy_vector) / (
                np.linalg.norm(vector[:len(strategy_vector)]) * np.linalg.norm(strategy_vector) + 1e-8
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_strategy = strategy
        
        return best_strategy
    
    def reset_episode(self):
        """Reset episode history"""
        # Apply updates for the episode
        if self.episode_history:
            self._update_episode()
        
        self.episode_history = []
        self.current_strategy = "balanced"
        self.current_state_key = None
    
    def _update_episode(self):
        """Update value function based on episode history"""
        # Skip if no history
        if not self.episode_history:
            return
        
        # Assume last reward is available
        last_reward = 0.0  # Default if no reward
        
        # Backward updates (like TD(λ))
        for i in range(len(self.episode_history) - 1, -1, -1):
            state_key, strategy = self.episode_history[i]
            
            # Initialize value if not exists
            if state_key not in self.strategy_values[strategy]:
                self.strategy_values[strategy][state_key] = 0.0
            
            # Update value function
            current_value = self.strategy_values[strategy][state_key]
            self.strategy_values[strategy][state_key] = current_value + self.learning_rate * (last_reward - current_value)
            
            # Decay reward for earlier states
            last_reward *= self.gamma

class SkillHierarchy:
    """
    Hierarchical skill learning system.
    
    Learns skills at different levels of abstraction.
    """
    
    def __init__(self, levels=3, feature_dim=128):
        """
        Initialize skill hierarchy.
        
        Args:
            levels: Number of hierarchy levels
            feature_dim: Dimension of feature vectors
        """
        self.levels = levels
        self.feature_dim = feature_dim
        
        # Skill definitions at each level
        self.skills = {}
        
        # Level 1: Basic actions
        self.skills[1] = [
            "attack", "defend", "build", "scout", "harvest"
        ]
        
        # Level 2: Tactical combinations
        self.skills[2] = [
            "rush", "expand", "tech_up", "harass", "contain"
        ]
        
        # Level 3: Strategic plans
        self.skills[3] = [
            "early_aggression", "macro_focused", "timing_attack", 
            "tech_switch", "late_game_dominance"
        ]
        
        # Skill proficiency (0-1 scale)
        self.skill_levels = {}
        for level in range(1, levels + 1):
            self.skill_levels[level] = {}
            for skill in self.skills[level]:
                self.skill_levels[level][skill] = 0.2  # Initial proficiency
        
        # Skill usage counts
        self.skill_usage = {}
        for level in range(1, levels + 1):
            self.skill_usage[level] = {}
            for skill in self.skills[level]:
                self.skill_usage[level][skill] = 0
        
        # Skill success rates
        self.skill_success = {}
        for level in range(1, levels + 1):
            self.skill_success[level] = {}
            for skill in self.skills[level]:
                self.skill_success[level][skill] = []
        
        # Current active skills at each level
        self.active_skills = {}
        for level in range(1, levels + 1):
            self.active_skills[level] = random.choice(self.skills[level])
    
    def update(self, state, reward=None):
        """
        Update skill hierarchy based on state and reward.
        
        Args:
            state: Current game state
            reward: Reward received (optional)
        """
        # Update skill usage
        for level in range(1, self.levels + 1):
            skill = self.active_skills[level]
            self.skill_usage[level][skill] += 1
        
        # Update skill success if reward provided
        if reward is not None:
            for level in range(1, self.levels + 1):
                skill = self.active_skills[level]
                self.skill_success[level][skill].append(reward > 0)
                
                # Update skill proficiency based on success rate
                if len(self.skill_success[level][skill]) >= 5:
                    success_rate = sum(self.skill_success[level][skill][-5:]) / 5
                    
                    # Adjust proficiency (with momentum)
                    current_level = self.skill_levels[level][skill]
                    target_level = min(1.0, 0.2 + success_rate * 0.8)  # Scale to 0.2-1.0
                    
                    # Smooth update
                    self.skill_levels[level][skill] = current_level * 0.9 + target_level * 0.1
        
        # Select new active skills based on state
        self._select_skills(state)
    
    def _select_skills(self, state):
        """
        Select appropriate skills for current state.
        
        Args:
            state: Current game state
        """
        # This is a simplified implementation
        # In a real system, this would use state features to select appropriate skills
        
        # Level 3 (strategic) selection - based on game phase
        game_phase = self._estimate_game_phase(state)
        if game_phase < 0.3:
            # Early game
            candidates = ["early_aggression", "macro_focused"]
        elif game_phase < 0.7:
            # Mid game
            candidates = ["timing_attack", "tech_switch", "macro_focused"]
        else:
            # Late game
            candidates = ["tech_switch", "late_game_dominance"]
        
        # Filter by proficiency and select
        viable_candidates = [skill for skill in candidates if self.skill_levels[3][skill] > 0.3]
        if viable_candidates:
            self.active_skills[3] = max(viable_candidates, 
                                       key=lambda s: self.skill_levels[3][s])
        else:
            self.active_skills[3] = random.choice(candidates)
        
        # Level 2 (tactical) selection - based on active strategy
        if self.active_skills[3] == "early_aggression":
            candidates = ["rush", "harass"]
        elif self.active_skills[3] == "macro_focused":
            candidates = ["expand", "tech_up"]
        elif self.active_skills[3] == "timing_attack":
            candidates = ["contain", "harass"]
        elif self.active_skills[3] == "tech_switch":
            candidates = ["tech_up", "contain"]
        else:  # late_game_dominance
            candidates = ["expand", "contain"]
        
        # Select based on proficiency
        self.active_skills[2] = max(candidates, 
                                   key=lambda s: self.skill_levels[2][s])
        
        # Level 1 (basic) selection - based on active tactic
        if self.active_skills[2] == "rush":
            candidates = ["attack", "build"]
        elif self.active_skills[2] == "expand":
            candidates = ["build", "harvest"]
        elif self.active_skills[2] == "tech_up":
            candidates = ["build", "defend"]
        elif self.active_skills[2] == "harass":
            candidates = ["attack", "scout"]
        else:  # contain
            candidates = ["attack", "defend"]
        
        # Select based on proficiency
        self.active_skills[1] = max(candidates, 
                                   key=lambda s: self.skill_levels[1][s])
    
    def _estimate_game_phase(self, state):
        """
        Estimate game phase from state.
        
        Args:
            state: Current game state
            
        Returns:
            Game phase (0-1 scale, 0=early, 1=late)
        """
        # This is a simplified implementation
        # In a real system, this would use supply, tech level, etc.
        
        # Use first element as a proxy for game progression
        return min(1.0, max(0.0, state[0]))
    
    def get_skill_levels(self):
        """Get current skill proficiency levels"""
        return self.skill_levels
    
    def get_active_skills(self):
        """Get currently active skills at each level"""
        return self.active_skills

class FederatedIntrextroLearning:
    """
    Federated version of Intrextro Learning.
    
    Enables distributed learning across multiple agents.
    """
    
    def __init__(self, config=None):
        """
        Initialize federated learning system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            "num_clients": 5,
            "aggregation_rounds": 10,
            "local_update_steps": 5,
            "feature_dim": 128
        }
        
        # Create client models
        self.clients = []
        for i in range(self.config.get("num_clients", 5)):
            self.clients.append(IntrextroLearningImplementation(self.config))
        
        # Global model (for aggregation)
        self.global_model = IntrextroLearningImplementation(self.config)
        
        # Training metrics
        self.client_rewards = [[] for _ in range(self.config.get("num_clients", 5))]
        self.global_rewards = []
        
        # Round counter
        self.round = 0
    
    def client_update(self, client_id, inputs):
        """
        Update a specific client.
        
        Args:
            client_id: ID of client to update
            inputs: Input data for client
            
        Returns:
            Client outputs
        """
        if client_id >= len(self.clients):
            raise ValueError(f"Invalid client ID: {client_id}")
        
        # Process inputs with client model
        outputs = self.clients[client_id].process(inputs)
        
        # Store reward if available
        if "reward" in inputs:
            self.client_rewards[client_id].append(inputs["reward"])
        
        return outputs
    
    def aggregate_models(self):
        """
        Aggregate client models into global model.
        
        This implements federated averaging.
        """
        self.round += 1
        
        # Skip if not enough rounds
        if self.round % self.config.get("aggregation_rounds", 10) != 0:
            return
        
        logger.info("Performing federated aggregation")
        
        # Aggregate curiosity modules
        self._aggregate_curiosity_modules()
        
        # Aggregate opponent models
        self._aggregate_opponent_models()
        
        # Aggregate meta-learners
        self._aggregate_meta_learners()
        
        # Distribute global model to clients
        self._distribute_global_model()
        
        logger.info("Federated aggregation complete")
    
    def _aggregate_curiosity_modules(self):
        """Aggregate curiosity modules from clients"""
        # This is a simplified implementation
        # In a real system, this would aggregate neural network weights
        
        # Collect prediction errors
        all_errors = []
        for client in self.clients:
            all_errors.extend(client.curiosity_module.prediction_errors)
        
        # Update global model
        if all_errors:
            self.global_model.curiosity_module.prediction_errors = deque(all_errors, maxlen=100)
    
    def _aggregate_opponent_models(self):
        """Aggregate opponent models from clients"""
        # Aggregate style probabilities
        avg_probabilities = np.zeros_like(self.global_model.opponent_modeler.style_probabilities)
        
        for client in self.clients:
            avg_probabilities += client.opponent_modeler.style_probabilities
        
        avg_probabilities /= len(self.clients)
        
        # Update global model
        self.global_model.opponent_modeler.style_probabilities = avg_probabilities
    
    def _aggregate_meta_learners(self):
        """Aggregate meta-learners from clients"""
        # Aggregate strategy values
        for strategy in self.global_model.meta_learner.strategies:
            # Initialize global values
            global_values = {}
            
            # Collect all state keys
            all_keys = set()
            for client in self.clients:
                all_keys.update(client.meta_learner.strategy_values[strategy].keys())
            
            # Average values for each state
            for state_key in all_keys:
                values = []
                for client in self.clients:
                    if state_key in client.meta_learner.strategy_values[strategy]:
                        values.append(client.meta_learner.strategy_values[strategy][state_key])
                
                if values:
                    global_values[state_key] = sum(values) / len(values)
            
            # Update global model
            self.global_model.meta_learner.strategy_values[strategy] = global_values
    
    def _distribute_global_model(self):
        """Distribute global model to clients"""
        # This is a simplified implementation
        # In a real system, this would distribute neural network weights
        
        for client in self.clients:
            # Copy opponent model
            client.opponent_modeler.style_probabilities = self.global_model.opponent_modeler.style_probabilities.copy()
            
            # Copy meta-learner values
            for strategy in client.meta_learner.strategies:
                client.meta_learner.strategy_values[strategy] = self.global_model.meta_learner.strategy_values[strategy].copy()
    
    def get_performance_metrics(self):
        """Get performance metrics across clients"""
        metrics = {
            "client_rewards": [np.mean(rewards[-10:]) if len(rewards) >= 10 else 0 for rewards in self.client_rewards],
            "global_reward": np.mean(self.global_rewards[-10:]) if len(self.global_rewards) >= 10 else 0,
            "round": self.round
        }
        
        return metrics

class QuantumInspiredFusion:
    """
    Quantum-inspired fusion module for Intrextro Learning.
    
    Implements quantum-inspired information fusion for multimodal inputs.
    """
    
    def __init__(self, feature_dim=128, num_modes=3):
        """
        Initialize quantum-inspired fusion.
        
        Args:
            feature_dim: Dimension of feature vectors
            num_modes: Number of input modalities
        """
        self.feature_dim = feature_dim
        self.num_modes = num_modes
        
        # Entanglement matrix (simplified quantum-inspired representation)
        self.entanglement = np.eye(num_modes) + 0.2 * np.ones((num_modes, num_modes))
        np.fill_diagonal(self.entanglement, 1.0)
        
        # Normalize
        row_sums = self.entanglement.sum(axis=1)
        self.entanglement = self.entanglement / row_sums[:, np.newaxis]
        
        # Mode weights
        self.mode_weights = np.ones(num_modes) / num_modes
        
        # Learning rate
        self.learning_rate = 0.01
    
    def fuse(self, inputs):
        """
        Fuse multiple input modalities.
        
        Args:
            inputs: List of input vectors for each modality
            
        Returns:
            Fused representation
        """
        if len(inputs) != self.num_modes:
            raise ValueError(f"Expected {self.num_modes} inputs, got {len(inputs)}")
        
        # Apply entanglement
        entangled = np.zeros((self.num_modes, self.feature_dim))
        
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                entangled[i] += self.entanglement[i, j] * inputs[j]
        
        # Apply mode weights
        fused = np.zeros(self.feature_dim)
        for i in range(self.num_modes):
            fused += self.mode_weights[i] * entangled[i]
        
        return fused
    
    def update(self, inputs, reward):
        """
        Update fusion parameters based on reward.
        
        Args:
            inputs: List of input vectors
            reward: Reward signal
        """
        if reward is None or len(inputs) != self.num_modes:
            return
        
        # Update mode weights based on reward
        # Increase weights for modes that correlate with positive rewards
        for i in range(self.num_modes):
            # Compute correlation with reward
            correlation = np.mean(inputs[i]) * reward
            
            # Update weight
            self.mode_weights[i] += self.learning_rate * correlation
        
        # Normalize weights
        self.mode_weights = self.mode_weights / np.sum(self.mode_weights)
        
        # Update entanglement matrix (simplified)
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                if i != j:
                    # Compute correlation between modes
                    mode_correlation = np.dot(inputs[i], inputs[j]) / (
                        np.linalg.norm(inputs[i]) * np.linalg.norm(inputs[j]) + 1e-8
                    )
                    
                    # Update entanglement based on correlation and reward
                    self.entanglement[i, j] += self.learning_rate * mode_correlation * reward
        
        # Normalize entanglement matrix
        row_sums = self.entanglement.sum(axis=1)
        self.entanglement = self.entanglement / row_sums[:, np.newaxis]
    
    def get_entanglement_metrics(self):
        """Get metrics about the entanglement matrix"""
        return {
            "avg_entanglement": np.mean(self.entanglement - np.eye(self.num_modes)),
            "max_entanglement": np.max(self.entanglement - np.eye(self.num_modes)),
            "mode_weights": self.mode_weights.tolist()
        }
    
    def interference(self, inputs):
        """
        Apply quantum-inspired interference between inputs.
        
        Args:
            inputs: List of input vectors
            
        Returns:
            Interference pattern
        """
        if len(inputs) != self.num_modes:
            raise ValueError(f"Expected {self.num_modes} inputs, got {len(inputs)}")
        
        # Normalize inputs
        normalized_inputs = []
        for inp in inputs:
            norm = np.linalg.norm(inp)
            if norm > 0:
                normalized_inputs.append(inp / norm)
            else:
                normalized_inputs.append(inp)
        
        # Compute interference pattern
        interference = np.zeros(self.feature_dim)
        
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                # Phase factor (simplified)
                phase = np.sin(np.sum(normalized_inputs[i] * normalized_inputs[j]))
                
                # Interference term
                interference += self.entanglement[i, j] * normalized_inputs[i] * normalized_inputs[j] * phase
        
        return interference

class AdaptiveExplorationModule:
    """
    Adaptive exploration module for Intrextro Learning.
    
    Implements adaptive exploration strategies based on learning progress.
    """
    
    def __init__(self, feature_dim=128):
        """
        Initialize adaptive exploration module.
        
        Args:
            feature_dim: Dimension of feature vectors
        """
        self.feature_dim = feature_dim
        
        # Exploration strategies
        self.strategies = [
            "random",           # Random exploration
            "novelty_seeking",  # Seek novel states
            "uncertainty",      # Target uncertain states
            "goal_directed",    # Directed exploration
            "skill_practice"    # Practice specific skills
        ]
        
        # Strategy weights
        self.strategy_weights = {
            "random": 0.2,
            "novelty_seeking": 0.3,
            "uncertainty": 0.2,
            "goal_directed": 0.2,
            "skill_practice": 0.1
        }
        
        # Learning progress for each strategy
        self.learning_progress = {
            strategy: deque(maxlen=10) for strategy in self.strategies
        }
        
        # Current strategy
        self.current_strategy = "random"
        
        # Exploration noise scale
        self.noise_scale = 0.2
        
        # Learning rate
        self.learning_rate = 0.1
    
    def select_exploration_strategy(self, state, learning_progress=None):
        """
        Select exploration strategy based on learning progress.
        
        Args:
            state: Current state
            learning_progress: Optional learning progress signal
            
        Returns:
            Selected strategy name
        """
        # Update learning progress if provided
        if learning_progress is not None:
            self.learning_progress[self.current_strategy].append(learning_progress)
            
            # Update strategy weights based on learning progress
            self._update_strategy_weights()
        
        # Select strategy probabilistically based on weights
        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        
        self.current_strategy = random.choices(strategies, weights=weights, k=1)[0]
        
        return self.current_strategy
    
    def _update_strategy_weights(self):
        """Update strategy weights based on learning progress"""
        # Compute average learning progress for each strategy
        avg_progress = {}
        for strategy, progress in self.learning_progress.items():
            if progress:
                avg_progress[strategy] = sum(progress) / len(progress)
            else:
                avg_progress[strategy] = 0.0
        
        # Softmax normalization
        total = sum(np.exp(list(avg_progress.values())))
        if total > 0:
            for strategy in self.strategies:
                self.strategy_weights[strategy] = np.exp(avg_progress[strategy]) / total
    
    def generate_exploration_noise(self, state):
        """
        Generate exploration noise based on current strategy.
        
        Args:
            state: Current state
            
        Returns:
            Exploration noise vector
        """
        # Base noise
        noise = np.random.randn(self.feature_dim) * self.noise_scale
        
        # Modify based on strategy
        if self.current_strategy == "random":
            # Pure random noise
            pass
        
        elif self.current_strategy == "novelty_seeking":
            # Directional noise away from familiar states
            # This is a simplified implementation
            noise *= 1.5  # Increase magnitude
        
        elif self.current_strategy == "uncertainty":
            # Focus on uncertain dimensions
            # This is a simplified implementation
            uncertainty_mask = np.random.binomial(1, 0.3, size=self.feature_dim)
            noise *= uncertainty_mask * 2.0
        
        elif self.current_strategy == "goal_directed":
            # Directed noise towards goal
            # This is a simplified implementation
            goal_direction = np.ones(self.feature_dim) * 0.1
            noise = noise * 0.5 + goal_direction
        
        elif self.current_strategy == "skill_practice":
            # Focused on skill-relevant dimensions
            # This is a simplified implementation
            skill_mask = np.random.binomial(1, 0.2, size=self.feature_dim)
            noise *= skill_mask * 3.0
        
        return noise
    
    def adapt_noise_scale(self, reward):
        """
        Adapt noise scale based on rewards.
        
        Args:
            reward: Reward signal
        """
        if reward is None:
            return
        
        # Decrease noise if getting good rewards
        if reward > 0:
            self.noise_scale = max(0.05, self.noise_scale * 0.99)
        else:
            # Increase noise if not getting rewards
            self.noise_scale = min(0.5, self.noise_scale * 1.01)

class EmergentBehaviorAnalyzer:
    """
    Analyzer for emergent behaviors in Intrextro Learning.
    
    Identifies and categorizes emergent behaviors and strategies.
    """
    
    def __init__(self, memory_size=1000, feature_dim=128):
        """
        Initialize emergent behavior analyzer.
        
        Args:
            memory_size: Size of behavior memory
            feature_dim: Dimension of feature vectors
        """
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        
        # Behavior memory
        self.behavior_memory = deque(maxlen=memory_size)
        
        # Behavior clusters
        self.behavior_clusters = []
        self.cluster_labels = []
        
        # Minimum samples for clustering
        self.min_samples = 50
        
        # Behavior transition graph
        self.transition_graph = {}
        
        # Current behavior
        self.current_behavior = None
        self.previous_behavior = None
    
    def update(self, state, action, reward):
        """
        Update behavior analysis with new observation.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        # Create behavior vector (state-action pair)
        behavior = np.concatenate([state[:10], action[:10]])
        
        # Add to memory
        self.behavior_memory.append({
            "behavior": behavior,
            "reward": reward
        })
        
        # Update clusters periodically
        if len(self.behavior_memory) >= self.min_samples and len(self.behavior_memory) % 20 == 0:
            self._update_clusters()
        
        # Identify current behavior
        if self.behavior_clusters:
            self.previous_behavior = self.current_behavior
            self.current_behavior = self._identify_behavior(behavior)
            
            # Update transition graph
            if self.previous_behavior is not None and self.current_behavior is not None:
                self._update_transition_graph(self.previous_behavior, self.current_behavior)
    
    def _update_clusters(self):
        """Update behavior clusters"""
        # Extract behaviors
        behaviors = np.array([item["behavior"] for item in self.behavior_memory])
        
        # Simple clustering (in a real implementation, use k-means or DBSCAN)
        # This is a very simplified implementation
        if not self.behavior_clusters:
            # Initialize first cluster
            self.behavior_clusters = [np.mean(behaviors[:10], axis=0)]
            self.cluster_labels = ["behavior_1"]
        
        # Find new clusters
        for behavior in behaviors:
            # Compute distances to existing clusters
            distances = [np.linalg.norm(behavior - cluster) for cluster in self.behavior_clusters]
            min_distance = min(distances) if distances else float('inf')
            
            # Create new cluster if behavior is far from existing ones
            if min_distance > 2.0 and len(self.behavior_clusters) < 10:
                self.behavior_clusters.append(behavior)
                self.cluster_labels.append(f"behavior_{len(self.behavior_clusters)}")
    
    def _identify_behavior(self, behavior):
        """
        Identify behavior cluster for a given behavior.
        
        Args:
            behavior: Behavior vector
            
        Returns:
            Behavior label
        """
        if not self.behavior_clusters:
            return None
        
        # Compute distances to clusters
        distances = [np.linalg.norm(behavior - cluster) for cluster in self.behavior_clusters]
        min_idx = np.argmin(distances)
        
        return self.cluster_labels[min_idx]
    
    def _update_transition_graph(self, from_behavior, to_behavior):
        """
        Update behavior transition graph.
        
        Args:
            from_behavior: Source behavior
            to_behavior: Target behavior
        """
        if from_behavior not in self.transition_graph:
            self.transition_graph[from_behavior] = {}
        
        if to_behavior not in self.transition_graph[from_behavior]:
            self.transition_graph[from_behavior][to_behavior] = 0
        
        self.transition_graph[from_behavior][to_behavior] += 1
    
    def get_behavior_statistics(self):
        """Get statistics about emergent behaviors"""
        if not self.behavior_clusters:
            return {"num_behaviors": 0}
        
        # Count behaviors
        behavior_counts = {}
        for item in self.behavior_memory:
            behavior = self._identify_behavior(item["behavior"])
            if behavior:
                if behavior not in behavior_counts:
                    behavior_counts[behavior] = 0
                behavior_counts[behavior] += 1
        
        # Compute average reward for each behavior
        behavior_rewards = {}
        for item in self.behavior_memory:
            behavior = self._identify_behavior(item["behavior"])
            if behavior:
                if behavior not in behavior_rewards:
                    behavior_rewards[behavior] = []
                behavior_rewards[behavior].append(item["reward"])
        
        avg_rewards = {
            behavior: sum(rewards) / len(rewards) 
            for behavior, rewards in behavior_rewards.items() if rewards
        }
        
        return {
            "num_behaviors": len(self.behavior_clusters),
            "behavior_counts": behavior_counts,
            "avg_rewards": avg_rewards,
            "current_behavior": self.current_behavior
        }
    
    def get_dominant_behaviors(self, top_n=3):
        """
        Get the most dominant behaviors.
        
        Args:
            top_n: Number of top behaviors to return
            
        Returns:
            List of dominant behavior labels
        """
        stats = self.get_behavior_statistics()
        
        if "behavior_counts" not in stats or not stats["behavior_counts"]:
            return []
        
        # Sort behaviors by count
        sorted_behaviors = sorted(
            stats["behavior_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N
        return [behavior for behavior, _ in sorted_behaviors[:top_n]]
    
    def get_most_rewarding_behaviors(self, top_n=3):
        """
        Get the most rewarding behaviors.
        
        Args:
            top_n: Number of top behaviors to return
            
        Returns:
            List of most rewarding behavior labels
        """
        stats = self.get_behavior_statistics()
        
        if "avg_rewards" not in stats or not stats["avg_rewards"]:
            return []
        
        # Sort behaviors by average reward
        sorted_behaviors = sorted(
            stats["avg_rewards"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N
        return [behavior for behavior, _ in sorted_behaviors[:top_n]]
    
    def get_behavior_transitions(self):
        """Get behavior transition probabilities"""
        transitions = {}
        
        for from_behavior, targets in self.transition_graph.items():
            total = sum(targets.values())
            transitions[from_behavior] = {
                to_behavior: count / total
                for to_behavior, count in targets.items()
            }
        
        return transitions
    
    def predict_next_behavior(self):
        """Predict next behavior based on transition graph"""
        if not self.current_behavior or self.current_behavior not in self.transition_graph:
            return None
        
        # Get transition probabilities
        transitions = self.transition_graph[self.current_behavior]
        
        if not transitions:
            return None
        
        # Select next behavior probabilistically
        behaviors = list(transitions.keys())
        probabilities = list(transitions.values())
        total = sum(probabilities)
        
        if total == 0:
            return None
        
        normalized_probs = [p / total for p in probabilities]
        
        return random.choices(behaviors, weights=normalized_probs, k=1)[0]

class IntrextroLearningIntegration:
    """
    Integration module for Intrextro Learning.
    
    Combines all components into a unified learning system.
    """
    
    def __init__(self, config=None):
        """
        Initialize integration module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            "feature_dim": 128,
            "memory_size": 10000,
            "curiosity_weight": 0.5,
            "adaptation_rate": 0.1,
            "meta_learning_rate": 0.01,
            "skill_hierarchy_levels": 3,
            "opponent_model_size": 50,
            "exploration_bonus_scale": 0.1,
            "num_modes": 3
        }
        
        # Initialize core learning system
        self.core = IntrextroLearningImplementation(self.config)
        
        # Initialize additional components
        self.quantum_fusion = QuantumInspiredFusion(
            feature_dim=self.config.get("feature_dim", 128),
            num_modes=self.config.get("num_modes", 3)
        )
        
        self.adaptive_exploration = AdaptiveExplorationModule(
            feature_dim=self.config.get("feature_dim", 128)
        )
        
        self.behavior_analyzer = EmergentBehaviorAnalyzer(
            memory_size=1000,
            feature_dim=self.config.get("feature_dim", 128)
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        
        # Integration weights
        self.integration_weights = {
            "core": 0.6,
            "quantum": 0.2,
            "exploration": 0.2
        }
        
        # Current state
        self.current_state = None
        self.current_action = None
    
    def process(self, inputs):
        """
        Process inputs and generate integrated adaptation.
        
        Args:
            inputs: Dictionary containing game state, reward, and other information
            
        Returns:
            Dictionary with adaptation vector and metrics
        """
        # Extract inputs
        game_state = inputs.get("game_state")
        reward = inputs.get("reward")
        opponent_state = inputs.get("opponent_state")
        strategy = inputs.get("strategy")
        
        # Update performance history if reward is provided
        if reward is not None:
            self.performance_history.append(reward)
        
        # Process game state if provided
        if game_state is not None:
            # Process with core system
            core_output = self.core.process(inputs)
            
            # Extract adaptation vector
            core_adaptation = core_output.get("adaptation_vector")
            
            # Generate exploration noise
            exploration_strategy = self.adaptive_exploration.select_exploration_strategy(
                game_state, 
                learning_progress=core_output.get("prediction_error")
            )
            
            exploration_noise = self.adaptive_exploration.generate_exploration_noise(game_state)
            
            # Apply quantum fusion
            if opponent_state is not None and core_adaptation is not None:
                # Prepare inputs for fusion
                fusion_inputs = [
                    game_state,
                    opponent_state,
                    core_adaptation
                ]
                
                # Apply fusion
                quantum_adaptation = self.quantum_fusion.fuse(fusion_inputs)
                
                # Update fusion based on reward
                if reward is not None:
                    self.quantum_fusion.update(fusion_inputs, reward)
            else:
                # Default quantum adaptation
                quantum_adaptation = np.zeros(self.config.get("feature_dim", 128))
            
            # Combine adaptations
            integrated_adaptation = (
                self.integration_weights["core"] * core_adaptation +
                self.integration_weights["quantum"] * quantum_adaptation +
                self.integration_weights["exploration"] * exploration_noise
            )
            
            # Normalize
            norm = np.linalg.norm(integrated_adaptation)
            if norm > 0:
                integrated_adaptation /= norm
            
            # Update behavior analyzer
            self.behavior_analyzer.update(
                game_state, 
                integrated_adaptation, 
                reward
            )
            
            # Update current state and action
            self.current_state = game_state
            self.current_action = integrated_adaptation
            
            # Adapt exploration noise scale
            self.adaptive_exploration.adapt_noise_scale(reward)
            
            # Prepare output
            output = {
                "adaptation_vector": integrated_adaptation,
                "core_metrics": core_output,
                "exploration_strategy": exploration_strategy,
                "quantum_metrics": self.quantum_fusion.get_entanglement_metrics(),
                "behavior_metrics": self.behavior_analyzer.get_behavior_statistics(),
                "dominant_behaviors": self.behavior_analyzer.get_dominant_behaviors(),
                "rewarding_behaviors": self.behavior_analyzer.get_most_rewarding_behaviors()
            }
            
            return output
        
        # Return current adaptation if no game state
        return {
            "adaptation_vector": self.current_action if self.current_action is not None 
                                else np.zeros(self.config.get("feature_dim", 128))
        }
    
    def update_integration_weights(self):
        """Update integration weights based on performance"""
        if len(self.performance_history) < 10:
            return
        
        # Compute recent performance
        recent_performance = sum(self.performance_history[-10:]) / 10
        
        # Adjust weights based on performance
        if recent_performance > 0:
            # If doing well, rely more on core system
            self.integration_weights["core"] = min(0.8, self.integration_weights["core"] + 0.01)
            self.integration_weights["exploration"] = max(0.1, self.integration_weights["exploration"] - 0.005)
        else:
            # If doing poorly, increase exploration
            self.integration_weights["core"] = max(0.4, self.integration_weights["core"] - 0.01)
            self.integration_weights["exploration"] = min(0.4, self.integration_weights["exploration"] + 0.01)
        
        # Ensure quantum weight is complementary
        self.integration_weights["quantum"] = 1.0 - self.integration_weights["core"] - self.integration_weights["exploration"]
    
    def reset(self):
        """Reset the learning system between episodes"""
        # Reset core system
        self.core.reset()
        
        # Reset additional components
        self.adaptive_exploration.noise_scale = 0.2
        
        # Keep behavior analyzer state for continuity
        
        # Update integration weights
        self.update_integration_weights()
        
        logger.info("Intrextro learning integration system reset")
    
    def save_model(self, path):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        model_data = {
            "core": self.core.get_serializable_data(),
            "quantum_fusion": {
                "entanglement": self.quantum_fusion.entanglement.tolist(),
                "mode_weights": self.quantum_fusion.mode_weights.tolist()
            },
            "integration_weights": self.integration_weights,
            "config": self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load configuration
        self.config = model_data.get("config", self.config)
        
        # Reinitialize core with loaded data
        self.core = IntrextroLearningImplementation(self.config)
        self.core.load_serializable_data(model_data.get("core", {}))
        
        # Load quantum fusion data
        if "quantum_fusion" in model_data:
            qf_data = model_data["quantum_fusion"]
            if "entanglement" in qf_data:
                self.quantum_fusion.entanglement = np.array(qf_data["entanglement"])
            if "mode_weights" in qf_data:
                self.quantum_fusion.mode_weights = np.array(qf_data["mode_weights"])
        
        # Load integration weights
        if "integration_weights" in model_data:
            self.integration_weights = model_data["integration_weights"]
        
        logger.info(f"Model loaded from {path}")
    
    def get_performance_metrics(self):
        """Get performance metrics for the system"""
        if not self.performance_history:
            return {"avg_reward": 0.0}
        
        return {
            "avg_reward": sum(self.performance_history) / len(self.performance_history),
            "recent_reward": sum(list(self.performance_history)[-10:]) / min(10, len(self.performance_history)),
            "integration_weights": self.integration_weights,
            "exploration_scale": self.adaptive_exploration.noise_scale
        }

class IntrextroLearningImplementation:
    """
    Core implementation of Intrextro Learning.
    
    Combines intrinsic and extrinsic motivation with adaptive learning.
    """
    
    def __init__(self, config=None):
        """
        Initialize Intrextro Learning system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            "feature_dim": 128,
            "memory_size": 10000,
            "curiosity_weight": 0.5,
            "adaptation_rate": 0.1,
            "meta_learning_rate": 0.01,
            "skill_hierarchy_levels": 3,
            "opponent_model_size": 50,
            "exploration_bonus_scale": 0.1
        }
        
        # Initialize feature dimension
        self.feature_dim = self.config.get("feature_dim", 128)
        
        # Initialize memory
        self.memory = ExperienceMemory(
            capacity=self.config.get("memory_size", 10000),
            feature_dim=self.feature_dim
        )
        
        # Initialize curiosity module
        self.curiosity_module = CuriosityModule(
            feature_dim=self.feature_dim
        )
        
        # Initialize opponent modeler
        self.opponent_modeler = ExpandedOpponentModeler(
            feature_dim=self.feature_dim,
            memory_size=self.config.get("opponent_model_size", 50)
        )
        
        # Initialize meta-learner
        self.meta_learner = MetaLearner(
            num_strategies=5,
            feature_dim=self.feature_dim
        )
        
        # Initialize skill hierarchy
        self.skill_hierarchy = SkillHierarchy(
            levels=self.config.get("skill_hierarchy_levels", 3),
            feature_dim=self.feature_dim
        )
        
        # Learning parameters
        self.curiosity_weight = self.config.get("curiosity_weight", 0.5)
        self.adaptation_rate = self.config.get("adaptation_rate", 0.1)
        self.exploration_bonus_scale = self.config.get("exploration_bonus_scale", 0.1)
        
        # Current adaptation vector
        self.current_adaptation = np.zeros(self.feature_dim)
        
        # Episode counter
        self.episode_count = 0
        
        # Step counter
        self.step_count = 0
        
        # Logger
        self.logger = logging.getLogger("IntrextroLearning")
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def process(self, inputs):
        """
        Process inputs and generate adaptation.
        
        Args:
            inputs: Dictionary containing game state, reward, and other information
            
        Returns:
            Dictionary with adaptation vector and metrics
        """
        # Extract inputs
        game_state = inputs.get("game_state")
        reward = inputs.get("reward")
        opponent_state = inputs.get("opponent_state")
        strategy = inputs.get("strategy")
        
        # Increment step counter
        self.step_count += 1
        
        # Process game state if provided
        if game_state is not None:
            # Convert to numpy array if needed
            if isinstance(game_state, torch.Tensor):
                game_state = game_state.detach().cpu().numpy()
            
            # Add to memory
            if reward is not None:
                self.memory.add(game_state, reward)
            
            # Update opponent model if opponent state is provided
            if opponent_state is not None:
                self.opponent_modeler.update(opponent_state)
            
            # Compute curiosity signal
            curiosity_signal, prediction_error = self.curiosity_module.compute_curiosity(game_state)
            
            # Update meta-learner
            if strategy is not None:
                self.meta_learner.update(game_state, strategy, reward)
            
            # Select strategy using meta-learner
            strategy_vector = self.meta_learner.select_strategy(game_state)
            
            # Update skill hierarchy
            self.skill_hierarchy.update(game_state, reward)
            
            # Get opponent adaptation
            opponent_adaptation = self.opponent_modeler.get_adaptation_strategy()
            
            # Compute intrinsic reward
            intrinsic_reward = self.curiosity_weight * curiosity_signal
            
            # Compute exploration bonus
            exploration_bonus = self.exploration_bonus_scale * np.random.randn(self.feature_dim)
            
            # Combine signals for adaptation
            adaptation = (
                (1.0 - self.curiosity_weight) * opponent_adaptation +  # Extrinsic adaptation
                intrinsic_reward * strategy_vector +                   # Intrinsic adaptation
                exploration_bonus                                      # Exploration
            )
            
            # Normalize adaptation vector
            norm = np.linalg.norm(adaptation)
            if norm > 0:
                adaptation = adaptation / norm
            
            # Update current adaptation with momentum
            self.current_adaptation = (
                (1.0 - self.adaptation_rate) * self.current_adaptation +
                self.adaptation_rate * adaptation
            )
            
            # Prepare output
            output = {
                "adaptation_vector": self.current_adaptation,
                "curiosity_signal": float(curiosity_signal),
                "prediction_error": float(prediction_error),
                "intrinsic_reward": float(intrinsic_reward),
                "opponent_style": self.opponent_modeler.get_dominant_style(),
                "active_skills": self.skill_hierarchy.get_active_skills(),
                "step_count": self.step_count
            }
            
            return output
        
        # Return current adaptation if no game state
        return {
            "adaptation_vector": self.current_adaptation,
            "step_count": self.step_count
        }
    
    def reset(self):
        """Reset the learning system between episodes"""
        # Increment episode counter
        self.episode_count += 1
        
        # Reset meta-learner episode
        self.meta_learner.reset_episode()
        
        # Log episode completion
        self.logger.info(f"Episode {self.episode_count} completed with {self.step_count} steps")
        
        # Reset step counter
        self.step_count = 0
    
    def get_serializable_data(self):
        """Get data that can be serialized for saving"""
        return {
            "memory_samples": self.memory.get_recent_samples(100),
            "curiosity_errors": list(self.curiosity_module.prediction_errors),
            "opponent_style_probs": self.opponent_modeler.style_probabilities.tolist(),
            "meta_learner_strategies": {
                strategy: {str(k): v for k, v in values.items()}
                for strategy, values in self.meta_learner.strategy_values.items()
            },
            "skill_levels": self.skill_hierarchy.get_skill_levels(),
            "current_adaptation": self.current_adaptation.tolist(),
            "episode_count": self.episode_count,
            "step_count": self.step_count
        }
    
    def load_serializable_data(self, data):
        """Load serialized data"""
        if not data:
            return
        
        # Load memory samples
        if "memory_samples" in data:
            for sample in data["memory_samples"]:
                self.memory.add(sample["state"], sample["reward"])
        
        # Load curiosity errors
        if "curiosity_errors" in data:
            self.curiosity_module.prediction_errors = deque(data["curiosity_errors"], maxlen=100)
        
        # Load opponent style probabilities
        if "opponent_style_probs" in data:
            self.opponent_modeler.style_probabilities = np.array(data["opponent_style_probs"])
        
        # Load meta-learner strategies
        if "meta_learner_strategies" in data:
            for strategy, values in data["meta_learner_strategies"].items():
                self.meta_learner.strategy_values[strategy] = {
                    eval(k): v for k, v in values.items()
                }
        
        # Load current adaptation
        if "current_adaptation" in data:
            self.current_adaptation = np.array(data["current_adaptation"])
        
        # Load counters
        if "episode_count" in data:
            self.episode_count = data["episode_count"]
        
        if "step_count" in data:
            self.step_count = data["step_count"]

class ExperienceMemory:
    """
    Memory module for storing and retrieving experiences.
    
    Implements prioritized experience replay.
    """
    
    def __init__(self, capacity=10000, feature_dim=128):
        """
        Initialize experience memory.
        
        Args:
            capacity: Maximum number of experiences to store
            feature_dim: Dimension of feature vectors
        """
        self.capacity = capacity
        self.feature_dim = feature_dim
        
        # Memory buffers
        self.states = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)
        
        # Current position in buffer
        self.position = 0
        
        # Number of experiences in memory
        self.size = 0
        
        # Priority parameters
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.epsilon = 1e-5  # Small constant to avoid zero priority
        
        # Recent high-reward experiences
        self.high_reward_threshold = 0.8
        self.high_reward_experiences = deque(maxlen=100)
    
    def add(self, state, reward):
        """
        Add experience to memory.
        
        Args:
            state: State vector
            reward: Reward value
        """
        # Store experience
        self.states[self.position] = state
        self.rewards[self.position] = reward
        
        # Set initial priority to maximum priority in buffer
        if self.size > 0:
            self.priorities[self.position] = np.max(self.priorities[:self.size])
        else:
            self.priorities[self.position] = 1.0
        
        # Track high-reward experiences
        if reward > self.high_reward_threshold:
            self.high_reward_experiences.append({
                "state": state,
                "reward": reward,
                "position": self.position
            })
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size=32):
        """
        Sample batch of experiences with prioritized replay.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary with states, rewards, indices, and weights
        """
        if self.size == 0:
            return None
        
        # Calculate sampling probabilities
        probs = self.priorities[:self.size] ** self.alpha
        probs /= np.sum(probs)
        
        # Sample indices
        indices = np.random.choice(self.size, size=min(batch_size, self.size), p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= np.max(weights)
        
        return {
            "states": self.states[indices],
            "rewards": self.rewards[indices],
            "indices": indices,
            "weights": weights
        }
    
    def update_priorities(self, indices, errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            errors: TD errors for each experience
        """
        for idx, error in zip(indices, errors):
            if idx < self.size:
                self.priorities[idx] = (abs(error) + self.epsilon) ** self.alpha
    
    def get_high_reward_experiences(self, n=10):
        """
        Get high-reward experiences.
        
        Args:
            n: Number of experiences to return
            
        Returns:
            List of high-reward experiences
        """
        if not self.high_reward_experiences:
            return []
        
        # Sort by reward
        sorted_experiences = sorted(
            self.high_reward_experiences,
            key=lambda x: x["reward"],
            reverse=True
        )
        
        return sorted_experiences[:n]
    
    def get_recent_samples(self, n=100):
        """
        Get most recent samples.
        
        Args:
            n: Number of samples to return
            
        Returns:
            List of recent samples
        """
        if self.size == 0:
            return []
        
        # Calculate indices of most recent samples
        if self.size < self.capacity:
            # Memory not full yet
            indices = np.arange(self.size - 1, max(-1, self.size - n - 1), -1)
        else:
            # Memory is full, need to wrap around
            start_idx = (self.position - 1) % self.capacity
            indices = [(start_idx - i) % self.capacity for i in range(min(n, self.capacity))]
        
        # Extract samples
        samples = []
        for idx in indices:
            samples.append({
                "state": self.states[idx].tolist(),
                "reward": float(self.rewards[idx])
            })
        
        return samples

class CuriosityModule:
    """
    Curiosity module for intrinsic motivation.
    
    Implements prediction-based curiosity.
    """
    
    def __init__(self, feature_dim=128):
        """
        Initialize curiosity module.
        
        Args:
            feature_dim: Dimension of feature vectors
        """
        self.feature_dim = feature_dim
        
        # Prediction model (simplified)
        # In a real implementation, this would be a neural network
        self.prediction_weights = np.random.randn(feature_dim, feature_dim) * 0.1
        
        # Learning rate
        self.learning_rate = 0.01
        
        # Prediction errors
        self.prediction_errors = deque(maxlen=100)
        
        # Novelty threshold
        self.novelty_threshold = 0.5
        
        # Decay factor for curiosity
        self.curiosity_decay = 0.99
        
        # State visitation counts (simplified)
        self.state_counts = {}
    
    def compute_curiosity(self, state):
        """
        Compute curiosity signal for a state.
        
        Args:
            state: State vector
            
        Returns:
            Tuple of (curiosity_signal, prediction_error)
        """
        # Make prediction
        prediction = self._predict_next_state(state)
        
        # Compute prediction error
        prediction_error = np.mean((prediction - state) ** 2)
        
        # Update prediction errors
        self.prediction_errors.append(prediction_error)
        
        # Update prediction model
        self._update_prediction_model(state, prediction, prediction_error)
        
        # Update state visitation count
        state_key = self._get_state_key(state)
        if state_key in self.state_counts:
            self.state_counts[state_key] += 1
        else:
            self.state_counts[state_key] = 1
        
        # Compute curiosity signal
        # Combine prediction error and state novelty
        prediction_curiosity = min(1.0, prediction_error / self.novelty_threshold)
        
        # Count-based novelty
        count = self.state_counts.get(state_key, 0)
        count_curiosity = 1.0 / (1.0 + count)
        
        # Combine both signals
        curiosity_signal = 0.7 * prediction_curiosity + 0.3 * count_curiosity
        
        return curiosity_signal, prediction_error
    
    def _predict_next_state(self, state):
        """
        Predict next state.
        
        Args:
            state: Current state
            
        Returns:
            Predicted next state
        """
        # Simple linear prediction
        prediction = np.dot(state, self.prediction_weights)
        
        # Add noise
        prediction += np.random.randn(self.feature_dim) * 0.01
        
        return prediction
    
    def _update_prediction_model(self, state, prediction, error):
        """
        Update prediction model.
        
        Args:
            state: Current state
            prediction: Predicted next state
            error: Prediction error
        """
        # Simplified update rule
        # In a real implementation, this would use backpropagation
        gradient = np.outer(state, prediction - state)
        self.prediction_weights -= self.learning_rate * gradient
    
    def _get_state_key(self, state):
        """
        Get discrete key for a continuous state.
        
        Args:
            state: State vector
            
        Returns:
            Tuple key for state
        """
        # Discretize state for counting
        # This is a simplified implementation
        discretized = tuple((state * 5).astype(int)[:5])
        return discretized
    
    def get_curiosity_stats(self):
        """Get statistics about curiosity module"""
        if not self.prediction_errors:
            return {"avg_error": 0.0, "max_error": 0.0}
        
        return {
            "avg_error": sum(self.prediction_errors) / len(self.prediction_errors),
            "max_error": max(self.prediction_errors),
            "unique_states": len(self.state_counts)
        }

class ExpandedOpponentModeler:
    """
    Expanded opponent modeling module.
    
    Models opponent behavior and adapts strategies.
    """
    
    def __init__(self, feature_dim=128, memory_size=50):
        """
        Initialize opponent modeler.
        
        Args:
            feature_dim: Dimension of feature vectors
            memory_size: Size of opponent memory
        """
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # Opponent memory
        self.opponent_memory = deque(maxlen=memory_size)
        
        # Opponent styles
        self.styles = [
            "aggressive",
            "defensive",
            "economic",
            "technological",
            "balanced"
        ]
        
        # Style probabilities
        self.style_probabilities = np.ones(len(self.styles)) / len(self.styles)
        
        # Style feature templates (simplified)
        self.style_templates = {
            "aggressive": np.array([0.8, 0.3, 0.2, 0.1, 0.6]),
            "defensive": np.array([0.3, 0.8, 0.4, 0.3, 0.2]),
            "economic": np.array([0.2, 0.4, 0.9, 0.3, 0.1]),
            "technological": np.array([0.1, 0.3, 0.5, 0.9, 0.2]),
            "balanced": np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        }
        
        # Counter strategies (simplified)
        self.counter_strategies = {
            "aggressive": np.array([0.2, 0.8, 0.3, 0.2, 0.5]),
            "defensive": np.array([0.7, 0.2, 0.4, 0.3, 0.4]),
            "economic": np.array([0.8, 0.3, 0.2, 0.4, 0.3]),
            "technological": np.array([0.6, 0.2, 0.7, 0.1, 0.4]),
            "balanced": np.array([0.6, 0.6, 0.6, 0.6, 0.6])
        }
        
        # Learning rate
        self.learning_rate = 0.1
        
        # Adaptation strength
        self.adaptation_strength = 0.5
    
    def update(self, opponent_state):
        """
        Update opponent model with new observation.
        
        Args:
            opponent_state: Opponent state vector
        """
        # Add to memory
        self.opponent_memory.append(opponent_state)
        
        # Update style probabilities
        self._update_style_probabilities()
    
    def _update_style_probabilities(self):
        """Update style probabilities based on observed opponent behavior"""
        if not self.opponent_memory:
            return
        
        # Get recent opponent states
        recent_states = list(self.opponent_memory)[-min(10, len(self.opponent_memory)):]
        
        # Average recent states
        avg_state = np.mean(recent_states, axis=0)
        
        # Extract relevant features (simplified)
        # In a real implementation, this would extract meaningful features
        features = avg_state[:5]
        
        # Compute similarity to each style
        similarities = []
        for style in self.styles:
            template = self.style_templates[style]
            similarity = 1.0 - np.mean(np.abs(features - template))
            similarities.append(max(0.0, similarity))
        
        # Normalize similarities
        total = sum(similarities)
        if total > 0:
            normalized = [s / total for s in similarities]
        else:
            normalized = [1.0 / len(similarities)] * len(similarities)
        
        # Update probabilities with momentum
        self.style_probabilities = (
            (1.0 - self.learning_rate) * self.style_probabilities +
            self.learning_rate * np.array(normalized)
        )
    
    def get_dominant_style(self):
        """Get the most likely opponent style"""
        return self.styles[np.argmax(self.style_probabilities)]
    
    def get_adaptation_strategy(self):
        """Get adaptation strategy based on opponent model"""
        # Compute weighted counter strategy
        adaptation = np.zeros(self.feature_dim)
        
        # Use only the first 5 dimensions for the counter strategy
        for i, style in enumerate(self.styles):
            counter = self.counter_strategies[style]
            adaptation[:5] += self.style_probabilities[i] * counter
        
        # Fill remaining dimensions with small random values
        adaptation[5:] = np.random.randn(self.feature_dim - 5) * 0.1
        
        # Scale adaptation
        adaptation *= self.adaptation_strength
        
        return adaptation
    
    def get_opponent_stats(self):
        """Get statistics about opponent model"""
        return {
            "dominant_style": self.get_dominant_style(),
            "style_probabilities": {
                style: float(prob)
                for style, prob in zip(self.styles, self.style_probabilities)
            },
            "memory_size": len(self.opponent_memory)
        }

class MetaLearner:
    """
    Meta-learning module for strategy selection.
    
    Learns which strategies work best in different situations.
    """
    
    def __init__(self, num_strategies=5, feature_dim=128):
        """
        Initialize meta-learner.
        
        Args:
            num_strategies: Number of strategies
            feature_dim: Dimension of feature vectors
        """
        self.num_strategies = num_strategies
        self.feature_dim = feature_dim
        
        # Strategies
        self.strategies = [
            "rush",
            "expand",
            "tech",
            "defend",
            "harass"
        ]
        
        # Strategy values (state -> value mapping)
        self.strategy_values = {}
        for strategy in self.strategies:
            self.strategy_values[strategy] = {}
        
        # Learning rate
        self.learning_rate = 0.1
        
        # Discount factor
        self.gamma = 0.9
        
        # Exploration factor
        self.epsilon = 0.2
        
        # Current episode memory
        self.episode_memory = []
        
        # Strategy feature templates (simplified)
        self.strategy_templates = {
            "rush": np.array([0.9, 0.1, 0.1, 0.2, 0.7]),
            "expand": np.array([0.2, 0.8, 0.7, 0.1, 0.2]),
            "tech": np.array([0.3, 0.4, 0.2, 0.9, 0.2]),
            "defend": np.array([0.2, 0.3, 0.1, 0.7, 0.7]),
            "harass": np.array([0.7, 0.2, 0.3, 0.3, 0.5])
        }
    
    def update(self, state, strategy, reward):
        """
        Update meta-learner with new observation.
        
        Args:
            state: State vector
            strategy: Strategy used
            reward: Reward received
        """
        # Add to episode memory
        self.episode_memory.append({
            "state": state,
            "strategy": strategy,
            "reward": reward
        })
        
        # Get state key
        state_key = self._get_state_key(state)
        
        # Update strategy value
        if state_key not in self.strategy_values[strategy]:
            self.strategy_values[strategy][state_key] = 0.0
        
        # Update value with reward
        self.strategy_values[strategy][state_key] += self.learning_rate * (
            reward - self.strategy_values[strategy][state_key]
        )
    
    def select_strategy(self, state):
        """
        Select strategy for current state.
        
        Args:
            state: State vector
            
        Returns:
            Strategy feature vector
        """
        # Get state key
        state_key = self._get_state_key(state)
        
        # Exploration
        if np.random.rand() < self.epsilon:
            strategy = random.choice(self.strategies)
        else:
            # Find best strategy for this state
            best_value = float('-inf')
            best_strategy = None
            
            for strategy in self.strategies:
                value = self.strategy_values[strategy].get(state_key, 0.0)
                if value > best_value:
                    best_value = value
                    best_strategy = strategy
            
            strategy = best_strategy or random.choice(self.strategies)
        
        # Convert strategy to feature vector
        strategy_vector = np.zeros(self.feature_dim)
        
        # Use template for first 5 dimensions
        strategy_vector[:5] = self.strategy_templates[strategy]
        
        # Add small random values for remaining dimensions
        strategy_vector[5:] = np.random.randn(self.feature_dim - 5) * 0.1
        
        return strategy_vector
    
    def _get_state_key(self, state):
        """
        Get discrete key for a continuous state.
        
        Args:
            state: State vector
            
        Returns:
            Tuple key for state
        """
        # Discretize state for mapping
        # This is a simplified implementation
        discretized = tuple((state[:5] * 5).astype(int))
        return discretized
    
    def reset_episode(self):
        """Reset episode memory"""
        # Process episode for learning
        self._process_episode()
        
        # Clear memory
        self.episode_memory = []
    
    def _process_episode(self):
        """Process episode for learning"""
        if not self.episode_memory:
            return
        
        # Compute returns
        returns = []
        G = 0
        
        # Process in reverse order
        for experience in reversed(self.episode_memory):
            G = experience["reward"] + self.gamma * G
            returns.insert(0, G)
        
        # Update strategy values with returns
        for i, experience in enumerate(self.episode_memory):
            state_key = self._get_state_key(experience["state"])
            strategy = experience["strategy"]
            
            if state_key not in self.strategy_values[strategy]:
                self.strategy_values[strategy][state_key] = 0.0
            
            # Update with return
            self.strategy_values[strategy][state_key] += self.learning_rate * (
                returns[i] - self.strategy_values[strategy][state_key]
            )
    
    def get_strategy_stats(self):
        """Get statistics about strategies"""
        # Count strategy values
        counts = {strategy: len(values) for strategy, values in self.strategy_values.items()}
        
        # Compute average values
        avg_values = {}
        for strategy, values in self.strategy_values.items():
            if values:
                avg_values[strategy] = sum(values.values()) / len(values)
            else:
                avg_values[strategy] = 0.0
        
        return {
            "strategy_counts": counts,
            "avg_values": avg_values,
            "epsilon": self.epsilon
        }
    
    def adapt_exploration(self, performance):
        """
        Adapt exploration rate based on performance.
        
        Args:
            performance: Performance metric
        """
        # Decrease epsilon if performance is good
        if performance > 0:
            self.epsilon = max(0.05, self.epsilon * 0.99)
        else:
            # Increase epsilon if performance is poor
            self.epsilon = min(0.5, self.epsilon * 1.01)

class SkillHierarchy:
    """
    Skill hierarchy for learning complex behaviors.
    
    Implements hierarchical skill learning and composition.
    """
    
    def __init__(self, levels=3, feature_dim=128):
        """
        Initialize skill hierarchy.
        
        Args:
            levels: Number of hierarchy levels
            feature_dim: Dimension of feature vectors
        """
        self.levels = levels
        self.feature_dim = feature_dim
        
        # Skills at each level
        self.skills = {level: {} for level in range(levels)}
        
        # Skill activation thresholds
        self.activation_thresholds = {level: 0.5 - 0.1 * level for level in range(levels)}
        
        # Skill learning rates
        self.learning_rates = {level: 0.1 / (level + 1) for level in range(levels)}
        
        # Active skills
        self.active_skills = {level: None for level in range(levels)}
        
        # Skill success counters
        self.skill_successes = {level: {} for level in range(levels)}
        self.skill_attempts = {level: {} for level in range(levels)}
    
    def update(self, state, reward):
        """
        Update skill hierarchy with new observation.
        
        Args:
            state: State vector
            reward: Reward value
        """
        # Update skills at each level
        for level in range(self.levels):
            # Get state key for this level
            state_key = self._get_state_key(state, level)
            
            # Initialize skill if not exists
            if state_key not in self.skills[level]:
                self.skills[level][state_key] = np.random.randn(self.feature_dim) * 0.1
                self.skill_successes[level][state_key] = 0
                self.skill_attempts[level][state_key] = 0
            
            # Update skill with reward
            if reward is not None:
                self.skills[level][state_key] += self.learning_rates[level] * reward * np.random.randn(self.feature_dim)
                
                # Update success counters
                self.skill_attempts[level][state_key] += 1
                if reward > 0:
                    self.skill_successes[level][state_key] += 1
            
            # Activate skill if above threshold
            activation = self._compute_activation(state, level)
            if activation > self.activation_thresholds[level]:
                self.active_skills[level] = state_key
            else:
                self.active_skills[level] = None
    
    def _get_state_key(self, state, level):
        """
        Get discrete key for a continuous state at a specific level.
        
        Args:
            state: State vector
            level: Hierarchy level
            
        Returns:
            Tuple key for state
        """
        # Discretize state with coarseness depending on level
        # Higher levels have coarser discretization
        granularity = 5 - level
        discretized = tuple((state[:5] * granularity).astype(int))
        return discretized
    
    def _compute_activation(self, state, level):
        """
        Compute activation level for a state at a specific level.
        
        Args:
            state: State vector
            level: Hierarchy level
            
        Returns:
            Activation value
        """
        # Get state key
        state_key = self._get_state_key(state, level)
        
        # Return 0 if skill doesn't exist
        if state_key not in self.skills[level]:
            return 0.0
        
        # Compute activation based on skill success rate
        attempts = self.skill_attempts[level].get(state_key, 0)
        successes = self.skill_successes[level].get(state_key, 0)
        
        if attempts > 0:
            success_rate = successes / attempts
        else:
            success_rate = 0.0
        
        # Combine with state similarity
        skill_vector = self.skills[level][state_key]
        similarity = np.dot(state, skill_vector) / (np.linalg.norm(state) * np.linalg.norm(skill_vector) + 1e-8)
        
        # Combine success rate and similarity
        activation = 0.7 * success_rate + 0.3 * max(0, similarity)
        
        return activation
    
    def get_active_skills(self):
        """Get currently active skills"""
        return {level: skill for level, skill in self.active_skills.items() if skill is not None}
    
    def get_skill_levels(self):
        """Get skill levels and success rates"""
        skill_levels = {}
        
        for level in range(self.levels):
            level_skills = {}
            
            for skill_key in self.skills[level]:
                attempts = self.skill_attempts[level].get(skill_key, 0)
                successes = self.skill_successes[level].get(skill_key, 0)
                
                if attempts > 0:
                    success_rate = successes / attempts
                else:
                    success_rate = 0.0
                
                # Convert tuple key to string for JSON serialization
                level_skills[str(skill_key)] = {
                    "success_rate": success_rate,
                    "attempts": attempts,
                    "successes": successes
                }
            
            skill_levels[f"level_{level}"] = level_skills
        
        return skill_levels
    
    def get_skill_vector(self, level=None):
        """
        Get skill vector for active skills.
        
        Args:
            level: Specific level to get skill for, or None for all levels
            
        Returns:
            Combined skill vector
        """
        if level is not None:
            # Return skill vector for specific level
            skill_key = self.active_skills[level]
            if skill_key is not None:
                return self.skills[level][skill_key]
            return np.zeros(self.feature_dim)
        
        # Combine skills from all active levels
        skill_vector = np.zeros(self.feature_dim)
        
        for level, skill_key in self.active_skills.items():
            if skill_key is not None:
                # Weight higher levels more
                weight = (level + 1) / self.levels
                skill_vector += weight * self.skills[level][skill_key]
        
        # Normalize
        norm = np.linalg.norm(skill_vector)
        if norm > 0:
            skill_vector /= norm
        
        return skill_vector

def main():
    """Main function for testing the expanded Intrextro Learning implementation"""
    # Initialize the system
    config = {
        "feature_dim": 64,
        "memory_size": 1000,
        "curiosity_weight": 0.6,
        "adaptation_rate": 0.1,
        "meta_learning_rate": 0.01,
        "skill_hierarchy_levels": 3,
        "opponent_model_size": 30,
        "exploration_bonus_scale": 0.2,
        "num_modes": 3
    }
    
    system = IntrextroLearningIntegration(config)
    
    # Simulate some inputs
    for i in range(100):
        # Generate random game state
        game_state = np.random.randn(64) * 0.1 + 0.5
        
        # Generate random opponent state
        opponent_state = np.random.randn(64) * 0.1 + 0.5
        
        # Generate random reward
        reward = np.random.randn() * 0.5
        
        # Process inputs
        inputs = {
            "game_state": game_state,
            "opponent_state": opponent_state,
            "reward": reward,
            "strategy": random.choice(["rush", "expand", "tech", "defend", "harass"])
        }
        
        output = system.process(inputs)
        
        # Print some metrics every 10 steps
        if i % 10 == 0:
            print(f"Step {i}:")
            print(f"  Curiosity signal: {output.get('curiosity_signal', 0):.4f}")
            print(f"  Opponent style: {output.get('opponent_style', 'unknown')}")
            print(f"  Active skills: {len(output.get('active_skills', {}))}")
            print(f"  Adaptation norm: {np.linalg.norm(output.get('adaptation_vector', np.zeros(64))):.4f}")
            print()
    
    # Reset system
    system.reset()
    
    # Print final performance metrics
    metrics = system.get_performance_metrics()
    print("Final performance metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()