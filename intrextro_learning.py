import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass,field
from typing import Any,Dict,List,Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from intrextro_learning import IntrextroLearning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration class
@dataclass
class LearningConfig:
    """Configuration for the learning system."""
    hidden_size: int = 64
    learning_rate: float = 0.001
    memory_size: int = 1000
    batch_size: int = 32
    num_layers: int = 2
    dropout_rate: float = 0.1
    activation: str = "relu"
    optimizer: str = "adam"
    use_cuda: bool = False
    seed: int = 42
    max_epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    layer_sizes: List[int] = field(default_factory=lambda: [128, 64])

    def __post_init__(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def get_device(self) -> torch.device:
        """Get the appropriate device based on configuration."""
        if self.use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

# Utility functions
def fix_dimension_mismatch(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Fix dimension mismatch by padding or truncating the tensor.

    Args:
        tensor: Input tensor
        target_size: Target size for the last dimension

    Returns:
        Tensor with the correct last dimension size
    """
    current_size = tensor.size(-1)

    if current_size == target_size:
        return tensor

    if current_size < target_size:
        # Pad the tensor
        padding_size = target_size - current_size
        if tensor.dim() == 2:
            padding = torch.zeros(tensor.size(0), padding_size)
            return torch.cat([tensor, padding], dim=1)
        elif tensor.dim() == 3:
            padding = torch.zeros(tensor.size(0), tensor.size(1), padding_size)
            return torch.cat([tensor, padding], dim=2)
        else:
            # For other dimensions, reshape, pad, and reshape back
            original_shape = tensor.shape
            reshaped = tensor.reshape(-1, current_size)
            padding = torch.zeros(reshaped.size(0), padding_size)
            padded = torch.cat([reshaped, padding], dim=1)
            new_shape = list(original_shape)
            new_shape[-1] = target_size
            return padded.reshape(new_shape)
    else:
        # Truncate the tensor
        if tensor.dim() == 2:
            return tensor[:, :target_size]
        elif tensor.dim() == 3:
            return tensor[:, :, :target_size]
        else:
            # For other dimensions, reshape, truncate, and reshape back
            original_shape = tensor.shape
            reshaped = tensor.reshape(-1, current_size)
            truncated = reshaped[:, :target_size]
            new_shape = list(original_shape)
            new_shape[-1] = target_size
            return truncated.reshape(new_shape)

def create_activation(name: str) -> nn.Module:
    """
    Create an activation function based on name.

    Args:
        name: Name of the activation function

    Returns:
        PyTorch activation module
    """
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
    }
    return activations.get(name.lower(), nn.ReLU())

def create_optimizer(name: str, parameters, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """
    Create an optimizer based on name.

    Args:
        name: Name of the optimizer
        parameters: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay factor

    Returns:
        PyTorch optimizer
    """
    optimizers = {
        "adam": lambda: torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay),
        "sgd": lambda: torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay),
        "rmsprop": lambda: torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay),
        "adagrad": lambda: torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay),
        "adadelta": lambda: torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay),
    }
    return optimizers.get(name.lower(), optimizers["adam"])()

# Base Core class
class BaseCore:
    """Base class for all learning cores."""

    def __init__(self, config: LearningConfig):
        """
        Initialize the base core.

        Args:
            config: Configuration for the learning core
        """
        self.config = config
        self.device = config.get_device()

    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process input data.

        Args:
            data: Input data to process

        Returns:
            Dictionary containing processing results
        """
        raise NotImplementedError("Subclasses must implement process method")

    def adapt(self, feedback: Dict[str, Any]) -> None:
        """
        Adapt the model based on feedback.

        Args:
            feedback: Dictionary containing feedback information
        """
        raise NotImplementedError("Subclasses must implement adapt method")

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the core.

        Returns:
            Dictionary containing the state
        """
        raise NotImplementedError("Subclasses must implement get_state method")

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: File path to save the model
        """
        raise NotImplementedError("Subclasses must implement save_model method")

    def load_model(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: File path to load the model from
        """
        raise NotImplementedError("Subclasses must implement load_model method")

# Dummy Core for fallback
class DummyCore(BaseCore):
    """Dummy core used as a fallback when other cores fail to initialize."""

    def __init__(self, config: LearningConfig, core_type: str):
        """
        Initialize the dummy core.

        Args:
            config: Configuration for the learning core
            core_type: Type of core this dummy is replacing
        """
        super().__init__(config)
        self.core_type = core_type
        logging.warning(f"Using DummyCore as fallback for {core_type}")

    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process input data with minimal functionality.

        Args:
            data: Input data to process

        Returns:
            Dictionary with basic information
        """
        return {
            "status": "dummy_processed",
            "core_type": self.core_type,
            "data_type": type(data).__name__,
            "timestamp": time.time()
        }

    def adapt(self, feedback: Dict[str, Any]) -> None:
        """
        Pretend to adapt the model.

        Args:
            feedback: Dictionary containing feedback information
        """
        logging.info(f"DummyCore ({self.core_type}) received adaptation feedback: {feedback.keys()}")

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the dummy core.

        Returns:
            Dictionary with basic state information
        """
        return {
            "core_type": self.core_type,
            "status": "dummy",
            "config": self.config.__dict__
        }

    def save_model(self, path: str) -> None:
        """
        Pretend to save the model.

        Args:
            path: File path to save the model
        """
        logging.info(f"DummyCore ({self.core_type}) pretending to save model to {path}")

    def load_model(self, path: str) -> None:
        """
        Pretend to load the model.

        Args:
            path: File path to load the model from
        """
        logging.info(f"DummyCore ({self.core_type}) pretending to load model from {path}")

# Meta Core
class MetaCore(BaseCore):
    """
    Meta-learning core that optimizes learning strategies.
    This core analyzes data and determines the best learning approach.
    """

    def __init__(self, config: LearningConfig):
        """
        Initialize the meta core.

        Args:
            config: Configuration for the learning core
        """
        super().__init__(config)

        # Meta-model for strategy selection
        self.strategy_network = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 64),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 5),  # 5 outputs for 5 core types
            nn.Softmax(dim=1)
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            create_activation(config.activation),
            nn.Linear(128, config.hidden_size),
        ).to(self.device)

        # Optimizer
        self.optimizer = create_optimizer(
            config.optimizer,
            list(self.strategy_network.parameters()) + list(self.feature_extractor.parameters()),
            config.learning_rate,
            config.weight_decay
        )

        # Performance history
        self.performance_history = []

    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process input data to determine optimal learning strategy.

        Args:
            data: Input data to process

        Returns:
            Dictionary with processing results
        """
        try:
            # Convert input to tensor
            input_tensor = self._prepare_input(data)

            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                strategy_scores = self.strategy_network(features)

            return {
                "strategy_scores": strategy_scores.cpu().numpy().tolist(),
                "features": features.cpu().numpy().tolist(),
                "recommended_strategy": int(torch.argmax(strategy_scores).item())
            }
        except Exception as e:
            logging.error(f"Error in MetaCore.process: {e}")
            return {"error": str(e)}

    def optimize_learning(self, data: Any) -> np.ndarray:
        """
        Optimize the learning approach for the given data.

        Args:
            data: Input data to analyze

        Returns:
            NumPy array with scores for each learning strategy
        """
        try:
            # Convert input to tensor
            input_tensor = self._prepare_input(data)

            # Extract features and get strategy scores
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                strategy_scores = self.strategy_network(features)

            return strategy_scores.cpu().numpy()
        except Exception as e:
            logging.error(f"Error in MetaCore.optimize_learning: {e}")
            # Return default scores with slight preference for meta core
            return np.array([0.3, 0.2, 0.2, 0.15, 0.15])

    def _prepare_input(self, data: Any) -> torch.Tensor:
        """
        Prepare input data for processing.

        Args:
            data: Input data in various formats

        Returns:
            Tensor ready for processing
        """
        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data, list):
            tensor = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, dict):
            # Convert dict to list of values
            tensor = torch.tensor(list(data.values()), dtype=torch.float32)
        else:
            # Try to convert to string and then to tensor
            tensor = torch.tensor([ord(c) for c in str(data)], dtype=torch.float32)

        # Ensure correct shape
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # Add batch dimension

        # Fix dimension mismatch
        if tensor.size(1) != self.config.hidden_size:
            tensor = fix_dimension_mismatch(tensor, self.config.hidden_size)

        return tensor.to(self.device)

    def adapt(self, feedback: Dict[str, Any]) -> None:
        """
        Adapt the meta-learning model based on feedback.

        Args:
            feedback: Dictionary containing feedback information
        """
        try:
            if "performance" in feedback and "strategy" in feedback:
                # Extract feedback
                performance = feedback["performance"]
                strategy = feedback["strategy"]

                # Store in history
                self.performance_history.append({
                    "strategy": strategy,
                    "performance": performance,
                    "timestamp": time.time()
                })

                # If we have data to learn from
                if "data" in feedback:
                    input_tensor = self._prepare_input(feedback["data"])

                    # Create target tensor (one-hot encoding of strategy)
                    target = torch.zeros(1, 5)
                    target[0, strategy] = 1.0

                    # Forward pass
                    features = self.feature_extractor(input_tensor)
                    strategy_scores = self.strategy_network(features)

                    # Compute loss (weighted by performance)
                    loss = F.cross_entropy(strategy_scores, target) * (1.0 - performance)

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(self.strategy_network.parameters()) +
                        list(self.feature_extractor.parameters()),
                        self.config.gradient_clip
                    )

                    self.optimizer.step()
            except Exception as e:
            logging.error(f"Error in MetaCore.adapt: {e}")

    def get_state(self) -> Dict[str,Any]:
        """
		Get the current state of the meta core.

		Returns:
			Dictionary containing the state
		"""
        return {
            "strategy_network": self.strategy_network.state_dict(),
            "feature_extractor": self.feature_extractor.state_dict(),
            "performance_history": self.performance_history[-10:],  # Last 10 entries
            "config": self.config.__dict__
        }

    def save_model(self,path: str) -> None:
        """
		Save the meta core model to disk.

		Args:
			path: File path to save the model
		"""
        try:
            state = {
                "strategy_network": self.strategy_network.state_dict(),
                "feature_extractor": self.feature_extractor.state_dict(),
                "performance_history": self.performance_history,
                "config": self.config.__dict__
            }
            torch.save(state,path)
            logging.info(f"MetaCore model saved to {path}")
        except Exception as e:
            logging.error(f"Error saving MetaCore model: {e}")

    def load_model(self,path: str) -> None:
        """
		Load the meta core model from disk.

		Args:
			path: File path to load the model from
		"""
        try:
            state = torch.load(path,map_location=self.device)
            self.strategy_network.load_state_dict(state["strategy_network"])
            self.feature_extractor.load_state_dict(state["feature_extractor"])
            self.performance_history = state.get("performance_history",[])
            logging.info(f"MetaCore model loaded from {path}")
        except Exception as e:
            logging.error(f"Error loading MetaCore model: {e}")


# Deep Core
class DeepCore(BaseCore):
    """
	Deep learning core for pattern recognition and feature extraction.
	This core uses deep neural networks for complex pattern analysis.
	"""

    def __init__(self,config: LearningConfig):
        """
		Initialize the deep core.

		Args:
			config: Configuration for the learning core
		"""
        super().__init__(config)

        # Feature extractor (encoder)
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_size,256),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256,128),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128,64),
        ).to(self.device)

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(64,128),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128,256),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256,config.hidden_size),
        ).to(self.device)

        # Pattern classifier
        self.classifier = nn.Sequential(
            nn.Linear(64,128),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128,10),  # 10 pattern classes
            nn.Softmax(dim=1)
        ).to(self.device)

        # Optimizer
        self.optimizer = create_optimizer(
            config.optimizer,
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifier.parameters()),
            config.learning_rate,
            config.weight_decay
        )

        # Training history
        self.training_history = []

    def process(self,data: Any) -> Dict[str,Any]:
        """
		Process input data using deep learning.

		Args:
			data: Input data to process

		Returns:
			Dictionary with processing results
		"""
        try:
            # Convert input to tensor
            input_tensor = self._prepare_input(data)

            # Process through the model
            with torch.no_grad():
                features = self.encoder(input_tensor)
                reconstruction = self.decoder(features)
                pattern_probs = self.classifier(features)

            return {
                "features": features.cpu().numpy().tolist(),
                "reconstruction": reconstruction.cpu().numpy().tolist(),
                "pattern_probabilities": pattern_probs.cpu().numpy().tolist(),
                "detected_pattern": int(torch.argmax(pattern_probs,dim=1).item())
            }
        except Exception as e:
            logging.error(f"Error in DeepCore.process: {e}")
            return {"error": str(e)}

    def process_patterns(self,data: Any) -> np.ndarray:
        """
		Process input data to extract patterns.

		Args:
			data: Input data to analyze

		Returns:
			NumPy array with extracted patterns
		"""
        try:
            # Convert input to tensor
            input_tensor = self._prepare_input(data)

            # Extract features
            with torch.no_grad():
                features = self.encoder(input_tensor)

            return features.cpu().numpy()
        except Exception as e:
            logging.error(f"Error in DeepCore.process_patterns: {e}")
            return np.zeros((1,64))

    def _prepare_input(self,data: Any) -> torch.Tensor:
        """
		Prepare input data for processing.

		Args:
			data: Input data in various formats

		Returns:
			Tensor ready for processing
		"""
        if isinstance(data,torch.Tensor):
            tensor = data
        elif isinstance(data,np.ndarray):
            tensor = torch.from_numpy(data).float()
        elif isinstance(data,list):
            tensor = torch.tensor(data,dtype=torch.float32)
        elif isinstance(data,dict):
            # Convert dict to list of values
            tensor = torch.tensor(list(data.values()),dtype=torch.float32)
        else:
            # Try to convert to string and then to tensor
            tensor = torch.tensor([ord(c) for c in str(data)],dtype=torch.float32)

        # Ensure correct shape
        if tensor.dim()==1:
            tensor = tensor.unsqueeze(0)  # Add batch dimension

        # Fix dimension mismatch
        if tensor.size(1)!=self.config.hidden_size:
            tensor = fix_dimension_mismatch(tensor,self.config.hidden_size)

        return tensor.to(self.device)

    def adapt(self,feedback: Dict[str,Any]) -> None:
        """
		Adapt the deep learning model based on feedback.

		Args:
			feedback: Dictionary containing feedback information
		"""
        try:
            if "data" in feedback:
                input_tensor = self._prepare_input(feedback["data"])

                # Forward pass
                features = self.encoder(input_tensor)
                reconstruction = self.decoder(features)
                pattern_probs = self.classifier(features)

                # Compute losses
                reconstruction_loss = F.mse_loss(reconstruction,input_tensor)

                # If target pattern is provided
                if "target_pattern" in feedback:
                    target = torch.tensor([feedback["target_pattern"]],device=self.device)
                    classification_loss = F.cross_entropy(pattern_probs,target)
                else:
                    classification_loss = torch.tensor(0.0,device=self.device)

                # Total loss
                total_loss = reconstruction_loss + classification_loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.decoder.parameters()) +
                    list(self.classifier.parameters()),
                    self.config.gradient_clip
                )

                self.optimizer.step()

                # Store training info
                self.training_history.append(
                    {
                        "reconstruction_loss": reconstruction_loss.item(),
                        "classification_loss": classification_loss.item(),
                        "total_loss": total_loss.item(),
                        "timestamp": time.time()
                    })
        except Exception as e:
            logging.error(f"Error in DeepCore.adapt: {e}")

    def get_state(self) -> Dict[str,Any]:
        """
		Get the current state of the deep core.

		Returns:
			Dictionary containing the state
		"""
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "classifier": self.classifier.state_dict(),
            "training_history": self.training_history[-10:],  # Last 10 entries
            "config": self.config.__dict__
        }

    def save_model(self,path: str) -> None:
        """
		Save the deep core model to disk.

		Args:
			path: File path to save the model
		"""
        try:
            state = {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "classifier": self.classifier.state_dict(),
                "training_history": self.training_history,
                "config": self.config.__dict__
            }
            torch.save(state,path)
            logging.info(f"DeepCore model saved to {path}")
        except Exception as e:
            logging.error(f"Error saving DeepCore model: {e}")

    def load_model(self,path: str) -> None:
        """
		Load the deep core model from disk.

		Args:
			path: File path to load the model from
		"""
        try:
            state = torch.load(path,map_location=self.device)
            self.encoder.load_state_dict(state["encoder"])
            self.decoder.load_state_dict(state["decoder"])
            self.classifier.load_state_dict(state["classifier"])
            self.training_history = state.get("training_history",[])
            logging.info(f"DeepCore model loaded from {path}")
        except Exception as e:
            logging.error(f"Error loading DeepCore model: {e}")


# Transfer Core
class TransferCore(BaseCore):
    """
	Transfer learning core for knowledge transfer between domains.
	This core adapts knowledge from a source domain to a target domain.
	"""

    def __init__(self,config: LearningConfig):
        """
		Initialize the transfer core.

		Args:
			config: Configuration for the learning core
		"""
        super().__init__(config)

        # Source domain encoder
        self.source_encoder = nn.Sequential(
            nn.Linear(config.hidden_size,128),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128,64),
        ).to(self.device)

        # Target domain encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(config.hidden_size,128),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128,64),
        ).to(self.device)

        # Shared representation network
        self.shared_network = nn.Sequential(
            nn.Linear(64,128),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128,64),
        ).to(self.device)

        # Domain classifier (for adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(64,32),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(32,1),
            nn.Sigmoid()
        ).to(self.device)

        # Task predictor
        self.task_predictor = nn.Sequential(
            nn.Linear(64,128),
            create_activation(config.activation),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128,config.hidden_size),
        ).to(self.device)

        # Optimizers
        self.encoder_optimizer = create_optimizer(
            config.optimizer,
            list(self.source_encoder.parameters()) +
            list(self.target_encoder.parameters()) +
            list(self.shared_network.parameters()),
            config.learning_rate,
            config.weight_decay
        )

        self.classifier_optimizer = create_optimizer(
            config.optimizer,
            self.domain_classifier.parameters(),
            config.learning_rate,
            config.weight_decay
        )

        self.predictor_optimizer = create_optimizer(
            config.optimizer,
            self.task_predictor.parameters(),
            config.learning_rate,
            config.weight_decay
        )

        # Transfer history
        self.transfer_history = []

    def process(self,data: Any) -> Dict[str,Any]:
        """
		Process input data using transfer learning.

		Args:
			data: Input data to process

		Returns:
			Dictionary with processing results
		"""
        try:
            # Convert input to tensor
            input_tensor = self._prepare_input(data)

            # Process through the model
            with torch.no_grad():
                # Assume it's from target domain
                features = self.target_encoder(input_tensor)
                shared_features = self.shared_network(features)
                prediction = self.task_predictor(shared_features)
                domain_prob = self.domain_classifier(shared_features)

            return {
                "features": features.cpu().numpy().tolist(),
                "shared_features": shared_features.cpu().numpy().tolist(),
                "prediction": prediction.cpu().numpy().tolist(),
                "domain_probability": domain_prob.item()
            }
        except Exception as e:
            logging.error(f"Error in TransferCore.process: {e}")
            return {"error": str(e)}

    def transfer_knowledge(self,source_data: Any,target_data: Any) -> Dict[str,Any]:
        """
		Transfer knowledge from source domain to target domain.

		Args:
			source_data: Data from source domain
			target_data: Data from target domain

		Returns:
			Dictionary with transfer results
		"""
        try:
            # Convert inputs to tensors
            source_tensor = self._prepare_input(source_data)
            target_tensor = self._prepare_input(target_data)

            # Process through the model
            with torch.no_grad():
                # Source domain
                source_features = self.source_encoder(source_tensor)
                source_shared = self.shared_network(source_features)
                source_prediction = self.task_predictor(source_shared)
                source_domain = self.domain_classifier(source
                source_domain = self.domain_classifier(source_shared)

                # Target domain
                target_features = self.target_encoder(target_tensor)
                target_shared = self.shared_network(target_features)
                target_prediction = self.task_predictor(target_shared)
                target_domain = self.domain_classifier(target_shared)

                return {
                    "source_features": source_features.cpu().numpy().tolist(),
                    "target_features": target_features.cpu().numpy().tolist(),
                    "source_prediction": source_prediction.cpu().numpy().tolist(),
                    "target_prediction": target_prediction.cpu().numpy().tolist(),
                    "domain_discrimination": {
                        "source": source_domain.item(),
                        "target": target_domain.item()
                    },
                    "transfer_success": 1.0 - abs(source_domain.item() - target_domain.item())
                }
                except Exception as e:
                logging.error(f"Error in TransferCore.transfer_knowledge: {e}")
                return {"error": str(e)}

        def _prepare_input(self,data: Any) -> torch.Tensor:
            """
			Prepare input data for processing.

			Args:
				data: Input data in various formats

			Returns:
				Tensor ready for processing
			"""
            if isinstance(data,torch.Tensor):
                tensor = data
            elif isinstance(data,np.ndarray):
                tensor = torch.from_numpy(data).float()
            elif isinstance(data,list):
                tensor = torch.tensor(data,dtype=torch.float32)
            elif isinstance(data,dict):
                # Convert dict to list of values
                tensor = torch.tensor(list(data.values()),dtype=torch.float32)
            else:
                # Try to convert to string and then to tensor
                tensor = torch.tensor([ord(c) for c in str(data)],dtype=torch.float32)

            # Ensure correct shape
            if tensor.dim()==1:
                tensor = tensor.unsqueeze(0)  # Add batch dimension

            # Fix dimension mismatch
            if tensor.size(1)!=self.config.hidden_size:
                tensor = fix_dimension_mismatch(tensor,self.config.hidden_size)

            return tensor.to(self.device)

        def adapt(self,feedback: Dict[str,Any]) -> None:
            """
			Adapt the transfer learning model based on feedback.

			Args:
				feedback: Dictionary containing feedback information
			"""
            try:
                if "source_data" in feedback and "target_data" in feedback:
                    source_tensor = self._prepare_input(feedback["source_data"])
                    target_tensor = self._prepare_input(feedback["target_data"])

                    # Create domain labels
                    source_domain_label = torch.ones(source_tensor.size(0),1,device=self.device)
                    target_domain_label = torch.zeros(target_tensor.size(0),1,device=self.device)

                    # Forward pass - Source
                    source_features = self.source_encoder(source_tensor)
                    source_shared = self.shared_network(source_features)
                    source_prediction = self.task_predictor(source_shared)
                    source_domain = self.domain_classifier(source_shared)

                    # Forward pass - Target
                    target_features = self.target_encoder(target_tensor)
                    target_shared = self.shared_network(target_features)
                    target_prediction = self.task_predictor(target_shared)
                    target_domain = self.domain_classifier(target_shared)

                    # Task loss (supervised on source)
                    task_loss = F.mse_loss(source_prediction,source_tensor)

                    # Domain classification loss
                    domain_loss = F.binary_cross_entropy(source_domain,source_domain_label) + \
                                  F.binary_cross_entropy(target_domain,target_domain_label)

                    # Feature alignment loss (minimize domain classification accuracy)
                    alignment_loss = -domain_loss

                    # Update domain classifier
                    self.classifier_optimizer.zero_grad()
                    domain_loss.backward(retain_graph=True)
                    self.classifier_optimizer.step()

                    # Update task predictor
                    self.predictor_optimizer.zero_grad()
                    task_loss.backward(retain_graph=True)
                    self.predictor_optimizer.step()

                    # Update encoders for feature alignment
                    self.encoder_optimizer.zero_grad()
                    alignment_loss.backward()
                    self.encoder_optimizer.step()

                    # Store training info
                    self.transfer_history.append(
                        {
                            "task_loss": task_loss.item(),
                            "domain_loss": domain_loss.item(),
                            "alignment_loss": alignment_loss.item(),
                            "timestamp": time.time()
                        })
            except Exception as e:
                logging.error(f"Error in TransferCore.adapt: {e}")

        def get_state(self) -> Dict[str,Any]:
            """
			Get the current state of the transfer core.

			Returns:
				Dictionary containing the state
			"""
            return {
                "source_encoder": self.source_encoder.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "shared_network": self.shared_network.state_dict(),
                "domain_classifier": self.domain_classifier.state_dict(),
                "task_predictor": self.task_predictor.state_dict(),
                "transfer_history": self.transfer_history[-10:],  # Last 10 entries
                "config": self.config.__dict__
            }

        def save_model(self,path: str) -> None:
            """
			Save the transfer core model to disk.

			Args:
				path: File path to save the model
			"""
            try:
                state = {
                    "source_encoder": self.source_encoder.state_dict(),
                    "target_encoder": self.target_encoder.state_dict(),
                    "shared_network": self.shared_network.state_dict(),
                    "domain_classifier": self.domain_classifier.state_dict(),
                    "task_predictor": self.task_predictor.state_dict(),
                    "transfer_history": self.transfer_history,
                    "config": self.config.__dict__
                }
                torch.save(state,path)
                logging.info(f"TransferCore model saved to {path}")
            except Exception as e:
                logging.error(f"Error saving TransferCore model: {e}")

        def load_model(self,path: str) -> None:
            """
			Load the transfer core model from disk.

			Args:
				path: File path to load the model from
			"""
            try:
                state = torch.load(path,map_location=self.device)
                self.source_encoder.load_state_dict(state["source_encoder"])
                self.target_encoder.load_state_dict(state["target_encoder"])
                self.shared_network.load_state_dict(state["shared_network"])
                self.domain_classifier.load_state_dict(state["domain_classifier"])
                self.task_predictor.load_state_dict(state["task_predictor"])
                self.transfer_history = state.get("transfer_history",[])
                logging.info(f"TransferCore model loaded from {path}")
            except Exception as e:
                logging.error(f"Error loading TransferCore model: {e}")

        # RL Core
        class RLCore(BaseCore):
            """
			Reinforcement learning core for decision making and optimization.
			This core learns optimal actions through interaction with an environment.
			"""

            def __init__(self,config: LearningConfig):
                """
				Initialize the RL core.

				Args:
					config: Configuration for the learning core
				"""
                super().__init__(config)

                # Policy network (actor)
                self.policy_network = nn.Sequential(
                    nn.Linear(config.hidden_size,128),
                    create_activation(config.activation),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(128,64),
                    create_activation(config.activation),
                    nn.Linear(64,config.hidden_size),
                    nn.Softmax(dim=1)
                ).to(self.device)

                # Value network (critic)
                self.value_network = nn.Sequential(
                    nn.Linear(config.hidden_size,128),
                    create_activation(config.activation),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(128,64),
                    create_activation(config.activation),
                    nn.Linear(64,1)
                ).to(self.device)

                # Action space
                self.action_space = torch.arange(config.hidden_size,dtype=torch.float32).to(self.device)

                # Optimizer
                self.optimizer = create_optimizer(
                    config.optimizer,
                    list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                    config.learning_rate,
                    config.weight_decay
                )

                # Experience buffer
                self.experience_buffer = deque(maxlen=config.memory_size)

            def process(self,data: Any) -> Dict[str,Any]:
                """
				Process input data using reinforcement learning.

				Args:
					data: Input data to process (state)

				Returns:
					Dictionary with processing results
				"""
                try:
                    # Convert input to tensor (state)
                    state_tensor = self._process_state(data)

                    # Select action based on policy
                    action = self._select_action(state_tensor)

                    # Estimate value
                    value = self._estimate_value(state_tensor)

                    return {
                        "action": action,
                        "value": value,
                        "state": state_tensor.cpu().numpy().tolist()
                    }
                except Exception as e:
                    logging.error(f"Error in RLCore.process: {e}")
                    return {"error": str(e)}

            def _select_action(self,state: torch.Tensor) -> np.ndarray:
                """
				Select an action based on the current state.

				Args:
					state: Current state tensor

				Returns:
					Selected action
				"""
                try:
                    # Ensure state has correct dimensions
                    if state.dim()==1:
                        state = state.unsqueeze(0)  # Add batch dimension

                    # Fix dimension mismatch if needed
                    if state.size(1)!=self.config.hidden_size:
                        state = fix_dimension_mismatch(state,self.config.hidden_size)

                    with torch.no_grad():
                        action_probs = self.policy_network(state)
                        # Sample action from probability distribution
                        action = torch.multinomial(action_probs,1).item()

                    return action
                except Exception as e:
                    logging.warning(f"Error selecting action: {e}")
                    # Return a random action as fallback
                    return np.random.randint(0,self.config.hidden_size)

            def _estimate_value(self,state: torch.Tensor) -> float:
                """
				Estimate the value of the current state.

				Args:
					state: Current state tensor

				Returns:
					Estimated value
				"""
                try:
                    # Ensure state has correct dimensions
                    if state.dim()==1:
                        state = state.unsqueeze(0)  # Add batch dimension

                    # Fix dimension mismatch if needed
                    if state.size(1)!=self.config.hidden_size:
                        state = fix_dimension_mismatch(state,self.config.hidden_size)

                    with torch.no_grad():
                        value = self.value_network(state)

                    return value.item()
                except Exception as e:
                    logging.warning(f"Error estimating value: {e}")
                    return 0.0

            def optimize_actions(self,state: Dict) -> Dict:
                """
				Optimize actions based on the provided state.

				Args:
					state: Current state dictionary

				Returns:
					Dictionary with optimized actions
				"""
                state_tensor = self._process_state(state)

                with torch.no_grad():
                    action_probs = self.policy_network(state_tensor)

                    if action_probs.dim() > 2:
                        action_probs = action_probs.reshape(-1,action_probs.size(-1))

                    if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                        action_probs = torch.where(
                            torch.isnan(action_probs) | torch.isinf(action_probs),
                            torch.tensor(1e-6,device=self.device),
                            action_probs
                        )
                        action_probs = action_probs / action_probs.sum(dim=-1,keepdim=True)

                    value = self.value_network(state_tensor)

                    if value.dim() > 2:
                        value = value.reshape(-1,value.size(-1))

                    try:
                        selected_action = torch.multinomial(action_probs,1)
                    except RuntimeError as e:
                        logging.warning(f"Multinomial sampling failed: {e}. Using argmax instead.")
                        selected_action = torch.argmax(action_probs,dim=-1,keepdim=True)

                    return {
                        "action": selected_action.item() if selected_action.numel()==1 else selected_action[0].item(),
                        "value": value.item() if value.numel()==1 else value[0].item(),
                        "action_probs": action_probs.detach().cpu().numpy().tolist()
                    }

            def _process_state(self,state: Dict) -> torch.Tensor:
                """
				Process the input state to ensure it is a tensor of the correct shape.

				Args:
					state: Input state in various formats

				Returns:
					Processed state tensor
				"""
                if isinstance(state,dict):
                    state_values = np.array(list(state.values()))
                    state_tensor = torch.FloatTensor(state_values)
                elif isinstance(state,np.ndarray):
                    state_tensor = torch.FloatTensor(state)
                else:
                    state_tensor = torch.FloatTensor(state)

                if state_tensor.dim()==1:
                    state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

                if state_tensor.size(1)!=self.config.hidden_size:
                    state_tensor = fix_dimension_mismatch(state_tensor,self.config.hidden_size)

                return state_tensor.to(self.device)

            def adapt(self,feedback: Dict[str,Any]) -> None:
                """
				Adapt the model based on external feedback.

				Args:
					feedback: Dictionary containing feedback information
				"""
                if "reward" in feedback:
                    self._update_policy(feedback)

            def _update_policy(self,feedback: Dict[str,Any]) -> None:
                """
				Update the policy based on the feedback provided.

				Args:
					feedback: Dictionary containing feedback information
				"""
                try:
                    reward = torch.tensor(feedback["reward"],device=self.device)
                    state = self._process_state(feedback["state"])

                    # Get current value and action probabilities
                    value = self.value_network(state)
                    action_probs = self.policy_network(state)

                    # Compute losses
                    # Advantage = reward - value
                    advantage = reward - value.detach()

                    # Get the action that was taken
                    action = torch.tensor(feedback["action"],device=self.device).long()

                    # Compute policy loss (policy gradient)
                    policy_loss = -torch.log(action_probs.squeeze(0)[action]) * advantage

                    # Compute value loss (MSE)
                    value_loss = F.mse_loss(value,reward)

                    # Total loss
                    total_loss = policy_loss + value_loss

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    total_loss.backward()

                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                        self.config.gradient_clip
                    )

                    self.optimizer.step()

                    # Store experience
                    self.experience_buffer.append(
                        {
                            "state": state.cpu().numpy(),
                            "action": action.item(),
                            "reward": reward.item(),
                            "value": value.item(),
                            "advantage": advantage.item(),
                            "policy_loss": policy_loss.item(),
                            "value_loss": value_loss.item(),
                            "total_loss": total_loss.item(),
                            "timestamp": time.time()
                        })
                except Exception as e:
                    logging.error(f"Error in RLCore._update_policy: {e}")

                def get_state(self) -> Dict[str,Any]:
                    """
					Get the current state of the RL core.

					Returns:
						Dictionary containing the state
					"""
                    return {
                        "policy_network": self.policy_network.state_dict(),
                        "value_network": self.value_network.state_dict(),
                        "experience_buffer": list(self.experience_buffer)[-10:],  # Last 10 entries
                        "config": self.config.__dict__
                    }

                def save_model(self,path: str) -> None:
                    """
					Save the RL core model to disk.

					Args:
						path: File path to save the model
					"""
                    try:
                        state = {
                            "policy_network": self.policy_network.state_dict(),
                            "value_network": self.value_network.state_dict(),
                            "experience_buffer": list(self.experience_buffer),
                            "config": self.config.__dict__
                        }
                        torch.save(state,path)
                        logging.info(f"RLCore model saved to {path}")
                    except Exception as e:
                        logging.error(f"Error saving RLCore model: {e}")

                def load_model(self,path: str) -> None:
                    """
					Load the RL core model from disk.

					Args:
						path: File path to load the model from
					"""
                    try:
                        state = torch.load(path,map_location=self.device)
                        self.policy_network.load_state_dict(state["policy_network"])
                        self.value_network.load_state_dict(state["value_network"])
                        self.experience_buffer = deque(state.get("experience_buffer",[]),maxlen=self.config.memory_size)
                        logging.info(f"RLCore model loaded from {path}")
                    except Exception as e:
                        logging.error(f"Error loading RLCore model: {e}")

            # Evolutionary Core
            class EvolutionaryCore(BaseCore):
                """
				Evolutionary learning core for population-based optimization.
				This core uses evolutionary algorithms to evolve solutions.
				"""

                def __init__(self,config: LearningConfig):
                    """
					Initialize the evolutionary core.

					Args:
						config: Configuration for the learning core
					"""
                    super().__init__(config)

                    # Population size
                    self.population_size = 50

                    # Mutation rate
                    self.mutation_rate = 0.1

                    # Crossover rate
                    self.crossover_rate = 0.7

                    # Number of elite individuals
                    self.num_elite = 5

                    # Initialize population
                    self.population = self._initialize_population()

                    # Fitness scores
                    self.fitness_scores = torch.zeros(self.population_size,device=self.device)

                    # Best solution
                    self.best_solution = None
                    self.best_fitness = float('-inf')

                    # Evolution history
                    self.evolution_history = []

                def _initialize_population(self) -> torch.Tensor:
                    """
					Initialize the population with random solutions.

					Returns:
						Tensor containing the population
					"""
                    return torch.randn(self.population_size,self.config.hidden_size,device=self.device)

                def process(self,data: Any) -> Dict[str,Any]:
                    """
					Process input data using evolutionary algorithms.

					Args:
						data: Input data to process

					Returns:
						Dictionary with processing results
					"""
                    try:
                        # Convert input to tensor
                        input_tensor = self._prepare_input(data)

                        # Find the best solution for the input
                        solution,fitness = self._find_best_solution(input_tensor)

                        return {
                            "solution": solution.cpu().numpy().tolist(),
                            "fitness": fitness.item(),
                            "best_solution": self.best_solution.cpu().numpy().tolist() if self.best_solution is not None else None,
                            "best_fitness": self.best_fitness
                        }
                    except Exception as e:
                        logging.error(f"Error in EvolutionaryCore.process: {e}")
                        return {"error": str(e)}

                def _find_best_solution(self,input_tensor: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
                    """
					Find the best solution for the given input.

					Args:
						input_tensor: Input tensor

					Returns:
						Tuple of (best solution, fitness)
					"""
                    # Calculate fitness for each individual in the population
                    fitness_scores = self._calculate_fitness(input_tensor)

                    # Get the best individual
                    best_idx = torch.argmax(fitness_scores)
                    best_individual = self.population[best_idx]
                    best_fitness = fitness_scores[best_idx]

                    return best_individual,best_fitness

                def _calculate_fitness(self,input_tensor: torch.Tensor) -> torch.Tensor:
                    """
					Calculate fitness scores for the population.

					Args:
						input_tensor: Input tensor

					Returns:
						Tensor of fitness scores
					"""
                    # Simple fitness function: negative mean squared error
                    # Higher is better
                    fitness_scores = -torch.mean((self.population - input_tensor.unsqueeze(0)) ** 2,dim=1)
                    return fitness_scores

                def _prepare_input(self,data: Any) -> torch.Tensor:
                    """
					Prepare input data for processing.

					Args:
						data: Input data in various formats

					Returns:
						Tensor ready for processing
					"""
                    if isinstance(data,torch.Tensor):
                        tensor = data
                    elif isinstance(data,np.ndarray):
                        tensor = torch.from_numpy(data).float()
                    elif isinstance(data,list):
                        tensor = torch.tensor(data,dtype=torch.float32)
                    elif isinstance(data,dict):
                        # Convert dict to list of values
                        tensor = torch.tensor(list(data.values()),dtype=torch.float32)
                    else:
                        # Try to convert to string and then to tensor
                        tensor = torch.tensor([ord(c) for c in str(data)],dtype=torch.float32)

                    # Ensure correct shape
                    if tensor.dim()==1:
                        tensor = tensor.unsqueeze(0)  # Add batch dimension

                    # Fix dimension mismatch
                    if tensor.size(1)!=self.config.hidden_size:
                        tensor = fix_dimension_mismatch(tensor,self.config.hidden_size)

                    return tensor.squeeze(0).to(self.device)  # Remove batch dimension

                def adapt(self,feedback: Dict[str,Any]) -> None:
                    """
					Adapt the evolutionary model based on feedback.

					Args:
						feedback: Dictionary containing feedback information
					"""
                    try:
                        if "fitness_scores" in feedback:
                            # Update fitness scores
                            fitness_scores = torch.tensor(feedback["fitness_scores"],device=self.device)
                            self.fitness_scores = fitness_scores

                            # Evolve the population
                            self._evolve_population()

                            # Update best solution
                            best_idx = torch.argmax(self.fitness_scores)
                            current_best = self.population[best_idx]
                            current_best_fitness = self.fitness_scores[best_idx].item()

                            if current_best_fitness > self.best_fitness:
                                self.best_solution = current_best.clone()
                                self.best_fitness = current_best_fitness

                            # Store evolution info
                            self.evolution_history.append(
                                {
                                    "mean_fitness": torch.mean(self.fitness_scores).item(),
                                    "max_fitness": torch.max(self.fitness_scores).item(),
                                    "min_fitness": torch.min(self.fitness_scores).item(),
                                    "best_fitness": self.best_fitness,
                                    "timestamp": time.time()
                                })
                        elif "target" in feedback and "data" in feedback:
                            # Calculate fitness based on target
                            input_tensor = self._prepare_input(feedback["data"])
                            target_tensor = self._prepare_input(feedback["target"])

                            # Calculate fitness as negative distance to target
                            self.fitness_scores = -torch.mean((self.population - target_tensor.unsqueeze(0)) ** 2,dim=1)

                            # Evolve the population
                            self._evolve_population()

                            # Update best solution
                            best_idx = torch.argmax(self.fitness_scores)
                            current_best = self.population[best_idx]
                            current_best_fitness = self.fitness_scores[best_idx].item()

                            if current_best_fitness > self.best_fitness:
                                self.best_solution = current_best.clone()
                                self.best_fitness = current_best_fitness

                            # Store evolution info
                            self.evolution_history.append(
                                {
                                    "mean_fitness": torch.mean(self.fitness_scores).item(),
                                    "max_fitness": torch.max(self.fitness_scores).item(),
                                    "min_fitness": torch.min(self.fitness_scores).item(),
                                    "best_fitness": self.best_fitness,
                                    "timestamp": time.time()
                                })
                    except Exception as e:
                        logging.error(f"Error in EvolutionaryCore.adapt: {e}")

                def _evolve_population(self) -> None:
                    """
					Evolve the population using selection, crossover, and mutation.
					"""
                    try:
                        # Selection (tournament selection)
                        selected_indices = self._tournament_selection()
                        selected_population = self.population[selected_indices]

                        # Elitism: keep the best individuals
                        elite_indices = torch.argsort(self.fitness_scores,descending=True)[:self.num_elite]
                        elite_individuals = self.population[elite_indices]

                        # Create new population
                        new_population = torch.zeros_like(self.population)

                        # Add elite individuals
                        new_population[:self.num_elite] = elite_individuals

                        # Crossover and mutation for the rest
                        for i in range(self.num_elite,self.population_size,2):
                            if i + 1 < self.population_size:
                                # Select two parents
                                parent1 = selected_population[i - self.num_elite]
                                parent2 = selected_population[i + 1 - self.num_elite]

                                # Crossover
                                if torch.rand(1).item() < self.crossover_rate:
                                    child1,child2 = self._crossover(parent1,parent2)
                                else:
                                    child1,child2 = parent1.clone(),parent2.clone()

                                # Mutation
                                child1 = self._mutate(child1)
                                child2 = self._mutate(child2)

                                # Add to new population
                                new_population[i] = child1
                                new_population[i + 1] = child2
                            else:
                                # If odd population size, just add the last parent with mutation
                                parent = selected_population[i - self.num_elite]
                                child = self._mutate(parent.clone())
                                new_population[i] = child

                        # Update population
                        self.population = new_population
                    except Exception as e:
                        logging.error(f"Error in EvolutionaryCore._evolve_population: {e}")

                def _tournament_selection(self,tournament_size: int = 3) -> torch.Tensor:
                    """
					Perform tournament selection.

					Args:
						tournament_size: Size of each tournament

					Returns:
						Indices of selected individuals
					"""
                    selected_indices = []

                    for _ in range(self.population_size - self.num_elite):
                        # Randomly select individuals for the tournament
                        tournament_indices = torch.randint(0,self.population_size,(tournament_size,))
                        tournament_fitness = self.fitness_scores[tournament_indices]

                        # Select the winner (highest fitness)
                        winner_idx = tournament_indices[torch.argmax(tournament_fitness)]
                        selected_indices.append(winner_idx)

                    return torch.tensor(selected_indices,device=self.device)

                def _crossover(self,parent1: torch.Tensor,parent2: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
                    """
					Perform crossover between two parents.

					Args:
						parent1: First parent
						parent2: Second parent

					Returns:
						Two children after crossover
					"""
                    # Single-point crossover
                    crossover_point = torch.randint(1,self.config.hidden_size - 1,(1,)).item()

                    child1 = torch.cat([parent1[:crossover_point],parent2[crossover_point:]])
                    child2 = torch.cat([parent2[:crossover_point],parent1[crossover_point:]])

                    return child1,child2

                def _mutate(self,individual: torch.Tensor) -> torch.Tensor:
                    """
					Perform mutation on an individual.

					Args:
						individual: Individual to mutate

					Returns:
						Mutated individual
					"""
                    # Generate mutation mask
                    mutation_mask = torch.rand_like(individual) < self.mutation_rate

                    # Generate random mutations
                    mutations = torch.randn_like(individual)

                    # Apply mutations
                    mutated_individual = torch.where(mutation_mask,individual + mutations,individual)

                    return mutated_individual

                def get_state(self) -> Dict[str,Any]:
                    """
					Get the current state of the evolutionary core.

					Returns:
						Dictionary containing the state
					"""
                    return {
                        "population": self.population.cpu().numpy().tolist(),
                        "fitness_scores": self.fitness_scores.cpu().numpy().tolist(),
                        "best_solution": self.best_solution.cpu().numpy().tolist() if self.best_solution is not None else None,
                        "best_fitness": self.best_fitness,
                        "evolution_history": self.evolution_history[-10:],  # Last 10 entries
                        "config": self.config.__dict__
                    }

                    def save_model(self,path: str) -> None:
                        """
						Save the evolutionary core model to disk.

						Args:
							path: File path to save the model
						"""
                        try:
                            state = {
                                "population": self.population.cpu().numpy(),
                                "fitness_scores": self.fitness_scores.cpu().numpy(),
                                "best_solution": self.best_solution.cpu().numpy() if self.best_solution is not None else None,
                                "best_fitness": self.best_fitness,
                                "evolution_history": self.evolution_history,
                                "config": self.config.__dict__
                            }
                            torch.save(state,path)
                            logging.info(f"EvolutionaryCore model saved to {path}")
                        except Exception as e:
                            logging.error(f"Error saving EvolutionaryCore model: {e}")

                    def load_model(self,path: str) -> None:
                        """
						Load the evolutionary core model from disk.

						Args:
							path: File path to load the model from
						"""
                        try:
                            state = torch.load(path,map_location=self.device)
                            self.population = torch.tensor(state["population"],device=self.device)
                            self.fitness_scores = torch.tensor(state["fitness_scores"],device=self.device)
                            self.best_solution = torch.tensor(state["best_solution"],device=self.device) if state[
                                                                                                            "best_fitness": self.best_fitness,
                                                                                                            "evolution_history": self.evolution_history[
                                                                                                                                 -10:],
                                                                                                            # Last 10 entries
                                                                                                            "config": self.config.__dict__
                            ]

                            def save_model(self,path: str) -> None:
                                """
								Save the evolutionary core model to disk.

								Args:
									path: File path to save the model
								"""
                                try:
                                    state = {
                                        "population": self.population.cpu().numpy(),
                                        "fitness_scores": self.fitness_scores.cpu().numpy(),
                                        "best_solution": self.best_solution.cpu().numpy() if self.best_solution is not None else None,
                                        "best_fitness": self.best_fitness,
                                        "evolution_history": self.evolution_history,
                                        "config": self.config.__dict__
                                    }
                                    torch.save(state,path)
                                    logging.info(f"EvolutionaryCore model saved to {path}")
                                except Exception as e:
                                    logging.error(f"Error saving EvolutionaryCore model: {e}")

                            def load_model(self,path: str) -> None:
                                """
								Load the evolutionary core model from disk.

								Args:
									path: File path to load the model from
								"""
                                try:
                                    state = torch.load(path,map_location=self.device)
                                    self.population = torch.tensor(state["population"],device=self.device)
                                    self.fitness_scores = torch.tensor(state["fitness_scores"],device=self.device)
                                    self.best_solution = torch.tensor(state["best_solution"],device=self.device) if \
                                    state["best_solution"] is not None else None
                                    self.best_fitness = state["best_fitness"]
                                    self.evolution_history = state.get("evolution_history",[])
                                    logging.info(f"EvolutionaryCore model loaded from {path}")
                                except Exception as e:
                                    logging.error(f"Error loading EvolutionaryCore model: {e}")

                            # Main IntrextroLearning System

                        class IntrextroLearning:
                            """
							Main learning system that integrates all cores.
							This system coordinates the different learning cores and manages their interactions.
							"""

                            def __init__(self,config: LearningConfig = None):
                                """
								Initialize the learning system.

								Args:
									config: Configuration for the learning system
								"""
                                self.config = config or LearningConfig()
                                self.device = self.config.get_device()

                                # Initialize cores
                                self.cores = self._initialize_cores()

                                # Core weights (importance of each core)
                                self.core_weights = torch.ones(len(self.cores),device=self.device) / len(self.cores)

                                # System state
                                self.state = {
                                    "iteration": 0,
                                    "last_processed": None,
                                    "performance_history": [],
                                    "active_core": None
                                }

                            def _initialize_cores(self) -> Dict[str,BaseCore]:
                                """
								Initialize all learning cores.

								Returns:
									Dictionary of initialized cores
								"""
                                cores = {}

                                # Initialize each core with error handling
                                core_classes = {
                                    "meta": MetaCore,
                                    "deep": DeepCore,
                                    "transfer": TransferCore,
                                    "rl": RLCore,
                                    "evolutionary": EvolutionaryCore
                                }

                                for name,core_class in core_classes.items():
                                    try:
                                        cores[name] = core_class(self.config)
                                        logging.info(f"Initialized {name} core")
                                    except Exception as e:
                                        logging.error(f"Error initializing {name} core: {e}")
                                        cores[name] = DummyCore(self.config,name)

                                return cores

                            def process(self,data: Any) -> Dict[str,Any]:
                                """
								Process input data using the learning system.

								Args:
									data: Input data to process

								Returns:
									Dictionary with processing results
								"""
                                try:
                                    # Update iteration counter
                                    self.state["iteration"] += 1

                                    # Determine which core to use
                                    core_name: str
                                    core_name,core_scores = self._select_core(data)
                                    self.state["active_core"] = core_name

                                    # Process with selected core
                                    result = self.cores[core_name].process(data)

                                    # Add metadata
                                    result.update(
                                        {
                                            "core": core_name,
                                            "core_scores": core_scores,
                                            "iteration": self.state["iteration"],
                                            "timestamp": time.time()
                                        })

                                    # Store last processed data
                                    self.state["last_processed"] = dict(data=data,result=result,core=core_name)

                                    return result
                                except Exception as e:
                                    logging.error(f"Error in IntrextroLearning.process: {e}")
                                    return {"error": str(e)}

                            def _select_core(self,data: Any) -> Tuple[str,List[float]]:
                                """
								Select the most appropriate core for the given data.

								Args:
									data: Input data

								Returns:
									Tuple of (selected core name, core scores)
								"""
                                try:
                                    # Use meta core to determine the best strategy
                                    meta_result = self.cores["meta"].optimize_learning(data)

                                    # Convert to list and normalize if needed
                                    if isinstance(meta_result,np.ndarray):
                                        core_scores = meta_result.tolist()
                                    else:
                                        core_scores = meta_result

                                    # Get core names
                                    core_names = list(self.cores.keys())

                                    # Select the core with the highest score
                                    selected_idx = np.argmax(core_scores)
                                    selected_core = core_names[selected_idx]

                                    return selected_core,core_scores
                                except Exception as e:
                                    logging.error(f"Error selecting core: {e}")
                                    # Default to deep core
                                    return "deep",[0.2,0.4,0.2,0.1,0.1]

                            def adapt(self,feedback: Dict[str,Any]) -> None:
                                """
								Adapt the learning system based on feedback.

								Args:
									feedback: Dictionary containing feedback information
								"""
                                try:
                                    # Store performance
                                    if "performance" in feedback:
                                        self.state["performance_history"].append(
                                            {
                                                "performance": feedback["performance"],
                                                "iteration": self.state["iteration"],
                                                "timestamp": time.time()
                                            })

                                    # If specific core is targeted
                                    if "core" in feedback:
                                        core_name = feedback["core"]
                                        if core_name in self.cores:
                                            self.cores[core_name].adapt(feedback)
                                    else:
                                        # Adapt all cores with the feedback
                                        for core_name,core in self.cores.items():
                                            core.adapt(feedback)

                                    # Adapt meta core with information about which core was used
                                    if self.state["active_core"] is not None:
                                        core_idx = list(self.cores.keys()).index(self.state["active_core"])
                                        meta_feedback = {
                                            "strategy": core_idx,
                                            "performance": feedback.get("performance",0.5),
                                            "data": feedback.get(
                                                "data",self.state["last_processed"]["data"] if self.state[
                                                    "last_processed"] else None)
                                        }
                                        self.cores["meta"].adapt(meta_feedback)
                                except Exception as e:
                                    logging.error(f"Error in IntrextroLearning.adapt: {e}")

                            def get_state(self) -> Dict[str,Any]:
                                """
								Get the current state of the learning system.

								Returns:
									Dictionary containing the state
								"""
                                core_states = {}
                                for name,core in self.cores.items():
                                    try:
                                        core_states[name] = core.get_state()
                                    except Exception as e:
                                        logging.error(f"Error getting state for {name} core: {e}")
                                        core_states[name] = {"error": str(e)}

                                return {
                                    "iteration": self.state["iteration"],
                                    "active_core": self.state["active_core"],
                                    "performance_history": self.state["performance_history"][-10:],  # Last 10 entries
                                    "core_weights": self.core_weights.cpu().numpy().tolist(),
                                    "cores": core_states,
                                    "config": self.config.__dict__
                                }

                            def save_model(self,path: str) -> None:
                                """
								Save the learning system to disk.

								Args:
									path: Directory path to save the models
								"""
                                try:
                                    # Create directory if it doesn't exist
                                    os.makedirs(path,exist_ok=True)

                                    # Save each core
                                    for name,core in self.cores.items():
                                        core_path = os.path.join(path,f"{name}_core.pt")
                                        try:
                                            core.save_model(core_path)
                                        except Exception as e:
                                            logging.error(f"Error saving {name} core: {e}")

                                    # Save system state
                                    system_state = {
                                        "iteration": self.state["iteration"],
                                        "performance_history": self.state["performance_history"],
                                        "active_core": self.state["active_core"],
                                        "core_weights": self.core_weights.cpu().numpy(),
                                        "config": self.config.__dict__
                                    }

                                    system_path = os.path.join(path,"system_state.pt")
                                    torch.save(system_state,system_path)

                                    logging.info(f"IntrextroLearning system saved to {path}")
                                except Exception as e:
                                    logging.error(f"Error saving IntrextroLearning system: {e}")

                            def load_model(self,path: str) -> None:
                                """
								Load the learning system from disk.

								Args:
									path: Directory path to load the models from
								"""
                                try:
                                    # Load each core
                                    for name,core in self.cores.items():
                                        core_path = os.path.join(path,f"{name}_core.pt")
                                        if os.path.exists(core_path):
                                            try:
                                                core.load_model(core_path)
                                            except Exception as e:
                                                logging.error(f"Error loading {name} core: {e}")

                                    # Load system state
                                    system_path = os.path.join(path,"system_state.pt")
                                    if os.path.exists(system_path):
                                        system_state = torch.load(system_path,map_location=self.device)

                                        self.state["iteration"] = system_state.get("iteration",0)
                                        self.state["performance_history"] = system_state.get("performance_history",[])
                                        self.state["active_core"] = system_state.get("active_core",None)

                                        if "core_weights" in system_state:
                                            self.core_weights = torch.tensor(
                                                system_state["core_weights"],device=self.device)

                                    logging.info(f"IntrextroLearning system loaded from {path}")
                                except Exception as e:
                                    logging.error(f"Error loading IntrextroLearning system: {e}")

                            # Example usage

                        if __name__=="__main__":
                            # Create configuration
                            config = LearningConfig(
                                hidden_size=64,
                                learning_rate=0.001,
                                use_cuda=torch.cuda.is_available()
                            )

                            # Initialize learning system
                            learning_system = IntrextroLearning(config)

                            # Example data
                            data = {
                                "feature1": 0.5,
                                "feature2": 0.3,
                                "feature3": 0.7,
                                "feature4": 0.2,
                                # Pad with zeros to match hidden_size
                                **{f"feature{i}": 0.0 for i in range(5,config.hidden_size + 1)}
                            }

                            # Process data
                            result = learning_system.process(data)
                            print("Processing result:",json.dumps(result,indent=2))

                            # Provide feedback
                            feedback = {
                                "performance": 0.8,
                                "data": data
                            }
                            learning_system.adapt(feedback)

                            # Get system state
                            state = learning_system.get_state()
                            print("System state:",json.dumps(state,indent=2))

                            # Save and load the model
                            learning_system.save_model("./model_save")
                            learning_system.load_model("./model_save")



