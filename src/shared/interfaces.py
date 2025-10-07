"""
Core interfaces and abstract base classes for the federated learning system.
Defines contracts that services must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch

from .models import (
    ModelUpdate, GlobalModel, ClientCapabilities, RegistrationResponse,
    ModelResponse, AckResponse, RoundConfig, TrainingStatus, TrainingMetrics,
    CompressedUpdate, ModelWeights, ClientID, RoundNumber
)


class CoordinatorServiceInterface(ABC):
    """Interface for the coordinator service."""
    
    @abstractmethod
    def register_client(self, client_id: ClientID, capabilities: ClientCapabilities) -> RegistrationResponse:
        """Register a new client with the coordinator."""
        pass
    
    @abstractmethod
    def get_global_model(self, client_id: ClientID, round_number: RoundNumber) -> ModelResponse:
        """Get the current global model for a client."""
        pass
    
    @abstractmethod
    def submit_model_update(self, client_id: ClientID, model_update: ModelUpdate) -> AckResponse:
        """Submit a model update from a client."""
        pass
    
    @abstractmethod
    def start_training_round(self, round_config: RoundConfig) -> bool:
        """Start a new training round."""
        pass
    
    @abstractmethod
    def get_training_status(self) -> TrainingStatus:
        """Get current training status."""
        pass


class ClientServiceInterface(ABC):
    """Interface for the client service."""
    
    @abstractmethod
    def initialize_local_model(self, global_model: torch.nn.Module) -> None:
        """Initialize local model from global model."""
        pass
    
    @abstractmethod
    def train_local_model(self, epochs: int, batch_size: int) -> TrainingMetrics:
        """Train the local model and return metrics."""
        pass
    
    @abstractmethod
    def apply_differential_privacy(self, model_update: ModelUpdate, epsilon: float) -> ModelUpdate:
        """Apply differential privacy noise to model update."""
        pass
    
    @abstractmethod
    def compress_model_update(self, model_update: ModelUpdate) -> CompressedUpdate:
        """Compress model update for efficient transmission."""
        pass
    
    @abstractmethod
    def sync_with_coordinator(self) -> bool:
        """Synchronize with coordinator for new rounds."""
        pass


class AggregationServiceInterface(ABC):
    """Interface for the aggregation service."""
    
    @abstractmethod
    def aggregate_updates(self, updates: List[ModelUpdate], weights: List[float]) -> GlobalModel:
        """Aggregate client updates using FedAvg algorithm."""
        pass
    
    @abstractmethod
    def validate_update(self, update: ModelUpdate) -> bool:
        """Validate a model update for consistency."""
        pass
    
    @abstractmethod
    def compress_global_model(self, model: GlobalModel) -> CompressedUpdate:
        """Compress global model for distribution."""
        pass
    
    @abstractmethod
    def calculate_convergence_metrics(self, old_model: GlobalModel, new_model: GlobalModel) -> float:
        """Calculate convergence metrics between model versions."""
        pass


class ModelInterface(ABC):
    """Interface for PyTorch models used in federated learning."""
    
    @abstractmethod
    def get_model_weights(self) -> ModelWeights:
        """Get model weights as a dictionary."""
        pass
    
    @abstractmethod
    def set_model_weights(self, weights: ModelWeights) -> None:
        """Set model weights from a dictionary."""
        pass
    
    @abstractmethod
    def get_parameter_count(self) -> int:
        """Get total number of model parameters."""
        pass
    
    @abstractmethod
    def estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        pass


class DataLoaderInterface(ABC):
    """Interface for data loading and preprocessing."""
    
    @abstractmethod
    def load_training_data(self, client_id: ClientID) -> torch.utils.data.DataLoader:
        """Load training data for a specific client."""
        pass
    
    @abstractmethod
    def load_validation_data(self) -> torch.utils.data.DataLoader:
        """Load validation data."""
        pass
    
    @abstractmethod
    def get_data_statistics(self, client_id: ClientID) -> Dict[str, Any]:
        """Get statistics about client's data distribution."""
        pass


class PrivacyEngineInterface(ABC):
    """Interface for differential privacy operations."""
    
    @abstractmethod
    def add_noise(self, gradients: ModelWeights, epsilon: float, delta: float) -> ModelWeights:
        """Add differential privacy noise to gradients."""
        pass
    
    @abstractmethod
    def clip_gradients(self, gradients: ModelWeights, max_norm: float) -> ModelWeights:
        """Clip gradients to bound sensitivity."""
        pass
    
    @abstractmethod
    def calculate_privacy_budget(self, epsilon: float, delta: float, steps: int) -> float:
        """Calculate privacy budget consumption."""
        pass
    
    @abstractmethod
    def validate_privacy_parameters(self, epsilon: float, delta: float) -> bool:
        """Validate privacy parameters are within acceptable bounds."""
        pass


class CompressionInterface(ABC):
    """Interface for model compression operations."""
    
    @abstractmethod
    def compress_weights(self, weights: ModelWeights) -> bytes:
        """Compress model weights to bytes."""
        pass
    
    @abstractmethod
    def decompress_weights(self, compressed_data: bytes) -> ModelWeights:
        """Decompress bytes back to model weights."""
        pass
    
    @abstractmethod
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        pass