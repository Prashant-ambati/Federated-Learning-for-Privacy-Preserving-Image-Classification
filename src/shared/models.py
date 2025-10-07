"""
Core data models for the federated learning system.
Defines the primary data structures used across all services.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import torch


class ComputePowerLevel(Enum):
    """Enumeration for client computational capabilities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy parameters."""
    epsilon: float  # Differential privacy parameter
    delta: float    # Differential privacy parameter
    max_grad_norm: float  # Gradient clipping threshold
    noise_multiplier: float
    
    def __post_init__(self):
        """Validate privacy parameters."""
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta < 0 or self.delta >= 1:
            raise ValueError("Delta must be in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        if self.noise_multiplier < 0:
            raise ValueError("Noise multiplier must be non-negative")


@dataclass
class ClientCapabilities:
    """Client computational and network capabilities."""
    compute_power: ComputePowerLevel
    network_bandwidth: int  # Mbps
    available_samples: int
    supported_models: List[str]
    privacy_requirements: PrivacyConfig


@dataclass
class ModelUpdate:
    """Model update from a client containing weights and metadata."""
    client_id: str
    round_number: int
    model_weights: Dict[str, torch.Tensor]
    num_samples: int
    training_loss: float
    privacy_budget_used: float
    compression_ratio: float
    timestamp: datetime
    
    def validate(self) -> bool:
        """Validate model update integrity."""
        if not self.client_id or self.round_number < 0:
            return False
        if self.num_samples <= 0 or self.training_loss < 0:
            return False
        if not (0 <= self.privacy_budget_used <= 1):
            return False
        if not (0 <= self.compression_ratio <= 1):
            return False
        return True


@dataclass
class GlobalModel:
    """Global model state and metadata."""
    round_number: int
    model_weights: Dict[str, torch.Tensor]
    accuracy_metrics: Dict[str, float]
    participating_clients: List[str]
    convergence_score: float
    created_at: datetime
    
    def get_accuracy(self, dataset: str = "test") -> Optional[float]:
        """Get accuracy for a specific dataset."""
        return self.accuracy_metrics.get(f"{dataset}_accuracy")


@dataclass
class TrainingMetrics:
    """Training metrics from local client training."""
    loss: float
    accuracy: float
    epochs_completed: int
    training_time: float
    samples_processed: int


@dataclass
class RegistrationResponse:
    """Response to client registration request."""
    success: bool
    client_id: str
    message: str
    global_model_version: int


@dataclass
class ModelResponse:
    """Response containing global model for client."""
    success: bool
    model_weights: Optional[Dict[str, torch.Tensor]]
    round_number: int
    message: str


@dataclass
class AckResponse:
    """Acknowledgment response for model updates."""
    success: bool
    message: str
    next_round_eta: Optional[datetime]


@dataclass
class RoundConfig:
    """Configuration for a training round."""
    round_number: int
    min_clients: int
    max_clients: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    timeout_seconds: int


@dataclass
class TrainingStatus:
    """Current training status and progress."""
    current_round: int
    active_clients: int
    round_progress: float  # 0.0 to 1.0
    global_accuracy: float
    convergence_score: float
    estimated_completion: Optional[datetime]


@dataclass
class CompressedUpdate:
    """Compressed model update for efficient transmission."""
    client_id: str
    round_number: int
    compressed_weights: bytes
    compression_metadata: Dict[str, Any]
    original_size: int
    compressed_size: int
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_size == 0:
            return 0.0
        return self.compressed_size / self.original_size


# Type aliases for better code readability
ModelWeights = Dict[str, torch.Tensor]
ClientID = str
RoundNumber = int