"""
Validation utilities for federated learning data models.
Provides comprehensive validation for model updates, privacy parameters, and system constraints.
"""

import torch
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .models import ModelUpdate, GlobalModel, PrivacyConfig, ClientCapabilities, ComputePowerLevel

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ModelUpdateValidator:
    """Validator for model updates from clients."""
    
    def __init__(self, max_weight_magnitude: float = 10.0, min_samples: int = 1):
        self.max_weight_magnitude = max_weight_magnitude
        self.min_samples = min_samples
    
    def validate_model_update(self, update: ModelUpdate) -> bool:
        """
        Comprehensive validation of a model update.
        
        Args:
            update: ModelUpdate to validate
            
        Returns:
            bool: True if valid, raises ValidationError if invalid
        """
        try:
            # Basic field validation
            self._validate_basic_fields(update)
            
            # Model weights validation
            self._validate_model_weights(update.model_weights)
            
            # Privacy and compression validation
            self._validate_privacy_and_compression(update)
            
            # Timestamp validation
            self._validate_timestamp(update.timestamp)
            
            logger.info(f"Model update validation passed for client {update.client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Model update validation failed for client {update.client_id}: {str(e)}")
            raise ValidationError(f"Model update validation failed: {str(e)}")
    
    def _validate_basic_fields(self, update: ModelUpdate) -> None:
        """Validate basic fields of model update."""
        if not update.client_id or not isinstance(update.client_id, str):
            raise ValidationError("Client ID must be a non-empty string")
        
        if update.round_number < 0:
            raise ValidationError("Round number must be non-negative")
        
        if update.num_samples < self.min_samples:
            raise ValidationError(f"Number of samples must be at least {self.min_samples}")
        
        if update.training_loss < 0:
            raise ValidationError("Training loss must be non-negative")
    
    def _validate_model_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Validate model weights structure and values."""
        if not weights:
            raise ValidationError("Model weights cannot be empty")
        
        for layer_name, weight_tensor in weights.items():
            if not isinstance(weight_tensor, torch.Tensor):
                raise ValidationError(f"Weight for layer {layer_name} must be a torch.Tensor")
            
            if torch.isnan(weight_tensor).any():
                raise ValidationError(f"NaN values found in layer {layer_name}")
            
            if torch.isinf(weight_tensor).any():
                raise ValidationError(f"Infinite values found in layer {layer_name}")
            
            max_magnitude = torch.abs(weight_tensor).max().item()
            if max_magnitude > self.max_weight_magnitude:
                raise ValidationError(
                    f"Weight magnitude {max_magnitude} exceeds maximum {self.max_weight_magnitude} in layer {layer_name}"
                )
    
    def _validate_privacy_and_compression(self, update: ModelUpdate) -> None:
        """Validate privacy budget and compression ratio."""
        if not (0 <= update.privacy_budget_used <= 1):
            raise ValidationError("Privacy budget used must be between 0 and 1")
        
        if not (0 <= update.compression_ratio <= 1):
            raise ValidationError("Compression ratio must be between 0 and 1")
    
    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Validate timestamp is reasonable."""
        now = datetime.now()
        max_age = timedelta(hours=24)  # Updates shouldn't be older than 24 hours
        future_tolerance = timedelta(minutes=5)  # Allow small clock skew
        
        if timestamp < now - max_age:
            raise ValidationError("Model update timestamp is too old")
        
        if timestamp > now + future_tolerance:
            raise ValidationError("Model update timestamp is in the future")


class GlobalModelValidator:
    """Validator for global model state."""
    
    def validate_global_model(self, model: GlobalModel) -> bool:
        """
        Validate global model structure and consistency.
        
        Args:
            model: GlobalModel to validate
            
        Returns:
            bool: True if valid, raises ValidationError if invalid
        """
        try:
            # Basic validation
            if model.round_number < 0:
                raise ValidationError("Round number must be non-negative")
            
            if not model.model_weights:
                raise ValidationError("Global model weights cannot be empty")
            
            # Validate participating clients
            if not model.participating_clients:
                raise ValidationError("Must have at least one participating client")
            
            # Validate convergence score
            if not (0 <= model.convergence_score <= 1):
                raise ValidationError("Convergence score must be between 0 and 1")
            
            # Validate accuracy metrics
            self._validate_accuracy_metrics(model.accuracy_metrics)
            
            logger.info(f"Global model validation passed for round {model.round_number}")
            return True
            
        except Exception as e:
            logger.error(f"Global model validation failed: {str(e)}")
            raise ValidationError(f"Global model validation failed: {str(e)}")
    
    def _validate_accuracy_metrics(self, metrics: Dict[str, float]) -> None:
        """Validate accuracy metrics are reasonable."""
        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Metric {metric_name} must be numeric")
            
            if metric_name.endswith("_accuracy") and not (0 <= value <= 1):
                raise ValidationError(f"Accuracy metric {metric_name} must be between 0 and 1")


class PrivacyConfigValidator:
    """Validator for differential privacy configuration."""
    
    def __init__(self, max_epsilon: float = 10.0, max_delta: float = 1e-3):
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
    
    def validate_privacy_config(self, config: PrivacyConfig) -> bool:
        """
        Validate privacy configuration parameters.
        
        Args:
            config: PrivacyConfig to validate
            
        Returns:
            bool: True if valid, raises ValidationError if invalid
        """
        try:
            # Epsilon validation
            if config.epsilon <= 0:
                raise ValidationError("Epsilon must be positive")
            
            if config.epsilon > self.max_epsilon:
                raise ValidationError(f"Epsilon {config.epsilon} exceeds maximum {self.max_epsilon}")
            
            # Delta validation
            if config.delta < 0 or config.delta >= 1:
                raise ValidationError("Delta must be in [0, 1)")
            
            if config.delta > self.max_delta:
                raise ValidationError(f"Delta {config.delta} exceeds maximum {self.max_delta}")
            
            # Gradient norm validation
            if config.max_grad_norm <= 0:
                raise ValidationError("Max gradient norm must be positive")
            
            # Noise multiplier validation
            if config.noise_multiplier < 0:
                raise ValidationError("Noise multiplier must be non-negative")
            
            logger.info("Privacy configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Privacy configuration validation failed: {str(e)}")
            raise ValidationError(f"Privacy configuration validation failed: {str(e)}")


class ClientCapabilitiesValidator:
    """Validator for client capabilities."""
    
    def validate_client_capabilities(self, capabilities: ClientCapabilities) -> bool:
        """
        Validate client capabilities are reasonable.
        
        Args:
            capabilities: ClientCapabilities to validate
            
        Returns:
            bool: True if valid, raises ValidationError if invalid
        """
        try:
            # Compute power validation
            if not isinstance(capabilities.compute_power, ComputePowerLevel):
                raise ValidationError("Compute power must be a valid ComputePowerLevel")
            
            # Network bandwidth validation
            if capabilities.network_bandwidth <= 0:
                raise ValidationError("Network bandwidth must be positive")
            
            if capabilities.network_bandwidth > 10000:  # 10 Gbps seems reasonable max
                raise ValidationError("Network bandwidth seems unrealistically high")
            
            # Sample count validation
            if capabilities.available_samples <= 0:
                raise ValidationError("Available samples must be positive")
            
            # Supported models validation
            if not capabilities.supported_models:
                raise ValidationError("Must support at least one model type")
            
            # Privacy requirements validation
            privacy_validator = PrivacyConfigValidator()
            privacy_validator.validate_privacy_config(capabilities.privacy_requirements)
            
            logger.info("Client capabilities validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Client capabilities validation failed: {str(e)}")
            raise ValidationError(f"Client capabilities validation failed: {str(e)}")


def validate_model_compatibility(weights1: Dict[str, torch.Tensor], 
                                weights2: Dict[str, torch.Tensor]) -> bool:
    """
    Validate that two sets of model weights are compatible for aggregation.
    
    Args:
        weights1: First set of model weights
        weights2: Second set of model weights
        
    Returns:
        bool: True if compatible, raises ValidationError if not
    """
    try:
        # Check same layer names
        if set(weights1.keys()) != set(weights2.keys()):
            raise ValidationError("Model weights have different layer names")
        
        # Check same shapes
        for layer_name in weights1.keys():
            if weights1[layer_name].shape != weights2[layer_name].shape:
                raise ValidationError(f"Layer {layer_name} has incompatible shapes")
        
        return True
        
    except Exception as e:
        logger.error(f"Model compatibility validation failed: {str(e)}")
        raise ValidationError(f"Model compatibility validation failed: {str(e)}")


def validate_training_round_config(min_clients: int, max_clients: int, 
                                 local_epochs: int, batch_size: int) -> bool:
    """
    Validate training round configuration parameters.
    
    Args:
        min_clients: Minimum number of clients required
        max_clients: Maximum number of clients allowed
        local_epochs: Number of local training epochs
        batch_size: Training batch size
        
    Returns:
        bool: True if valid, raises ValidationError if invalid
    """
    try:
        if min_clients <= 0:
            raise ValidationError("Minimum clients must be positive")
        
        if max_clients < min_clients:
            raise ValidationError("Maximum clients must be >= minimum clients")
        
        if local_epochs <= 0:
            raise ValidationError("Local epochs must be positive")
        
        if batch_size <= 0:
            raise ValidationError("Batch size must be positive")
        
        return True
        
    except Exception as e:
        logger.error(f"Training round config validation failed: {str(e)}")
        raise ValidationError(f"Training round config validation failed: {str(e)}")