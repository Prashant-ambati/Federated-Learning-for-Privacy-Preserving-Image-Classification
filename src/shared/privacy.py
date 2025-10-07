"""
Differential privacy mechanisms for federated learning.
Implements noise injection, gradient clipping, and privacy budget tracking.
"""

import torch
import numpy as np
import math
from typing import Dict, Tuple, Optional, List, Any
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from .models import ModelWeights, PrivacyConfig
from .interfaces import PrivacyEngineInterface

logger = logging.getLogger(__name__)


class PrivacyError(Exception):
    """Custom exception for privacy-related errors."""
    pass


class PrivacyBudgetTracker:
    """Tracks privacy budget consumption over time."""
    
    def __init__(self, initial_epsilon: float, initial_delta: float):
        """
        Initialize privacy budget tracker.
        
        Args:
            initial_epsilon: Initial epsilon budget
            initial_delta: Initial delta budget
        """
        self.initial_epsilon = initial_epsilon
        self.initial_delta = initial_delta
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.consumption_history = []
        self.start_time = datetime.now()
    
    def consume_budget(self, epsilon: float, delta: float, operation: str = "training"):
        """
        Consume privacy budget for an operation.
        
        Args:
            epsilon: Epsilon to consume
            delta: Delta to consume
            operation: Description of the operation
        """
        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        
        self.consumption_history.append({
            'timestamp': datetime.now().isoformat(),
            'epsilon': epsilon,
            'delta': delta,
            'operation': operation,
            'total_epsilon': self.consumed_epsilon,
            'total_delta': self.consumed_delta
        })
        
        logger.debug(f"Privacy budget consumed - ε: {epsilon:.6f}, δ: {delta:.6f} for {operation}")
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        remaining_epsilon = max(0, self.initial_epsilon - self.consumed_epsilon)
        remaining_delta = max(0, self.initial_delta - self.consumed_delta)
        return remaining_epsilon, remaining_delta
    
    def is_budget_exhausted(self, required_epsilon: float = 0, required_delta: float = 0) -> bool:
        """Check if privacy budget is exhausted."""
        remaining_epsilon, remaining_delta = self.get_remaining_budget()
        return remaining_epsilon < required_epsilon or remaining_delta < required_delta
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive budget status."""
        remaining_epsilon, remaining_delta = self.get_remaining_budget()
        
        return {
            'initial_epsilon': self.initial_epsilon,
            'initial_delta': self.initial_delta,
            'consumed_epsilon': self.consumed_epsilon,
            'consumed_delta': self.consumed_delta,
            'remaining_epsilon': remaining_epsilon,
            'remaining_delta': remaining_delta,
            'epsilon_utilization': self.consumed_epsilon / self.initial_epsilon,
            'delta_utilization': self.consumed_delta / self.initial_delta,
            'operations_count': len(self.consumption_history),
            'tracking_duration': (datetime.now() - self.start_time).total_seconds()
        }


class GradientClipper:
    """Clips gradients to bound sensitivity for differential privacy."""
    
    def __init__(self, max_grad_norm: float):
        """
        Initialize gradient clipper.
        
        Args:
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.max_grad_norm = max_grad_norm
    
    def clip_gradients(self, gradients: ModelWeights) -> Tuple[ModelWeights, float]:
        """
        Clip gradients to bound L2 norm.
        
        Args:
            gradients: Model gradients to clip
            
        Returns:
            Tuple of (clipped_gradients, actual_norm)
        """
        try:
            # Calculate total gradient norm
            total_norm = 0.0
            for grad in gradients.values():
                if grad is not None:
                    total_norm += grad.norm().item() ** 2
            total_norm = math.sqrt(total_norm)
            
            # Clip if necessary
            clipped_gradients = {}
            if total_norm > self.max_grad_norm:
                clip_coef = self.max_grad_norm / total_norm
                for name, grad in gradients.items():
                    if grad is not None:
                        clipped_gradients[name] = grad * clip_coef
                    else:
                        clipped_gradients[name] = grad
                
                logger.debug(f"Gradients clipped: norm {total_norm:.4f} -> {self.max_grad_norm:.4f}")
            else:
                clipped_gradients = {name: grad.clone() if grad is not None else grad 
                                   for name, grad in gradients.items()}
            
            return clipped_gradients, min(total_norm, self.max_grad_norm)
            
        except Exception as e:
            logger.error(f"Gradient clipping failed: {str(e)}")
            raise PrivacyError(f"Gradient clipping failed: {str(e)}")
    
    def estimate_sensitivity(self, gradients_batch: List[ModelWeights]) -> float:
        """
        Estimate sensitivity from a batch of gradients.
        
        Args:
            gradients_batch: List of gradient dictionaries
            
        Returns:
            float: Estimated sensitivity
        """
        if not gradients_batch:
            return 0.0
        
        max_norm = 0.0
        for gradients in gradients_batch:
            total_norm = 0.0
            for grad in gradients.values():
                if grad is not None:
                    total_norm += grad.norm().item() ** 2
            total_norm = math.sqrt(total_norm)
            max_norm = max(max_norm, total_norm)
        
        return max_norm


class GaussianNoiseGenerator:
    """Generates Gaussian noise for differential privacy."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize noise generator.
        
        Args:
            device: Device to generate noise on
        """
        self.device = device or torch.device('cpu')
    
    def generate_noise(self, 
                      shape: torch.Size, 
                      sensitivity: float, 
                      epsilon: float, 
                      delta: float) -> torch.Tensor:
        """
        Generate Gaussian noise for differential privacy.
        
        Args:
            shape: Shape of noise tensor to generate
            sensitivity: Sensitivity of the function
            epsilon: Privacy parameter epsilon
            delta: Privacy parameter delta
            
        Returns:
            torch.Tensor: Gaussian noise tensor
        """
        try:
            # Calculate noise scale using Gaussian mechanism
            if epsilon <= 0:
                raise ValueError("Epsilon must be positive")
            
            if delta <= 0 or delta >= 1:
                raise ValueError("Delta must be in (0, 1)")
            
            # Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            
            # Generate noise
            noise = torch.normal(mean=0.0, std=sigma, size=shape, device=self.device)
            
            logger.debug(f"Generated Gaussian noise with σ={sigma:.6f}")
            return noise
            
        except Exception as e:
            logger.error(f"Noise generation failed: {str(e)}")
            raise PrivacyError(f"Noise generation failed: {str(e)}")
    
    def add_noise_to_gradients(self, 
                              gradients: ModelWeights, 
                              sensitivity: float, 
                              epsilon: float, 
                              delta: float) -> ModelWeights:
        """
        Add Gaussian noise to model gradients.
        
        Args:
            gradients: Model gradients
            sensitivity: Sensitivity bound
            epsilon: Privacy parameter epsilon
            delta: Privacy parameter delta
            
        Returns:
            ModelWeights: Noisy gradients
        """
        try:
            noisy_gradients = {}
            
            for name, grad in gradients.items():
                if grad is not None:
                    # Generate noise with same shape as gradient
                    noise = self.generate_noise(grad.shape, sensitivity, epsilon, delta)
                    noisy_gradients[name] = grad + noise
                else:
                    noisy_gradients[name] = grad
            
            logger.debug(f"Added Gaussian noise to {len(noisy_gradients)} gradient tensors")
            return noisy_gradients
            
        except Exception as e:
            logger.error(f"Adding noise to gradients failed: {str(e)}")
            raise PrivacyError(f"Adding noise to gradients failed: {str(e)}")


class DifferentialPrivacyEngine(PrivacyEngineInterface):
    """Main engine for differential privacy operations."""
    
    def __init__(self, 
                 privacy_config: PrivacyConfig,
                 device: Optional[torch.device] = None):
        """
        Initialize differential privacy engine.
        
        Args:
            privacy_config: Privacy configuration parameters
            device: Device for computations
        """
        self.config = privacy_config
        self.device = device or torch.device('cpu')
        
        # Initialize components
        self.clipper = GradientClipper(privacy_config.max_grad_norm)
        self.noise_generator = GaussianNoiseGenerator(device)
        self.budget_tracker = PrivacyBudgetTracker(
            privacy_config.epsilon, 
            privacy_config.delta
        )
        
        logger.info(f"DP Engine initialized - ε={privacy_config.epsilon}, "
                   f"δ={privacy_config.delta}, max_norm={privacy_config.max_grad_norm}")
    
    def add_noise(self, gradients: ModelWeights, epsilon: float, delta: float) -> ModelWeights:
        """Add differential privacy noise to gradients."""
        try:
            # Validate privacy parameters
            if not self.validate_privacy_parameters(epsilon, delta):
                raise PrivacyError("Invalid privacy parameters")
            
            # Check budget availability
            if self.budget_tracker.is_budget_exhausted(epsilon, delta):
                raise PrivacyError("Privacy budget exhausted")
            
            # Clip gradients first
            clipped_gradients, actual_norm = self.clipper.clip_gradients(gradients)
            
            # Add noise using the clipped norm as sensitivity
            noisy_gradients = self.noise_generator.add_noise_to_gradients(
                clipped_gradients, actual_norm, epsilon, delta
            )
            
            # Update budget
            self.budget_tracker.consume_budget(epsilon, delta, "gradient_noise")
            
            logger.debug(f"Applied DP noise - ε={epsilon:.6f}, δ={delta:.6f}")
            return noisy_gradients
            
        except Exception as e:
            logger.error(f"Adding DP noise failed: {str(e)}")
            raise PrivacyError(f"Adding DP noise failed: {str(e)}")
    
    def clip_gradients(self, gradients: ModelWeights, max_norm: float) -> ModelWeights:
        """Clip gradients to bound sensitivity."""
        clipper = GradientClipper(max_norm)
        clipped_gradients, _ = clipper.clip_gradients(gradients)
        return clipped_gradients
    
    def calculate_privacy_budget(self, epsilon: float, delta: float, steps: int) -> float:
        """Calculate privacy budget consumption for multiple steps."""
        # For composition, we use advanced composition theorem
        # This is a simplified version - in practice, you might want to use
        # more sophisticated composition bounds
        
        if steps <= 1:
            return epsilon
        
        # Advanced composition (simplified)
        # ε' ≈ ε * sqrt(2 * steps * ln(1/δ)) + steps * ε * (e^ε - 1)
        composition_epsilon = epsilon * math.sqrt(2 * steps * math.log(1 / delta))
        composition_epsilon += steps * epsilon * (math.exp(epsilon) - 1)
        
        return composition_epsilon
    
    def validate_privacy_parameters(self, epsilon: float, delta: float) -> bool:
        """Validate privacy parameters are within acceptable bounds."""
        try:
            if epsilon <= 0:
                logger.error("Epsilon must be positive")
                return False
            
            if epsilon > 10.0:  # Very high epsilon
                logger.warning(f"Epsilon {epsilon} is very high, privacy may be weak")
            
            if delta <= 0 or delta >= 1:
                logger.error("Delta must be in (0, 1)")
                return False
            
            if delta > 1e-3:  # High delta
                logger.warning(f"Delta {delta} is high, privacy may be weak")
            
            return True
            
        except Exception as e:
            logger.error(f"Privacy parameter validation failed: {str(e)}")
            return False
    
    def get_privacy_analysis(self) -> Dict[str, Any]:
        """Get comprehensive privacy analysis."""
        budget_status = self.budget_tracker.get_budget_status()
        
        # Calculate privacy strength indicators
        epsilon_strength = "strong" if self.config.epsilon < 1.0 else "moderate" if self.config.epsilon < 5.0 else "weak"
        delta_strength = "strong" if self.config.delta < 1e-5 else "moderate" if self.config.delta < 1e-3 else "weak"
        
        return {
            'privacy_config': {
                'epsilon': self.config.epsilon,
                'delta': self.config.delta,
                'max_grad_norm': self.config.max_grad_norm,
                'noise_multiplier': self.config.noise_multiplier
            },
            'budget_status': budget_status,
            'privacy_strength': {
                'epsilon_strength': epsilon_strength,
                'delta_strength': delta_strength,
                'overall_strength': min(epsilon_strength, delta_strength, key=lambda x: ['strong', 'moderate', 'weak'].index(x))
            },
            'recommendations': self._get_privacy_recommendations()
        }
    
    def _get_privacy_recommendations(self) -> List[str]:
        """Get privacy configuration recommendations."""
        recommendations = []
        
        if self.config.epsilon > 5.0:
            recommendations.append("Consider reducing epsilon for stronger privacy")
        
        if self.config.delta > 1e-3:
            recommendations.append("Consider reducing delta for better privacy guarantees")
        
        if self.config.max_grad_norm > 10.0:
            recommendations.append("Consider reducing gradient clipping norm to improve privacy")
        
        remaining_epsilon, remaining_delta = self.budget_tracker.get_remaining_budget()
        if remaining_epsilon < self.config.epsilon * 0.1:
            recommendations.append("Privacy budget nearly exhausted, consider resetting or reducing usage")
        
        if not recommendations:
            recommendations.append("Privacy configuration looks good")
        
        return recommendations
    
    def reset_budget(self, new_epsilon: Optional[float] = None, new_delta: Optional[float] = None):
        """Reset privacy budget tracker."""
        epsilon = new_epsilon or self.config.epsilon
        delta = new_delta or self.config.delta
        
        self.budget_tracker = PrivacyBudgetTracker(epsilon, delta)
        
        if new_epsilon:
            self.config.epsilon = new_epsilon
        if new_delta:
            self.config.delta = new_delta
        
        logger.info(f"Privacy budget reset - ε={epsilon}, δ={delta}")


class PrivacyAccountant:
    """Advanced privacy accounting for federated learning."""
    
    def __init__(self):
        """Initialize privacy accountant."""
        self.privacy_ledger = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0
    
    def add_mechanism(self, 
                     mechanism_type: str, 
                     epsilon: float, 
                     delta: float, 
                     sensitivity: float,
                     noise_scale: float,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Add a privacy mechanism to the ledger.
        
        Args:
            mechanism_type: Type of mechanism ('gaussian', 'laplace', etc.)
            epsilon: Epsilon parameter
            delta: Delta parameter
            sensitivity: Function sensitivity
            noise_scale: Scale of noise added
            metadata: Additional metadata
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'mechanism_type': mechanism_type,
            'epsilon': epsilon,
            'delta': delta,
            'sensitivity': sensitivity,
            'noise_scale': noise_scale,
            'metadata': metadata or {}
        }
        
        self.privacy_ledger.append(entry)
        self.total_epsilon += epsilon
        self.total_delta += delta
        
        logger.debug(f"Privacy mechanism added: {mechanism_type} (ε={epsilon:.6f}, δ={delta:.6f})")
    
    def get_total_privacy_cost(self) -> Tuple[float, float]:
        """Get total privacy cost using composition."""
        # This is a simplified composition - in practice, you'd use more
        # sophisticated bounds like RDP or zCDP
        return self.total_epsilon, self.total_delta
    
    def get_privacy_ledger(self) -> List[Dict[str, Any]]:
        """Get complete privacy ledger."""
        return self.privacy_ledger.copy()
    
    def export_ledger(self, filepath: str):
        """Export privacy ledger to file."""
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'total_epsilon': self.total_epsilon,
                    'total_delta': self.total_delta,
                    'ledger': self.privacy_ledger
                }, f, indent=2)
            logger.info(f"Privacy ledger exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export privacy ledger: {str(e)}")


def create_privacy_engine(epsilon: float = 1.0, 
                         delta: float = 1e-5, 
                         max_grad_norm: float = 1.0,
                         noise_multiplier: float = 1.0,
                         device: Optional[torch.device] = None) -> DifferentialPrivacyEngine:
    """
    Factory function to create differential privacy engine.
    
    Args:
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise multiplier
        device: Device for computations
        
    Returns:
        DifferentialPrivacyEngine: Configured privacy engine
    """
    privacy_config = PrivacyConfig(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier
    )
    
    return DifferentialPrivacyEngine(privacy_config, device)


def estimate_privacy_parameters(target_accuracy: float = 0.9, 
                               dataset_size: int = 10000,
                               num_rounds: int = 100) -> Dict[str, float]:
    """
    Estimate privacy parameters for target accuracy and dataset.
    
    Args:
        target_accuracy: Target model accuracy
        dataset_size: Size of the dataset
        num_rounds: Number of federated learning rounds
        
    Returns:
        Dict: Recommended privacy parameters
    """
    # This is a simplified heuristic - in practice, you'd use more
    # sophisticated analysis based on the specific model and dataset
    
    # Larger datasets can tolerate more noise
    base_epsilon = 1.0 if dataset_size > 5000 else 2.0
    
    # Adjust for target accuracy
    if target_accuracy > 0.95:
        epsilon = base_epsilon * 2  # Need less noise for high accuracy
    elif target_accuracy < 0.85:
        epsilon = base_epsilon * 0.5  # Can tolerate more noise
    else:
        epsilon = base_epsilon
    
    # Adjust for number of rounds (composition)
    epsilon = epsilon / math.sqrt(num_rounds)
    
    # Standard delta
    delta = 1.0 / dataset_size
    
    # Gradient clipping norm
    max_grad_norm = 1.0 if target_accuracy > 0.9 else 2.0
    
    return {
        'epsilon': epsilon,
        'delta': delta,
        'max_grad_norm': max_grad_norm,
        'noise_multiplier': 1.0
    }