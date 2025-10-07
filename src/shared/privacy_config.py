"""
Privacy parameter configuration and management for federated learning.
Provides utilities for configuring, validating, and optimizing privacy parameters.
"""

import yaml
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import math

from .models import PrivacyConfig
from .privacy import DifferentialPrivacyEngine, PrivacyBudgetTracker

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Predefined privacy levels for easy configuration."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CUSTOM = "custom"


@dataclass
class PrivacyUtilityTradeoff:
    """Represents privacy-utility tradeoff metrics."""
    epsilon: float
    delta: float
    expected_accuracy: float
    privacy_strength: str
    utility_score: float
    recommendation: str


class PrivacyConfigManager:
    """Manages privacy configuration and parameter optimization."""
    
    # Predefined privacy configurations
    PRIVACY_PRESETS = {
        PrivacyLevel.HIGH: {
            'epsilon': 0.5,
            'delta': 1e-6,
            'max_grad_norm': 0.5,
            'noise_multiplier': 2.0
        },
        PrivacyLevel.MEDIUM: {
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'noise_multiplier': 1.0
        },
        PrivacyLevel.LOW: {
            'epsilon': 3.0,
            'delta': 1e-4,
            'max_grad_norm': 2.0,
            'noise_multiplier': 0.5
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize privacy configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file
        self.current_config = None
        self.config_history = []
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def create_config(self, 
                     privacy_level: Union[PrivacyLevel, str] = PrivacyLevel.MEDIUM,
                     custom_params: Optional[Dict[str, float]] = None) -> PrivacyConfig:
        """
        Create privacy configuration.
        
        Args:
            privacy_level: Privacy level or 'custom'
            custom_params: Custom parameters if privacy_level is 'custom'
            
        Returns:
            PrivacyConfig: Privacy configuration
        """
        try:
            if isinstance(privacy_level, str):
                privacy_level = PrivacyLevel(privacy_level)
            
            if privacy_level == PrivacyLevel.CUSTOM:
                if not custom_params:
                    raise ValueError("Custom parameters required for custom privacy level")
                params = custom_params
            else:
                params = self.PRIVACY_PRESETS[privacy_level].copy()
                
                # Override with custom params if provided
                if custom_params:
                    params.update(custom_params)
            
            config = PrivacyConfig(**params)
            self.current_config = config
            
            # Add to history
            self.config_history.append({
                'timestamp': self._get_timestamp(),
                'privacy_level': privacy_level.value,
                'config': asdict(config)
            })
            
            logger.info(f"Created privacy config: {privacy_level.value} - "
                       f"ε={config.epsilon}, δ={config.delta}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to create privacy config: {str(e)}")
            raise ValueError(f"Failed to create privacy config: {str(e)}")
    
    def optimize_for_accuracy(self, 
                             target_accuracy: float,
                             dataset_size: int,
                             num_rounds: int = 100,
                             model_complexity: str = "medium") -> PrivacyConfig:
        """
        Optimize privacy parameters for target accuracy.
        
        Args:
            target_accuracy: Target model accuracy (0.0-1.0)
            dataset_size: Size of the training dataset
            num_rounds: Number of federated learning rounds
            model_complexity: Model complexity ("low", "medium", "high")
            
        Returns:
            PrivacyConfig: Optimized privacy configuration
        """
        try:
            # Base parameters based on target accuracy
            if target_accuracy >= 0.95:
                base_epsilon = 2.0
                base_max_norm = 2.0
            elif target_accuracy >= 0.90:
                base_epsilon = 1.0
                base_max_norm = 1.0
            elif target_accuracy >= 0.85:
                base_epsilon = 0.8
                base_max_norm = 0.8
            else:
                base_epsilon = 0.5
                base_max_norm = 0.5
            
            # Adjust for dataset size
            if dataset_size < 1000:
                epsilon_multiplier = 1.5  # Need less noise for small datasets
            elif dataset_size > 10000:
                epsilon_multiplier = 0.8  # Can tolerate more noise
            else:
                epsilon_multiplier = 1.0
            
            # Adjust for model complexity
            complexity_multipliers = {
                "low": 0.8,    # Simple models need less privacy protection
                "medium": 1.0,
                "high": 1.2    # Complex models need more privacy protection
            }
            complexity_mult = complexity_multipliers.get(model_complexity, 1.0)
            
            # Adjust for composition over rounds
            composition_factor = 1.0 / math.sqrt(num_rounds)
            
            # Calculate final parameters
            epsilon = base_epsilon * epsilon_multiplier * complexity_mult * composition_factor
            delta = min(1.0 / dataset_size, 1e-5)  # Standard delta
            max_grad_norm = base_max_norm * complexity_mult
            noise_multiplier = 1.0 / epsilon  # Inverse relationship
            
            # Ensure reasonable bounds
            epsilon = max(0.1, min(epsilon, 10.0))
            delta = max(1e-7, min(delta, 1e-3))
            max_grad_norm = max(0.1, min(max_grad_norm, 5.0))
            noise_multiplier = max(0.1, min(noise_multiplier, 5.0))
            
            config = PrivacyConfig(
                epsilon=epsilon,
                delta=delta,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier
            )
            
            self.current_config = config
            
            logger.info(f"Optimized privacy config for accuracy {target_accuracy:.2f}: "
                       f"ε={epsilon:.4f}, δ={delta:.2e}")
            
            return config
            
        except Exception as e:
            logger.error(f"Privacy optimization failed: {str(e)}")
            raise ValueError(f"Privacy optimization failed: {str(e)}")
    
    def analyze_privacy_utility_tradeoff(self, 
                                       epsilon_range: Tuple[float, float] = (0.1, 5.0),
                                       num_points: int = 10,
                                       dataset_size: int = 10000) -> List[PrivacyUtilityTradeoff]:
        """
        Analyze privacy-utility tradeoff across epsilon range.
        
        Args:
            epsilon_range: Range of epsilon values to analyze
            num_points: Number of points to analyze
            dataset_size: Dataset size for delta calculation
            
        Returns:
            List[PrivacyUtilityTradeoff]: Tradeoff analysis results
        """
        try:
            results = []
            epsilon_min, epsilon_max = epsilon_range
            
            for i in range(num_points):
                # Calculate epsilon for this point
                if num_points == 1:
                    epsilon = epsilon_min
                else:
                    epsilon = epsilon_min + (epsilon_max - epsilon_min) * i / (num_points - 1)
                
                # Standard delta
                delta = 1.0 / dataset_size
                
                # Estimate expected accuracy (simplified heuristic)
                # Higher epsilon generally means higher accuracy
                base_accuracy = 0.85
                epsilon_boost = min(0.1, epsilon * 0.02)  # Diminishing returns
                expected_accuracy = min(0.99, base_accuracy + epsilon_boost)
                
                # Privacy strength assessment
                if epsilon < 1.0:
                    privacy_strength = "Strong"
                elif epsilon < 3.0:
                    privacy_strength = "Moderate"
                else:
                    privacy_strength = "Weak"
                
                # Utility score (combination of accuracy and privacy)
                privacy_score = max(0, 1 - epsilon / 5.0)  # Higher epsilon = lower privacy score
                accuracy_score = expected_accuracy
                utility_score = (privacy_score + accuracy_score) / 2
                
                # Recommendation
                if epsilon < 0.5:
                    recommendation = "Very strong privacy, may impact accuracy"
                elif epsilon < 1.0:
                    recommendation = "Good balance of privacy and utility"
                elif epsilon < 3.0:
                    recommendation = "Moderate privacy, good accuracy"
                else:
                    recommendation = "Weak privacy, high accuracy"
                
                tradeoff = PrivacyUtilityTradeoff(
                    epsilon=epsilon,
                    delta=delta,
                    expected_accuracy=expected_accuracy,
                    privacy_strength=privacy_strength,
                    utility_score=utility_score,
                    recommendation=recommendation
                )
                
                results.append(tradeoff)
            
            logger.info(f"Analyzed privacy-utility tradeoff for {num_points} points")
            return results
            
        except Exception as e:
            logger.error(f"Privacy-utility analysis failed: {str(e)}")
            raise ValueError(f"Privacy-utility analysis failed: {str(e)}")
    
    def validate_config(self, config: PrivacyConfig) -> Dict[str, Any]:
        """
        Validate privacy configuration.
        
        Args:
            config: Privacy configuration to validate
            
        Returns:
            Dict: Validation results
        """
        try:
            validation_results = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'recommendations': []
            }
            
            # Validate epsilon
            if config.epsilon <= 0:
                validation_results['errors'].append("Epsilon must be positive")
                validation_results['valid'] = False
            elif config.epsilon > 10.0:
                validation_results['warnings'].append("Very high epsilon may provide weak privacy")
            elif config.epsilon < 0.1:
                validation_results['warnings'].append("Very low epsilon may severely impact utility")
            
            # Validate delta
            if config.delta <= 0 or config.delta >= 1:
                validation_results['errors'].append("Delta must be in (0, 1)")
                validation_results['valid'] = False
            elif config.delta > 1e-3:
                validation_results['warnings'].append("High delta may weaken privacy guarantees")
            
            # Validate gradient norm
            if config.max_grad_norm <= 0:
                validation_results['errors'].append("Max gradient norm must be positive")
                validation_results['valid'] = False
            elif config.max_grad_norm > 10.0:
                validation_results['warnings'].append("Very high gradient norm may reduce privacy")
            
            # Validate noise multiplier
            if config.noise_multiplier < 0:
                validation_results['errors'].append("Noise multiplier must be non-negative")
                validation_results['valid'] = False
            
            # Generate recommendations
            if config.epsilon > 1.0 and config.delta > 1e-5:
                validation_results['recommendations'].append(
                    "Consider reducing either epsilon or delta for stronger privacy"
                )
            
            if config.max_grad_norm > 2.0:
                validation_results['recommendations'].append(
                    "Consider reducing gradient clipping norm for better privacy"
                )
            
            if not validation_results['recommendations']:
                validation_results['recommendations'].append("Configuration looks good")
            
            logger.info(f"Privacy config validation: {'passed' if validation_results['valid'] else 'failed'}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Privacy config validation failed: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'recommendations': []
            }
    
    def save_config(self, config: PrivacyConfig, filepath: str, format: str = "yaml"):
        """
        Save privacy configuration to file.
        
        Args:
            config: Privacy configuration to save
            filepath: File path to save to
            format: File format ("yaml" or "json")
        """
        try:
            config_dict = asdict(config)
            config_dict['metadata'] = {
                'created_at': self._get_timestamp(),
                'version': '1.0'
            }
            
            if format.lower() == "yaml":
                with open(filepath, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Privacy config saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save privacy config: {str(e)}")
            raise ValueError(f"Failed to save privacy config: {str(e)}")
    
    def load_config(self, filepath: str) -> PrivacyConfig:
        """
        Load privacy configuration from file.
        
        Args:
            filepath: File path to load from
            
        Returns:
            PrivacyConfig: Loaded privacy configuration
        """
        try:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                with open(filepath, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # Remove metadata if present
            config_dict.pop('metadata', None)
            
            config = PrivacyConfig(**config_dict)
            self.current_config = config
            
            logger.info(f"Privacy config loaded from {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load privacy config: {str(e)}")
            raise ValueError(f"Failed to load privacy config: {str(e)}")
    
    def get_config_summary(self, config: Optional[PrivacyConfig] = None) -> Dict[str, Any]:
        """
        Get summary of privacy configuration.
        
        Args:
            config: Privacy configuration (uses current if None)
            
        Returns:
            Dict: Configuration summary
        """
        config = config or self.current_config
        if not config:
            return {'error': 'No configuration available'}
        
        # Determine privacy level
        privacy_level = "Custom"
        for level, preset in self.PRIVACY_PRESETS.items():
            if (abs(config.epsilon - preset['epsilon']) < 0.01 and
                abs(config.delta - preset['delta']) < 1e-7):
                privacy_level = level.value.title()
                break
        
        # Privacy strength
        if config.epsilon < 1.0:
            strength = "Strong"
        elif config.epsilon < 3.0:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        return {
            'privacy_level': privacy_level,
            'privacy_strength': strength,
            'parameters': {
                'epsilon': config.epsilon,
                'delta': config.delta,
                'max_grad_norm': config.max_grad_norm,
                'noise_multiplier': config.noise_multiplier
            },
            'estimated_accuracy_impact': self._estimate_accuracy_impact(config),
            'recommendations': self._get_config_recommendations(config)
        }
    
    def _estimate_accuracy_impact(self, config: PrivacyConfig) -> str:
        """Estimate accuracy impact of privacy configuration."""
        if config.epsilon > 2.0:
            return "Low impact"
        elif config.epsilon > 1.0:
            return "Moderate impact"
        elif config.epsilon > 0.5:
            return "High impact"
        else:
            return "Very high impact"
    
    def _get_config_recommendations(self, config: PrivacyConfig) -> List[str]:
        """Get recommendations for privacy configuration."""
        recommendations = []
        
        if config.epsilon > 5.0:
            recommendations.append("Consider reducing epsilon for stronger privacy")
        elif config.epsilon < 0.1:
            recommendations.append("Consider increasing epsilon to improve utility")
        
        if config.delta > 1e-4:
            recommendations.append("Consider reducing delta for better privacy guarantees")
        
        if config.max_grad_norm > 3.0:
            recommendations.append("Consider reducing gradient clipping norm")
        
        if not recommendations:
            recommendations.append("Configuration is well-balanced")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_config_history(self) -> List[Dict[str, Any]]:
        """Get configuration history."""
        return self.config_history.copy()


def create_privacy_config_for_dataset(dataset_name: str, 
                                     target_accuracy: float = 0.9) -> PrivacyConfig:
    """
    Create privacy configuration optimized for specific dataset.
    
    Args:
        dataset_name: Name of the dataset ("mnist", "cifar10", etc.)
        target_accuracy: Target accuracy
        
    Returns:
        PrivacyConfig: Optimized privacy configuration
    """
    dataset_configs = {
        'mnist': {
            'dataset_size': 60000,
            'model_complexity': 'low',
            'base_epsilon': 1.0
        },
        'cifar10': {
            'dataset_size': 50000,
            'model_complexity': 'medium',
            'base_epsilon': 1.5
        },
        'cifar100': {
            'dataset_size': 50000,
            'model_complexity': 'high',
            'base_epsilon': 2.0
        }
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_configs:
        logger.warning(f"Unknown dataset {dataset_name}, using default config")
        dataset_name = 'mnist'
    
    config_info = dataset_configs[dataset_name]
    
    manager = PrivacyConfigManager()
    return manager.optimize_for_accuracy(
        target_accuracy=target_accuracy,
        dataset_size=config_info['dataset_size'],
        model_complexity=config_info['model_complexity']
    )


def compare_privacy_configs(configs: List[Tuple[str, PrivacyConfig]]) -> Dict[str, Any]:
    """
    Compare multiple privacy configurations.
    
    Args:
        configs: List of (name, config) tuples
        
    Returns:
        Dict: Comparison results
    """
    if not configs:
        return {'error': 'No configurations to compare'}
    
    comparison = {
        'configurations': {},
        'ranking': {
            'by_privacy': [],
            'by_utility': [],
            'by_balance': []
        },
        'recommendations': []
    }
    
    for name, config in configs:
        manager = PrivacyConfigManager()
        summary = manager.get_config_summary(config)
        
        # Privacy score (lower epsilon = higher privacy)
        privacy_score = max(0, 1 - config.epsilon / 5.0)
        
        # Utility score (higher epsilon = higher utility, but with diminishing returns)
        utility_score = min(1.0, config.epsilon / 3.0)
        
        # Balance score
        balance_score = (privacy_score + utility_score) / 2
        
        comparison['configurations'][name] = {
            'config': summary,
            'scores': {
                'privacy': privacy_score,
                'utility': utility_score,
                'balance': balance_score
            }
        }
    
    # Create rankings
    config_items = list(comparison['configurations'].items())
    
    comparison['ranking']['by_privacy'] = sorted(
        config_items, 
        key=lambda x: x[1]['scores']['privacy'], 
        reverse=True
    )
    
    comparison['ranking']['by_utility'] = sorted(
        config_items, 
        key=lambda x: x[1]['scores']['utility'], 
        reverse=True
    )
    
    comparison['ranking']['by_balance'] = sorted(
        config_items, 
        key=lambda x: x[1]['scores']['balance'], 
        reverse=True
    )
    
    # Generate recommendations
    best_privacy = comparison['ranking']['by_privacy'][0][0]
    best_utility = comparison['ranking']['by_utility'][0][0]
    best_balance = comparison['ranking']['by_balance'][0][0]
    
    comparison['recommendations'] = [
        f"For strongest privacy: {best_privacy}",
        f"For best utility: {best_utility}",
        f"For best balance: {best_balance}"
    ]
    
    return comparison