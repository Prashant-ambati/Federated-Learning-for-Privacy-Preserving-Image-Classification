"""
Convergence detection and early stopping for federated learning.
Implements various convergence metrics and early stopping criteria.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from collections import deque
import math

from ..shared.models import GlobalModel, ModelWeights

logger = logging.getLogger(__name__)


class ConvergenceError(Exception):
    """Custom exception for convergence detection errors."""
    pass


class ConvergenceMetrics:
    """Container for convergence metrics."""
    
    def __init__(self):
        self.weight_change_norm = 0.0
        self.relative_weight_change = 0.0
        self.accuracy_change = 0.0
        self.loss_change = 0.0
        self.convergence_score = 0.0
        self.is_converged = False
        self.confidence = 0.0


class ConvergenceDetector:
    """Detects convergence in federated learning training."""
    
    def __init__(self, 
                 patience: int = 5,
                 min_delta: float = 1e-4,
                 window_size: int = 3,
                 convergence_threshold: float = 1e-3):
        """
        Initialize convergence detector.
        
        Args:
            patience: Number of rounds to wait for improvement
            min_delta: Minimum change to qualify as improvement
            window_size: Window size for moving average
            convergence_threshold: Threshold for convergence detection
        """
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.convergence_threshold = convergence_threshold
        
        # History tracking
        self.accuracy_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.weight_change_history = deque(maxlen=100)
        self.convergence_history = deque(maxlen=100)
        
        # State tracking
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        
        logger.info(f"Convergence detector initialized - patience: {patience}, "
                   f"min_delta: {min_delta}, threshold: {convergence_threshold}")
    
    def calculate_convergence_metrics(self, 
                                    current_model: GlobalModel,
                                    previous_model: Optional[GlobalModel] = None) -> ConvergenceMetrics:
        """
        Calculate comprehensive convergence metrics.
        
        Args:
            current_model: Current global model
            previous_model: Previous global model (if available)
            
        Returns:
            ConvergenceMetrics: Calculated convergence metrics
        """
        try:
            metrics = ConvergenceMetrics()
            
            # Get current accuracy and loss
            current_accuracy = current_model.get_accuracy() or 0.0
            current_loss = self._extract_loss_from_model(current_model)
            
            # Update history
            self.accuracy_history.append(current_accuracy)
            self.loss_history.append(current_loss)
            
            if previous_model is not None:
                # Calculate weight change metrics
                weight_metrics = self._calculate_weight_change_metrics(
                    current_model.model_weights, 
                    previous_model.model_weights
                )
                metrics.weight_change_norm = weight_metrics['norm']
                metrics.relative_weight_change = weight_metrics['relative']
                
                # Calculate accuracy and loss changes
                prev_accuracy = previous_model.get_accuracy() or 0.0
                prev_loss = self._extract_loss_from_model(previous_model)
                
                metrics.accuracy_change = current_accuracy - prev_accuracy
                metrics.loss_change = current_loss - prev_loss
                
                # Update weight change history
                self.weight_change_history.append(metrics.weight_change_norm)
            
            # Calculate overall convergence score
            metrics.convergence_score = self._calculate_convergence_score(metrics)
            
            # Determine if converged
            metrics.is_converged, metrics.confidence = self._check_convergence(metrics)
            
            # Update convergence history
            self.convergence_history.append({
                'round': current_model.round_number,
                'accuracy': current_accuracy,
                'loss': current_loss,
                'convergence_score': metrics.convergence_score,
                'is_converged': metrics.is_converged,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update best metrics
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.rounds_without_improvement = 0
            else:
                self.rounds_without_improvement += 1
            
            if current_loss < self.best_loss:
                self.best_loss = current_loss
            
            logger.debug(f"Convergence metrics - Score: {metrics.convergence_score:.6f}, "
                        f"Converged: {metrics.is_converged}, Confidence: {metrics.confidence:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Convergence calculation failed: {str(e)}")
            raise ConvergenceError(f"Convergence calculation failed: {str(e)}")
    
    def should_stop_early(self) -> Tuple[bool, str]:
        """
        Check if training should stop early.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        try:
            # Check patience-based early stopping
            if self.rounds_without_improvement >= self.patience:
                return True, f"No improvement for {self.patience} rounds"
            
            # Check convergence-based stopping
            if len(self.convergence_history) >= self.window_size:
                recent_scores = [h['convergence_score'] for h in list(self.convergence_history)[-self.window_size:]]
                avg_score = np.mean(recent_scores)
                
                if avg_score < self.convergence_threshold:
                    return True, f"Convergence threshold reached (score: {avg_score:.6f})"
            
            # Check if accuracy has plateaued
            if len(self.accuracy_history) >= self.window_size * 2:
                recent_acc = list(self.accuracy_history)[-self.window_size:]
                older_acc = list(self.accuracy_history)[-self.window_size*2:-self.window_size]
                
                recent_avg = np.mean(recent_acc)
                older_avg = np.mean(older_acc)
                
                if abs(recent_avg - older_avg) < self.min_delta:
                    return True, f"Accuracy plateaued (change: {abs(recent_avg - older_avg):.6f})"
            
            return False, "Continue training"
            
        except Exception as e:
            logger.error(f"Early stopping check failed: {str(e)}")
            return False, "Error in early stopping check"
    
    def _calculate_weight_change_metrics(self, 
                                       current_weights: ModelWeights,
                                       previous_weights: ModelWeights) -> Dict[str, float]:
        """Calculate weight change metrics between two models."""
        total_norm = 0.0
        total_current_norm = 0.0
        
        for layer_name in current_weights.keys():
            if layer_name in previous_weights:
                current = current_weights[layer_name]
                previous = previous_weights[layer_name]
                
                # Calculate L2 norm of difference
                diff_norm = torch.norm(current - previous).item()
                current_norm = torch.norm(current).item()
                
                total_norm += diff_norm ** 2
                total_current_norm += current_norm ** 2
        
        # Calculate metrics
        weight_change_norm = math.sqrt(total_norm)
        total_weight_norm = math.sqrt(total_current_norm)
        
        relative_change = weight_change_norm / total_weight_norm if total_weight_norm > 0 else 0.0
        
        return {
            'norm': weight_change_norm,
            'relative': relative_change
        }
    
    def _extract_loss_from_model(self, model: GlobalModel) -> float:
        """Extract loss value from model metrics."""
        # Try to get loss from accuracy metrics
        for key, value in model.accuracy_metrics.items():
            if 'loss' in key.lower():
                return float(value)
        
        # Default to 0 if no loss found
        return 0.0
    
    def _calculate_convergence_score(self, metrics: ConvergenceMetrics) -> float:
        """Calculate overall convergence score."""
        # Combine different metrics into a single score
        # Lower score indicates better convergence
        
        score = 0.0
        
        # Weight change component (normalized)
        if metrics.relative_weight_change > 0:
            score += metrics.relative_weight_change
        
        # Accuracy change component (negative change is bad)
        if metrics.accuracy_change < 0:
            score += abs(metrics.accuracy_change)
        
        # Loss change component (positive change is bad)
        if metrics.loss_change > 0:
            score += metrics.loss_change
        
        return score
    
    def _check_convergence(self, metrics: ConvergenceMetrics) -> Tuple[bool, float]:
        """Check if the model has converged."""
        # Simple convergence check based on score
        is_converged = metrics.convergence_score < self.convergence_threshold
        
        # Calculate confidence based on recent history
        confidence = 0.0
        if len(self.convergence_history) >= 3:
            recent_scores = [h['convergence_score'] for h in list(self.convergence_history)[-3:]]
            # Higher confidence if scores are consistently low
            avg_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            if avg_score < self.convergence_threshold:
                confidence = max(0.0, 1.0 - std_score)
            else:
                confidence = 0.0
        
        return is_converged, confidence
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get comprehensive convergence summary."""
        if not self.convergence_history:
            return {'message': 'No convergence data available'}
        
        recent_history = list(self.convergence_history)[-10:]  # Last 10 rounds
        
        return {
            'current_status': {
                'converged': self.converged,
                'best_accuracy': self.best_accuracy,
                'best_loss': self.best_loss,
                'rounds_without_improvement': self.rounds_without_improvement,
                'total_rounds': len(self.convergence_history)
            },
            'recent_performance': {
                'avg_accuracy': np.mean([h['accuracy'] for h in recent_history]),
                'avg_loss': np.mean([h['loss'] for h in recent_history]),
                'avg_convergence_score': np.mean([h['convergence_score'] for h in recent_history]),
                'convergence_trend': self._calculate_trend([h['convergence_score'] for h in recent_history])
            },
            'early_stopping': {
                'patience': self.patience,
                'min_delta': self.min_delta,
                'should_stop': self.should_stop_early()[0],
                'stop_reason': self.should_stop_early()[1]
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope < -0.001:
            return "improving"
        elif slope > 0.001:
            return "degrading"
        else:
            return "stable"
    
    def reset(self):
        """Reset convergence detector state."""
        self.accuracy_history.clear()
        self.loss_history.clear()
        self.weight_change_history.clear()
        self.convergence_history.clear()
        
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        
        logger.info("Convergence detector reset")


class AdaptiveConvergenceDetector(ConvergenceDetector):
    """Adaptive convergence detector that adjusts thresholds based on training progress."""
    
    def __init__(self, 
                 patience: int = 5,
                 min_delta: float = 1e-4,
                 window_size: int = 3,
                 convergence_threshold: float = 1e-3,
                 adaptation_rate: float = 0.1):
        """
        Initialize adaptive convergence detector.
        
        Args:
            patience: Number of rounds to wait for improvement
            min_delta: Minimum change to qualify as improvement
            window_size: Window size for moving average
            convergence_threshold: Initial threshold for convergence detection
            adaptation_rate: Rate at which thresholds adapt
        """
        super().__init__(patience, min_delta, window_size, convergence_threshold)
        self.initial_threshold = convergence_threshold
        self.adaptation_rate = adaptation_rate
        
        logger.info(f"Adaptive convergence detector initialized with adaptation rate: {adaptation_rate}")
    
    def calculate_convergence_metrics(self, 
                                    current_model: GlobalModel,
                                    previous_model: Optional[GlobalModel] = None) -> ConvergenceMetrics:
        """Calculate convergence metrics with adaptive thresholds."""
        metrics = super().calculate_convergence_metrics(current_model, previous_model)
        
        # Adapt threshold based on training progress
        self._adapt_threshold()
        
        return metrics
    
    def _adapt_threshold(self):
        """Adapt convergence threshold based on training history."""
        if len(self.convergence_history) < 5:
            return  # Need some history to adapt
        
        # Calculate recent convergence score variance
        recent_scores = [h['convergence_score'] for h in list(self.convergence_history)[-5:]]
        score_variance = np.var(recent_scores)
        
        # Adapt threshold based on variance
        # High variance -> increase threshold (be more lenient)
        # Low variance -> decrease threshold (be more strict)
        
        if score_variance > 0.01:  # High variance
            self.convergence_threshold = min(
                self.initial_threshold * 2,
                self.convergence_threshold * (1 + self.adaptation_rate)
            )
        elif score_variance < 0.001:  # Low variance
            self.convergence_threshold = max(
                self.initial_threshold * 0.5,
                self.convergence_threshold * (1 - self.adaptation_rate)
            )
        
        logger.debug(f"Adapted convergence threshold to: {self.convergence_threshold:.6f}")


def create_convergence_detector(detector_type: str = "standard", **kwargs) -> ConvergenceDetector:
    """
    Factory function to create convergence detector.
    
    Args:
        detector_type: Type of detector ("standard" or "adaptive")
        **kwargs: Detector-specific parameters
        
    Returns:
        ConvergenceDetector: Configured convergence detector
    """
    if detector_type == "adaptive":
        return AdaptiveConvergenceDetector(**kwargs)
    else:
        return ConvergenceDetector(**kwargs)


def analyze_convergence_patterns(convergence_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze convergence patterns from training history.
    
    Args:
        convergence_history: List of convergence data points
        
    Returns:
        Dict: Analysis results
    """
    if not convergence_history:
        return {'error': 'No convergence history provided'}
    
    # Extract metrics
    rounds = [h['round'] for h in convergence_history]
    accuracies = [h['accuracy'] for h in convergence_history]
    losses = [h['loss'] for h in convergence_history]
    scores = [h['convergence_score'] for h in convergence_history]
    
    # Calculate statistics
    analysis = {
        'training_rounds': len(convergence_history),
        'accuracy_stats': {
            'initial': accuracies[0] if accuracies else 0,
            'final': accuracies[-1] if accuracies else 0,
            'max': max(accuracies) if accuracies else 0,
            'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
        },
        'loss_stats': {
            'initial': losses[0] if losses else 0,
            'final': losses[-1] if losses else 0,
            'min': min(losses) if losses else 0,
            'reduction': losses[0] - losses[-1] if len(losses) > 1 else 0
        },
        'convergence_stats': {
            'final_score': scores[-1] if scores else 0,
            'min_score': min(scores) if scores else 0,
            'avg_score': np.mean(scores) if scores else 0,
            'trend': _calculate_trend_analysis(scores)
        }
    }
    
    # Determine convergence quality
    if analysis['accuracy_stats']['improvement'] > 0.05:
        convergence_quality = "excellent"
    elif analysis['accuracy_stats']['improvement'] > 0.02:
        convergence_quality = "good"
    elif analysis['accuracy_stats']['improvement'] > 0.01:
        convergence_quality = "fair"
    else:
        convergence_quality = "poor"
    
    analysis['convergence_quality'] = convergence_quality
    
    return analysis


def _calculate_trend_analysis(values: List[float]) -> Dict[str, Any]:
    """Calculate detailed trend analysis."""
    if len(values) < 3:
        return {'trend': 'insufficient_data'}
    
    # Calculate moving averages
    window = min(3, len(values) // 3)
    if window < 1:
        window = 1
    
    moving_avg = []
    for i in range(window, len(values) + 1):
        avg = np.mean(values[i-window:i])
        moving_avg.append(avg)
    
    # Calculate overall trend
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    
    # Determine trend direction and strength
    if slope < -0.01:
        trend_direction = "strongly_improving"
    elif slope < -0.001:
        trend_direction = "improving"
    elif slope > 0.01:
        trend_direction = "strongly_degrading"
    elif slope > 0.001:
        trend_direction = "degrading"
    else:
        trend_direction = "stable"
    
    return {
        'trend': trend_direction,
        'slope': slope,
        'r_squared': np.corrcoef(x, values)[0, 1] ** 2 if len(values) > 1 else 0,
        'volatility': np.std(values) if values else 0
    }