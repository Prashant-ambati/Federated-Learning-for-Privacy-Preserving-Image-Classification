"""
FedAvg (Federated Averaging) algorithm implementation.
Implements weighted averaging of client model updates for federated learning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import copy

from ..shared.models import ModelUpdate, GlobalModel, ModelWeights
from ..shared.interfaces import AggregationServiceInterface
from ..shared.validation import ModelUpdateValidator, validate_model_compatibility

logger = logging.getLogger(__name__)


class FedAvgError(Exception):
    """Custom exception for FedAvg aggregation errors."""
    pass


class FedAvgAggregator(AggregationServiceInterface):
    """
    Implements the FedAvg algorithm for federated learning.
    
    The FedAvg algorithm performs weighted averaging of client model updates
    based on the number of training samples each client used.
    """
    
    def __init__(self, 
                 min_clients: int = 2,
                 max_clients: Optional[int] = None,
                 validate_updates: bool = True):
        """
        Initialize FedAvg aggregator.
        
        Args:
            min_clients: Minimum number of clients required for aggregation
            max_clients: Maximum number of clients to include (None for no limit)
            validate_updates: Whether to validate model updates before aggregation
        """
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.validate_updates = validate_updates
        
        # Validation and statistics
        self.validator = ModelUpdateValidator() if validate_updates else None
        self.aggregation_history = []
        
        logger.info(f"FedAvg aggregator initialized - min_clients: {min_clients}, "
                   f"max_clients: {max_clients}, validation: {validate_updates}")
    
    def aggregate_updates(self, 
                         updates: List[ModelUpdate], 
                         weights: Optional[List[float]] = None) -> GlobalModel:
        """
        Aggregate client model updates using FedAvg algorithm.
        
        Args:
            updates: List of model updates from clients
            weights: Optional custom weights (if None, uses sample counts)
            
        Returns:
            GlobalModel: Aggregated global model
        """
        try:
            start_time = datetime.now()
            
            # Validate inputs
            self._validate_aggregation_inputs(updates, weights)
            
            # Filter and validate updates
            valid_updates = self._filter_and_validate_updates(updates)
            
            if len(valid_updates) < self.min_clients:
                raise FedAvgError(f"Insufficient valid updates: {len(valid_updates)} < {self.min_clients}")
            
            # Limit number of clients if specified
            if self.max_clients and len(valid_updates) > self.max_clients:
                # Select clients with most samples (or randomly if tied)
                valid_updates = sorted(valid_updates, key=lambda x: x.num_samples, reverse=True)
                valid_updates = valid_updates[:self.max_clients]
                logger.info(f"Limited to {self.max_clients} clients with most samples")
            
            # Calculate aggregation weights
            if weights is None:
                aggregation_weights = self._calculate_sample_weights(valid_updates)
            else:
                aggregation_weights = self._normalize_weights(weights[:len(valid_updates)])
            
            # Perform weighted averaging
            aggregated_weights = self._weighted_average(valid_updates, aggregation_weights)
            
            # Calculate aggregation metrics
            total_samples = sum(update.num_samples for update in valid_updates)
            avg_training_loss = sum(update.training_loss * weight 
                                  for update, weight in zip(valid_updates, aggregation_weights))
            
            # Create global model
            global_model = GlobalModel(
                round_number=valid_updates[0].round_number,  # Assume all same round
                model_weights=aggregated_weights,
                accuracy_metrics={},  # Will be filled by evaluation
                participating_clients=[update.client_id for update in valid_updates],
                convergence_score=0.0,  # Will be calculated separately
                created_at=datetime.now()
            )
            
            # Record aggregation statistics
            aggregation_time = (datetime.now() - start_time).total_seconds()
            self._record_aggregation_stats(valid_updates, aggregation_weights, 
                                         total_samples, avg_training_loss, aggregation_time)
            
            logger.info(f"FedAvg aggregation completed - {len(valid_updates)} clients, "
                       f"{total_samples} total samples, {aggregation_time:.2f}s")
            
            return global_model
            
        except Exception as e:
            logger.error(f"FedAvg aggregation failed: {str(e)}")
            raise FedAvgError(f"FedAvg aggregation failed: {str(e)}")
    
    def validate_update(self, update: ModelUpdate) -> bool:
        """Validate a single model update."""
        try:
            if not self.validate_updates or not self.validator:
                return True
            
            return self.validator.validate_model_update(update)
            
        except Exception as e:
            logger.error(f"Update validation failed for client {update.client_id}: {str(e)}")
            return False
    
    def compress_global_model(self, model: GlobalModel) -> bytes:
        """Compress global model for distribution (placeholder implementation)."""
        # This would typically use the compression service
        import pickle
        return pickle.dumps(model.model_weights)
    
    def calculate_convergence_metrics(self, 
                                    old_model: GlobalModel, 
                                    new_model: GlobalModel) -> float:
        """
        Calculate convergence metrics between two global models.
        
        Args:
            old_model: Previous global model
            new_model: New global model
            
        Returns:
            float: Convergence score (0.0 = no change, 1.0 = maximum change)
        """
        try:
            if not old_model or not new_model:
                return 1.0  # Maximum change if no previous model
            
            total_diff = 0.0
            total_norm = 0.0
            
            for layer_name in new_model.model_weights.keys():
                if layer_name in old_model.model_weights:
                    old_weights = old_model.model_weights[layer_name]
                    new_weights = new_model.model_weights[layer_name]
                    
                    # Calculate L2 difference
                    diff = torch.norm(new_weights - old_weights).item()
                    norm = torch.norm(new_weights).item()
                    
                    total_diff += diff
                    total_norm += norm
            
            # Normalize by total model norm
            if total_norm > 0:
                convergence_score = total_diff / total_norm
            else:
                convergence_score = 0.0
            
            # Clamp to [0, 1] range
            convergence_score = min(1.0, max(0.0, convergence_score))
            
            logger.debug(f"Convergence score: {convergence_score:.6f}")
            return convergence_score
            
        except Exception as e:
            logger.error(f"Convergence calculation failed: {str(e)}")
            return 0.0
    
    def _validate_aggregation_inputs(self, 
                                   updates: List[ModelUpdate], 
                                   weights: Optional[List[float]]):
        """Validate aggregation inputs."""
        if not updates:
            raise FedAvgError("No model updates provided")
        
        if weights is not None:
            if len(weights) != len(updates):
                raise FedAvgError("Number of weights must match number of updates")
            
            if any(w < 0 for w in weights):
                raise FedAvgError("All weights must be non-negative")
            
            if sum(weights) == 0:
                raise FedAvgError("Sum of weights cannot be zero")
    
    def _filter_and_validate_updates(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Filter and validate model updates."""
        valid_updates = []
        
        for update in updates:
            try:
                # Basic validation
                if update.num_samples <= 0:
                    logger.warning(f"Skipping update from {update.client_id}: invalid sample count")
                    continue
                
                if update.training_loss < 0:
                    logger.warning(f"Skipping update from {update.client_id}: invalid training loss")
                    continue
                
                # Detailed validation if enabled
                if self.validate_updates and not self.validate_update(update):
                    logger.warning(f"Skipping update from {update.client_id}: validation failed")
                    continue
                
                valid_updates.append(update)
                
            except Exception as e:
                logger.error(f"Error validating update from {update.client_id}: {str(e)}")
                continue
        
        # Check model compatibility
        if len(valid_updates) > 1:
            reference_weights = valid_updates[0].model_weights
            for i, update in enumerate(valid_updates[1:], 1):
                try:
                    validate_model_compatibility(reference_weights, update.model_weights)
                except Exception as e:
                    logger.warning(f"Removing incompatible update from {update.client_id}: {str(e)}")
                    valid_updates.pop(i)
        
        return valid_updates
    
    def _calculate_sample_weights(self, updates: List[ModelUpdate]) -> List[float]:
        """Calculate weights based on number of samples."""
        total_samples = sum(update.num_samples for update in updates)
        
        if total_samples == 0:
            # Equal weights if no sample information
            return [1.0 / len(updates)] * len(updates)
        
        weights = [update.num_samples / total_samples for update in updates]
        return weights
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1.0."""
        total_weight = sum(weights)
        
        if total_weight == 0:
            return [1.0 / len(weights)] * len(weights)
        
        return [w / total_weight for w in weights]
    
    def _weighted_average(self, 
                         updates: List[ModelUpdate], 
                         weights: List[float]) -> ModelWeights:
        """Perform weighted averaging of model weights."""
        if not updates:
            raise FedAvgError("No updates to aggregate")
        
        # Initialize aggregated weights with zeros
        aggregated_weights = {}
        reference_weights = updates[0].model_weights
        
        for layer_name, layer_weights in reference_weights.items():
            aggregated_weights[layer_name] = torch.zeros_like(layer_weights)
        
        # Weighted sum
        for update, weight in zip(updates, weights):
            for layer_name, layer_weights in update.model_weights.items():
                if layer_name in aggregated_weights:
                    aggregated_weights[layer_name] += weight * layer_weights
                else:
                    logger.warning(f"Layer {layer_name} not found in reference model")
        
        return aggregated_weights
    
    def _record_aggregation_stats(self, 
                                updates: List[ModelUpdate],
                                weights: List[float],
                                total_samples: int,
                                avg_training_loss: float,
                                aggregation_time: float):
        """Record aggregation statistics."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'num_clients': len(updates),
            'total_samples': total_samples,
            'avg_training_loss': avg_training_loss,
            'aggregation_time': aggregation_time,
            'client_weights': {
                update.client_id: weight 
                for update, weight in zip(updates, weights)
            },
            'client_samples': {
                update.client_id: update.num_samples 
                for update in updates
            }
        }
        
        self.aggregation_history.append(stats)
        
        # Keep only recent history (last 100 aggregations)
        if len(self.aggregation_history) > 100:
            self.aggregation_history = self.aggregation_history[-100:]
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        if not self.aggregation_history:
            return {'message': 'No aggregation history available'}
        
        recent_stats = self.aggregation_history[-10:]  # Last 10 aggregations
        
        return {
            'total_aggregations': len(self.aggregation_history),
            'recent_aggregations': len(recent_stats),
            'avg_clients_per_round': np.mean([s['num_clients'] for s in recent_stats]),
            'avg_samples_per_round': np.mean([s['total_samples'] for s in recent_stats]),
            'avg_aggregation_time': np.mean([s['aggregation_time'] for s in recent_stats]),
            'avg_training_loss': np.mean([s['avg_training_loss'] for s in recent_stats]),
            'client_participation': self._calculate_client_participation()
        }
    
    def _calculate_client_participation(self) -> Dict[str, Any]:
        """Calculate client participation statistics."""
        if not self.aggregation_history:
            return {}
        
        all_clients = set()
        client_counts = {}
        
        for stats in self.aggregation_history:
            for client_id in stats['client_weights'].keys():
                all_clients.add(client_id)
                client_counts[client_id] = client_counts.get(client_id, 0) + 1
        
        total_rounds = len(self.aggregation_history)
        
        return {
            'unique_clients': len(all_clients),
            'avg_participation_rate': np.mean(list(client_counts.values())) / total_rounds,
            'most_active_clients': sorted(client_counts.items(), 
                                        key=lambda x: x[1], reverse=True)[:5]
        }


class AdaptiveFedAvg(FedAvgAggregator):
    """
    Adaptive FedAvg that adjusts aggregation based on client performance.
    """
    
    def __init__(self, 
                 min_clients: int = 2,
                 max_clients: Optional[int] = None,
                 validate_updates: bool = True,
                 performance_weight: float = 0.1):
        """
        Initialize adaptive FedAvg aggregator.
        
        Args:
            min_clients: Minimum number of clients required
            max_clients: Maximum number of clients to include
            validate_updates: Whether to validate updates
            performance_weight: Weight for performance-based adjustment (0.0-1.0)
        """
        super().__init__(min_clients, max_clients, validate_updates)
        self.performance_weight = performance_weight
        self.client_performance_history = {}
        
        logger.info(f"Adaptive FedAvg initialized with performance weight: {performance_weight}")
    
    def aggregate_updates(self, 
                         updates: List[ModelUpdate], 
                         weights: Optional[List[float]] = None) -> GlobalModel:
        """Aggregate updates with adaptive weighting."""
        try:
            # Update client performance history
            self._update_performance_history(updates)
            
            # Calculate adaptive weights if not provided
            if weights is None:
                weights = self._calculate_adaptive_weights(updates)
            
            # Use parent aggregation with adaptive weights
            return super().aggregate_updates(updates, weights)
            
        except Exception as e:
            logger.error(f"Adaptive FedAvg aggregation failed: {str(e)}")
            raise FedAvgError(f"Adaptive FedAvg aggregation failed: {str(e)}")
    
    def _update_performance_history(self, updates: List[ModelUpdate]):
        """Update client performance history."""
        for update in updates:
            client_id = update.client_id
            
            if client_id not in self.client_performance_history:
                self.client_performance_history[client_id] = {
                    'losses': [],
                    'sample_counts': [],
                    'participation_count': 0
                }
            
            history = self.client_performance_history[client_id]
            history['losses'].append(update.training_loss)
            history['sample_counts'].append(update.num_samples)
            history['participation_count'] += 1
            
            # Keep only recent history
            max_history = 10
            if len(history['losses']) > max_history:
                history['losses'] = history['losses'][-max_history:]
                history['sample_counts'] = history['sample_counts'][-max_history:]
    
    def _calculate_adaptive_weights(self, updates: List[ModelUpdate]) -> List[float]:
        """Calculate adaptive weights based on samples and performance."""
        # Start with sample-based weights
        sample_weights = self._calculate_sample_weights(updates)
        
        if self.performance_weight == 0:
            return sample_weights
        
        # Calculate performance-based adjustments
        performance_adjustments = []
        
        for update in updates:
            client_id = update.client_id
            
            if client_id in self.client_performance_history:
                history = self.client_performance_history[client_id]
                
                # Better performance (lower loss) gets higher weight
                avg_loss = np.mean(history['losses'])
                max_loss = max(h['losses'] for h in self.client_performance_history.values() 
                              if h['losses'])
                
                if max_loss > 0:
                    # Invert loss (lower loss = higher adjustment)
                    performance_adj = 1.0 - (avg_loss / max_loss)
                else:
                    performance_adj = 1.0
            else:
                performance_adj = 1.0  # Neutral for new clients
            
            performance_adjustments.append(performance_adj)
        
        # Combine sample weights with performance adjustments
        adaptive_weights = []
        for sample_weight, perf_adj in zip(sample_weights, performance_adjustments):
            adaptive_weight = (1 - self.performance_weight) * sample_weight + \
                            self.performance_weight * perf_adj
            adaptive_weights.append(adaptive_weight)
        
        # Normalize
        return self._normalize_weights(adaptive_weights)


def create_fedavg_aggregator(aggregator_type: str = "standard", **kwargs) -> FedAvgAggregator:
    """
    Factory function to create FedAvg aggregator.
    
    Args:
        aggregator_type: Type of aggregator ("standard" or "adaptive")
        **kwargs: Aggregator-specific parameters
        
    Returns:
        FedAvgAggregator: Configured aggregator
    """
    if aggregator_type == "adaptive":
        return AdaptiveFedAvg(**kwargs)
    else:
        return FedAvgAggregator(**kwargs)


def benchmark_aggregation_performance(num_clients_list: List[int] = [5, 10, 25, 50],
                                    model_size: int = 1000000) -> Dict[str, Any]:
    """
    Benchmark FedAvg aggregation performance.
    
    Args:
        num_clients_list: List of client counts to test
        model_size: Approximate model size (number of parameters)
        
    Returns:
        Dict: Benchmark results
    """
    results = {}
    
    for num_clients in num_clients_list:
        try:
            # Create dummy model updates
            updates = []
            for i in range(num_clients):
                # Create dummy weights
                dummy_weights = {
                    'layer1': torch.randn(model_size // 4),
                    'layer2': torch.randn(model_size // 4),
                    'layer3': torch.randn(model_size // 4),
                    'layer4': torch.randn(model_size // 4)
                }
                
                update = ModelUpdate(
                    client_id=f"client_{i}",
                    round_number=1,
                    model_weights=dummy_weights,
                    num_samples=np.random.randint(100, 1000),
                    training_loss=np.random.uniform(0.1, 2.0),
                    privacy_budget_used=0.1,
                    compression_ratio=0.8,
                    timestamp=datetime.now()
                )
                updates.append(update)
            
            # Benchmark aggregation
            aggregator = FedAvgAggregator(validate_updates=False)
            
            import time
            start_time = time.time()
            global_model = aggregator.aggregate_updates(updates)
            aggregation_time = time.time() - start_time
            
            results[f"{num_clients}_clients"] = {
                'aggregation_time': aggregation_time,
                'throughput': num_clients / aggregation_time,
                'memory_usage': sum(w.numel() * w.element_size() 
                                  for w in global_model.model_weights.values()),
                'participating_clients': len(global_model.participating_clients)
            }
            
            logger.info(f"Benchmark {num_clients} clients: {aggregation_time:.4f}s")
            
        except Exception as e:
            logger.error(f"Benchmark failed for {num_clients} clients: {str(e)}")
            results[f"{num_clients}_clients"] = {'error': str(e)}
    
    return results