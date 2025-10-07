"""
Training status and metrics tracking for federated learning coordinator.
Collects, stores, and provides access to training metrics and system status.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
import json
import statistics

from ..shared.models import TrainingStatus, ModelUpdate, GlobalModel

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics for a single training round."""
    round_number: int
    start_time: datetime
    end_time: Optional[datetime]
    participating_clients: int
    successful_clients: int
    failed_clients: int
    total_samples: int
    avg_training_loss: float
    global_accuracy: float
    convergence_score: float
    aggregation_time: float
    round_duration: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data


@dataclass
class ClientMetrics:
    """Metrics for a single client."""
    client_id: str
    rounds_participated: int
    successful_rounds: int
    failed_rounds: int
    total_samples_contributed: int
    avg_training_loss: float
    avg_response_time: float
    last_seen: datetime
    reliability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['last_seen'] = self.last_seen.isoformat()
        return data


@dataclass
class SystemMetrics:
    """Overall system metrics."""
    total_rounds: int
    active_clients: int
    total_clients_ever: int
    avg_clients_per_round: float
    total_samples_processed: int
    current_global_accuracy: float
    best_global_accuracy: float
    avg_round_duration: float
    system_uptime: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates training metrics."""
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            history_size: Maximum number of historical records to keep
        """
        self.history_size = history_size
        
        # Round metrics
        self.round_metrics: Dict[int, RoundMetrics] = {}
        self.round_history = deque(maxlen=history_size)
        
        # Client metrics
        self.client_metrics: Dict[str, ClientMetrics] = {}
        self.client_history = defaultdict(lambda: deque(maxlen=100))
        
        # System metrics
        self.system_start_time = datetime.now()
        self.total_rounds = 0
        self.total_clients_ever = set()
        
        # Real-time tracking
        self.active_rounds: Dict[int, Dict[str, Any]] = {}
        self.current_status = TrainingStatus(
            current_round=0,
            active_clients=0,
            round_progress=0.0,
            global_accuracy=0.0,
            convergence_score=0.0,
            estimated_completion=None
        )
        
        # Synchronization
        self.lock = threading.RLock()
        
        logger.info("Metrics collector initialized")
    
    def start_round(self, 
                   round_number: int, 
                   participating_clients: List[str],
                   global_model: Optional[GlobalModel] = None):
        """Start tracking a new round."""
        with self.lock:
            start_time = datetime.now()
            
            # Initialize round tracking
            self.active_rounds[round_number] = {
                'start_time': start_time,
                'participating_clients': set(participating_clients),
                'completed_clients': set(),
                'failed_clients': set(),
                'client_updates': {},
                'total_samples': 0
            }
            
            # Update current status
            self.current_status.current_round = round_number
            self.current_status.active_clients = len(participating_clients)
            self.current_status.round_progress = 0.0
            
            if global_model:
                self.current_status.global_accuracy = global_model.get_accuracy() or 0.0
                self.current_status.convergence_score = global_model.convergence_score
            
            logger.info(f"Started tracking round {round_number} with {len(participating_clients)} clients")
    
    def record_client_update(self, client_id: str, update: ModelUpdate):
        """Record a client model update."""
        with self.lock:
            round_number = update.round_number
            
            if round_number not in self.active_rounds:
                logger.warning(f"Received update for inactive round {round_number}")
                return
            
            round_data = self.active_rounds[round_number]
            
            # Record update
            round_data['client_updates'][client_id] = {
                'timestamp': update.timestamp,
                'training_loss': update.training_loss,
                'num_samples': update.num_samples,
                'privacy_budget_used': update.privacy_budget_used
            }
            
            round_data['completed_clients'].add(client_id)
            round_data['total_samples'] += update.num_samples
            
            # Update client metrics
            self._update_client_metrics(client_id, update)
            
            # Update round progress
            progress = len(round_data['completed_clients']) / len(round_data['participating_clients'])
            self.current_status.round_progress = progress
            
            logger.debug(f"Recorded update from client {client_id} for round {round_number}")
    
    def record_client_failure(self, client_id: str, round_number: int, failure_reason: str):
        """Record a client failure."""
        with self.lock:
            if round_number not in self.active_rounds:
                return
            
            round_data = self.active_rounds[round_number]
            round_data['failed_clients'].add(client_id)
            
            # Update client metrics
            if client_id in self.client_metrics:
                self.client_metrics[client_id].failed_rounds += 1
            
            logger.debug(f"Recorded failure for client {client_id} in round {round_number}: {failure_reason}")
    
    def complete_round(self, 
                      round_number: int, 
                      global_model: GlobalModel,
                      aggregation_time: float):
        """Complete round tracking and calculate final metrics."""
        with self.lock:
            if round_number not in self.active_rounds:
                logger.warning(f"Attempting to complete inactive round {round_number}")
                return
            
            round_data = self.active_rounds[round_number]
            end_time = datetime.now()
            
            # Calculate round metrics
            participating_clients = len(round_data['participating_clients'])
            successful_clients = len(round_data['completed_clients'])
            failed_clients = len(round_data['failed_clients'])
            total_samples = round_data['total_samples']
            
            # Calculate average training loss
            updates = round_data['client_updates'].values()
            if updates:
                weighted_loss = sum(u['training_loss'] * u['num_samples'] for u in updates)
                avg_training_loss = weighted_loss / total_samples if total_samples > 0 else 0.0
            else:
                avg_training_loss = 0.0
            
            # Create round metrics
            round_metrics = RoundMetrics(
                round_number=round_number,
                start_time=round_data['start_time'],
                end_time=end_time,
                participating_clients=participating_clients,
                successful_clients=successful_clients,
                failed_clients=failed_clients,
                total_samples=total_samples,
                avg_training_loss=avg_training_loss,
                global_accuracy=global_model.get_accuracy() or 0.0,
                convergence_score=global_model.convergence_score,
                aggregation_time=aggregation_time,
                round_duration=(end_time - round_data['start_time']).total_seconds()
            )
            
            # Store metrics
            self.round_metrics[round_number] = round_metrics
            self.round_history.append(round_metrics)
            
            # Update system metrics
            self.total_rounds += 1
            self.total_clients_ever.update(round_data['participating_clients'])
            
            # Update current status
            self.current_status.global_accuracy = round_metrics.global_accuracy
            self.current_status.convergence_score = round_metrics.convergence_score
            self.current_status.round_progress = 1.0
            
            # Clean up active round
            del self.active_rounds[round_number]
            
            logger.info(f"Completed round {round_number} metrics: "
                       f"accuracy={round_metrics.global_accuracy:.4f}, "
                       f"clients={successful_clients}/{participating_clients}")
    
    def get_current_status(self) -> TrainingStatus:
        """Get current training status."""
        with self.lock:
            return TrainingStatus(
                current_round=self.current_status.current_round,
                active_clients=self.current_status.active_clients,
                round_progress=self.current_status.round_progress,
                global_accuracy=self.current_status.global_accuracy,
                convergence_score=self.current_status.convergence_score,
                estimated_completion=self._estimate_completion_time()
            )
    
    def get_round_metrics(self, round_number: int) -> Optional[RoundMetrics]:
        """Get metrics for a specific round."""
        with self.lock:
            return self.round_metrics.get(round_number)
    
    def get_client_metrics(self, client_id: str) -> Optional[ClientMetrics]:
        """Get metrics for a specific client."""
        with self.lock:
            return self.client_metrics.get(client_id)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get overall system metrics."""
        with self.lock:
            # Calculate averages
            if self.round_history:
                avg_clients_per_round = statistics.mean(
                    r.participating_clients for r in self.round_history
                )
                total_samples = sum(r.total_samples for r in self.round_history)
                
                completed_rounds = [r for r in self.round_history if r.round_duration is not None]
                avg_round_duration = statistics.mean(
                    r.round_duration for r in completed_rounds
                ) if completed_rounds else 0.0
                
                best_accuracy = max(r.global_accuracy for r in self.round_history)
                current_accuracy = self.round_history[-1].global_accuracy if self.round_history else 0.0
            else:
                avg_clients_per_round = 0.0
                total_samples = 0
                avg_round_duration = 0.0
                best_accuracy = 0.0
                current_accuracy = 0.0
            
            system_uptime = (datetime.now() - self.system_start_time).total_seconds()
            
            return SystemMetrics(
                total_rounds=self.total_rounds,
                active_clients=len([c for c in self.client_metrics.values() 
                                  if (datetime.now() - c.last_seen).total_seconds() < 3600]),
                total_clients_ever=len(self.total_clients_ever),
                avg_clients_per_round=avg_clients_per_round,
                total_samples_processed=total_samples,
                current_global_accuracy=current_accuracy,
                best_global_accuracy=best_accuracy,
                avg_round_duration=avg_round_duration,
                system_uptime=system_uptime
            )
    
    def get_recent_rounds(self, count: int = 10) -> List[RoundMetrics]:
        """Get recent round metrics."""
        with self.lock:
            return list(self.round_history)[-count:]
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress over time."""
        with self.lock:
            if not self.round_history:
                return {'rounds': [], 'accuracy_trend': [], 'loss_trend': []}
            
            rounds = []
            accuracy_trend = []
            loss_trend = []
            
            for round_metrics in self.round_history:
                rounds.append(round_metrics.round_number)
                accuracy_trend.append(round_metrics.global_accuracy)
                loss_trend.append(round_metrics.avg_training_loss)
            
            return {
                'rounds': rounds,
                'accuracy_trend': accuracy_trend,
                'loss_trend': loss_trend,
                'convergence_trend': [r.convergence_score for r in self.round_history]
            }
    
    def get_client_participation_stats(self) -> Dict[str, Any]:
        """Get client participation statistics."""
        with self.lock:
            if not self.client_metrics:
                return {'total_clients': 0, 'active_clients': 0, 'participation_distribution': {}}
            
            # Calculate participation distribution
            participation_counts = defaultdict(int)
            for client_metrics in self.client_metrics.values():
                participation_counts[client_metrics.rounds_participated] += 1
            
            # Active clients (seen in last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            active_clients = sum(
                1 for c in self.client_metrics.values()
                if c.last_seen > cutoff_time
            )
            
            return {
                'total_clients': len(self.client_metrics),
                'active_clients': active_clients,
                'participation_distribution': dict(participation_counts),
                'avg_reliability': statistics.mean(
                    c.reliability_score for c in self.client_metrics.values()
                ) if self.client_metrics else 0.0
            }
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file."""
        with self.lock:
            try:
                export_data = {
                    'system_metrics': self.get_system_metrics().to_dict(),
                    'round_metrics': [r.to_dict() for r in self.round_history],
                    'client_metrics': {
                        client_id: metrics.to_dict()
                        for client_id, metrics in self.client_metrics.items()
                    },
                    'training_progress': self.get_training_progress(),
                    'export_timestamp': datetime.now().isoformat()
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Metrics exported to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")
    
    def _update_client_metrics(self, client_id: str, update: ModelUpdate):
        """Update client-specific metrics."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = ClientMetrics(
                client_id=client_id,
                rounds_participated=0,
                successful_rounds=0,
                failed_rounds=0,
                total_samples_contributed=0,
                avg_training_loss=0.0,
                avg_response_time=0.0,
                last_seen=datetime.now(),
                reliability_score=1.0
            )
        
        metrics = self.client_metrics[client_id]
        
        # Update metrics
        metrics.rounds_participated += 1
        metrics.successful_rounds += 1
        metrics.total_samples_contributed += update.num_samples
        metrics.last_seen = update.timestamp
        
        # Update average training loss
        if metrics.avg_training_loss == 0:
            metrics.avg_training_loss = update.training_loss
        else:
            # Exponential moving average
            metrics.avg_training_loss = (metrics.avg_training_loss * 0.9) + (update.training_loss * 0.1)
        
        # Update reliability score
        if metrics.rounds_participated > 0:
            metrics.reliability_score = metrics.successful_rounds / metrics.rounds_participated
        
        # Add to client history
        self.client_history[client_id].append({
            'timestamp': update.timestamp.isoformat(),
            'round_number': update.round_number,
            'training_loss': update.training_loss,
            'num_samples': update.num_samples
        })
    
    def _estimate_completion_time(self) -> Optional[datetime]:
        """Estimate completion time for current round."""
        if not self.active_rounds:
            return None
        
        current_round = max(self.active_rounds.keys())
        round_data = self.active_rounds[current_round]
        
        # Calculate average time per client based on completed clients
        completed_count = len(round_data['completed_clients'])
        if completed_count == 0:
            return None
        
        elapsed_time = datetime.now() - round_data['start_time']
        avg_time_per_client = elapsed_time.total_seconds() / completed_count
        
        # Estimate remaining time
        remaining_clients = len(round_data['participating_clients']) - completed_count
        estimated_remaining_seconds = remaining_clients * avg_time_per_client
        
        return datetime.now() + timedelta(seconds=estimated_remaining_seconds)


class MetricsTracker:
    """Main metrics tracking service for the coordinator."""
    
    def __init__(self, 
                 collector: Optional[MetricsCollector] = None,
                 update_interval: float = 10.0):
        """
        Initialize metrics tracker.
        
        Args:
            collector: Metrics collector instance
            update_interval: Metrics update interval in seconds
        """
        self.collector = collector or MetricsCollector()
        self.update_interval = update_interval
        
        # Background tracking
        self.running = False
        self.tracker_thread = None
        
        # Callbacks
        self.on_metrics_updated = None
        
        logger.info("Metrics tracker initialized")
    
    def start(self):
        """Start metrics tracking."""
        try:
            if self.running:
                return
            
            self.running = True
            
            # Start tracking thread
            self.tracker_thread = threading.Thread(
                target=self._tracking_loop,
                daemon=True
            )
            self.tracker_thread.start()
            
            logger.info("Metrics tracker started")
            
        except Exception as e:
            logger.error(f"Failed to start metrics tracker: {e}")
            raise
    
    def stop(self):
        """Stop metrics tracking."""
        try:
            if not self.running:
                return
            
            logger.info("Stopping metrics tracker")
            self.running = False
            
            # Wait for tracking thread
            if self.tracker_thread and self.tracker_thread.is_alive():
                self.tracker_thread.join(timeout=5.0)
            
            logger.info("Metrics tracker stopped")
            
        except Exception as e:
            logger.error(f"Error stopping metrics tracker: {e}")
    
    def get_collector(self) -> MetricsCollector:
        """Get the metrics collector instance."""
        return self.collector
    
    def _tracking_loop(self):
        """Background tracking loop."""
        while self.running:
            try:
                time.sleep(self.update_interval)
                
                if not self.running:
                    break
                
                # Trigger metrics update callback
                if self.on_metrics_updated:
                    current_status = self.collector.get_current_status()
                    self.on_metrics_updated(current_status)
                
            except Exception as e:
                logger.error(f"Error in metrics tracking loop: {e}")
    
    def set_callback(self, on_metrics_updated: Optional[callable] = None):
        """Set metrics update callback."""
        self.on_metrics_updated = on_metrics_updated