"""
Training round management for federated learning coordinator.
Manages the lifecycle of federated learning rounds including client coordination,
model distribution, update collection, and aggregation.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import uuid

from ..shared.models import (
    ModelUpdate, GlobalModel, RoundConfig, TrainingStatus,
    ClientCapabilities, PrivacyConfig
)
from ..aggregation.fedavg import FedAvgAggregator
from ..aggregation.convergence import ConvergenceDetector

logger = logging.getLogger(__name__)


class RoundState(Enum):
    """Training round states."""
    WAITING = "waiting"
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class ClientState(Enum):
    """Client states within a round."""
    REGISTERED = "registered"
    INVITED = "invited"
    TRAINING = "training"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TrainingRound:
    """Represents a single federated learning training round."""
    
    def __init__(self, 
                 round_number: int,
                 config: RoundConfig,
                 global_model: GlobalModel):
        """
        Initialize training round.
        
        Args:
            round_number: Round number
            config: Round configuration
            global_model: Global model for this round
        """
        self.round_number = round_number
        self.config = config
        self.global_model = global_model
        
        # Round state
        self.state = RoundState.WAITING
        self.start_time = None
        self.end_time = None
        self.timeout_time = None
        
        # Client management
        self.invited_clients: Set[str] = set()
        self.participating_clients: Dict[str, ClientState] = {}
        self.client_updates: Dict[str, ModelUpdate] = {}
        self.client_capabilities: Dict[str, ClientCapabilities] = {}
        
        # Results
        self.aggregated_model = None
        self.convergence_metrics = None
        self.round_metrics = {}
        
        # Synchronization
        self.lock = threading.RLock()
        
        logger.info(f"Training round {round_number} initialized")
    
    def add_client(self, client_id: str, capabilities: ClientCapabilities):
        """Add client to the round."""
        with self.lock:
            self.invited_clients.add(client_id)
            self.participating_clients[client_id] = ClientState.INVITED
            self.client_capabilities[client_id] = capabilities
            
            logger.debug(f"Added client {client_id} to round {self.round_number}")
    
    def remove_client(self, client_id: str):
        """Remove client from the round."""
        with self.lock:
            self.invited_clients.discard(client_id)
            self.participating_clients.pop(client_id, None)
            self.client_updates.pop(client_id, None)
            self.client_capabilities.pop(client_id, None)
            
            logger.debug(f"Removed client {client_id} from round {self.round_number}")
    
    def update_client_state(self, client_id: str, state: ClientState):
        """Update client state."""
        with self.lock:
            if client_id in self.participating_clients:
                old_state = self.participating_clients[client_id]
                self.participating_clients[client_id] = state
                
                logger.debug(f"Client {client_id} state: {old_state.value} -> {state.value}")
    
    def submit_update(self, client_id: str, update: ModelUpdate) -> bool:
        """Submit model update from client."""
        with self.lock:
            if client_id not in self.participating_clients:
                logger.warning(f"Update from unregistered client {client_id}")
                return False
            
            if self.state not in [RoundState.IN_PROGRESS, RoundState.COLLECTING]:
                logger.warning(f"Update from {client_id} rejected - round state: {self.state.value}")
                return False
            
            self.client_updates[client_id] = update
            self.update_client_state(client_id, ClientState.COMPLETED)
            
            logger.info(f"Received update from client {client_id}")
            return True
    
    def get_progress(self) -> float:
        """Get round progress (0.0 to 1.0)."""
        with self.lock:
            if not self.participating_clients:
                return 0.0
            
            completed_clients = sum(
                1 for state in self.participating_clients.values()
                if state in [ClientState.COMPLETED, ClientState.FAILED, ClientState.TIMEOUT]
            )
            
            return completed_clients / len(self.participating_clients)
    
    def is_ready_for_aggregation(self) -> bool:
        """Check if round is ready for aggregation."""
        with self.lock:
            completed_updates = len(self.client_updates)
            min_clients = self.config.min_clients
            
            return completed_updates >= min_clients
    
    def get_summary(self) -> Dict[str, Any]:
        """Get round summary."""
        with self.lock:
            return {
                'round_number': self.round_number,
                'state': self.state.value,
                'invited_clients': len(self.invited_clients),
                'participating_clients': len(self.participating_clients),
                'completed_updates': len(self.client_updates),
                'progress': self.get_progress(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
            }


class RoundManager:
    """Manages federated learning training rounds."""
    
    def __init__(self, 
                 aggregator: Optional[FedAvgAggregator] = None,
                 convergence_detector: Optional[ConvergenceDetector] = None,
                 max_concurrent_rounds: int = 1):
        """
        Initialize round manager.
        
        Args:
            aggregator: Model aggregation service
            convergence_detector: Convergence detection service
            max_concurrent_rounds: Maximum number of concurrent rounds
        """
        self.aggregator = aggregator or FedAvgAggregator()
        self.convergence_detector = convergence_detector or ConvergenceDetector()
        self.max_concurrent_rounds = max_concurrent_rounds
        
        # Round management
        self.rounds: Dict[int, TrainingRound] = {}
        self.current_round_number = 0
        self.active_rounds: Set[int] = set()
        self.round_history = deque(maxlen=100)
        
        # Client management
        self.registered_clients: Dict[str, ClientCapabilities] = {}
        self.client_round_assignments: Dict[str, int] = {}
        
        # Configuration
        self.default_config = RoundConfig(
            round_number=0,
            min_clients=2,
            max_clients=50,
            local_epochs=5,
            batch_size=32,
            learning_rate=0.001,
            timeout_seconds=300
        )
        
        # Callbacks
        self.on_round_started: Optional[Callable] = None
        self.on_round_completed: Optional[Callable] = None
        self.on_convergence_detected: Optional[Callable] = None
        
        # Synchronization
        self.lock = threading.RLock()
        self.running = False
        self.manager_thread = None
        
        logger.info("Round manager initialized")
    
    def start(self):
        """Start the round manager."""
        try:
            with self.lock:
                if self.running:
                    logger.warning("Round manager already running")
                    return
                
                self.running = True
                
                # Start management thread
                self.manager_thread = threading.Thread(
                    target=self._management_loop,
                    daemon=True
                )
                self.manager_thread.start()
                
                logger.info("Round manager started")
                
        except Exception as e:
            logger.error(f"Failed to start round manager: {e}")
            raise
    
    def stop(self):
        """Stop the round manager."""
        try:
            with self.lock:
                if not self.running:
                    return
                
                logger.info("Stopping round manager")
                self.running = False
                
                # Wait for management thread
                if self.manager_thread and self.manager_thread.is_alive():
                    self.manager_thread.join(timeout=5.0)
                
                # Complete any active rounds
                for round_num in list(self.active_rounds):
                    round_obj = self.rounds.get(round_num)
                    if round_obj:
                        round_obj.state = RoundState.FAILED
                        round_obj.end_time = datetime.now()
                
                logger.info("Round manager stopped")
                
        except Exception as e:
            logger.error(f"Error stopping round manager: {e}")
    
    def register_client(self, client_id: str, capabilities: ClientCapabilities) -> bool:
        """
        Register client with round manager.
        
        Args:
            client_id: Client identifier
            capabilities: Client capabilities
            
        Returns:
            bool: True if registration successful
        """
        try:
            with self.lock:
                self.registered_clients[client_id] = capabilities
                
                logger.info(f"Registered client {client_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            return False
    
    def unregister_client(self, client_id: str):
        """Unregister client from round manager."""
        try:
            with self.lock:
                # Remove from registered clients
                self.registered_clients.pop(client_id, None)
                
                # Remove from current round assignment
                assigned_round = self.client_round_assignments.pop(client_id, None)
                
                # Remove from active round
                if assigned_round and assigned_round in self.rounds:
                    self.rounds[assigned_round].remove_client(client_id)
                
                logger.info(f"Unregistered client {client_id}")
                
        except Exception as e:
            logger.error(f"Failed to unregister client {client_id}: {e}")
    
    def create_round(self, 
                    config: Optional[RoundConfig] = None,
                    global_model: Optional[GlobalModel] = None) -> Optional[TrainingRound]:
        """
        Create a new training round.
        
        Args:
            config: Round configuration (uses default if None)
            global_model: Global model for the round
            
        Returns:
            TrainingRound: Created round or None if failed
        """
        try:
            with self.lock:
                # Check concurrent round limit
                if len(self.active_rounds) >= self.max_concurrent_rounds:
                    logger.warning("Maximum concurrent rounds reached")
                    return None
                
                # Create round configuration
                round_config = config or self._create_default_config()
                round_config.round_number = self.current_round_number
                
                # Create or use provided global model
                if global_model is None:
                    global_model = self._create_initial_global_model()
                
                # Create training round
                training_round = TrainingRound(
                    self.current_round_number,
                    round_config,
                    global_model
                )
                
                # Store round
                self.rounds[self.current_round_number] = training_round
                self.active_rounds.add(self.current_round_number)
                
                logger.info(f"Created training round {self.current_round_number}")
                
                # Increment round number for next round
                self.current_round_number += 1
                
                return training_round
                
        except Exception as e:
            logger.error(f"Failed to create training round: {e}")
            return None
    
    def start_round(self, round_number: int) -> bool:
        """
        Start a training round.
        
        Args:
            round_number: Round number to start
            
        Returns:
            bool: True if round started successfully
        """
        try:
            with self.lock:
                if round_number not in self.rounds:
                    logger.error(f"Round {round_number} not found")
                    return False
                
                training_round = self.rounds[round_number]
                
                if training_round.state != RoundState.WAITING:
                    logger.warning(f"Round {round_number} not in waiting state")
                    return False
                
                # Select clients for the round
                selected_clients = self._select_clients_for_round(training_round.config)
                
                if len(selected_clients) < training_round.config.min_clients:
                    logger.error(f"Insufficient clients for round {round_number}")
                    return False
                
                # Add clients to round
                for client_id in selected_clients:
                    capabilities = self.registered_clients[client_id]
                    training_round.add_client(client_id, capabilities)
                    self.client_round_assignments[client_id] = round_number
                
                # Update round state
                training_round.state = RoundState.STARTING
                training_round.start_time = datetime.now()
                training_round.timeout_time = training_round.start_time + timedelta(
                    seconds=training_round.config.timeout_seconds
                )
                
                logger.info(f"Started round {round_number} with {len(selected_clients)} clients")
                
                # Trigger callback
                if self.on_round_started:
                    self.on_round_started(training_round)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to start round {round_number}: {e}")
            return False
    
    def submit_model_update(self, client_id: str, update: ModelUpdate) -> bool:
        """
        Submit model update from client.
        
        Args:
            client_id: Client identifier
            update: Model update
            
        Returns:
            bool: True if update accepted
        """
        try:
            with self.lock:
                # Find client's assigned round
                assigned_round = self.client_round_assignments.get(client_id)
                
                if assigned_round is None:
                    logger.warning(f"No round assignment for client {client_id}")
                    return False
                
                if assigned_round not in self.rounds:
                    logger.error(f"Assigned round {assigned_round} not found for client {client_id}")
                    return False
                
                # Submit update to round
                training_round = self.rounds[assigned_round]
                success = training_round.submit_update(client_id, update)
                
                if success:
                    # Check if round is ready for aggregation
                    if training_round.is_ready_for_aggregation():
                        self._trigger_aggregation(assigned_round)
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to submit update from client {client_id}: {e}")
            return False
    
    def get_round_status(self, round_number: int) -> Optional[Dict[str, Any]]:
        """Get status of a specific round."""
        with self.lock:
            if round_number not in self.rounds:
                return None
            
            return self.rounds[round_number].get_summary()
    
    def get_training_status(self) -> TrainingStatus:
        """Get overall training status."""
        with self.lock:
            # Find current active round
            current_round = max(self.active_rounds) if self.active_rounds else 0
            
            # Calculate metrics
            active_clients = len(self.registered_clients)
            round_progress = 0.0
            global_accuracy = 0.0
            convergence_score = 0.0
            
            if current_round in self.rounds:
                training_round = self.rounds[current_round]
                round_progress = training_round.get_progress()
                
                if training_round.global_model:
                    global_accuracy = training_round.global_model.get_accuracy() or 0.0
                    convergence_score = training_round.global_model.convergence_score
            
            # Estimate completion time
            estimated_completion = None
            if current_round in self.rounds:
                training_round = self.rounds[current_round]
                if training_round.timeout_time:
                    estimated_completion = training_round.timeout_time
            
            return TrainingStatus(
                current_round=current_round,
                active_clients=active_clients,
                round_progress=round_progress,
                global_accuracy=global_accuracy,
                convergence_score=convergence_score,
                estimated_completion=estimated_completion
            )
    
    def get_client_assignment(self, client_id: str) -> Optional[int]:
        """Get client's current round assignment."""
        with self.lock:
            return self.client_round_assignments.get(client_id)
    
    def _management_loop(self):
        """Main management loop."""
        while self.running:
            try:
                time.sleep(1.0)  # Check every second
                
                with self.lock:
                    current_time = datetime.now()
                    
                    # Check for round timeouts
                    for round_num in list(self.active_rounds):
                        training_round = self.rounds[round_num]
                        
                        if (training_round.timeout_time and 
                            current_time > training_round.timeout_time and
                            training_round.state in [RoundState.IN_PROGRESS, RoundState.COLLECTING]):
                            
                            logger.warning(f"Round {round_num} timed out")
                            self._handle_round_timeout(round_num)
                    
                    # Auto-start rounds if needed
                    if not self.active_rounds and self.registered_clients:
                        self._auto_start_round()
                
            except Exception as e:
                logger.error(f"Error in management loop: {e}")
    
    def _select_clients_for_round(self, config: RoundConfig) -> List[str]:
        """Select clients for a training round."""
        available_clients = list(self.registered_clients.keys())
        
        # Remove clients already assigned to active rounds
        for client_id in list(available_clients):
            if client_id in self.client_round_assignments:
                available_clients.remove(client_id)
        
        # Limit to max clients
        max_clients = min(config.max_clients, len(available_clients))
        
        # Simple selection - could be enhanced with capability-based selection
        selected = available_clients[:max_clients]
        
        logger.debug(f"Selected {len(selected)} clients from {len(available_clients)} available")
        return selected
    
    def _trigger_aggregation(self, round_number: int):
        """Trigger model aggregation for a round."""
        try:
            training_round = self.rounds[round_number]
            
            if training_round.state != RoundState.IN_PROGRESS:
                training_round.state = RoundState.COLLECTING
            
            # Start aggregation in background
            threading.Thread(
                target=self._perform_aggregation,
                args=(round_number,),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"Failed to trigger aggregation for round {round_number}: {e}")
    
    def _perform_aggregation(self, round_number: int):
        """Perform model aggregation for a round."""
        try:
            with self.lock:
                training_round = self.rounds[round_number]
                training_round.state = RoundState.AGGREGATING
                
                logger.info(f"Starting aggregation for round {round_number}")
                
                # Get model updates
                updates = list(training_round.client_updates.values())
                
                if not updates:
                    logger.error(f"No updates available for aggregation in round {round_number}")
                    training_round.state = RoundState.FAILED
                    return
                
                # Perform aggregation
                aggregated_model = self.aggregator.aggregate_updates(updates)
                
                # Calculate convergence metrics
                convergence_metrics = None
                if training_round.global_model:
                    convergence_metrics = self.convergence_detector.calculate_convergence_metrics(
                        aggregated_model, training_round.global_model
                    )
                    aggregated_model.convergence_score = convergence_metrics.convergence_score
                
                # Update round
                training_round.aggregated_model = aggregated_model
                training_round.convergence_metrics = convergence_metrics
                training_round.state = RoundState.COMPLETED
                training_round.end_time = datetime.now()
                
                # Remove from active rounds
                self.active_rounds.discard(round_number)
                
                # Add to history
                self.round_history.append(training_round.get_summary())
                
                # Clear client assignments
                for client_id in training_round.participating_clients:
                    self.client_round_assignments.pop(client_id, None)
                
                logger.info(f"Completed aggregation for round {round_number}")
                
                # Trigger callbacks
                if self.on_round_completed:
                    self.on_round_completed(training_round)
                
                if (convergence_metrics and convergence_metrics.is_converged and 
                    self.on_convergence_detected):
                    self.on_convergence_detected(convergence_metrics)
                
        except Exception as e:
            logger.error(f"Aggregation failed for round {round_number}: {e}")
            with self.lock:
                if round_number in self.rounds:
                    self.rounds[round_number].state = RoundState.FAILED
                    self.active_rounds.discard(round_number)
    
    def _handle_round_timeout(self, round_number: int):
        """Handle round timeout."""
        try:
            training_round = self.rounds[round_number]
            
            # Mark timed out clients
            for client_id, state in training_round.participating_clients.items():
                if state not in [ClientState.COMPLETED, ClientState.FAILED]:
                    training_round.update_client_state(client_id, ClientState.TIMEOUT)
            
            # Check if we have minimum updates for aggregation
            if training_round.is_ready_for_aggregation():
                logger.info(f"Round {round_number} timed out but has sufficient updates")
                self._trigger_aggregation(round_number)
            else:
                logger.warning(f"Round {round_number} timed out with insufficient updates")
                training_round.state = RoundState.FAILED
                training_round.end_time = datetime.now()
                self.active_rounds.discard(round_number)
                
                # Clear client assignments
                for client_id in training_round.participating_clients:
                    self.client_round_assignments.pop(client_id, None)
            
        except Exception as e:
            logger.error(f"Failed to handle timeout for round {round_number}: {e}")
    
    def _auto_start_round(self):
        """Automatically start a new round if conditions are met."""
        try:
            if len(self.registered_clients) >= self.default_config.min_clients:
                # Create and start new round
                training_round = self.create_round()
                if training_round:
                    self.start_round(training_round.round_number)
                    
        except Exception as e:
            logger.error(f"Failed to auto-start round: {e}")
    
    def _create_default_config(self) -> RoundConfig:
        """Create default round configuration."""
        return RoundConfig(
            round_number=self.current_round_number,
            min_clients=self.default_config.min_clients,
            max_clients=self.default_config.max_clients,
            local_epochs=self.default_config.local_epochs,
            batch_size=self.default_config.batch_size,
            learning_rate=self.default_config.learning_rate,
            timeout_seconds=self.default_config.timeout_seconds
        )
    
    def _create_initial_global_model(self) -> GlobalModel:
        """Create initial global model."""
        from ..shared.models_pytorch import ModelFactory
        
        # Create a simple CNN model
        model = ModelFactory.create_model("simple_cnn", num_classes=10)
        
        return GlobalModel(
            round_number=0,
            model_weights=model.get_model_weights(),
            accuracy_metrics={"test_accuracy": 0.1},
            participating_clients=[],
            convergence_score=1.0,
            created_at=datetime.now()
        )
    
    def set_callbacks(self,
                     on_round_started: Optional[Callable] = None,
                     on_round_completed: Optional[Callable] = None,
                     on_convergence_detected: Optional[Callable] = None):
        """Set callback functions for round events."""
        self.on_round_started = on_round_started
        self.on_round_completed = on_round_completed
        self.on_convergence_detected = on_convergence_detected
        
        logger.info("Round manager callbacks configured")