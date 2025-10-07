"""
Federated training workflow for clients.
Orchestrates the complete federated learning process from a client perspective.
"""

import torch
import asyncio
import threading
import time
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum

from ..shared.models import (
    ModelUpdate, GlobalModel, ClientCapabilities, PrivacyConfig,
    TrainingMetrics, ComputePowerLevel
)
from ..shared.models_pytorch import ModelFactory, FederatedCNNBase
from ..shared.training import LocalTrainer, FederatedTrainingConfig, create_adaptive_config
from ..shared.privacy import DifferentialPrivacyEngine, create_privacy_engine
from ..shared.compression import ModelCompressionService, create_compression_service
from ..shared.data_loader import create_data_loader, DataLoaderInterface
from .grpc_client import FederatedLearningClient, ClientConnectionManager

logger = logging.getLogger(__name__)


class ClientState(Enum):
    """Client training states."""
    IDLE = "idle"
    CONNECTING = "connecting"
    REGISTERED = "registered"
    WAITING_FOR_ROUND = "waiting_for_round"
    DOWNLOADING_MODEL = "downloading_model"
    TRAINING = "training"
    APPLYING_PRIVACY = "applying_privacy"
    UPLOADING_UPDATE = "uploading_update"
    ROUND_COMPLETE = "round_complete"
    ERROR = "error"


class FederatedTrainer:
    """Main federated training orchestrator for clients."""
    
    def __init__(self, 
                 client_id: str,
                 coordinator_address: str,
                 capabilities: ClientCapabilities,
                 model_type: str = "simple_cnn",
                 dataset_name: str = "mnist",
                 data_dir: str = "./data",
                 checkpoint_dir: str = "./checkpoints"):
        """
        Initialize federated trainer.
        
        Args:
            client_id: Unique client identifier
            coordinator_address: Coordinator server address
            capabilities: Client computational capabilities
            model_type: Type of model to use
            dataset_name: Dataset name ("mnist", "cifar10")
            data_dir: Directory for data storage
            checkpoint_dir: Directory for model checkpoints
        """
        self.client_id = client_id
        self.coordinator_address = coordinator_address
        self.capabilities = capabilities
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Training components
        self.model: Optional[FederatedCNNBase] = None
        self.local_trainer: Optional[LocalTrainer] = None
        self.privacy_engine: Optional[DifferentialPrivacyEngine] = None
        self.compression_service: Optional[ModelCompressionService] = None
        self.data_loader: Optional[DataLoaderInterface] = None
        
        # Communication
        self.grpc_client: Optional[FederatedLearningClient] = None
        self.connection_manager: Optional[ClientConnectionManager] = None
        
        # Training state
        self.state = ClientState.IDLE
        self.current_round = 0
        self.global_model: Optional[GlobalModel] = None
        self.round_config: Optional[Dict[str, Any]] = None
        self.training_config: Optional[FederatedTrainingConfig] = None
        
        # Metrics and history
        self.training_history = []
        self.round_metrics = {}
        
        # Callbacks
        self.on_state_changed: Optional[Callable] = None
        self.on_round_started: Optional[Callable] = None
        self.on_round_completed: Optional[Callable] = None
        self.on_training_completed: Optional[Callable] = None
        
        # Control
        self.running = False
        self.training_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        logger.info(f"Federated trainer initialized for client {client_id}")
    
    def initialize(self) -> bool:
        """
        Initialize all training components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing federated trainer components")
            
            # Initialize model
            self.model = ModelFactory.create_model(
                self.model_type,
                num_classes=10 if self.dataset_name in ["mnist", "cifar10"] else 10
            )
            
            # Initialize local trainer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.local_trainer = LocalTrainer(
                model=self.model,
                device=device,
                checkpoint_dir=self.checkpoint_dir
            )
            
            # Initialize privacy engine
            privacy_config = self.capabilities.privacy_requirements
            self.privacy_engine = create_privacy_engine(
                epsilon=privacy_config.epsilon,
                delta=privacy_config.delta,
                max_grad_norm=privacy_config.max_grad_norm,
                noise_multiplier=privacy_config.noise_multiplier,
                device=device
            )
            
            # Initialize compression service
            self.compression_service = create_compression_service("lz4")
            
            # Initialize data loader
            self.data_loader = create_data_loader(
                dataset_name=self.dataset_name,
                data_dir=self.data_dir,
                num_clients=100,  # Assume up to 100 clients
                partition_strategy="non_iid",
                batch_size=32,
                validation_split=0.1,
                download=True
            )
            
            # Initialize gRPC client
            self.grpc_client = FederatedLearningClient(
                server_address=self.coordinator_address,
                client_id=self.client_id,
                capabilities=self.capabilities
            )
            
            # Set up callbacks
            self.grpc_client.set_callbacks(
                on_model_received=self._on_model_received,
                on_round_started=self._on_round_started,
                on_training_completed=self._on_training_completed
            )
            
            # Initialize connection manager
            self.connection_manager = ClientConnectionManager(
                client=self.grpc_client,
                heartbeat_interval=30.0,
                auto_reconnect=True
            )
            
            logger.info("Federated trainer initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize federated trainer: {e}")
            return False
    
    def start_training(self) -> bool:
        """
        Start federated training process.
        
        Returns:
            bool: True if training started successfully
        """
        try:
            with self.lock:
                if self.running:
                    logger.warning("Training already running")
                    return False
                
                if not self.model or not self.grpc_client:
                    logger.error("Trainer not initialized")
                    return False
                
                self.running = True
                self._set_state(ClientState.CONNECTING)
                
                # Start connection manager
                self.connection_manager.start()
                
                # Start training thread
                self.training_thread = threading.Thread(
                    target=self._training_loop,
                    daemon=True
                )
                self.training_thread.start()
                
                logger.info("Federated training started")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            self.running = False
            return False
    
    def stop_training(self):
        """Stop federated training process."""
        try:
            with self.lock:
                if not self.running:
                    return
                
                logger.info("Stopping federated training")
                self.running = False
                
                # Stop connection manager
                if self.connection_manager:
                    self.connection_manager.stop()
                
                # Wait for training thread
                if self.training_thread and self.training_thread.is_alive():
                    self.training_thread.join(timeout=10.0)
                
                self._set_state(ClientState.IDLE)
                logger.info("Federated training stopped")
                
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        with self.lock:
            return {
                'client_id': self.client_id,
                'state': self.state.value,
                'current_round': self.current_round,
                'running': self.running,
                'model_type': self.model_type,
                'dataset': self.dataset_name,
                'rounds_completed': len(self.training_history),
                'last_accuracy': self.round_metrics.get('accuracy', 0.0),
                'last_loss': self.round_metrics.get('loss', 0.0)
            }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        with self.lock:
            return self.training_history.copy()
    
    def _training_loop(self):
        """Main training loop."""
        while self.running:
            try:
                if self.state == ClientState.CONNECTING:
                    # Wait for connection to be established
                    time.sleep(1.0)
                    if self.grpc_client.registered:
                        self._set_state(ClientState.REGISTERED)
                
                elif self.state == ClientState.REGISTERED:
                    # Wait for round invitation or start new round
                    self._wait_for_round()
                
                elif self.state == ClientState.WAITING_FOR_ROUND:
                    # Check for new round
                    self._check_for_new_round()
                
                elif self.state == ClientState.DOWNLOADING_MODEL:
                    # Download global model
                    self._download_global_model()
                
                elif self.state == ClientState.TRAINING:
                    # Perform local training
                    self._perform_local_training()
                
                elif self.state == ClientState.APPLYING_PRIVACY:
                    # Apply differential privacy
                    self._apply_differential_privacy()
                
                elif self.state == ClientState.UPLOADING_UPDATE:
                    # Upload model update
                    self._upload_model_update()
                
                elif self.state == ClientState.ROUND_COMPLETE:
                    # Round completed, wait for next
                    self._complete_round()
                
                elif self.state == ClientState.ERROR:
                    # Handle error state
                    self._handle_error()
                
                else:
                    # Default wait
                    time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                self._set_state(ClientState.ERROR)
                time.sleep(5.0)  # Wait before retrying
    
    def _wait_for_round(self):
        """Wait for round invitation."""
        try:
            # Try to join a training round
            round_config = self.grpc_client.join_training_round()
            
            if round_config:
                self.round_config = round_config
                self.current_round = round_config['round_number']
                
                # Create adaptive training configuration
                self.training_config = create_adaptive_config({
                    'compute_power': self.capabilities.compute_power.value,
                    'network_bandwidth': self.capabilities.network_bandwidth,
                    'available_samples': self.capabilities.available_samples
                })
                
                # Override with round config
                self.training_config.local_epochs = round_config.get('local_epochs', 5)
                self.training_config.batch_size = round_config.get('batch_size', 32)
                self.training_config.learning_rate = round_config.get('learning_rate', 0.001)
                
                logger.info(f"Joined round {self.current_round}")
                self._set_state(ClientState.DOWNLOADING_MODEL)
            else:
                # No round available, wait
                self._set_state(ClientState.WAITING_FOR_ROUND)
                time.sleep(5.0)
                
        except Exception as e:
            logger.error(f"Failed to join round: {e}")
            self._set_state(ClientState.ERROR)
    
    def _check_for_new_round(self):
        """Check for new training round."""
        try:
            # Get training status to check for new rounds
            status = self.grpc_client.get_training_status()
            
            if status and status['current_round'] > self.current_round:
                logger.info(f"New round available: {status['current_round']}")
                self._set_state(ClientState.REGISTERED)
            else:
                time.sleep(10.0)  # Wait 10 seconds before checking again
                
        except Exception as e:
            logger.error(f"Failed to check for new round: {e}")
            time.sleep(10.0)
    
    def _download_global_model(self):
        """Download global model from coordinator."""
        try:
            logger.info(f"Downloading global model for round {self.current_round}")
            
            global_model = self.grpc_client.get_global_model(self.current_round)
            
            if global_model:
                self.global_model = global_model
                
                # Update local model with global weights
                self.model.set_model_weights(global_model.model_weights)
                
                logger.info("Global model downloaded and applied")
                self._set_state(ClientState.TRAINING)
            else:
                logger.error("Failed to download global model")
                self._set_state(ClientState.ERROR)
                
        except Exception as e:
            logger.error(f"Failed to download global model: {e}")
            self._set_state(ClientState.ERROR)
    
    def _perform_local_training(self):
        """Perform local model training."""
        try:
            logger.info(f"Starting local training for round {self.current_round}")
            
            # Load training data
            train_loader = self.data_loader.load_training_data(self.client_id)
            val_loader = self.data_loader.load_validation_data(self.client_id)
            
            # Perform training
            training_metrics = self.local_trainer.train_local_model(
                train_loader=train_loader,
                epochs=self.training_config.local_epochs,
                learning_rate=self.training_config.learning_rate,
                optimizer_type=self.training_config.optimizer_type,
                validation_loader=val_loader,
                save_checkpoints=self.training_config.save_checkpoints
            )
            
            # Store metrics
            self.round_metrics = {
                'round': self.current_round,
                'loss': training_metrics.loss,
                'accuracy': training_metrics.accuracy,
                'epochs': training_metrics.epochs_completed,
                'training_time': training_metrics.training_time,
                'samples': training_metrics.samples_processed
            }
            
            logger.info(f"Local training completed - Loss: {training_metrics.loss:.4f}, "
                       f"Accuracy: {training_metrics.accuracy:.4f}")
            
            self._set_state(ClientState.APPLYING_PRIVACY)
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
            self._set_state(ClientState.ERROR)
    
    def _apply_differential_privacy(self):
        """Apply differential privacy to model update."""
        try:
            logger.info("Applying differential privacy")
            
            # Get model gradients (difference from global model)
            current_weights = self.model.get_model_weights()
            global_weights = self.global_model.model_weights
            
            # Calculate weight differences (gradients)
            gradients = {}
            for layer_name in current_weights.keys():
                if layer_name in global_weights:
                    gradients[layer_name] = current_weights[layer_name] - global_weights[layer_name]
                else:
                    gradients[layer_name] = current_weights[layer_name]
            
            # Apply differential privacy
            privacy_config = self.capabilities.privacy_requirements
            noisy_gradients = self.privacy_engine.add_noise(
                gradients,
                epsilon=privacy_config.epsilon,
                delta=privacy_config.delta
            )
            
            # Apply noisy gradients back to model
            noisy_weights = {}
            for layer_name in global_weights.keys():
                if layer_name in noisy_gradients:
                    noisy_weights[layer_name] = global_weights[layer_name] + noisy_gradients[layer_name]
                else:
                    noisy_weights[layer_name] = global_weights[layer_name]
            
            # Update model with noisy weights
            self.model.set_model_weights(noisy_weights)
            
            logger.info("Differential privacy applied")
            self._set_state(ClientState.UPLOADING_UPDATE)
            
        except Exception as e:
            logger.error(f"Failed to apply differential privacy: {e}")
            self._set_state(ClientState.ERROR)
    
    def _upload_model_update(self):
        """Upload model update to coordinator."""
        try:
            logger.info("Uploading model update")
            
            # Create model update
            model_update = ModelUpdate(
                client_id=self.client_id,
                round_number=self.current_round,
                model_weights=self.model.get_model_weights(),
                num_samples=self.round_metrics['samples'],
                training_loss=self.round_metrics['loss'],
                privacy_budget_used=self.capabilities.privacy_requirements.epsilon,
                compression_ratio=0.8,  # Placeholder
                timestamp=datetime.now()
            )
            
            # Submit update
            success = self.grpc_client.submit_model_update(model_update)
            
            if success:
                logger.info("Model update uploaded successfully")
                self._set_state(ClientState.ROUND_COMPLETE)
            else:
                logger.error("Failed to upload model update")
                self._set_state(ClientState.ERROR)
                
        except Exception as e:
            logger.error(f"Failed to upload model update: {e}")
            self._set_state(ClientState.ERROR)
    
    def _complete_round(self):
        """Complete current round and prepare for next."""
        try:
            # Record round in history
            round_record = {
                'round_number': self.current_round,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.round_metrics.copy(),
                'training_config': self.training_config.to_dict() if self.training_config else {}
            }
            
            self.training_history.append(round_record)
            
            logger.info(f"Round {self.current_round} completed")
            
            # Trigger callback
            if self.on_round_completed:
                self.on_round_completed(round_record)
            
            # Wait for next round
            self._set_state(ClientState.WAITING_FOR_ROUND)
            
        except Exception as e:
            logger.error(f"Failed to complete round: {e}")
            self._set_state(ClientState.ERROR)
    
    def _handle_error(self):
        """Handle error state."""
        logger.warning("In error state, attempting recovery")
        
        # Wait before attempting recovery
        time.sleep(10.0)
        
        # Try to reconnect
        if self.grpc_client and not self.grpc_client.connected:
            try:
                if self.grpc_client.connect() and self.grpc_client.register():
                    self._set_state(ClientState.REGISTERED)
                    return
            except Exception as e:
                logger.error(f"Recovery attempt failed: {e}")
        
        # If still in error, continue waiting
        time.sleep(30.0)
    
    def _set_state(self, new_state: ClientState):
        """Set client state and trigger callback."""
        with self.lock:
            old_state = self.state
            self.state = new_state
            
            logger.debug(f"State changed: {old_state.value} -> {new_state.value}")
            
            # Trigger callback
            if self.on_state_changed:
                self.on_state_changed(old_state, new_state)
    
    def _on_model_received(self, global_model: GlobalModel):
        """Callback for when global model is received."""
        logger.debug("Global model received callback")
    
    def _on_round_started(self, round_config: Dict[str, Any]):
        """Callback for when round starts."""
        logger.debug(f"Round started callback: {round_config}")
        
        if self.on_round_started:
            self.on_round_started(round_config)
    
    def _on_training_completed(self):
        """Callback for when training is completed."""
        logger.debug("Training completed callback")
        
        if self.on_training_completed:
            self.on_training_completed()
    
    def set_callbacks(self,
                     on_state_changed: Optional[Callable] = None,
                     on_round_started: Optional[Callable] = None,
                     on_round_completed: Optional[Callable] = None,
                     on_training_completed: Optional[Callable] = None):
        """Set callback functions for training events."""
        self.on_state_changed = on_state_changed
        self.on_round_started = on_round_started
        self.on_round_completed = on_round_completed
        self.on_training_completed = on_training_completed
        
        logger.info("Federated trainer callbacks configured")


def create_federated_trainer(client_id: str,
                           coordinator_address: str,
                           compute_power: str = "medium",
                           network_bandwidth: int = 10,
                           available_samples: int = 1000,
                           model_type: str = "simple_cnn",
                           dataset_name: str = "mnist",
                           privacy_epsilon: float = 1.0,
                           privacy_delta: float = 1e-5) -> FederatedTrainer:
    """
    Factory function to create federated trainer.
    
    Args:
        client_id: Unique client identifier
        coordinator_address: Coordinator server address
        compute_power: Client compute power level
        network_bandwidth: Network bandwidth in Mbps
        available_samples: Number of available training samples
        model_type: Type of model to use
        dataset_name: Dataset name
        privacy_epsilon: Privacy epsilon parameter
        privacy_delta: Privacy delta parameter
        
    Returns:
        FederatedTrainer: Configured federated trainer
    """
    # Create client capabilities
    compute_power_map = {
        "low": ComputePowerLevel.LOW,
        "medium": ComputePowerLevel.MEDIUM,
        "high": ComputePowerLevel.HIGH
    }
    
    from ..shared.models import PrivacyConfig
    
    capabilities = ClientCapabilities(
        compute_power=compute_power_map.get(compute_power, ComputePowerLevel.MEDIUM),
        network_bandwidth=network_bandwidth,
        available_samples=available_samples,
        supported_models=[model_type],
        privacy_requirements=PrivacyConfig(
            epsilon=privacy_epsilon,
            delta=privacy_delta,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )
    )
    
    return FederatedTrainer(
        client_id=client_id,
        coordinator_address=coordinator_address,
        capabilities=capabilities,
        model_type=model_type,
        dataset_name=dataset_name
    )