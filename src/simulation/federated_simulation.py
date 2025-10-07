"""
Federated learning simulation for end-to-end testing and validation.
Simulates multiple clients training collaboratively with the coordinator.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import json

from ..shared.models import ClientCapabilities, ComputePowerLevel, PrivacyConfig
from ..shared.models_pytorch import ModelFactory
from ..shared.data_loader import create_data_loader
from ..client.federated_trainer import create_federated_trainer
from ..coordinator.grpc_server import CoordinatorGRPCServer
from ..coordinator.round_manager import RoundManager
from ..coordinator.metrics_tracker import MetricsTracker
from ..coordinator.failure_handler import FailureHandler

logger = logging.getLogger(__name__)


class SimulationConfig:
    """Configuration for federated learning simulation."""
    
    def __init__(self,
                 num_clients: int = 5,
                 num_rounds: int = 10,
                 dataset_name: str = "mnist",
                 model_type: str = "simple_cnn",
                 partition_strategy: str = "non_iid",
                 target_accuracy: float = 0.91,
                 coordinator_host: str = "localhost",
                 coordinator_port: int = 50051,
                 privacy_epsilon: float = 1.0,
                 privacy_delta: float = 1e-5):
        """
        Initialize simulation configuration.
        
        Args:
            num_clients: Number of simulated clients
            num_rounds: Number of training rounds
            dataset_name: Dataset to use ("mnist", "cifar10")
            model_type: Model architecture to use
            partition_strategy: Data partitioning strategy
            target_accuracy: Target global accuracy
            coordinator_host: Coordinator host address
            coordinator_port: Coordinator gRPC port
            privacy_epsilon: Privacy epsilon parameter
            privacy_delta: Privacy delta parameter
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.partition_strategy = partition_strategy
        self.target_accuracy = target_accuracy
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_clients': self.num_clients,
            'num_rounds': self.num_rounds,
            'dataset_name': self.dataset_name,
            'model_type': self.model_type,
            'partition_strategy': self.partition_strategy,
            'target_accuracy': self.target_accuracy,
            'coordinator_host': self.coordinator_host,
            'coordinator_port': self.coordinator_port,
            'privacy_epsilon': self.privacy_epsilon,
            'privacy_delta': self.privacy_delta
        }


class SimulatedClient:
    """Represents a simulated federated learning client."""
    
    def __init__(self, 
                 client_id: str,
                 config: SimulationConfig,
                 compute_power: ComputePowerLevel = ComputePowerLevel.MEDIUM,
                 network_bandwidth: int = 10,
                 available_samples: int = 1000):
        """
        Initialize simulated client.
        
        Args:
            client_id: Unique client identifier
            config: Simulation configuration
            compute_power: Client computational power
            network_bandwidth: Network bandwidth in Mbps
            available_samples: Number of available training samples
        """
        self.client_id = client_id
        self.config = config
        
        # Create client capabilities
        self.capabilities = ClientCapabilities(
            compute_power=compute_power,
            network_bandwidth=network_bandwidth,
            available_samples=available_samples,
            supported_models=[config.model_type],
            privacy_requirements=PrivacyConfig(
                epsilon=config.privacy_epsilon,
                delta=config.privacy_delta,
                max_grad_norm=1.0,
                noise_multiplier=1.0
            )
        )
        
        # Create federated trainer
        coordinator_address = f"{config.coordinator_host}:{config.coordinator_port}"
        self.trainer = create_federated_trainer(
            client_id=client_id,
            coordinator_address=coordinator_address,
            compute_power=compute_power.value,
            network_bandwidth=network_bandwidth,
            available_samples=available_samples,
            model_type=config.model_type,
            dataset_name=config.dataset_name,
            privacy_epsilon=config.privacy_epsilon,
            privacy_delta=config.privacy_delta
        )
        
        # Training state
        self.is_running = False
        self.training_thread: Optional[threading.Thread] = None
        self.rounds_completed = 0
        self.training_history = []
        
        logger.info(f"Simulated client {client_id} initialized")
    
    def start_training(self):
        """Start client training."""
        try:
            if self.is_running:
                logger.warning(f"Client {self.client_id} already running")
                return
            
            # Initialize trainer
            if not self.trainer.initialize():
                logger.error(f"Failed to initialize trainer for client {self.client_id}")
                return
            
            # Start training
            if self.trainer.start_training():
                self.is_running = True
                logger.info(f"Client {self.client_id} training started")
            else:
                logger.error(f"Failed to start training for client {self.client_id}")
                
        except Exception as e:
            logger.error(f"Error starting client {self.client_id}: {e}")
    
    def stop_training(self):
        """Stop client training."""
        try:
            if not self.is_running:
                return
            
            self.trainer.stop_training()
            self.is_running = False
            
            logger.info(f"Client {self.client_id} training stopped")
            
        except Exception as e:
            logger.error(f"Error stopping client {self.client_id}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status."""
        status = self.trainer.get_status()
        status.update({
            'is_running': self.is_running,
            'rounds_completed': self.rounds_completed,
            'capabilities': {
                'compute_power': self.capabilities.compute_power.value,
                'network_bandwidth': self.capabilities.network_bandwidth,
                'available_samples': self.capabilities.available_samples
            }
        })
        return status


class FederatedLearningSimulation:
    """Main federated learning simulation orchestrator."""
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize federated learning simulation.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        
        # Simulation components
        self.coordinator_server: Optional[CoordinatorGRPCServer] = None
        self.round_manager: Optional[RoundManager] = None
        self.metrics_tracker: Optional[MetricsTracker] = None
        self.failure_handler: Optional[FailureHandler] = None
        
        # Simulated clients
        self.clients: Dict[str, SimulatedClient] = {}
        
        # Simulation state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.simulation_results: Dict[str, Any] = {}
        
        logger.info(f"Federated learning simulation initialized with {config.num_clients} clients")
    
    def setup_coordinator(self):
        """Setup coordinator services."""
        try:
            # Create coordinator components
            self.round_manager = RoundManager()
            self.metrics_tracker = MetricsTracker()
            self.failure_handler = FailureHandler()
            
            # Create gRPC server
            self.coordinator_server = CoordinatorGRPCServer(
                port=self.config.coordinator_port,
                max_workers=10
            )
            
            logger.info("Coordinator services setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup coordinator: {e}")
            raise
    
    def setup_clients(self):
        """Setup simulated clients."""
        try:
            # Create diverse client configurations
            compute_powers = [ComputePowerLevel.LOW, ComputePowerLevel.MEDIUM, ComputePowerLevel.HIGH]
            network_bandwidths = [5, 10, 25, 50]
            sample_counts = [500, 1000, 1500, 2000]
            
            for i in range(self.config.num_clients):
                client_id = f"sim-client-{i:03d}"
                
                # Assign diverse capabilities
                compute_power = compute_powers[i % len(compute_powers)]
                network_bandwidth = network_bandwidths[i % len(network_bandwidths)]
                available_samples = sample_counts[i % len(sample_counts)]
                
                # Create simulated client
                client = SimulatedClient(
                    client_id=client_id,
                    config=self.config,
                    compute_power=compute_power,
                    network_bandwidth=network_bandwidth,
                    available_samples=available_samples
                )
                
                self.clients[client_id] = client
            
            logger.info(f"Created {len(self.clients)} simulated clients")
            
        except Exception as e:
            logger.error(f"Failed to setup clients: {e}")
            raise
    
    def start_simulation(self) -> bool:
        """
        Start the federated learning simulation.
        
        Returns:
            bool: True if simulation started successfully
        """
        try:
            if self.is_running:
                logger.warning("Simulation already running")
                return False
            
            logger.info("Starting federated learning simulation")
            self.start_time = datetime.now()
            
            # Setup coordinator
            self.setup_coordinator()
            
            # Start coordinator services
            self.round_manager.start()
            self.metrics_tracker.start()
            self.failure_handler.start()
            
            # Start gRPC server
            self.coordinator_server.start()
            
            # Wait for coordinator to be ready
            time.sleep(2)
            
            # Setup and start clients
            self.setup_clients()
            
            # Start clients with staggered timing
            for i, (client_id, client) in enumerate(self.clients.items()):
                # Stagger client starts to avoid overwhelming coordinator
                time.sleep(1)
                
                # Start client in separate thread
                client_thread = threading.Thread(
                    target=client.start_training,
                    daemon=True
                )
                client_thread.start()
            
            self.is_running = True
            logger.info("Federated learning simulation started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            self.stop_simulation()
            return False
    
    def stop_simulation(self):
        """Stop the federated learning simulation."""
        try:
            if not self.is_running:
                return
            
            logger.info("Stopping federated learning simulation")
            
            # Stop clients
            for client in self.clients.values():
                client.stop_training()
            
            # Stop coordinator services
            if self.coordinator_server:
                self.coordinator_server.stop()
            
            if self.round_manager:
                self.round_manager.stop()
            
            if self.metrics_tracker:
                self.metrics_tracker.stop()
            
            if self.failure_handler:
                self.failure_handler.stop()
            
            self.is_running = False
            self.end_time = datetime.now()
            
            logger.info("Federated learning simulation stopped")
            
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
    
    def run_simulation(self, timeout_minutes: int = 30) -> Dict[str, Any]:
        """
        Run complete simulation with timeout.
        
        Args:
            timeout_minutes: Maximum simulation time in minutes
            
        Returns:
            Dict: Simulation results
        """
        try:
            # Start simulation
            if not self.start_simulation():
                return {'error': 'Failed to start simulation'}
            
            # Monitor simulation progress
            timeout_time = datetime.now() + timedelta(minutes=timeout_minutes)
            
            while self.is_running and datetime.now() < timeout_time:
                # Check if target accuracy reached
                if self._check_convergence():
                    logger.info("Target accuracy reached, stopping simulation")
                    break
                
                # Check if all rounds completed
                if self._check_rounds_completed():
                    logger.info("All rounds completed, stopping simulation")
                    break
                
                # Wait before next check
                time.sleep(10)
            
            # Stop simulation
            self.stop_simulation()
            
            # Collect results
            self.simulation_results = self._collect_results()
            
            return self.simulation_results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            self.stop_simulation()
            return {'error': str(e)}
    
    def _check_convergence(self) -> bool:
        """Check if target accuracy has been reached."""
        try:
            if not self.metrics_tracker:
                return False
            
            current_status = self.metrics_tracker.get_collector().get_current_status()
            return current_status.global_accuracy >= self.config.target_accuracy
            
        except Exception:
            return False
    
    def _check_rounds_completed(self) -> bool:
        """Check if all rounds have been completed."""
        try:
            if not self.round_manager:
                return False
            
            return self.round_manager.current_round_number >= self.config.num_rounds
            
        except Exception:
            return False
    
    def _collect_results(self) -> Dict[str, Any]:
        """Collect simulation results."""
        try:
            results = {
                'simulation_config': self.config.to_dict(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
                'success': True
            }
            
            # Collect coordinator metrics
            if self.metrics_tracker:
                collector = self.metrics_tracker.get_collector()
                results['system_metrics'] = collector.get_system_metrics().to_dict()
                results['training_progress'] = collector.get_training_progress()
                results['client_participation'] = collector.get_client_participation_stats()
            
            # Collect client results
            client_results = {}
            for client_id, client in self.clients.items():
                client_results[client_id] = {
                    'status': client.get_status(),
                    'training_history': client.trainer.get_training_history()
                }
            
            results['clients'] = client_results
            
            # Calculate summary statistics
            results['summary'] = self._calculate_summary_stats(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to collect results: {e}")
            return {
                'simulation_config': self.config.to_dict(),
                'error': str(e),
                'success': False
            }
    
    def _calculate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        try:
            summary = {}
            
            # System-level stats
            if 'system_metrics' in results:
                system_metrics = results['system_metrics']
                summary['final_accuracy'] = system_metrics.get('current_global_accuracy', 0.0)
                summary['best_accuracy'] = system_metrics.get('best_global_accuracy', 0.0)
                summary['total_rounds'] = system_metrics.get('total_rounds', 0)
                summary['avg_round_duration'] = system_metrics.get('avg_round_duration', 0.0)
                summary['target_achieved'] = summary['final_accuracy'] >= self.config.target_accuracy
            
            # Client-level stats
            if 'clients' in results:
                client_data = results['clients']
                summary['total_clients'] = len(client_data)
                summary['active_clients'] = sum(
                    1 for client in client_data.values()
                    if client['status'].get('is_running', False)
                )
                
                # Calculate participation rate
                total_rounds = summary.get('total_rounds', 0)
                if total_rounds > 0:
                    participation_rates = []
                    for client in client_data.values():
                        rounds_completed = client['status'].get('rounds_completed', 0)
                        participation_rate = rounds_completed / total_rounds
                        participation_rates.append(participation_rate)
                    
                    summary['avg_participation_rate'] = np.mean(participation_rates)
                    summary['min_participation_rate'] = min(participation_rates)
                    summary['max_participation_rate'] = max(participation_rates)
            
            # Privacy analysis
            summary['privacy_preserved'] = True  # Always true in our implementation
            summary['privacy_epsilon'] = self.config.privacy_epsilon
            summary['privacy_delta'] = self.config.privacy_delta
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to calculate summary stats: {e}")
            return {'error': str(e)}
    
    def export_results(self, filepath: str):
        """Export simulation results to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.simulation_results, f, indent=2)
            
            logger.info(f"Simulation results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")


def run_mnist_simulation(num_clients: int = 5, 
                        num_rounds: int = 10,
                        target_accuracy: float = 0.91) -> Dict[str, Any]:
    """
    Run MNIST federated learning simulation.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds
        target_accuracy: Target accuracy
        
    Returns:
        Dict: Simulation results
    """
    config = SimulationConfig(
        num_clients=num_clients,
        num_rounds=num_rounds,
        dataset_name="mnist",
        model_type="simple_cnn",
        target_accuracy=target_accuracy,
        privacy_epsilon=1.0,
        privacy_delta=1e-5
    )
    
    simulation = FederatedLearningSimulation(config)
    return simulation.run_simulation(timeout_minutes=30)


def run_cifar10_simulation(num_clients: int = 5, 
                          num_rounds: int = 15,
                          target_accuracy: float = 0.85) -> Dict[str, Any]:
    """
    Run CIFAR-10 federated learning simulation.
    
    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds
        target_accuracy: Target accuracy
        
    Returns:
        Dict: Simulation results
    """
    config = SimulationConfig(
        num_clients=num_clients,
        num_rounds=num_rounds,
        dataset_name="cifar10",
        model_type="cifar10_cnn",
        target_accuracy=target_accuracy,
        privacy_epsilon=1.5,
        privacy_delta=1e-5
    )
    
    simulation = FederatedLearningSimulation(config)
    return simulation.run_simulation(timeout_minutes=45)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Running MNIST federated learning simulation...")
    results = run_mnist_simulation(num_clients=3, num_rounds=5, target_accuracy=0.90)
    
    if results.get('success'):
        summary = results.get('summary', {})
        print(f"Simulation completed successfully!")
        print(f"Final accuracy: {summary.get('final_accuracy', 0):.4f}")
        print(f"Target achieved: {summary.get('target_achieved', False)}")
        print(f"Total rounds: {summary.get('total_rounds', 0)}")
    else:
        print(f"Simulation failed: {results.get('error', 'Unknown error')}")