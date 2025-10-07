"""
gRPC server implementation for the federated learning coordinator.
Handles client registration, model distribution, and update collection.
"""

import grpc
from concurrent import futures
import logging
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

# Add proto path for imports
import sys
import os
proto_path = os.path.join(os.path.dirname(__file__), '..', '..', 'proto')
sys.path.insert(0, proto_path)

import federated_learning_pb2 as pb2
import federated_learning_pb2_grpc as pb2_grpc

from ..shared.models import (
    ModelUpdate, GlobalModel, PrivacyConfig, ClientCapabilities,
    ComputePowerLevel, RoundConfig, TrainingStatus
)
from ..shared.grpc_utils import ProtobufConverter, GRPCError
from ..aggregation.fedavg import FedAvgAggregator
from ..aggregation.convergence import ConvergenceDetector

logger = logging.getLogger(__name__)


class FederatedLearningServicer(pb2_grpc.FederatedLearningServicer):
    """gRPC service implementation for federated learning coordinator."""
    
    def __init__(self, coordinator_service):
        """
        Initialize gRPC servicer.
        
        Args:
            coordinator_service: Coordinator service instance
        """
        self.coordinator = coordinator_service
        self.converter = ProtobufConverter()
        
        # Client management
        self.registered_clients = {}
        self.client_states = {}
        self.client_lock = threading.RLock()
        
        # Training state
        self.current_round = 0
        self.round_config = None
        self.global_model = None
        self.pending_updates = {}
        self.training_lock = threading.RLock()
        
        # Aggregation and convergence
        self.aggregator = FedAvgAggregator(min_clients=2, max_clients=50)
        self.convergence_detector = ConvergenceDetector(patience=5)
        
        logger.info("Federated learning gRPC servicer initialized")
    
    def RegisterClient(self, request, context):
        """Handle client registration."""
        try:
            client_id = request.client_id
            logger.info(f"Client registration request from {client_id}")
            
            # Convert capabilities
            capabilities = self.converter.client_capabilities_from_proto(request.capabilities)
            
            with self.client_lock:
                # Register client
                self.registered_clients[client_id] = {
                    'capabilities': capabilities,
                    'registration_time': datetime.now(),
                    'client_version': request.client_version,
                    'metadata': dict(request.metadata)
                }
                
                # Initialize client state
                self.client_states[client_id] = pb2.CLIENT_STATE_REGISTERED
            
            # Create server info
            server_info = pb2.ServerInfo(
                server_version="1.0.0",
                supported_models=["simple_cnn", "cifar10_cnn", "federated_resnet"],
                supported_datasets=["mnist", "cifar10"],
                default_privacy_config=self.converter.privacy_config_to_proto(
                    PrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0, noise_multiplier=1.0)
                )
            )
            
            response = pb2.RegistrationResponse(
                success=True,
                message=f"Client {client_id} registered successfully",
                assigned_client_id=client_id,
                global_model_version=self.current_round,
                server_info=server_info
            )
            
            logger.info(f"Client {client_id} registered successfully")
            return response
            
        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            return pb2.RegistrationResponse(
                success=False,
                message=f"Registration failed: {str(e)}",
                assigned_client_id="",
                global_model_version=0
            )
    
    def UpdateClientCapabilities(self, request, context):
        """Handle client capability updates."""
        try:
            client_id = request.client_id if hasattr(request, 'client_id') else "unknown"
            
            with self.client_lock:
                if client_id in self.registered_clients:
                    capabilities = self.converter.client_capabilities_from_proto(request)
                    self.registered_clients[client_id]['capabilities'] = capabilities
                    
                    logger.info(f"Updated capabilities for client {client_id}")
                    return pb2.UpdateResponse(success=True, message="Capabilities updated")
                else:
                    return pb2.UpdateResponse(success=False, message="Client not registered")
                    
        except Exception as e:
            logger.error(f"Capability update failed: {e}")
            return pb2.UpdateResponse(success=False, message=f"Update failed: {str(e)}")
    
    def GetGlobalModel(self, request, context):
        """Handle global model requests."""
        try:
            client_id = request.client_id
            requested_round = request.round_number
            
            logger.debug(f"Global model request from {client_id} for round {requested_round}")
            
            with self.client_lock:
                if client_id not in self.registered_clients:
                    return pb2.ModelResponse(
                        success=False,
                        message="Client not registered",
                        model_weights=b"",
                        round_number=0
                    )
            
            with self.training_lock:
                if self.global_model is None:
                    # Create initial model if none exists
                    self.global_model = self._create_initial_global_model()
                
                # Update client state
                self.client_states[client_id] = pb2.CLIENT_STATE_TRAINING
                
                # Convert global model to proto response
                response = self.converter.global_model_to_proto(self.global_model)
                response.success = True
                response.message = f"Global model for round {self.current_round}"
                
                logger.debug(f"Sent global model to {client_id}")
                return response
                
        except Exception as e:
            logger.error(f"Get global model failed: {e}")
            return pb2.ModelResponse(
                success=False,
                message=f"Failed to get global model: {str(e)}",
                model_weights=b"",
                round_number=0
            )
    
    def SubmitModelUpdate(self, request, context):
        """Handle model update submissions."""
        try:
            client_id = request.client_id
            round_number = request.round_number
            
            logger.info(f"Model update from {client_id} for round {round_number}")
            
            with self.client_lock:
                if client_id not in self.registered_clients:
                    return pb2.UpdateAck(
                        success=False,
                        message="Client not registered",
                        next_round_eta=0,
                        round_status=pb2.ROUND_STATUS_UNKNOWN
                    )
            
            # Convert model update
            model_update = self.converter.model_update_from_proto(request)
            
            with self.training_lock:
                # Store update
                if round_number not in self.pending_updates:
                    self.pending_updates[round_number] = {}
                
                self.pending_updates[round_number][client_id] = model_update
                
                # Update client state
                self.client_states[client_id] = pb2.CLIENT_STATE_WAITING
                
                # Check if we have enough updates for aggregation
                current_updates = self.pending_updates.get(self.current_round, {})
                min_clients = getattr(self.round_config, 'min_clients', 2) if self.round_config else 2
                
                if len(current_updates) >= min_clients:
                    # Trigger aggregation in background
                    threading.Thread(target=self._perform_aggregation, daemon=True).start()
                
                # Calculate next round ETA
                next_eta = int((datetime.now() + timedelta(minutes=5)).timestamp())
                
                response = pb2.UpdateAck(
                    success=True,
                    message=f"Update received for round {round_number}",
                    next_round_eta=next_eta,
                    round_status=pb2.ROUND_STATUS_IN_PROGRESS
                )
                
                logger.info(f"Accepted update from {client_id}")
                return response
                
        except Exception as e:
            logger.error(f"Submit model update failed: {e}")
            return pb2.UpdateAck(
                success=False,
                message=f"Update submission failed: {str(e)}",
                next_round_eta=0,
                round_status=pb2.ROUND_STATUS_UNKNOWN
            )
    
    def JoinTrainingRound(self, request, context):
        """Handle training round join requests."""
        try:
            client_id = request.client_id
            requested_round = request.requested_round
            
            logger.info(f"Round join request from {client_id} for round {requested_round}")
            
            with self.client_lock:
                if client_id not in self.registered_clients:
                    return pb2.RoundJoinResponse(
                        success=False,
                        message="Client not registered",
                        assigned_round=0
                    )
            
            # Create round config if needed
            if self.round_config is None:
                self.round_config = self._create_default_round_config()
            
            # Convert round config to proto
            proto_config = pb2.RoundConfig(
                round_number=self.current_round,
                min_clients=2,
                max_clients=50,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.001,
                timeout_seconds=300,
                optimizer_type="adam",
                privacy_config=self.converter.privacy_config_to_proto(
                    PrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0, noise_multiplier=1.0)
                )
            )
            
            response = pb2.RoundJoinResponse(
                success=True,
                message=f"Joined round {self.current_round}",
                assigned_round=self.current_round,
                round_config=proto_config
            )
            
            logger.info(f"Client {client_id} joined round {self.current_round}")
            return response
            
        except Exception as e:
            logger.error(f"Join training round failed: {e}")
            return pb2.RoundJoinResponse(
                success=False,
                message=f"Failed to join round: {str(e)}",
                assigned_round=0
            )
    
    def GetRoundConfig(self, request, context):
        """Handle round configuration requests."""
        try:
            client_id = request.client_id
            round_number = request.round_number
            
            if self.round_config is None:
                self.round_config = self._create_default_round_config()
            
            proto_config = pb2.RoundConfig(
                round_number=round_number,
                min_clients=2,
                max_clients=50,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.001,
                timeout_seconds=300,
                optimizer_type="adam",
                privacy_config=self.converter.privacy_config_to_proto(
                    PrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0, noise_multiplier=1.0)
                )
            )
            
            return proto_config
            
        except Exception as e:
            logger.error(f"Get round config failed: {e}")
            return pb2.RoundConfig()
    
    def GetTrainingStatus(self, request, context):
        """Handle training status requests."""
        try:
            client_id = request.client_id
            
            with self.training_lock:
                # Calculate round progress
                current_updates = self.pending_updates.get(self.current_round, {})
                min_clients = 2
                progress = min(1.0, len(current_updates) / min_clients)
                
                # Get global accuracy
                global_accuracy = 0.0
                if self.global_model:
                    global_accuracy = self.global_model.get_accuracy() or 0.0
                
                # Create client statuses
                client_statuses = []
                with self.client_lock:
                    for cid, state in self.client_states.items():
                        client_status = pb2.ClientStatus(
                            client_id=cid,
                            state=state,
                            last_accuracy=0.0,  # Would be filled from history
                            last_loss=0.0,      # Would be filled from history
                            last_update_time=int(datetime.now().timestamp()),
                            rounds_participated=1  # Would be calculated from history
                        )
                        client_statuses.append(client_status)
                
                response = pb2.TrainingStatus(
                    current_round=self.current_round,
                    active_clients=len(self.registered_clients),
                    round_progress=progress,
                    global_accuracy=global_accuracy,
                    convergence_score=0.0,  # Would be calculated
                    estimated_completion=int((datetime.now() + timedelta(minutes=10)).timestamp()),
                    round_status=pb2.ROUND_STATUS_IN_PROGRESS,
                    client_statuses=client_statuses
                )
                
                return response
                
        except Exception as e:
            logger.error(f"Get training status failed: {e}")
            return pb2.TrainingStatus()
    
    def HealthCheck(self, request, context):
        """Handle health check requests."""
        try:
            response = pb2.HealthResponse(
                healthy=True,
                status="OK",
                details={
                    "service": "federated_learning_coordinator",
                    "version": "1.0.0",
                    "registered_clients": str(len(self.registered_clients)),
                    "current_round": str(self.current_round)
                },
                timestamp=int(datetime.now().timestamp())
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return pb2.HealthResponse(
                healthy=False,
                status=f"ERROR: {str(e)}",
                timestamp=int(datetime.now().timestamp())
            )
    
    def GetMetrics(self, request, context):
        """Handle metrics requests."""
        try:
            # Create sample metrics
            metrics = {}
            
            # Client count metric
            client_count_metric = pb2.MetricData(
                metric_name="registered_clients",
                points=[
                    pb2.MetricPoint(
                        timestamp=int(datetime.now().timestamp()),
                        value=float(len(self.registered_clients))
                    )
                ]
            )
            metrics["registered_clients"] = client_count_metric
            
            # Round metric
            round_metric = pb2.MetricData(
                metric_name="current_round",
                points=[
                    pb2.MetricPoint(
                        timestamp=int(datetime.now().timestamp()),
                        value=float(self.current_round)
                    )
                ]
            )
            metrics["current_round"] = round_metric
            
            response = pb2.MetricsResponse(
                success=True,
                metrics=metrics
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Get metrics failed: {e}")
            return pb2.MetricsResponse(success=False)
    
    def _create_initial_global_model(self) -> GlobalModel:
        """Create initial global model."""
        from ..shared.models_pytorch import ModelFactory
        import torch
        
        # Create a simple CNN model
        model = ModelFactory.create_model("simple_cnn", num_classes=10)
        
        global_model = GlobalModel(
            round_number=0,
            model_weights=model.get_model_weights(),
            accuracy_metrics={"test_accuracy": 0.1},  # Initial random accuracy
            participating_clients=[],
            convergence_score=1.0,
            created_at=datetime.now()
        )
        
        logger.info("Created initial global model")
        return global_model
    
    def _create_default_round_config(self) -> RoundConfig:
        """Create default round configuration."""
        return RoundConfig(
            round_number=self.current_round,
            min_clients=2,
            max_clients=50,
            local_epochs=5,
            batch_size=32,
            learning_rate=0.001,
            timeout_seconds=300
        )
    
    def _perform_aggregation(self):
        """Perform model aggregation in background."""
        try:
            with self.training_lock:
                current_updates = self.pending_updates.get(self.current_round, {})
                
                if len(current_updates) < 2:
                    logger.warning("Not enough updates for aggregation")
                    return
                
                logger.info(f"Starting aggregation for round {self.current_round} with {len(current_updates)} updates")
                
                # Perform aggregation
                updates_list = list(current_updates.values())
                previous_model = self.global_model
                
                # Aggregate updates
                self.global_model = self.aggregator.aggregate_updates(updates_list)
                
                # Calculate convergence
                if previous_model:
                    convergence_metrics = self.convergence_detector.calculate_convergence_metrics(
                        self.global_model, previous_model
                    )
                    self.global_model.convergence_score = convergence_metrics.convergence_score
                
                # Move to next round
                self.current_round += 1
                
                # Clear pending updates for completed round
                if self.current_round - 1 in self.pending_updates:
                    del self.pending_updates[self.current_round - 1]
                
                # Reset client states
                with self.client_lock:
                    for client_id in self.client_states:
                        self.client_states[client_id] = pb2.CLIENT_STATE_REGISTERED
                
                logger.info(f"Aggregation completed for round {self.current_round - 1}")
                
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")


class CoordinatorGRPCServer:
    """Coordinator gRPC server manager."""
    
    def __init__(self, port: int = 50051, max_workers: int = 10):
        """
        Initialize coordinator gRPC server.
        
        Args:
            port: Port to listen on
            max_workers: Maximum number of worker threads
        """
        self.port = port
        self.max_workers = max_workers
        self.server = None
        self.servicer = None
        
        logger.info(f"Coordinator gRPC server initialized on port {port}")
    
    def start(self, coordinator_service=None):
        """Start the gRPC server."""
        try:
            # Create servicer
            self.servicer = FederatedLearningServicer(coordinator_service)
            
            # Create server
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),     # 100MB
                ]
            )
            
            # Add servicer to server
            pb2_grpc.add_FederatedLearningServicer_to_server(self.servicer, self.server)
            
            # Add port
            listen_addr = f'[::]:{self.port}'
            self.server.add_insecure_port(listen_addr)
            
            # Start server
            self.server.start()
            
            logger.info(f"Coordinator gRPC server started on {listen_addr}")
            
        except Exception as e:
            logger.error(f"Failed to start coordinator gRPC server: {e}")
            raise
    
    def stop(self, grace_period: int = 5):
        """Stop the gRPC server."""
        try:
            if self.server:
                logger.info("Stopping coordinator gRPC server...")
                self.server.stop(grace_period)
                self.server = None
                logger.info("Coordinator gRPC server stopped")
        except Exception as e:
            logger.error(f"Error stopping coordinator gRPC server: {e}")
    
    def wait_for_termination(self):
        """Wait for server termination."""
        if self.server:
            self.server.wait_for_termination()
    
    def get_servicer(self) -> FederatedLearningServicer:
        """Get the gRPC servicer instance."""
        return self.servicer    de
f _bytes_to_dict(self, data: bytes) -> Dict:
        """Convert bytes to dictionary (placeholder)."""
        import pickle
        return pickle.loads(data)
    
    async def _trigger_aggregation(self):
        """Trigger model aggregation in background."""
        try:
            logger.info(f"Starting aggregation for round {self.current_round}")
            
            with self.round_lock:
                if not self.round_updates:
                    logger.warning("No updates available for aggregation")
                    return
                
                # Record round start time
                if self.metrics['last_round_start'] is None:
                    self.metrics['last_round_start'] = time.time()
                
                # Perform aggregation
                updates_list = list(self.round_updates.values())
                aggregated_model = self.aggregator.aggregate_updates(updates_list)
                
                # Check convergence
                previous_model = self.global_model
                convergence_result = self.convergence_detector.check_convergence(
                    aggregated_model, previous_model
                )
                
                # Update global model
                self.global_model = aggregated_model
                self.current_round += 1
                
                # Update metrics
                round_time = time.time() - self.metrics['last_round_start']
                self.metrics['average_round_time'] = (
                    (self.metrics['average_round_time'] * self.metrics['total_rounds_completed'] + round_time) /
                    (self.metrics['total_rounds_completed'] + 1)
                )
                self.metrics['total_rounds_completed'] += 1
                self.metrics['last_round_start'] = None
                
                # Clear round updates for next round
                self.round_updates.clear()
                
                # Update client states
                with self.client_lock:
                    for client_id in self.registered_clients:
                        self.registered_clients[client_id]['state'] = 'idle'
                
                logger.info(f"Round {self.current_round - 1} aggregation completed. "
                           f"Convergence: {convergence_result.converged}")
                
                # Check if training should stop
                if convergence_result.converged:
                    self.training_active = False
                    logger.info("Training converged, stopping federated learning")
                
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            # Reset round state on failure
            with self.round_lock:
                self.round_updates.clear()
            
            with self.client_lock:
                for client_id in self.registered_clients:
                    self.registered_clients[client_id]['state'] = 'error'


class FederatedLearningStreamServicer(fl_pb2_grpc.FederatedLearningStreamServicer):
    """Streaming gRPC servicer for real-time updates."""
    
    def __init__(self, main_servicer: FederatedLearningServicer):
        """Initialize streaming servicer."""
        self.main_servicer = main_servicer
        self.active_streams = {}
        logger.info("FederatedLearningStreamServicer initialized")
    
    async def StreamTrainingProgress(self, request: fl_pb2.TrainingProgressRequest, context):
        """Stream training progress updates to client."""
        client_id = request.client_id
        logger.info(f"Starting training progress stream for client {client_id}")
        
        try:
            while True:
                # Get current training status
                status_request = fl_pb2.TrainingStatusRequest(
                    client_id=client_id,
                    include_detailed_metrics=request.include_client_details
                )
                
                status_response = await self.main_servicer.GetTrainingStatus(status_request, context)
                
                # Convert to progress update
                progress_update = fl_pb2.TrainingProgressUpdate(
                    round_number=status_response.current_round,
                    round_status=status_response.round_status,
                    progress_percent=status_response.round_progress * 100,
                    clients_completed=len(self.main_servicer.round_updates),
                    total_clients=status_response.active_clients,
                    estimated_completion=status_response.estimated_completion,
                    current_metrics=status_response.global_metrics
                )
                
                yield progress_update
                
                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            logger.error(f"Training progress stream failed for client {client_id}: {str(e)}")
    
    async def StreamModelUpdates(self, request_iterator, context):
        """Stream model updates from clients."""
        try:
            async for update_request in request_iterator:
                # Process each update
                acknowledgment = await self.main_servicer.SubmitModelUpdate(update_request, context)
                yield acknowledgment
                
        except Exception as e:
            logger.error(f"Model updates stream failed: {str(e)}")
            yield fl_pb2.UpdateAcknowledgment(
                success=False,
                message=f"Stream failed: {str(e)}",
                next_round_eta=0,
                round_status=fl_pb2.ROUND_STATUS_FAILED
            )
    
    async def StreamMetrics(self, request: fl_pb2.MetricsStreamRequest, context):
        """Stream system metrics to client."""
        client_id = request.client_id
        logger.info(f"Starting metrics stream for client {client_id}")
        
        try:
            while True:
                # Get current metrics
                metrics_request = fl_pb2.MetricsRequest(
                    client_id=client_id,
                    metric_types=list(request.metric_names)
                )
                
                metrics_response = await self.main_servicer.GetMetrics(metrics_request, context)
                
                # Stream each metric
                for metric_data in metrics_response.metrics:
                    if not request.metric_names or metric_data.metric_name in request.metric_names:
                        yield metric_data
                
                # Wait for next update
                await asyncio.sleep(request.update_interval_seconds)
                
        except Exception as e:
            logger.error(f"Metrics stream failed for client {client_id}: {str(e)}")


class ConfigurationServicer(fl_pb2_grpc.ConfigurationServiceServicer):
    """Configuration management servicer."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize configuration servicer."""
        self.config = config
        self.config_lock = threading.RLock()
        logger.info("ConfigurationServicer initialized")
    
    async def GetConfiguration(self, request: fl_pb2.ConfigurationRequest, context) -> fl_pb2.ConfigurationResponse:
        """Get configuration for client."""
        try:
            client_id = request.client_id
            config_type = request.configuration_type
            
            with self.config_lock:
                if config_type == "training":
                    config_data = {
                        "local_epochs": str(self.config.get('federated_learning', {}).get('local_epochs', 5)),
                        "batch_size": str(self.config.get('federated_learning', {}).get('batch_size', 32)),
                        "learning_rate": str(self.config.get('federated_learning', {}).get('learning_rate', 0.001)),
                        "optimizer_type": self.config.get('federated_learning', {}).get('optimizer_type', 'adam')
                    }
                elif config_type == "privacy":
                    config_data = {
                        "epsilon": str(self.config.get('privacy', {}).get('epsilon', 1.0)),
                        "delta": str(self.config.get('privacy', {}).get('delta', 1e-5)),
                        "max_grad_norm": str(self.config.get('privacy', {}).get('max_grad_norm', 1.0)),
                        "noise_multiplier": str(self.config.get('privacy', {}).get('noise_multiplier', 1.0))
                    }
                else:
                    return fl_pb2.ConfigurationResponse(
                        success=False,
                        configuration={},
                        message=f"Unknown configuration type: {config_type}"
                    )
            
            return fl_pb2.ConfigurationResponse(
                success=True,
                configuration=config_data,
                message="Configuration retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Get configuration failed: {str(e)}")
            return fl_pb2.ConfigurationResponse(
                success=False,
                configuration={},
                message=f"Failed to get configuration: {str(e)}"
            )
    
    async def UpdateConfiguration(self, request: fl_pb2.ConfigurationUpdateRequest, context) -> fl_pb2.ConfigurationUpdateResponse:
        """Update configuration."""
        try:
            client_id = request.client_id
            config_type = request.configuration_type
            new_config = dict(request.configuration)
            
            # Validate configuration first
            validation_request = fl_pb2.ConfigurationValidationRequest(
                configuration_type=config_type,
                configuration=new_config
            )
            validation_response = await self.ValidateConfiguration(validation_request, context)
            
            if not validation_response.valid:
                return fl_pb2.ConfigurationUpdateResponse(
                    success=False,
                    message="Configuration validation failed",
                    validation_errors=validation_response.errors
                )
            
            # Update configuration
            with self.config_lock:
                if config_type == "training":
                    fl_config = self.config.setdefault('federated_learning', {})
                    fl_config.update({
                        'local_epochs': int(new_config.get('local_epochs', fl_config.get('local_epochs', 5))),
                        'batch_size': int(new_config.get('batch_size', fl_config.get('batch_size', 32))),
                        'learning_rate': float(new_config.get('learning_rate', fl_config.get('learning_rate', 0.001))),
                        'optimizer_type': new_config.get('optimizer_type', fl_config.get('optimizer_type', 'adam'))
                    })
                elif config_type == "privacy":
                    privacy_config = self.config.setdefault('privacy', {})
                    privacy_config.update({
                        'epsilon': float(new_config.get('epsilon', privacy_config.get('epsilon', 1.0))),
                        'delta': float(new_config.get('delta', privacy_config.get('delta', 1e-5))),
                        'max_grad_norm': float(new_config.get('max_grad_norm', privacy_config.get('max_grad_norm', 1.0))),
                        'noise_multiplier': float(new_config.get('noise_multiplier', privacy_config.get('noise_multiplier', 1.0)))
                    })
            
            logger.info(f"Configuration updated by client {client_id}: {config_type}")
            
            return fl_pb2.ConfigurationUpdateResponse(
                success=True,
                message="Configuration updated successfully",
                validation_errors=[]
            )
            
        except Exception as e:
            logger.error(f"Update configuration failed: {str(e)}")
            return fl_pb2.ConfigurationUpdateResponse(
                success=False,
                message=f"Failed to update configuration: {str(e)}",
                validation_errors=[str(e)]
            )
    
    async def ValidateConfiguration(self, request: fl_pb2.ConfigurationValidationRequest, context) -> fl_pb2.ConfigurationValidationResponse:
        """Validate configuration."""
        try:
            config_type = request.configuration_type
            config_data = dict(request.configuration)
            
            errors = []
            warnings = []
            recommendations = []
            
            if config_type == "training":
                # Validate training parameters
                try:
                    local_epochs = int(config_data.get('local_epochs', 5))
                    if local_epochs < 1 or local_epochs > 100:
                        errors.append("local_epochs must be between 1 and 100")
                except ValueError:
                    errors.append("local_epochs must be a valid integer")
                
                try:
                    batch_size = int(config_data.get('batch_size', 32))
                    if batch_size < 1 or batch_size > 1024:
                        errors.append("batch_size must be between 1 and 1024")
                    elif batch_size > 256:
                        warnings.append("Large batch sizes may impact convergence")
                except ValueError:
                    errors.append("batch_size must be a valid integer")
                
                try:
                    learning_rate = float(config_data.get('learning_rate', 0.001))
                    if learning_rate <= 0 or learning_rate > 1:
                        errors.append("learning_rate must be between 0 and 1")
                    elif learning_rate > 0.1:
                        warnings.append("High learning rates may cause instability")
                except ValueError:
                    errors.append("learning_rate must be a valid float")
                
            elif config_type == "privacy":
                # Validate privacy parameters
                try:
                    epsilon = float(config_data.get('epsilon', 1.0))
                    if epsilon <= 0:
                        errors.append("epsilon must be positive")
                    elif epsilon > 10:
                        warnings.append("High epsilon values provide less privacy")
                    elif epsilon < 0.1:
                        recommendations.append("Very low epsilon may significantly impact utility")
                except ValueError:
                    errors.append("epsilon must be a valid float")
                
                try:
                    delta = float(config_data.get('delta', 1e-5))
                    if delta <= 0 or delta >= 1:
                        errors.append("delta must be between 0 and 1")
                except ValueError:
                    errors.append("delta must be a valid float")
            
            else:
                errors.append(f"Unknown configuration type: {config_type}")
            
            return fl_pb2.ConfigurationValidationResponse(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return fl_pb2.ConfigurationValidationResponse(
                valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                recommendations=[]
            )


class GRPCServer:
    """Main gRPC server for federated learning coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize gRPC server."""
        self.config = config
        self.server = None
        self.servicer = None
        self.stream_servicer = None
        self.config_servicer = None
        
        # Server configuration
        self.host = config.get('grpc', {}).get('host', '0.0.0.0')
        self.port = config.get('grpc', {}).get('port', 50051)
        self.max_workers = config.get('grpc', {}).get('max_workers', 10)
        
        logger.info(f"GRPCServer initialized - {self.host}:{self.port}")
    
    async def start(self):
        """Start the gRPC server."""
        try:
            # Create servicers
            self.servicer = FederatedLearningServicer(self.config)
            self.stream_servicer = FederatedLearningStreamServicer(self.servicer)
            self.config_servicer = ConfigurationServicer(self.config)
            
            # Create server
            self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))
            
            # Add servicers
            fl_pb2_grpc.add_FederatedLearningServicer_to_server(self.servicer, self.server)
            fl_pb2_grpc.add_FederatedLearningStreamServicer_to_server(self.stream_servicer, self.server)
            fl_pb2_grpc.add_ConfigurationServiceServicer_to_server(self.config_servicer, self.server)
            
            # Configure server options
            options = [
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000)
            ]
            
            for option in options:
                self.server.add_generic_rpc_handlers((
                    grpc.method_handlers_generic_handler('', {option[0]: option[1]})
                ))
            
            # Add port
            listen_addr = f'{self.host}:{self.port}'
            self.server.add_insecure_port(listen_addr)
            
            # Start server
            await self.server.start()
            logger.info(f"gRPC server started on {listen_addr}")
            
            # Initialize global model if needed
            if self.servicer.global_model is None:
                await self._initialize_global_model()
            
        except Exception as e:
            logger.error(f"Failed to start gRPC server: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the gRPC server."""
        if self.server:
            logger.info("Stopping gRPC server...")
            await self.server.stop(grace=5)
            logger.info("gRPC server stopped")
    
    async def wait_for_termination(self):
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()
    
    async def _initialize_global_model(self):
        """Initialize global model with default weights."""
        try:
            from ..shared.models_pytorch import create_model
            from ..shared.models import GlobalModel
            
            # Create initial model
            model_type = self.config.get('federated_learning', {}).get('model_type', 'simple_cnn')
            model = create_model(model_type)
            
            # Convert to global model format
            model_weights = {name: param.data for name, param in model.named_parameters()}
            
            self.servicer.global_model = GlobalModel(
                round_number=0,
                model_weights=model_weights,
                accuracy_metrics={'test_accuracy': 0.0, 'validation_accuracy': 0.0},
                aggregation_info={'num_clients': 0, 'total_samples': 0}
            )
            
            logger.info("Global model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize global model: {str(e)}")


# Utility functions
def create_grpc_server(config: Dict[str, Any]) -> GRPCServer:
    """Factory function to create gRPC server."""
    return GRPCServer(config)


async def run_grpc_server(config: Dict[str, Any]):
    """Run gRPC server with proper lifecycle management."""
    server = create_grpc_server(config)
    
    try:
        await server.start()
        logger.info("gRPC server is running...")
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"gRPC server error: {str(e)}")
    finally:
        await server.stop()


if __name__ == "__main__":
    # Example configuration
    config = {
        'grpc': {
            'host': '0.0.0.0',
            'port': 50051,
            'max_workers': 10
        },
        'federated_learning': {
            'min_clients': 2,
            'max_clients': 50,
            'local_epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer_type': 'adam',
            'model_type': 'simple_cnn',
            'round_timeout': 300,
            'total_rounds': 100
        },
        'privacy': {
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'noise_multiplier': 1.0
        }
    }
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    asyncio.run(run_grpc_server(config))