"""
gRPC utilities for federated learning communication.
Handles protocol buffer conversion and gRPC service utilities.
"""

import grpc
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import torch
import pickle
import sys
import os

# Add proto directory to path for imports
proto_path = os.path.join(os.path.dirname(__file__), '..', '..', 'proto')
sys.path.insert(0, proto_path)

from .models import (
    ModelUpdate, GlobalModel, PrivacyConfig, ClientCapabilities, 
    TrainingMetrics, ComputePowerLevel, ModelWeights
)
from .serialization import ModelUpdateSerializer, GlobalModelSerializer

logger = logging.getLogger(__name__)


class GRPCError(Exception):
    """Custom exception for gRPC-related errors."""
    pass


class ProtobufConverter:
    """Converts between Python objects and Protocol Buffer messages."""
    
    def __init__(self):
        """Initialize protobuf converter."""
        self.model_serializer = ModelUpdateSerializer()
        self.global_serializer = GlobalModelSerializer()
    
    def privacy_config_to_proto(self, config: PrivacyConfig):
        """Convert PrivacyConfig to protobuf message."""
        try:
            # Import here to avoid circular imports
            import federated_learning_pb2 as pb2
            
            return pb2.PrivacyConfig(
                epsilon=config.epsilon,
                delta=config.delta,
                max_grad_norm=config.max_grad_norm,
                noise_multiplier=config.noise_multiplier
            )
        except Exception as e:
            logger.error(f"Failed to convert PrivacyConfig to proto: {e}")
            raise GRPCError(f"PrivacyConfig conversion failed: {e}")
    
    def privacy_config_from_proto(self, proto_config) -> PrivacyConfig:
        """Convert protobuf message to PrivacyConfig."""
        try:
            return PrivacyConfig(
                epsilon=proto_config.epsilon,
                delta=proto_config.delta,
                max_grad_norm=proto_config.max_grad_norm,
                noise_multiplier=proto_config.noise_multiplier
            )
        except Exception as e:
            logger.error(f"Failed to convert proto to PrivacyConfig: {e}")
            raise GRPCError(f"PrivacyConfig conversion failed: {e}")
    
    def client_capabilities_to_proto(self, capabilities: ClientCapabilities):
        """Convert ClientCapabilities to protobuf message."""
        try:
            import federated_learning_pb2 as pb2
            
            # Convert compute power level
            compute_power_map = {
                ComputePowerLevel.LOW: pb2.COMPUTE_POWER_LOW,
                ComputePowerLevel.MEDIUM: pb2.COMPUTE_POWER_MEDIUM,
                ComputePowerLevel.HIGH: pb2.COMPUTE_POWER_HIGH
            }
            
            proto_capabilities = pb2.ClientCapabilities(
                compute_power=compute_power_map.get(capabilities.compute_power, pb2.COMPUTE_POWER_MEDIUM),
                network_bandwidth_mbps=capabilities.network_bandwidth,
                available_samples=capabilities.available_samples,
                supported_models=capabilities.supported_models,
                privacy_requirements=self.privacy_config_to_proto(capabilities.privacy_requirements)
            )
            
            return proto_capabilities
            
        except Exception as e:
            logger.error(f"Failed to convert ClientCapabilities to proto: {e}")
            raise GRPCError(f"ClientCapabilities conversion failed: {e}")
    
    def client_capabilities_from_proto(self, proto_capabilities) -> ClientCapabilities:
        """Convert protobuf message to ClientCapabilities."""
        try:
            import federated_learning_pb2 as pb2
            
            # Convert compute power level
            compute_power_map = {
                pb2.COMPUTE_POWER_LOW: ComputePowerLevel.LOW,
                pb2.COMPUTE_POWER_MEDIUM: ComputePowerLevel.MEDIUM,
                pb2.COMPUTE_POWER_HIGH: ComputePowerLevel.HIGH
            }
            
            return ClientCapabilities(
                compute_power=compute_power_map.get(proto_capabilities.compute_power, ComputePowerLevel.MEDIUM),
                network_bandwidth=proto_capabilities.network_bandwidth_mbps,
                available_samples=proto_capabilities.available_samples,
                supported_models=list(proto_capabilities.supported_models),
                privacy_requirements=self.privacy_config_from_proto(proto_capabilities.privacy_requirements)
            )
            
        except Exception as e:
            logger.error(f"Failed to convert proto to ClientCapabilities: {e}")
            raise GRPCError(f"ClientCapabilities conversion failed: {e}")
    
    def model_update_to_proto(self, update: ModelUpdate):
        """Convert ModelUpdate to protobuf message."""
        try:
            import federated_learning_pb2 as pb2
            
            # Serialize model weights
            serialized_update = self.model_serializer.serialize_model_update(update)
            model_weights_bytes = serialized_update['model_weights'].encode('utf-8')
            
            # Create training metrics
            training_metrics = pb2.TrainingMetrics(
                loss=update.training_loss,
                accuracy=0.0,  # Will be filled if available
                epochs_completed=0,  # Will be filled if available
                training_time_seconds=0.0,  # Will be filled if available
                samples_processed=update.num_samples
            )
            
            # Create update metadata
            metadata = pb2.UpdateMetadata(
                compression_algorithm="lz4",  # Default
                original_size_bytes=len(model_weights_bytes),
                compressed_size_bytes=int(len(model_weights_bytes) * update.compression_ratio),
                privacy_epsilon_used=update.privacy_budget_used,
                privacy_delta_used=0.0,  # Will be filled if available
                client_version="1.0"
            )
            
            proto_update = pb2.ModelUpdate(
                client_id=update.client_id,
                round_number=update.round_number,
                model_weights=model_weights_bytes,
                num_samples=update.num_samples,
                training_loss=update.training_loss,
                privacy_budget_used=update.privacy_budget_used,
                compression_ratio=update.compression_ratio,
                timestamp=int(update.timestamp.timestamp()),
                training_metrics=training_metrics,
                metadata=metadata
            )
            
            return proto_update
            
        except Exception as e:
            logger.error(f"Failed to convert ModelUpdate to proto: {e}")
            raise GRPCError(f"ModelUpdate conversion failed: {e}")
    
    def model_update_from_proto(self, proto_update) -> ModelUpdate:
        """Convert protobuf message to ModelUpdate."""
        try:
            # Deserialize model weights
            model_weights_hex = proto_update.model_weights.decode('utf-8')
            
            # Create a serialized update dict for deserialization
            serialized_update = {
                'client_id': proto_update.client_id,
                'round_number': proto_update.round_number,
                'model_weights': model_weights_hex,
                'num_samples': proto_update.num_samples,
                'training_loss': proto_update.training_loss,
                'privacy_budget_used': proto_update.privacy_budget_used,
                'compression_ratio': proto_update.compression_ratio,
                'timestamp': datetime.fromtimestamp(proto_update.timestamp).isoformat()
            }
            
            return self.model_serializer.deserialize_model_update(serialized_update)
            
        except Exception as e:
            logger.error(f"Failed to convert proto to ModelUpdate: {e}")
            raise GRPCError(f"ModelUpdate conversion failed: {e}")
    
    def global_model_to_proto(self, model: GlobalModel):
        """Convert GlobalModel to protobuf message."""
        try:
            import federated_learning_pb2 as pb2
            
            # Serialize global model
            serialized_model = self.global_serializer.serialize_global_model(model)
            model_weights_bytes = serialized_model['model_weights'].encode('utf-8')
            
            # Create model metadata
            metadata = pb2.ModelMetadata(
                model_type="federated_cnn",  # Default
                parameter_count=sum(w.numel() for w in model.model_weights.values()),
                model_size_bytes=len(model_weights_bytes),
                compression_algorithm="lz4",
                compression_ratio=0.8,  # Default
                accuracy_metrics=model.accuracy_metrics
            )
            
            proto_response = pb2.ModelResponse(
                success=True,
                message="Global model retrieved successfully",
                model_weights=model_weights_bytes,
                round_number=model.round_number,
                metadata=metadata
            )
            
            return proto_response
            
        except Exception as e:
            logger.error(f"Failed to convert GlobalModel to proto: {e}")
            raise GRPCError(f"GlobalModel conversion failed: {e}")
    
    def global_model_from_proto(self, proto_response) -> GlobalModel:
        """Convert protobuf message to GlobalModel."""
        try:
            # Deserialize global model
            model_weights_hex = proto_response.model_weights.decode('utf-8')
            
            # Create a serialized model dict for deserialization
            serialized_model = {
                'round_number': proto_response.round_number,
                'model_weights': model_weights_hex,
                'accuracy_metrics': dict(proto_response.metadata.accuracy_metrics),
                'participating_clients': [],  # Will be filled if available
                'convergence_score': 0.0,  # Will be filled if available
                'created_at': datetime.now().isoformat()
            }
            
            return self.global_serializer.deserialize_global_model(serialized_model)
            
        except Exception as e:
            logger.error(f"Failed to convert proto to GlobalModel: {e}")
            raise GRPCError(f"GlobalModel conversion failed: {e}")


class GRPCClientManager:
    """Manages gRPC client connections and communication."""
    
    def __init__(self, server_address: str, max_retries: int = 3):
        """
        Initialize gRPC client manager.
        
        Args:
            server_address: Server address in format "host:port"
            max_retries: Maximum number of connection retries
        """
        self.server_address = server_address
        self.max_retries = max_retries
        self.channel = None
        self.stub = None
        self.converter = ProtobufConverter()
        
        logger.info(f"gRPC client manager initialized for {server_address}")
    
    def connect(self):
        """Establish gRPC connection."""
        try:
            # Create channel with options
            options = [
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000)
            ]
            
            self.channel = grpc.insecure_channel(self.server_address, options=options)
            
            # Import and create stub
            import federated_learning_pb2_grpc as pb2_grpc
            self.stub = pb2_grpc.FederatedLearningStub(self.channel)
            
            # Test connection
            self._test_connection()
            
            logger.info(f"gRPC connection established to {self.server_address}")
            
        except Exception as e:
            logger.error(f"Failed to connect to gRPC server: {e}")
            raise GRPCError(f"Connection failed: {e}")
    
    def disconnect(self):
        """Close gRPC connection."""
        try:
            if self.channel:
                self.channel.close()
                self.channel = None
                self.stub = None
                logger.info("gRPC connection closed")
        except Exception as e:
            logger.error(f"Error closing gRPC connection: {e}")
    
    def _test_connection(self):
        """Test gRPC connection with health check."""
        try:
            import federated_learning_pb2 as pb2
            
            request = pb2.HealthRequest(service_name="federated_learning")
            response = self.stub.HealthCheck(request, timeout=5.0)
            
            if not response.healthy:
                raise GRPCError(f"Server health check failed: {response.status}")
                
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                logger.warning("Health check not implemented, assuming server is healthy")
            else:
                raise GRPCError(f"Health check failed: {e}")
    
    def register_client(self, client_id: str, capabilities: ClientCapabilities) -> Dict[str, Any]:
        """Register client with coordinator."""
        try:
            import federated_learning_pb2 as pb2
            
            request = pb2.ClientRegistration(
                client_id=client_id,
                capabilities=self.converter.client_capabilities_to_proto(capabilities),
                client_version="1.0"
            )
            
            response = self.stub.RegisterClient(request)
            
            return {
                'success': response.success,
                'message': response.message,
                'assigned_client_id': response.assigned_client_id,
                'global_model_version': response.global_model_version
            }
            
        except grpc.RpcError as e:
            logger.error(f"Client registration failed: {e}")
            raise GRPCError(f"Registration failed: {e}")
    
    def get_global_model(self, client_id: str, round_number: int) -> GlobalModel:
        """Get global model from coordinator."""
        try:
            import federated_learning_pb2 as pb2
            
            request = pb2.ModelRequest(
                client_id=client_id,
                round_number=round_number,
                model_type="federated_cnn"
            )
            
            response = self.stub.GetGlobalModel(request)
            
            if not response.success:
                raise GRPCError(f"Failed to get global model: {response.message}")
            
            return self.converter.global_model_from_proto(response)
            
        except grpc.RpcError as e:
            logger.error(f"Get global model failed: {e}")
            raise GRPCError(f"Get global model failed: {e}")
    
    def submit_model_update(self, update: ModelUpdate) -> Dict[str, Any]:
        """Submit model update to coordinator."""
        try:
            proto_update = self.converter.model_update_to_proto(update)
            response = self.stub.SubmitModelUpdate(proto_update)
            
            return {
                'success': response.success,
                'message': response.message,
                'next_round_eta': response.next_round_eta
            }
            
        except grpc.RpcError as e:
            logger.error(f"Submit model update failed: {e}")
            raise GRPCError(f"Submit model update failed: {e}")
    
    def get_training_status(self, client_id: str) -> Dict[str, Any]:
        """Get training status from coordinator."""
        try:
            import federated_learning_pb2 as pb2
            
            request = pb2.StatusRequest(client_id=client_id)
            response = self.stub.GetTrainingStatus(request)
            
            return {
                'current_round': response.current_round,
                'active_clients': response.active_clients,
                'round_progress': response.round_progress,
                'global_accuracy': response.global_accuracy,
                'convergence_score': response.convergence_score
            }
            
        except grpc.RpcError as e:
            logger.error(f"Get training status failed: {e}")
            raise GRPCError(f"Get training status failed: {e}")


class GRPCServerManager:
    """Manages gRPC server setup and lifecycle."""
    
    def __init__(self, port: int = 50051, max_workers: int = 10):
        """
        Initialize gRPC server manager.
        
        Args:
            port: Port to listen on
            max_workers: Maximum number of worker threads
        """
        self.port = port
        self.max_workers = max_workers
        self.server = None
        self.converter = ProtobufConverter()
        
        logger.info(f"gRPC server manager initialized on port {port}")
    
    def start_server(self, service_implementation):
        """Start gRPC server with service implementation."""
        try:
            import federated_learning_pb2_grpc as pb2_grpc
            
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers)
            )
            
            # Add service implementation
            pb2_grpc.add_FederatedLearningServicer_to_server(
                service_implementation, self.server
            )
            
            # Add port
            listen_addr = f'[::]:{self.port}'
            self.server.add_insecure_port(listen_addr)
            
            # Start server
            self.server.start()
            
            logger.info(f"gRPC server started on {listen_addr}")
            
        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            raise GRPCError(f"Server start failed: {e}")
    
    def stop_server(self, grace_period: int = 5):
        """Stop gRPC server."""
        try:
            if self.server:
                self.server.stop(grace_period)
                self.server = None
                logger.info("gRPC server stopped")
        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")
    
    def wait_for_termination(self):
        """Wait for server termination."""
        if self.server:
            self.server.wait_for_termination()


def create_grpc_client(server_address: str) -> GRPCClientManager:
    """
    Factory function to create gRPC client.
    
    Args:
        server_address: Server address in format "host:port"
        
    Returns:
        GRPCClientManager: Configured gRPC client
    """
    return GRPCClientManager(server_address)


def create_grpc_server(port: int = 50051, max_workers: int = 10) -> GRPCServerManager:
    """
    Factory function to create gRPC server.
    
    Args:
        port: Port to listen on
        max_workers: Maximum number of worker threads
        
    Returns:
        GRPCServerManager: Configured gRPC server
    """
    return GRPCServerManager(port, max_workers)