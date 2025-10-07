"""
gRPC client implementation for federated learning clients.
Handles communication with the coordinator server.
"""

import grpc
import logging
import time
import threading
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
import random

# Add proto path for imports
import sys
import os
proto_path = os.path.join(os.path.dirname(__file__), '..', '..', 'proto')
sys.path.insert(0, proto_path)

import federated_learning_pb2 as pb2
import federated_learning_pb2_grpc as pb2_grpc

from ..shared.models import (
    ModelUpdate, GlobalModel, PrivacyConfig, ClientCapabilities,
    ComputePowerLevel, TrainingMetrics
)
from ..shared.grpc_utils import ProtobufConverter, GRPCError

logger = logging.getLogger(__name__)


class FederatedLearningClient:
    """gRPC client for federated learning communication."""
    
    def __init__(self, 
                 server_address: str,
                 client_id: str,
                 capabilities: ClientCapabilities,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize federated learning client.
        
        Args:
            server_address: Coordinator server address (host:port)
            client_id: Unique client identifier
            capabilities: Client computational capabilities
            max_retries: Maximum number of connection retries
            retry_delay: Delay between retries in seconds
        """
        self.server_address = server_address
        self.client_id = client_id
        self.capabilities = capabilities
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Connection management
        self.channel = None
        self.stub = None
        self.connected = False
        self.connection_lock = threading.RLock()
        
        # Protocol buffer converter
        self.converter = ProtobufConverter()
        
        # Client state
        self.registered = False
        self.current_round = 0
        self.round_config = None
        self.global_model = None
        
        # Callbacks
        self.on_model_received = None
        self.on_round_started = None
        self.on_training_completed = None
        
        logger.info(f"Federated learning client initialized: {client_id} -> {server_address}")
    
    def connect(self) -> bool:
        """
        Establish connection to coordinator server.
        
        Returns:
            bool: True if connection successful
        """
        try:
            with self.connection_lock:
                if self.connected:
                    return True
                
                logger.info(f"Connecting to coordinator at {self.server_address}")
                
                # Create gRPC channel with options
                options = [
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),     # 100MB
                ]
                
                self.channel = grpc.insecure_channel(self.server_address, options=options)
                self.stub = pb2_grpc.FederatedLearningStub(self.channel)
                
                # Test connection with health check
                if self._test_connection():
                    self.connected = True
                    logger.info("Successfully connected to coordinator")
                    return True
                else:
                    self._cleanup_connection()
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            self._cleanup_connection()
            return False
    
    def disconnect(self):
        """Disconnect from coordinator server."""
        try:
            with self.connection_lock:
                if self.connected:
                    logger.info("Disconnecting from coordinator")
                    self._cleanup_connection()
                    self.connected = False
                    self.registered = False
                    logger.info("Disconnected from coordinator")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def register(self) -> bool:
        """
        Register client with coordinator.
        
        Returns:
            bool: True if registration successful
        """
        try:
            if not self.connected:
                if not self.connect():
                    return False
            
            logger.info(f"Registering client {self.client_id}")
            
            # Create registration request
            request = pb2.ClientRegistration(
                client_id=self.client_id,
                capabilities=self.converter.client_capabilities_to_proto(self.capabilities),
                client_version="1.0.0",
                metadata={
                    "registration_time": datetime.now().isoformat(),
                    "client_type": "python_client"
                }
            )
            
            # Send registration request with retry
            response = self._call_with_retry(self.stub.RegisterClient, request)
            
            if response and response.success:
                self.registered = True
                self.current_round = response.global_model_version
                
                logger.info(f"Client {self.client_id} registered successfully")
                logger.info(f"Assigned to round {self.current_round}")
                
                return True
            else:
                error_msg = response.message if response else "Unknown error"
                logger.error(f"Registration failed: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    def get_global_model(self, round_number: Optional[int] = None) -> Optional[GlobalModel]:
        """
        Get global model from coordinator.
        
        Args:
            round_number: Specific round number (uses current if None)
            
        Returns:
            GlobalModel: Global model or None if failed
        """
        try:
            if not self.registered:
                logger.error("Client not registered")
                return None
            
            round_num = round_number or self.current_round
            
            logger.debug(f"Requesting global model for round {round_num}")
            
            # Create model request
            request = pb2.ModelRequest(
                client_id=self.client_id,
                round_number=round_num,
                model_type="federated_cnn"
            )
            
            # Send request with retry
            response = self._call_with_retry(self.stub.GetGlobalModel, request)
            
            if response and response.success:
                # Convert response to GlobalModel
                self.global_model = self.converter.global_model_from_proto(response)
                
                logger.info(f"Received global model for round {round_num}")
                
                # Trigger callback if set
                if self.on_model_received:
                    self.on_model_received(self.global_model)
                
                return self.global_model
            else:
                error_msg = response.message if response else "Unknown error"
                logger.error(f"Failed to get global model: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Get global model failed: {e}")
            return None
    
    def submit_model_update(self, model_update: ModelUpdate) -> bool:
        """
        Submit model update to coordinator.
        
        Args:
            model_update: Model update to submit
            
        Returns:
            bool: True if submission successful
        """
        try:
            if not self.registered:
                logger.error("Client not registered")
                return False
            
            logger.info(f"Submitting model update for round {model_update.round_number}")
            
            # Convert model update to proto
            proto_update = self.converter.model_update_to_proto(model_update)
            
            # Send update with retry
            response = self._call_with_retry(self.stub.SubmitModelUpdate, proto_update)
            
            if response and response.success:
                logger.info("Model update submitted successfully")
                
                # Update next round ETA if provided
                if response.next_round_eta > 0:
                    next_eta = datetime.fromtimestamp(response.next_round_eta)
                    logger.info(f"Next round estimated at: {next_eta}")
                
                return True
            else:
                error_msg = response.message if response else "Unknown error"
                logger.error(f"Model update submission failed: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Submit model update failed: {e}")
            return False
    
    def join_training_round(self, round_number: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Join a training round.
        
        Args:
            round_number: Round number to join (uses current if None)
            
        Returns:
            Dict: Round configuration or None if failed
        """
        try:
            if not self.registered:
                logger.error("Client not registered")
                return None
            
            round_num = round_number or self.current_round
            
            logger.info(f"Joining training round {round_num}")
            
            # Create join request
            request = pb2.RoundJoinRequest(
                client_id=self.client_id,
                requested_round=round_num
            )
            
            # Send request with retry
            response = self._call_with_retry(self.stub.JoinTrainingRound, request)
            
            if response and response.success:
                self.current_round = response.assigned_round
                
                # Extract round configuration
                config = response.round_config
                self.round_config = {
                    'round_number': config.round_number,
                    'min_clients': config.min_clients,
                    'max_clients': config.max_clients,
                    'local_epochs': config.local_epochs,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'timeout_seconds': config.timeout_seconds,
                    'optimizer_type': config.optimizer_type,
                    'privacy_config': self.converter.privacy_config_from_proto(config.privacy_config)
                }
                
                logger.info(f"Joined round {self.current_round}")
                
                # Trigger callback if set
                if self.on_round_started:
                    self.on_round_started(self.round_config)
                
                return self.round_config
            else:
                error_msg = response.message if response else "Unknown error"
                logger.error(f"Failed to join training round: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Join training round failed: {e}")
            return None
    
    def get_training_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current training status from coordinator.
        
        Returns:
            Dict: Training status or None if failed
        """
        try:
            if not self.registered:
                logger.error("Client not registered")
                return None
            
            # Create status request
            request = pb2.StatusRequest(client_id=self.client_id)
            
            # Send request with retry
            response = self._call_with_retry(self.stub.GetTrainingStatus, request)
            
            if response:
                status = {
                    'current_round': response.current_round,
                    'active_clients': response.active_clients,
                    'round_progress': response.round_progress,
                    'global_accuracy': response.global_accuracy,
                    'convergence_score': response.convergence_score,
                    'estimated_completion': datetime.fromtimestamp(response.estimated_completion) if response.estimated_completion > 0 else None,
                    'round_status': response.round_status
                }
                
                return status
            else:
                logger.error("Failed to get training status")
                return None
                
        except Exception as e:
            logger.error(f"Get training status failed: {e}")
            return None
    
    def update_capabilities(self, new_capabilities: ClientCapabilities) -> bool:
        """
        Update client capabilities.
        
        Args:
            new_capabilities: Updated client capabilities
            
        Returns:
            bool: True if update successful
        """
        try:
            if not self.registered:
                logger.error("Client not registered")
                return False
            
            logger.info("Updating client capabilities")
            
            # Convert capabilities to proto
            proto_capabilities = self.converter.client_capabilities_to_proto(new_capabilities)
            
            # Send update with retry
            response = self._call_with_retry(self.stub.UpdateClientCapabilities, proto_capabilities)
            
            if response and response.success:
                self.capabilities = new_capabilities
                logger.info("Client capabilities updated successfully")
                return True
            else:
                error_msg = response.message if response else "Unknown error"
                logger.error(f"Failed to update capabilities: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Update capabilities failed: {e}")
            return False
    
    def set_callbacks(self, 
                     on_model_received: Optional[Callable] = None,
                     on_round_started: Optional[Callable] = None,
                     on_training_completed: Optional[Callable] = None):
        """
        Set callback functions for client events.
        
        Args:
            on_model_received: Called when global model is received
            on_round_started: Called when training round starts
            on_training_completed: Called when training is completed
        """
        self.on_model_received = on_model_received
        self.on_round_started = on_round_started
        self.on_training_completed = on_training_completed
        
        logger.info("Client callbacks configured")
    
    def _test_connection(self) -> bool:
        """Test connection with health check."""
        try:
            request = pb2.HealthRequest(service_name="federated_learning")
            response = self.stub.HealthCheck(request, timeout=5.0)
            
            if response.healthy:
                logger.debug("Health check passed")
                return True
            else:
                logger.warning(f"Health check failed: {response.status}")
                return False
                
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                logger.warning("Health check not implemented, assuming server is healthy")
                return True
            else:
                logger.error(f"Health check failed: {e}")
                return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def _call_with_retry(self, method, request, timeout: float = 30.0):
        """Call gRPC method with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if not self.connected:
                    if not self.connect():
                        raise GRPCError("Not connected to server")
                
                # Call the method
                response = method(request, timeout=timeout)
                return response
                
            except grpc.RpcError as e:
                last_exception = e
                
                if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                    # Retryable errors
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"gRPC call failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}")
                        time.sleep(delay)
                        
                        # Reset connection for next attempt
                        self._cleanup_connection()
                        continue
                    else:
                        logger.error(f"gRPC call failed after {self.max_retries + 1} attempts: {e}")
                        break
                else:
                    # Non-retryable errors
                    logger.error(f"gRPC call failed with non-retryable error: {e}")
                    break
                    
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error in gRPC call: {e}")
                break
        
        # If we get here, all retries failed
        if last_exception:
            raise GRPCError(f"gRPC call failed: {last_exception}")
        else:
            raise GRPCError("gRPC call failed with unknown error")
    
    def _cleanup_connection(self):
        """Clean up gRPC connection resources."""
        try:
            if self.channel:
                self.channel.close()
            self.channel = None
            self.stub = None
            self.connected = False
        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class ClientConnectionManager:
    """Manages client connection lifecycle and automatic reconnection."""
    
    def __init__(self, client: FederatedLearningClient, 
                 heartbeat_interval: float = 30.0,
                 auto_reconnect: bool = True):
        """
        Initialize connection manager.
        
        Args:
            client: Federated learning client
            heartbeat_interval: Heartbeat interval in seconds
            auto_reconnect: Whether to automatically reconnect
        """
        self.client = client
        self.heartbeat_interval = heartbeat_interval
        self.auto_reconnect = auto_reconnect
        
        self.running = False
        self.heartbeat_thread = None
        
        logger.info("Client connection manager initialized")
    
    def start(self):
        """Start connection management."""
        try:
            self.running = True
            
            # Connect client
            if not self.client.connect():
                raise GRPCError("Failed to establish initial connection")
            
            # Register client
            if not self.client.register():
                raise GRPCError("Failed to register client")
            
            # Start heartbeat thread
            if self.heartbeat_interval > 0:
                self.heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop, 
                    daemon=True
                )
                self.heartbeat_thread.start()
            
            logger.info("Connection manager started")
            
        except Exception as e:
            logger.error(f"Failed to start connection manager: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop connection management."""
        try:
            logger.info("Stopping connection manager")
            self.running = False
            
            # Wait for heartbeat thread to finish
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=5.0)
            
            # Disconnect client
            self.client.disconnect()
            
            logger.info("Connection manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping connection manager: {e}")
    
    def _heartbeat_loop(self):
        """Heartbeat loop to maintain connection."""
        while self.running:
            try:
                time.sleep(self.heartbeat_interval)
                
                if not self.running:
                    break
                
                # Send heartbeat (get training status)
                status = self.client.get_training_status()
                
                if status is None and self.auto_reconnect:
                    logger.warning("Heartbeat failed, attempting reconnection")
                    self._attempt_reconnection()
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                if self.auto_reconnect:
                    self._attempt_reconnection()
    
    def _attempt_reconnection(self):
        """Attempt to reconnect client."""
        try:
            logger.info("Attempting to reconnect client")
            
            # Disconnect and reconnect
            self.client.disconnect()
            
            if self.client.connect() and self.client.register():
                logger.info("Client reconnected successfully")
            else:
                logger.error("Failed to reconnect client")
                
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}")


def create_federated_client(server_address: str,
                           client_id: str,
                           capabilities: ClientCapabilities) -> FederatedLearningClient:
    """
    Factory function to create federated learning client.
    
    Args:
        server_address: Coordinator server address
        client_id: Unique client identifier
        capabilities: Client capabilities
        
    Returns:
        FederatedLearningClient: Configured client
    """
    return FederatedLearningClient(server_address, client_id, capabilities)