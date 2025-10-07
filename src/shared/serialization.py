"""
Serialization utilities for federated learning data models.
Handles conversion between Python objects and network-transmittable formats.
"""

import pickle
import json
import torch
import io
from typing import Dict, Any, Optional, Union
import logging
from datetime import datetime

from .models import ModelUpdate, GlobalModel, ModelWeights

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass


class ModelWeightSerializer:
    """Handles serialization of PyTorch model weights."""
    
    @staticmethod
    def serialize_weights(weights: ModelWeights) -> bytes:
        """
        Serialize PyTorch model weights to bytes.
        
        Args:
            weights: Dictionary of layer names to tensors
            
        Returns:
            bytes: Serialized weights
        """
        try:
            # Use PyTorch's built-in serialization
            buffer = io.BytesIO()
            torch.save(weights, buffer)
            serialized_data = buffer.getvalue()
            
            logger.debug(f"Serialized {len(weights)} layers, {len(serialized_data)} bytes")
            return serialized_data
            
        except Exception as e:
            logger.error(f"Failed to serialize model weights: {str(e)}")
            raise SerializationError(f"Weight serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize_weights(data: bytes) -> ModelWeights:
        """
        Deserialize bytes back to PyTorch model weights.
        
        Args:
            data: Serialized weight data
            
        Returns:
            ModelWeights: Dictionary of layer names to tensors
        """
        try:
            buffer = io.BytesIO(data)
            weights = torch.load(buffer, map_location='cpu')  # Load to CPU first
            
            # Validate the deserialized data
            if not isinstance(weights, dict):
                raise SerializationError("Deserialized data is not a dictionary")
            
            for layer_name, tensor in weights.items():
                if not isinstance(tensor, torch.Tensor):
                    raise SerializationError(f"Layer {layer_name} is not a tensor")
            
            logger.debug(f"Deserialized {len(weights)} layers")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to deserialize model weights: {str(e)}")
            raise SerializationError(f"Weight deserialization failed: {str(e)}")


class ModelUpdateSerializer:
    """Handles serialization of ModelUpdate objects."""
    
    def __init__(self):
        self.weight_serializer = ModelWeightSerializer()
    
    def serialize_model_update(self, update: ModelUpdate) -> Dict[str, Any]:
        """
        Serialize ModelUpdate to a dictionary suitable for JSON/gRPC transmission.
        
        Args:
            update: ModelUpdate object to serialize
            
        Returns:
            Dict[str, Any]: Serialized update data
        """
        try:
            # Serialize model weights separately
            serialized_weights = self.weight_serializer.serialize_weights(update.model_weights)
            
            serialized_update = {
                'client_id': update.client_id,
                'round_number': update.round_number,
                'model_weights': serialized_weights.hex(),  # Convert bytes to hex string
                'num_samples': update.num_samples,
                'training_loss': float(update.training_loss),
                'privacy_budget_used': float(update.privacy_budget_used),
                'compression_ratio': float(update.compression_ratio),
                'timestamp': update.timestamp.isoformat()
            }
            
            logger.debug(f"Serialized model update for client {update.client_id}")
            return serialized_update
            
        except Exception as e:
            logger.error(f"Failed to serialize model update: {str(e)}")
            raise SerializationError(f"Model update serialization failed: {str(e)}")
    
    def deserialize_model_update(self, data: Dict[str, Any]) -> ModelUpdate:
        """
        Deserialize dictionary back to ModelUpdate object.
        
        Args:
            data: Serialized update data
            
        Returns:
            ModelUpdate: Reconstructed ModelUpdate object
        """
        try:
            # Deserialize model weights
            weights_hex = data['model_weights']
            weights_bytes = bytes.fromhex(weights_hex)
            model_weights = self.weight_serializer.deserialize_weights(weights_bytes)
            
            # Reconstruct ModelUpdate object
            update = ModelUpdate(
                client_id=data['client_id'],
                round_number=int(data['round_number']),
                model_weights=model_weights,
                num_samples=int(data['num_samples']),
                training_loss=float(data['training_loss']),
                privacy_budget_used=float(data['privacy_budget_used']),
                compression_ratio=float(data['compression_ratio']),
                timestamp=datetime.fromisoformat(data['timestamp'])
            )
            
            logger.debug(f"Deserialized model update for client {update.client_id}")
            return update
            
        except Exception as e:
            logger.error(f"Failed to deserialize model update: {str(e)}")
            raise SerializationError(f"Model update deserialization failed: {str(e)}")


class GlobalModelSerializer:
    """Handles serialization of GlobalModel objects."""
    
    def __init__(self):
        self.weight_serializer = ModelWeightSerializer()
    
    def serialize_global_model(self, model: GlobalModel) -> Dict[str, Any]:
        """
        Serialize GlobalModel to a dictionary.
        
        Args:
            model: GlobalModel object to serialize
            
        Returns:
            Dict[str, Any]: Serialized model data
        """
        try:
            # Serialize model weights
            serialized_weights = self.weight_serializer.serialize_weights(model.model_weights)
            
            serialized_model = {
                'round_number': model.round_number,
                'model_weights': serialized_weights.hex(),
                'accuracy_metrics': model.accuracy_metrics,
                'participating_clients': model.participating_clients,
                'convergence_score': float(model.convergence_score),
                'created_at': model.created_at.isoformat()
            }
            
            logger.debug(f"Serialized global model for round {model.round_number}")
            return serialized_model
            
        except Exception as e:
            logger.error(f"Failed to serialize global model: {str(e)}")
            raise SerializationError(f"Global model serialization failed: {str(e)}")
    
    def deserialize_global_model(self, data: Dict[str, Any]) -> GlobalModel:
        """
        Deserialize dictionary back to GlobalModel object.
        
        Args:
            data: Serialized model data
            
        Returns:
            GlobalModel: Reconstructed GlobalModel object
        """
        try:
            # Deserialize model weights
            weights_hex = data['model_weights']
            weights_bytes = bytes.fromhex(weights_hex)
            model_weights = self.weight_serializer.deserialize_weights(weights_bytes)
            
            # Reconstruct GlobalModel object
            model = GlobalModel(
                round_number=int(data['round_number']),
                model_weights=model_weights,
                accuracy_metrics=data['accuracy_metrics'],
                participating_clients=data['participating_clients'],
                convergence_score=float(data['convergence_score']),
                created_at=datetime.fromisoformat(data['created_at'])
            )
            
            logger.debug(f"Deserialized global model for round {model.round_number}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to deserialize global model: {str(e)}")
            raise SerializationError(f"Global model deserialization failed: {str(e)}")


class CompactSerializer:
    """Handles compact serialization for efficient network transmission."""
    
    @staticmethod
    def serialize_to_bytes(obj: Any) -> bytes:
        """
        Serialize any Python object to bytes using pickle.
        
        Args:
            obj: Object to serialize
            
        Returns:
            bytes: Serialized object
        """
        try:
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to serialize object: {str(e)}")
            raise SerializationError(f"Object serialization failed: {str(e)}")
    
    @staticmethod
    def deserialize_from_bytes(data: bytes) -> Any:
        """
        Deserialize bytes back to Python object using pickle.
        
        Args:
            data: Serialized data
            
        Returns:
            Any: Deserialized object
        """
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize object: {str(e)}")
            raise SerializationError(f"Object deserialization failed: {str(e)}")


def get_serialization_size(obj: Any) -> int:
    """
    Get the serialized size of an object in bytes.
    
    Args:
        obj: Object to measure
        
    Returns:
        int: Size in bytes
    """
    try:
        serialized = CompactSerializer.serialize_to_bytes(obj)
        return len(serialized)
    except Exception as e:
        logger.error(f"Failed to measure serialization size: {str(e)}")
        return 0


def validate_serialization_roundtrip(obj: Any) -> bool:
    """
    Validate that an object can be serialized and deserialized correctly.
    
    Args:
        obj: Object to test
        
    Returns:
        bool: True if roundtrip is successful
    """
    try:
        serialized = CompactSerializer.serialize_to_bytes(obj)
        deserialized = CompactSerializer.deserialize_from_bytes(serialized)
        
        # For tensors, check if they're equal
        if isinstance(obj, torch.Tensor):
            return torch.equal(obj, deserialized)
        
        # For dictionaries of tensors (model weights)
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            if set(obj.keys()) != set(deserialized.keys()):
                return False
            return all(torch.equal(obj[k], deserialized[k]) for k in obj.keys())
        
        # For other objects, use equality
        return obj == deserialized
        
    except Exception as e:
        logger.error(f"Serialization roundtrip test failed: {str(e)}")
        return False