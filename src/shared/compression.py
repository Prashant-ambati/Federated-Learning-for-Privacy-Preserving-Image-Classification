"""
Model compression utilities for efficient federated learning communication.
Implements various compression techniques to reduce network overhead.
"""

import torch
import numpy as np
import lz4.frame
import pickle
import io
from typing import Dict, Tuple, Any, Optional
import logging
from abc import ABC, abstractmethod

from .models import ModelWeights, CompressedUpdate
from .interfaces import CompressionInterface

logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """Custom exception for compression errors."""
    pass


class BaseCompressor(ABC):
    """Abstract base class for model compression algorithms."""
    
    @abstractmethod
    def compress(self, weights: ModelWeights) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress model weights.
        
        Args:
            weights: Dictionary of layer names to tensors
            
        Returns:
            Tuple of (compressed_data, metadata)
        """
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> ModelWeights:
        """
        Decompress model weights.
        
        Args:
            compressed_data: Compressed weight data
            metadata: Compression metadata
            
        Returns:
            ModelWeights: Decompressed weights
        """
        pass
    
    @abstractmethod
    def get_compression_name(self) -> str:
        """Get the name of the compression algorithm."""
        pass


class LZ4Compressor(BaseCompressor):
    """LZ4-based compression for model weights."""
    
    def __init__(self, compression_level: int = 1):
        """
        Initialize LZ4 compressor.
        
        Args:
            compression_level: LZ4 compression level (1-12, higher = better compression)
        """
        self.compression_level = max(1, min(12, compression_level))
    
    def compress(self, weights: ModelWeights) -> Tuple[bytes, Dict[str, Any]]:
        """Compress model weights using LZ4."""
        try:
            # Serialize weights to bytes first
            buffer = io.BytesIO()
            torch.save(weights, buffer)
            serialized_data = buffer.getvalue()
            
            # Compress using LZ4
            compressed_data = lz4.frame.compress(
                serialized_data, 
                compression_level=self.compression_level
            )
            
            metadata = {
                'algorithm': self.get_compression_name(),
                'original_size': len(serialized_data),
                'compressed_size': len(compressed_data),
                'compression_level': self.compression_level
            }
            
            logger.debug(f"LZ4 compressed {len(serialized_data)} -> {len(compressed_data)} bytes")
            return compressed_data, metadata
            
        except Exception as e:
            logger.error(f"LZ4 compression failed: {str(e)}")
            raise CompressionError(f"LZ4 compression failed: {str(e)}")
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> ModelWeights:
        """Decompress LZ4-compressed model weights."""
        try:
            # Decompress using LZ4
            decompressed_data = lz4.frame.decompress(compressed_data)
            
            # Deserialize back to weights
            buffer = io.BytesIO(decompressed_data)
            weights = torch.load(buffer, map_location='cpu')
            
            logger.debug(f"LZ4 decompressed {len(compressed_data)} -> {len(decompressed_data)} bytes")
            return weights
            
        except Exception as e:
            logger.error(f"LZ4 decompression failed: {str(e)}")
            raise CompressionError(f"LZ4 decompression failed: {str(e)}")
    
    def get_compression_name(self) -> str:
        return "lz4"


class QuantizationCompressor(BaseCompressor):
    """Quantization-based compression for model weights."""
    
    def __init__(self, bits: int = 8, symmetric: bool = True):
        """
        Initialize quantization compressor.
        
        Args:
            bits: Number of bits for quantization (1-32)
            symmetric: Whether to use symmetric quantization
        """
        self.bits = max(1, min(32, bits))
        self.symmetric = symmetric
        self.levels = 2 ** self.bits
    
    def compress(self, weights: ModelWeights) -> Tuple[bytes, Dict[str, Any]]:
        """Compress model weights using quantization."""
        try:
            compressed_weights = {}
            quantization_params = {}
            
            for layer_name, tensor in weights.items():
                # Quantize tensor
                quantized_tensor, scale, zero_point = self._quantize_tensor(tensor)
                compressed_weights[layer_name] = quantized_tensor
                quantization_params[layer_name] = {
                    'scale': scale,
                    'zero_point': zero_point,
                    'original_shape': tensor.shape,
                    'original_dtype': str(tensor.dtype)
                }
            
            # Serialize compressed weights
            buffer = io.BytesIO()
            torch.save(compressed_weights, buffer)
            compressed_data = buffer.getvalue()
            
            metadata = {
                'algorithm': self.get_compression_name(),
                'bits': self.bits,
                'symmetric': self.symmetric,
                'quantization_params': quantization_params
            }
            
            logger.debug(f"Quantization compressed to {self.bits} bits")
            return compressed_data, metadata
            
        except Exception as e:
            logger.error(f"Quantization compression failed: {str(e)}")
            raise CompressionError(f"Quantization compression failed: {str(e)}")
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> ModelWeights:
        """Decompress quantized model weights."""
        try:
            # Deserialize compressed weights
            buffer = io.BytesIO(compressed_data)
            compressed_weights = torch.load(buffer, map_location='cpu')
            
            # Dequantize weights
            decompressed_weights = {}
            quantization_params = metadata['quantization_params']
            
            for layer_name, quantized_tensor in compressed_weights.items():
                params = quantization_params[layer_name]
                decompressed_tensor = self._dequantize_tensor(
                    quantized_tensor, 
                    params['scale'], 
                    params['zero_point'],
                    params['original_shape'],
                    params['original_dtype']
                )
                decompressed_weights[layer_name] = decompressed_tensor
            
            logger.debug(f"Quantization decompressed from {metadata['bits']} bits")
            return decompressed_weights
            
        except Exception as e:
            logger.error(f"Quantization decompression failed: {str(e)}")
            raise CompressionError(f"Quantization decompression failed: {str(e)}")
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """Quantize a single tensor."""
        # Calculate quantization parameters
        if self.symmetric:
            max_val = torch.abs(tensor).max().item()
            scale = (2 * max_val) / (self.levels - 1)
            zero_point = (self.levels - 1) // 2
        else:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            scale = (max_val - min_val) / (self.levels - 1)
            zero_point = -round(min_val / scale)
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, 0, self.levels - 1)
        
        # Convert to appropriate integer type
        if self.bits <= 8:
            quantized = quantized.to(torch.uint8)
        elif self.bits <= 16:
            quantized = quantized.to(torch.int16)
        else:
            quantized = quantized.to(torch.int32)
        
        return quantized, scale, zero_point
    
    def _dequantize_tensor(self, quantized: torch.Tensor, scale: float, 
                          zero_point: int, original_shape: torch.Size, 
                          original_dtype: str) -> torch.Tensor:
        """Dequantize a single tensor."""
        # Dequantize
        dequantized = (quantized.float() - zero_point) * scale
        
        # Restore original shape and dtype
        dequantized = dequantized.reshape(original_shape)
        if 'float32' in original_dtype:
            dequantized = dequantized.to(torch.float32)
        elif 'float64' in original_dtype:
            dequantized = dequantized.to(torch.float64)
        
        return dequantized
    
    def get_compression_name(self) -> str:
        return f"quantization_{self.bits}bit"


class TopKSparsificationCompressor(BaseCompressor):
    """Top-K sparsification compression for model weights."""
    
    def __init__(self, sparsity_ratio: float = 0.9):
        """
        Initialize Top-K sparsification compressor.
        
        Args:
            sparsity_ratio: Fraction of weights to zero out (0.0-1.0)
        """
        self.sparsity_ratio = max(0.0, min(1.0, sparsity_ratio))
    
    def compress(self, weights: ModelWeights) -> Tuple[bytes, Dict[str, Any]]:
        """Compress model weights using Top-K sparsification."""
        try:
            compressed_weights = {}
            sparsification_params = {}
            
            for layer_name, tensor in weights.items():
                # Apply Top-K sparsification
                sparse_tensor, indices, original_shape = self._sparsify_tensor(tensor)
                compressed_weights[layer_name] = {
                    'values': sparse_tensor,
                    'indices': indices
                }
                sparsification_params[layer_name] = {
                    'original_shape': original_shape,
                    'original_dtype': str(tensor.dtype),
                    'sparsity_ratio': self.sparsity_ratio
                }
            
            # Serialize compressed weights
            buffer = io.BytesIO()
            pickle.dump(compressed_weights, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = buffer.getvalue()
            
            metadata = {
                'algorithm': self.get_compression_name(),
                'sparsity_ratio': self.sparsity_ratio,
                'sparsification_params': sparsification_params
            }
            
            logger.debug(f"Top-K sparsification with ratio {self.sparsity_ratio}")
            return compressed_data, metadata
            
        except Exception as e:
            logger.error(f"Top-K sparsification failed: {str(e)}")
            raise CompressionError(f"Top-K sparsification failed: {str(e)}")
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> ModelWeights:
        """Decompress Top-K sparsified model weights."""
        try:
            # Deserialize compressed weights
            buffer = io.BytesIO(compressed_data)
            compressed_weights = pickle.load(buffer)
            
            # Reconstruct sparse weights
            decompressed_weights = {}
            sparsification_params = metadata['sparsification_params']
            
            for layer_name, sparse_data in compressed_weights.items():
                params = sparsification_params[layer_name]
                decompressed_tensor = self._desparsify_tensor(
                    sparse_data['values'],
                    sparse_data['indices'],
                    params['original_shape'],
                    params['original_dtype']
                )
                decompressed_weights[layer_name] = decompressed_tensor
            
            logger.debug(f"Top-K desparsification completed")
            return decompressed_weights
            
        except Exception as e:
            logger.error(f"Top-K desparsification failed: {str(e)}")
            raise CompressionError(f"Top-K desparsification failed: {str(e)}")
    
    def _sparsify_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """Apply Top-K sparsification to a tensor."""
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        # Calculate number of elements to keep
        num_elements = flat_tensor.numel()
        k = int(num_elements * (1 - self.sparsity_ratio))
        
        if k == 0:
            # Keep at least one element
            k = 1
        
        # Get top-k elements by absolute value
        _, indices = torch.topk(torch.abs(flat_tensor), k)
        values = flat_tensor[indices]
        
        return values, indices, original_shape
    
    def _desparsify_tensor(self, values: torch.Tensor, indices: torch.Tensor, 
                          original_shape: torch.Size, original_dtype: str) -> torch.Tensor:
        """Reconstruct tensor from sparse representation."""
        # Create zero tensor with original shape
        num_elements = torch.prod(torch.tensor(original_shape)).item()
        flat_tensor = torch.zeros(num_elements, dtype=values.dtype)
        
        # Fill in the non-zero values
        flat_tensor[indices] = values
        
        # Reshape to original shape
        tensor = flat_tensor.reshape(original_shape)
        
        # Restore original dtype
        if 'float32' in original_dtype:
            tensor = tensor.to(torch.float32)
        elif 'float64' in original_dtype:
            tensor = tensor.to(torch.float64)
        
        return tensor
    
    def get_compression_name(self) -> str:
        return f"topk_sparsification_{self.sparsity_ratio}"


class ModelCompressionService(CompressionInterface):
    """Main service for model compression operations."""
    
    def __init__(self, algorithm: str = "lz4", **kwargs):
        """
        Initialize compression service.
        
        Args:
            algorithm: Compression algorithm to use
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.compressor = self._create_compressor(algorithm, **kwargs)
    
    def _create_compressor(self, algorithm: str, **kwargs) -> BaseCompressor:
        """Create compressor instance based on algorithm name."""
        if algorithm == "lz4":
            return LZ4Compressor(**kwargs)
        elif algorithm == "quantization":
            return QuantizationCompressor(**kwargs)
        elif algorithm == "topk":
            return TopKSparsificationCompressor(**kwargs)
        else:
            raise ValueError(f"Unknown compression algorithm: {algorithm}")
    
    def compress_weights(self, weights: ModelWeights) -> bytes:
        """Compress model weights to bytes."""
        try:
            compressed_data, metadata = self.compressor.compress(weights)
            
            # Package compressed data with metadata
            package = {
                'compressed_data': compressed_data,
                'metadata': metadata
            }
            
            # Serialize the package
            buffer = io.BytesIO()
            pickle.dump(package, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Weight compression failed: {str(e)}")
            raise CompressionError(f"Weight compression failed: {str(e)}")
    
    def decompress_weights(self, compressed_data: bytes) -> ModelWeights:
        """Decompress bytes back to model weights."""
        try:
            # Deserialize the package
            buffer = io.BytesIO(compressed_data)
            package = pickle.load(buffer)
            
            # Extract compressed data and metadata
            data = package['compressed_data']
            metadata = package['metadata']
            
            # Create appropriate compressor for decompression
            algorithm = metadata['algorithm']
            if algorithm != self.compressor.get_compression_name():
                # Create compressor for the specific algorithm
                if algorithm.startswith('quantization'):
                    bits = metadata['bits']
                    symmetric = metadata['symmetric']
                    compressor = QuantizationCompressor(bits=bits, symmetric=symmetric)
                elif algorithm.startswith('topk'):
                    sparsity_ratio = metadata['sparsity_ratio']
                    compressor = TopKSparsificationCompressor(sparsity_ratio=sparsity_ratio)
                else:
                    compressor = self.compressor
            else:
                compressor = self.compressor
            
            # Decompress weights
            return compressor.decompress(data, metadata)
            
        except Exception as e:
            logger.error(f"Weight decompression failed: {str(e)}")
            raise CompressionError(f"Weight decompression failed: {str(e)}")
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        if original_size == 0:
            return 0.0
        return compressed_size / original_size
    
    def estimate_compression_ratio(self, weights: ModelWeights) -> float:
        """Estimate compression ratio for given weights."""
        try:
            # Calculate original size
            original_size = sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
            
            # Compress and measure
            compressed_data = self.compress_weights(weights)
            compressed_size = len(compressed_data)
            
            return self.get_compression_ratio(original_size, compressed_size)
            
        except Exception as e:
            logger.error(f"Compression ratio estimation failed: {str(e)}")
            return 1.0  # No compression


def create_compression_service(algorithm: str = "lz4", **kwargs) -> ModelCompressionService:
    """
    Factory function to create compression service.
    
    Args:
        algorithm: Compression algorithm ("lz4", "quantization", "topk")
        **kwargs: Algorithm-specific parameters
        
    Returns:
        ModelCompressionService: Configured compression service
    """
    return ModelCompressionService(algorithm=algorithm, **kwargs)


def benchmark_compression_algorithms(weights: ModelWeights) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different compression algorithms on given weights.
    
    Args:
        weights: Model weights to benchmark
        
    Returns:
        Dict: Benchmark results for each algorithm
    """
    algorithms = [
        ("lz4", {}),
        ("quantization", {"bits": 8}),
        ("quantization", {"bits": 4}),
        ("topk", {"sparsity_ratio": 0.9}),
        ("topk", {"sparsity_ratio": 0.95})
    ]
    
    results = {}
    original_size = sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
    
    for algorithm, params in algorithms:
        try:
            service = create_compression_service(algorithm, **params)
            
            # Measure compression
            import time
            start_time = time.time()
            compressed_data = service.compress_weights(weights)
            compression_time = time.time() - start_time
            
            # Measure decompression
            start_time = time.time()
            decompressed_weights = service.decompress_weights(compressed_data)
            decompression_time = time.time() - start_time
            
            # Calculate metrics
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size
            
            # Verify accuracy (for quantization and sparsification)
            accuracy_loss = 0.0
            if algorithm in ["quantization", "topk"]:
                total_error = 0.0
                total_elements = 0
                for layer_name in weights.keys():
                    error = torch.abs(weights[layer_name] - decompressed_weights[layer_name]).sum().item()
                    total_error += error
                    total_elements += weights[layer_name].numel()
                accuracy_loss = total_error / total_elements
            
            key = f"{algorithm}_{params}" if params else algorithm
            results[key] = {
                'compression_ratio': compression_ratio,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'accuracy_loss': accuracy_loss
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed for {algorithm}: {str(e)}")
            continue
    
    return results