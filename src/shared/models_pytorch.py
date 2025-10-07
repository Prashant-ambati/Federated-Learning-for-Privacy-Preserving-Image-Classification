"""
PyTorch CNN model architectures for federated learning image classification.
Implements various CNN architectures suitable for MNIST, CIFAR-10, and other datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging

from .interfaces import ModelInterface
from .models import ModelWeights

logger = logging.getLogger(__name__)


class FederatedCNNBase(nn.Module, ModelInterface):
    """Base class for federated learning CNN models."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "base_cnn"
    
    def get_model_weights(self) -> ModelWeights:
        """Get model weights as a dictionary."""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_model_weights(self, weights: ModelWeights) -> None:
        """Set model weights from a dictionary."""
        state_dict = self.state_dict()
        for name, weight in weights.items():
            if name in state_dict:
                state_dict[name].copy_(weight)
            else:
                logger.warning(f"Weight {name} not found in model state dict")
    
    def get_parameter_count(self) -> int:
        """Get total number of model parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return param_size + buffer_size
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'name': self.model_name,
            'parameters': self.get_parameter_count(),
            'memory_bytes': self.estimate_memory_usage(),
            'layers': len(list(self.named_modules())),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class SimpleCNN(FederatedCNNBase):
    """Simple CNN for MNIST dataset."""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super().__init__()
        self.model_name = "simple_cnn"
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # For 28x28 input -> 7x7 after 2 pooling
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CIFAR10CNN(FederatedCNNBase):
    """CNN architecture optimized for CIFAR-10 dataset."""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.model_name = "cifar10_cnn"
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Second conv block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Third conv block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # For 32x32 input -> 4x4 after 3 pooling
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Second conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Third conv block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResNetBlock(nn.Module):
    """Basic ResNet block for deeper architectures."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FederatedResNet(FederatedCNNBase):
    """Lightweight ResNet for federated learning."""
    
    def __init__(self, num_classes: int = 10, num_blocks: List[int] = [2, 2, 2], 
                 input_channels: int = 3):
        super().__init__()
        self.model_name = "federated_resnet"
        self.num_classes = num_classes
        self.in_channels = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        
        # Global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple ResNet blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.fc(x)
        
        return x


class MobileNetBlock(nn.Module):
    """Depthwise separable convolution block for MobileNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class LightweightMobileNet(FederatedCNNBase):
    """Lightweight MobileNet for resource-constrained federated clients."""
    
    def __init__(self, num_classes: int = 10, width_multiplier: float = 1.0, 
                 input_channels: int = 3):
        super().__init__()
        self.model_name = "lightweight_mobilenet"
        self.num_classes = num_classes
        
        # Calculate channel numbers based on width multiplier
        def make_divisible(v: int, divisor: int = 8) -> int:
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # Initial conv layer
        input_channel = make_divisible(32 * width_multiplier)
        self.conv1 = nn.Conv2d(input_channels, input_channel, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        # MobileNet blocks configuration: (out_channels, stride)
        block_config = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
        ]
        
        # Build MobileNet blocks
        layers = []
        for out_channels, stride in block_config:
            out_channels = make_divisible(out_channels * width_multiplier)
            layers.append(MobileNetBlock(input_channel, out_channels, stride))
            input_channel = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(input_channel, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # MobileNet blocks
        x = self.features(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class ModelFactory:
    """Factory class for creating CNN models."""
    
    AVAILABLE_MODELS = {
        'simple_cnn': SimpleCNN,
        'cifar10_cnn': CIFAR10CNN,
        'federated_resnet': FederatedResNet,
        'lightweight_mobilenet': LightweightMobileNet
    }
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> FederatedCNNBase:
        """
        Create a model instance by name.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific parameters
            
        Returns:
            FederatedCNNBase: Model instance
        """
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.AVAILABLE_MODELS.keys())}")
        
        model_class = cls.AVAILABLE_MODELS[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def get_model_for_dataset(cls, dataset: str, **kwargs) -> FederatedCNNBase:
        """
        Get recommended model for a specific dataset.
        
        Args:
            dataset: Dataset name ('mnist', 'cifar10', etc.)
            **kwargs: Model-specific parameters
            
        Returns:
            FederatedCNNBase: Recommended model instance
        """
        dataset = dataset.lower()
        
        if dataset == 'mnist':
            return cls.create_model('simple_cnn', num_classes=10, **kwargs)
        elif dataset == 'cifar10':
            return cls.create_model('cifar10_cnn', num_classes=10, **kwargs)
        elif dataset == 'cifar100':
            return cls.create_model('federated_resnet', num_classes=100, **kwargs)
        else:
            logger.warning(f"Unknown dataset {dataset}, using simple CNN")
            return cls.create_model('simple_cnn', **kwargs)
    
    @classmethod
    def get_lightweight_model(cls, num_classes: int = 10, **kwargs) -> FederatedCNNBase:
        """
        Get a lightweight model for resource-constrained clients.
        
        Args:
            num_classes: Number of output classes
            **kwargs: Model-specific parameters
            
        Returns:
            FederatedCNNBase: Lightweight model instance
        """
        return cls.create_model('lightweight_mobilenet', 
                               num_classes=num_classes, 
                               width_multiplier=0.5, 
                               **kwargs)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls.AVAILABLE_MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model without instantiating it.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict: Model information
        """
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create a temporary instance to get info
        model = cls.create_model(model_name)
        info = model.get_model_info()
        del model  # Clean up
        
        return info


def benchmark_models(input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32), 
                    num_classes: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark all available models.
    
    Args:
        input_shape: Input tensor shape (batch, channels, height, width)
        num_classes: Number of output classes
        
    Returns:
        Dict: Benchmark results for each model
    """
    results = {}
    factory = ModelFactory()
    
    for model_name in factory.list_available_models():
        try:
            # Create model
            model = factory.create_model(model_name, num_classes=num_classes)
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Measure inference time
            import time
            start_time = time.time()
            with torch.no_grad():
                output = model(dummy_input)
            inference_time = time.time() - start_time
            
            # Get model info
            info = model.get_model_info()
            info['inference_time'] = inference_time
            info['output_shape'] = output.shape
            
            results[model_name] = info
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    return results


def validate_model_compatibility(model1: FederatedCNNBase, 
                                model2: FederatedCNNBase) -> bool:
    """
    Validate that two models are compatible for federated learning.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        bool: True if compatible
    """
    try:
        # Check same architecture
        if type(model1) != type(model2):
            return False
        
        # Check same parameter structure
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        
        if set(params1.keys()) != set(params2.keys()):
            return False
        
        # Check same parameter shapes
        for name in params1.keys():
            if params1[name].shape != params2[name].shape:
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Model compatibility check failed: {str(e)}")
        return False