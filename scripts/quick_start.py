#!/usr/bin/env python3
"""
Quick start script for the Federated Learning System.
Demonstrates basic functionality and validates the installation.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all core modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test shared modules
        from shared.models import ModelUpdate, GlobalModel, PrivacyConfig
        from shared.models_pytorch import ModelFactory
        from shared.privacy import create_privacy_engine
        from shared.compression import create_compression_service
        from shared.training import LocalTrainer
        
        # Test aggregation
        from aggregation.fedavg import FedAvgAggregator
        
        logger.info("✓ All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test CNN model creation."""
    logger.info("Testing model creation...")
    
    try:
        from shared.models_pytorch import ModelFactory
        
        # Test different models
        models_to_test = ['simple_cnn', 'cifar10_cnn', 'lightweight_mobilenet']
        
        for model_name in models_to_test:
            model = ModelFactory.create_model(model_name, num_classes=10)
            info = model.get_model_info()
            logger.info(f"✓ {model_name}: {info['parameters']} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False


def test_privacy_engine():
    """Test differential privacy engine."""
    logger.info("Testing privacy engine...")
    
    try:
        from shared.privacy import create_privacy_engine
        import torch
        
        # Create privacy engine
        privacy_engine = create_privacy_engine(epsilon=1.0, delta=1e-5)
        
        # Test noise addition
        dummy_gradients = {
            'layer1': torch.randn(100),
            'layer2': torch.randn(50)
        }
        
        noisy_gradients = privacy_engine.add_noise(dummy_gradients, 0.5, 1e-6)
        
        logger.info("✓ Privacy engine working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Privacy engine failed: {e}")
        return False


def test_compression():
    """Test model compression."""
    logger.info("Testing compression...")
    
    try:
        from shared.compression import create_compression_service
        import torch
        
        # Create compression service
        compression_service = create_compression_service("lz4")
        
        # Test compression
        dummy_weights = {
            'layer1': torch.randn(1000),
            'layer2': torch.randn(500)
        }
        
        compressed = compression_service.compress_weights(dummy_weights)
        decompressed = compression_service.decompress_weights(compressed)
        
        # Check if decompression worked
        assert len(decompressed) == len(dummy_weights)
        
        logger.info("✓ Compression working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Compression failed: {e}")
        return False


def test_fedavg():
    """Test FedAvg aggregation."""
    logger.info("Testing FedAvg aggregation...")
    
    try:
        from aggregation.fedavg import FedAvgAggregator
        from shared.models import ModelUpdate
        from datetime import datetime
        import torch
        
        # Create aggregator
        aggregator = FedAvgAggregator(min_clients=2, validate_updates=False)
        
        # Create dummy updates
        updates = []
        for i in range(3):
            dummy_weights = {
                'layer1': torch.randn(100),
                'layer2': torch.randn(50)
            }
            
            update = ModelUpdate(
                client_id=f"client_{i}",
                round_number=1,
                model_weights=dummy_weights,
                num_samples=100 + i * 50,
                training_loss=0.5 - i * 0.1,
                privacy_budget_used=0.1,
                compression_ratio=0.8,
                timestamp=datetime.now()
            )
            updates.append(update)
        
        # Test aggregation
        global_model = aggregator.aggregate_updates(updates)
        
        logger.info(f"✓ FedAvg aggregation: {len(global_model.participating_clients)} clients")
        return True
        
    except Exception as e:
        logger.error(f"✗ FedAvg failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting Federated Learning System Quick Start")
    logger.info("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_privacy_engine,
        test_compression,
        test_fedavg
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
    
    logger.info("=" * 60)
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Start infrastructure: docker-compose up -d postgres redis")
        logger.info("2. Run coordinator: python -m src.coordinator.main")
        logger.info("3. Run client: python -m src.client.main")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the installation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())