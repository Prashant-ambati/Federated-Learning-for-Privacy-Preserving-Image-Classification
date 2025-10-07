"""
Local training utilities for federated learning clients.
Implements PyTorch training loops with metrics collection and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime
import json

from .models import TrainingMetrics
from .models_pytorch import FederatedCNNBase

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Custom exception for training errors."""
    pass


class LocalTrainer:
    """Handles local model training for federated learning clients."""
    
    def __init__(self, 
                 model: FederatedCNNBase,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: Optional[str] = None):
        """
        Initialize local trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use for training (CPU/GPU)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        # Create checkpoint directory
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"LocalTrainer initialized on device: {self.device}")
    
    def train_local_model(self, 
                         train_loader: DataLoader,
                         epochs: int,
                         learning_rate: float = 0.001,
                         optimizer_type: str = 'adam',
                         loss_function: Optional[nn.Module] = None,
                         validation_loader: Optional[DataLoader] = None,
                         save_checkpoints: bool = True,
                         early_stopping_patience: Optional[int] = None) -> TrainingMetrics:
        """
        Train the local model for specified epochs.
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
            loss_function: Loss function (default: CrossEntropyLoss)
            validation_loader: Optional validation data loader
            save_checkpoints: Whether to save model checkpoints
            early_stopping_patience: Early stopping patience (None to disable)
            
        Returns:
            TrainingMetrics: Training results and metrics
        """
        try:
            start_time = time.time()
            
            # Setup optimizer and loss function
            optimizer = self._create_optimizer(optimizer_type, learning_rate)
            criterion = loss_function or nn.CrossEntropyLoss()
            
            # Training state tracking
            best_val_loss = float('inf')
            patience_counter = 0
            total_samples = 0
            epoch_losses = []
            epoch_accuracies = []
            
            logger.info(f"Starting local training for {epochs} epochs")
            
            for epoch in range(epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_loss, train_acc, samples_processed = self._train_epoch(
                    train_loader, optimizer, criterion
                )
                total_samples += samples_processed
                
                # Validation phase
                val_loss, val_acc = None, None
                if validation_loader:
                    val_loss, val_acc = self._validate_epoch(validation_loader, criterion)
                
                # Record metrics
                epoch_losses.append(train_loss)
                epoch_accuracies.append(train_acc)
                
                # Log progress
                log_msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(log_msg)
                
                # Save checkpoint
                if save_checkpoints and self.checkpoint_dir:
                    self._save_checkpoint(epoch, train_loss, val_loss)
                
                # Early stopping
                if early_stopping_patience and validation_loader:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
            
            training_time = time.time() - start_time
            
            # Calculate final metrics
            final_loss = epoch_losses[-1] if epoch_losses else 0.0
            final_accuracy = epoch_accuracies[-1] if epoch_accuracies else 0.0
            
            metrics = TrainingMetrics(
                loss=final_loss,
                accuracy=final_accuracy,
                epochs_completed=len(epoch_losses),
                training_time=training_time,
                samples_processed=total_samples
            )
            
            # Store training history
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'epochs': len(epoch_losses),
                'final_loss': final_loss,
                'final_accuracy': final_accuracy,
                'training_time': training_time,
                'samples_processed': total_samples
            })
            
            logger.info(f"Local training completed - Final Loss: {final_loss:.4f}, "
                       f"Final Accuracy: {final_accuracy:.4f}, Time: {training_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Local training failed: {str(e)}")
            raise TrainingError(f"Local training failed: {str(e)}")
    
    def _train_epoch(self, 
                    train_loader: DataLoader, 
                    optimizer: optim.Optimizer, 
                    criterion: nn.Module) -> Tuple[float, float, int]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            # Log batch progress occasionally
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy, total_samples
    
    def _validate_epoch(self, 
                       val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                # Move data to device
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples
        
        return val_loss, val_accuracy
    
    def _create_optimizer(self, optimizer_type: str, learning_rate: float) -> optim.Optimizer:
        """Create optimizer based on type."""
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat(),
            'model_info': self.model.get_model_info()
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dict: Checkpoint metadata
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_epoch = checkpoint['epoch']
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return {
                'epoch': checkpoint['epoch'],
                'train_loss': checkpoint['train_loss'],
                'val_loss': checkpoint.get('val_loss'),
                'timestamp': checkpoint.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise TrainingError(f"Failed to load checkpoint: {str(e)}")
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            self.model.eval()
            
            correct_predictions = 0
            total_samples = 0
            class_correct = {}
            class_total = {}
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(data)
                    _, predicted = torch.max(outputs, 1)
                    
                    total_samples += targets.size(0)
                    correct_predictions += (predicted == targets).sum().item()
                    
                    # Per-class accuracy
                    for i in range(targets.size(0)):
                        label = targets[i].item()
                        class_correct[label] = class_correct.get(label, 0) + (predicted[i] == targets[i]).item()
                        class_total[label] = class_total.get(label, 0) + 1
            
            overall_accuracy = correct_predictions / total_samples
            
            # Calculate per-class accuracies
            class_accuracies = {}
            for class_id in class_total.keys():
                class_accuracies[f'class_{class_id}_accuracy'] = class_correct[class_id] / class_total[class_id]
            
            metrics = {
                'overall_accuracy': overall_accuracy,
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                **class_accuracies
            }
            
            logger.info(f"Model evaluation completed - Accuracy: {overall_accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise TrainingError(f"Model evaluation failed: {str(e)}")
    
    def get_model_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Get current model gradients.
        
        Returns:
            Dict: Model gradients by parameter name
        """
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients
    
    def set_model_gradients(self, gradients: Dict[str, torch.Tensor]):
        """
        Set model gradients.
        
        Args:
            gradients: Gradients to set by parameter name
        """
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone()
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()
    
    def save_training_history(self, filepath: str):
        """Save training history to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            logger.info(f"Training history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save training history: {str(e)}")
    
    def reset_training_state(self):
        """Reset training state and history."""
        self.current_epoch = 0
        self.training_history = []
        logger.info("Training state reset")


class FederatedTrainingConfig:
    """Configuration for federated training parameters."""
    
    def __init__(self,
                 local_epochs: int = 5,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 optimizer_type: str = 'adam',
                 early_stopping_patience: Optional[int] = None,
                 save_checkpoints: bool = True,
                 validation_split: float = 0.1):
        """
        Initialize federated training configuration.
        
        Args:
            local_epochs: Number of local training epochs per round
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer
            early_stopping_patience: Early stopping patience
            save_checkpoints: Whether to save checkpoints
            validation_split: Fraction of data to use for validation
        """
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.early_stopping_patience = early_stopping_patience
        self.save_checkpoints = save_checkpoints
        self.validation_split = validation_split
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'optimizer_type': self.optimizer_type,
            'early_stopping_patience': self.early_stopping_patience,
            'save_checkpoints': self.save_checkpoints,
            'validation_split': self.validation_split
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FederatedTrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_adaptive_config(client_capabilities: Dict[str, Any]) -> FederatedTrainingConfig:
    """
    Create adaptive training configuration based on client capabilities.
    
    Args:
        client_capabilities: Client computational capabilities
        
    Returns:
        FederatedTrainingConfig: Adaptive configuration
    """
    compute_power = client_capabilities.get('compute_power', 'medium')
    network_bandwidth = client_capabilities.get('network_bandwidth', 10)  # Mbps
    available_samples = client_capabilities.get('available_samples', 1000)
    
    # Adapt parameters based on capabilities
    if compute_power == 'high':
        local_epochs = 10
        batch_size = 64
        learning_rate = 0.001
    elif compute_power == 'medium':
        local_epochs = 5
        batch_size = 32
        learning_rate = 0.001
    else:  # low
        local_epochs = 3
        batch_size = 16
        learning_rate = 0.0005
    
    # Adjust batch size based on available samples
    if available_samples < 500:
        batch_size = min(batch_size, 16)
    elif available_samples > 5000:
        batch_size = min(batch_size * 2, 128)
    
    # Adjust based on network bandwidth
    if network_bandwidth < 5:  # Low bandwidth
        local_epochs = max(local_epochs + 2, 7)  # More local training
    
    return FederatedTrainingConfig(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type='adam',
        early_stopping_patience=None,
        save_checkpoints=True,
        validation_split=0.1
    )


def validate_training_data(train_loader: DataLoader) -> Dict[str, Any]:
    """
    Validate training data loader.
    
    Args:
        train_loader: Training data loader to validate
        
    Returns:
        Dict: Validation results
    """
    try:
        # Check if data loader is not empty
        if len(train_loader) == 0:
            raise ValueError("Training data loader is empty")
        
        # Sample one batch to check data format
        data_iter = iter(train_loader)
        sample_batch = next(data_iter)
        
        if len(sample_batch) != 2:
            raise ValueError("Expected (data, targets) tuple from data loader")
        
        data, targets = sample_batch
        
        # Validate data shapes and types
        if not isinstance(data, torch.Tensor):
            raise ValueError("Data must be a torch.Tensor")
        
        if not isinstance(targets, torch.Tensor):
            raise ValueError("Targets must be a torch.Tensor")
        
        if len(data.shape) != 4:  # (batch, channels, height, width)
            raise ValueError(f"Expected 4D data tensor, got shape {data.shape}")
        
        batch_size, channels, height, width = data.shape
        num_classes = len(torch.unique(targets))
        
        validation_results = {
            'valid': True,
            'num_batches': len(train_loader),
            'batch_size': batch_size,
            'data_shape': (channels, height, width),
            'num_classes': num_classes,
            'data_type': str(data.dtype),
            'targets_type': str(targets.dtype)
        }
        
        logger.info(f"Training data validation passed: {validation_results}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Training data validation failed: {str(e)}")
        return {
            'valid': False,
            'error': str(e)
        }