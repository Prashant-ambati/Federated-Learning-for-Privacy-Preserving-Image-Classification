"""
Data loading and preprocessing for federated learning.
Handles MNIST, CIFAR-10 datasets with federated partitioning.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
import random
from collections import defaultdict

from .interfaces import DataLoaderInterface

logger = logging.getLogger(__name__)


class FederatedDataset(Dataset):
    """Base class for federated datasets."""
    
    def __init__(self, 
                 base_dataset: Dataset,
                 client_id: str,
                 indices: List[int]):
        """
        Initialize federated dataset.
        
        Args:
            base_dataset: Base PyTorch dataset
            client_id: Client identifier
            indices: Data indices assigned to this client
        """
        self.base_dataset = base_dataset
        self.client_id = client_id
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        # Count classes
        class_counts = defaultdict(int)
        for idx in self.indices:
            _, label = self.base_dataset[idx]
            class_counts[int(label)] += 1
        
        return {
            'client_id': self.client_id,
            'total_samples': len(self.indices),
            'class_distribution': dict(class_counts),
            'num_classes': len(class_counts)
        }


class DataPartitioner:
    """Partitions datasets for federated learning."""
    
    def __init__(self, 
                 dataset: Dataset,
                 num_clients: int,
                 partition_strategy: str = "iid",
                 alpha: float = 0.5,
                 min_samples_per_client: int = 10):
        """
        Initialize data partitioner.
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients
            partition_strategy: Partitioning strategy ("iid", "non_iid", "pathological")
            alpha: Dirichlet distribution parameter for non-IID partitioning
            min_samples_per_client: Minimum samples per client
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        
        # Extract labels
        self.labels = self._extract_labels()
        self.num_classes = len(set(self.labels))
        
        # Create partitions
        self.client_indices = self._create_partitions()
        
        logger.info(f"Created {partition_strategy} partitions for {num_clients} clients")
    
    def _extract_labels(self) -> List[int]:
        """Extract labels from dataset."""
        labels = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            labels.append(int(label))
        return labels
    
    def _create_partitions(self) -> Dict[int, List[int]]:
        """Create data partitions based on strategy."""
        if self.partition_strategy == "iid":
            return self._create_iid_partitions()
        elif self.partition_strategy == "non_iid":
            return self._create_non_iid_partitions()
        elif self.partition_strategy == "pathological":
            return self._create_pathological_partitions()
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")
    
    def _create_iid_partitions(self) -> Dict[int, List[int]]:
        """Create IID (independent and identically distributed) partitions."""
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        
        # Split indices evenly
        samples_per_client = len(indices) // self.num_clients
        client_indices = {}
        
        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            if client_id == self.num_clients - 1:
                # Last client gets remaining samples
                end_idx = len(indices)
            else:
                end_idx = start_idx + samples_per_client
            
            client_indices[client_id] = indices[start_idx:end_idx]
        
        return client_indices
    
    def _create_non_iid_partitions(self) -> Dict[int, List[int]]:
        """Create non-IID partitions using Dirichlet distribution."""
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        
        client_indices = defaultdict(list)
        
        # For each class, distribute samples using Dirichlet distribution
        for class_label, indices in class_indices.items():
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Ensure minimum samples per client
            min_proportion = self.min_samples_per_client / len(indices)
            proportions = np.maximum(proportions, min_proportion)
            proportions = proportions / proportions.sum()  # Renormalize
            
            # Distribute indices
            np.random.shuffle(indices)
            start_idx = 0
            
            for client_id in range(self.num_clients):
                num_samples = int(proportions[client_id] * len(indices))
                if client_id == self.num_clients - 1:
                    # Last client gets remaining samples
                    end_idx = len(indices)
                else:
                    end_idx = start_idx + num_samples
                
                client_indices[client_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Shuffle each client's indices
        for client_id in client_indices:
            random.shuffle(client_indices[client_id])
        
        return dict(client_indices)
    
    def _create_pathological_partitions(self) -> Dict[int, List[int]]:
        """Create pathological non-IID partitions (each client has only 1-2 classes)."""
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        
        client_indices = defaultdict(list)
        classes_per_client = max(1, self.num_classes // self.num_clients)
        
        # Assign classes to clients
        class_assignments = {}
        class_list = list(class_indices.keys())
        random.shuffle(class_list)
        
        for client_id in range(self.num_clients):
            start_class = (client_id * classes_per_client) % self.num_classes
            assigned_classes = []
            
            for i in range(classes_per_client):
                class_idx = (start_class + i) % self.num_classes
                assigned_classes.append(class_list[class_idx])
            
            class_assignments[client_id] = assigned_classes
        
        # Distribute samples within assigned classes
        for client_id, assigned_classes in class_assignments.items():
            for class_label in assigned_classes:
                indices = class_indices[class_label].copy()
                random.shuffle(indices)
                
                # Give each client a portion of the class
                samples_per_client = len(indices) // sum(
                    1 for cid, classes in class_assignments.items()
                    if class_label in classes
                )
                
                client_indices[client_id].extend(indices[:samples_per_client])
        
        # Ensure minimum samples per client
        for client_id in range(self.num_clients):
            if len(client_indices[client_id]) < self.min_samples_per_client:
                # Add random samples to meet minimum
                all_indices = set(range(len(self.dataset)))
                used_indices = set()
                for indices in client_indices.values():
                    used_indices.update(indices)
                
                available_indices = list(all_indices - used_indices)
                needed_samples = self.min_samples_per_client - len(client_indices[client_id])
                
                if available_indices:
                    additional_samples = random.sample(
                        available_indices, 
                        min(needed_samples, len(available_indices))
                    )
                    client_indices[client_id].extend(additional_samples)
        
        return dict(client_indices)
    
    def get_client_dataset(self, client_id: int) -> FederatedDataset:
        """Get dataset for a specific client."""
        if client_id not in self.client_indices:
            raise ValueError(f"Client {client_id} not found")
        
        return FederatedDataset(
            self.dataset,
            str(client_id),
            self.client_indices[client_id]
        )
    
    def get_partition_statistics(self) -> Dict[str, Any]:
        """Get statistics about the partitioning."""
        stats = {
            'num_clients': self.num_clients,
            'partition_strategy': self.partition_strategy,
            'total_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'client_statistics': {}
        }
        
        for client_id, indices in self.client_indices.items():
            client_dataset = FederatedDataset(self.dataset, str(client_id), indices)
            stats['client_statistics'][client_id] = client_dataset.get_statistics()
        
        return stats


class MNISTDataLoader(DataLoaderInterface):
    """Data loader for MNIST dataset."""
    
    def __init__(self, 
                 data_dir: str = "./data",
                 num_clients: int = 10,
                 partition_strategy: str = "iid",
                 batch_size: int = 32,
                 validation_split: float = 0.1,
                 download: bool = True):
        """
        Initialize MNIST data loader.
        
        Args:
            data_dir: Directory to store data
            num_clients: Number of federated clients
            partition_strategy: Data partitioning strategy
            batch_size: Batch size for data loaders
            validation_split: Fraction of data for validation
            download: Whether to download data if not present
        """
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load datasets
        self.train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=download,
            transform=self.train_transform
        )
        
        self.test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=self.test_transform
        )
        
        # Create federated partitions
        self.partitioner = DataPartitioner(
            self.train_dataset,
            num_clients,
            partition_strategy
        )
        
        logger.info(f"MNIST data loader initialized with {num_clients} clients")
    
    def load_training_data(self, client_id: str) -> DataLoader:
        """Load training data for a specific client."""
        try:
            client_idx = int(client_id.split('-')[-1]) if '-' in client_id else int(client_id)
            
            if client_idx >= self.num_clients:
                raise ValueError(f"Client ID {client_idx} exceeds number of clients {self.num_clients}")
            
            # Get client dataset
            client_dataset = self.partitioner.get_client_dataset(client_idx)
            
            # Split into train and validation
            if self.validation_split > 0:
                dataset_size = len(client_dataset)
                val_size = int(dataset_size * self.validation_split)
                train_size = dataset_size - val_size
                
                train_dataset, _ = torch.utils.data.random_split(
                    client_dataset, [train_size, val_size]
                )
            else:
                train_dataset = client_dataset
            
            # Create data loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            logger.info(f"Loaded training data for client {client_id}: {len(train_dataset)} samples")
            return train_loader
            
        except Exception as e:
            logger.error(f"Failed to load training data for client {client_id}: {e}")
            raise
    
    def load_validation_data(self, client_id: Optional[str] = None) -> DataLoader:
        """Load validation data."""
        try:
            if client_id:
                # Client-specific validation data
                client_idx = int(client_id.split('-')[-1]) if '-' in client_id else int(client_id)
                client_dataset = self.partitioner.get_client_dataset(client_idx)
                
                if self.validation_split > 0:
                    dataset_size = len(client_dataset)
                    val_size = int(dataset_size * self.validation_split)
                    train_size = dataset_size - val_size
                    
                    _, val_dataset = torch.utils.data.random_split(
                        client_dataset, [train_size, val_size]
                    )
                else:
                    # Use a small portion for validation
                    val_size = min(100, len(client_dataset) // 10)
                    val_dataset = Subset(client_dataset, list(range(val_size)))
            else:
                # Global validation data (test set)
                val_dataset = self.test_dataset
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            logger.debug(f"Loaded validation data: {len(val_dataset)} samples")
            return val_loader
            
        except Exception as e:
            logger.error(f"Failed to load validation data: {e}")
            raise
    
    def get_data_statistics(self, client_id: str) -> Dict[str, Any]:
        """Get statistics about client's data distribution."""
        try:
            client_idx = int(client_id.split('-')[-1]) if '-' in client_id else int(client_id)
            client_dataset = self.partitioner.get_client_dataset(client_idx)
            
            return client_dataset.get_statistics()
            
        except Exception as e:
            logger.error(f"Failed to get data statistics for client {client_id}: {e}")
            return {}


class CIFAR10DataLoader(DataLoaderInterface):
    """Data loader for CIFAR-10 dataset."""
    
    def __init__(self, 
                 data_dir: str = "./data",
                 num_clients: int = 10,
                 partition_strategy: str = "iid",
                 batch_size: int = 32,
                 validation_split: float = 0.1,
                 download: bool = True):
        """
        Initialize CIFAR-10 data loader.
        
        Args:
            data_dir: Directory to store data
            num_clients: Number of federated clients
            partition_strategy: Data partitioning strategy
            batch_size: Batch size for data loaders
            validation_split: Fraction of data for validation
            download: Whether to download data if not present
        """
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.partition_strategy = partition_strategy
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Define transforms with data augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=self.train_transform
        )
        
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=self.test_transform
        )
        
        # Create federated partitions
        self.partitioner = DataPartitioner(
            self.train_dataset,
            num_clients,
            partition_strategy
        )
        
        logger.info(f"CIFAR-10 data loader initialized with {num_clients} clients")
    
    def load_training_data(self, client_id: str) -> DataLoader:
        """Load training data for a specific client."""
        try:
            client_idx = int(client_id.split('-')[-1]) if '-' in client_id else int(client_id)
            
            if client_idx >= self.num_clients:
                raise ValueError(f"Client ID {client_idx} exceeds number of clients {self.num_clients}")
            
            # Get client dataset
            client_dataset = self.partitioner.get_client_dataset(client_idx)
            
            # Split into train and validation
            if self.validation_split > 0:
                dataset_size = len(client_dataset)
                val_size = int(dataset_size * self.validation_split)
                train_size = dataset_size - val_size
                
                train_dataset, _ = torch.utils.data.random_split(
                    client_dataset, [train_size, val_size]
                )
            else:
                train_dataset = client_dataset
            
            # Create data loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            logger.info(f"Loaded CIFAR-10 training data for client {client_id}: {len(train_dataset)} samples")
            return train_loader
            
        except Exception as e:
            logger.error(f"Failed to load CIFAR-10 training data for client {client_id}: {e}")
            raise
    
    def load_validation_data(self, client_id: Optional[str] = None) -> DataLoader:
        """Load validation data."""
        try:
            if client_id:
                # Client-specific validation data
                client_idx = int(client_id.split('-')[-1]) if '-' in client_id else int(client_id)
                client_dataset = self.partitioner.get_client_dataset(client_idx)
                
                if self.validation_split > 0:
                    dataset_size = len(client_dataset)
                    val_size = int(dataset_size * self.validation_split)
                    train_size = dataset_size - val_size
                    
                    _, val_dataset = torch.utils.data.random_split(
                        client_dataset, [train_size, val_size]
                    )
                else:
                    # Use a small portion for validation
                    val_size = min(100, len(client_dataset) // 10)
                    val_dataset = Subset(client_dataset, list(range(val_size)))
            else:
                # Global validation data (test set)
                val_dataset = self.test_dataset
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            logger.debug(f"Loaded CIFAR-10 validation data: {len(val_dataset)} samples")
            return val_loader
            
        except Exception as e:
            logger.error(f"Failed to load CIFAR-10 validation data: {e}")
            raise
    
    def get_data_statistics(self, client_id: str) -> Dict[str, Any]:
        """Get statistics about client's data distribution."""
        try:
            client_idx = int(client_id.split('-')[-1]) if '-' in client_id else int(client_id)
            client_dataset = self.partitioner.get_client_dataset(client_idx)
            
            return client_dataset.get_statistics()
            
        except Exception as e:
            logger.error(f"Failed to get CIFAR-10 data statistics for client {client_id}: {e}")
            return {}


def create_data_loader(dataset_name: str, 
                      data_dir: str = "./data",
                      num_clients: int = 10,
                      partition_strategy: str = "iid",
                      batch_size: int = 32,
                      validation_split: float = 0.1,
                      download: bool = True) -> DataLoaderInterface:
    """
    Factory function to create data loader.
    
    Args:
        dataset_name: Name of dataset ("mnist", "cifar10")
        data_dir: Directory to store data
        num_clients: Number of federated clients
        partition_strategy: Data partitioning strategy
        batch_size: Batch size for data loaders
        validation_split: Fraction of data for validation
        download: Whether to download data if not present
        
    Returns:
        DataLoaderInterface: Configured data loader
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        return MNISTDataLoader(
            data_dir=data_dir,
            num_clients=num_clients,
            partition_strategy=partition_strategy,
            batch_size=batch_size,
            validation_split=validation_split,
            download=download
        )
    elif dataset_name == "cifar10":
        return CIFAR10DataLoader(
            data_dir=data_dir,
            num_clients=num_clients,
            partition_strategy=partition_strategy,
            batch_size=batch_size,
            validation_split=validation_split,
            download=download
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def analyze_data_distribution(data_loader: DataLoaderInterface, 
                             client_ids: List[str]) -> Dict[str, Any]:
    """
    Analyze data distribution across clients.
    
    Args:
        data_loader: Data loader instance
        client_ids: List of client IDs to analyze
        
    Returns:
        Dict: Analysis results
    """
    analysis = {
        'clients': {},
        'overall_statistics': {
            'total_clients': len(client_ids),
            'class_distribution': defaultdict(int),
            'samples_per_client': []
        }
    }
    
    for client_id in client_ids:
        try:
            stats = data_loader.get_data_statistics(client_id)
            analysis['clients'][client_id] = stats
            
            # Update overall statistics
            analysis['overall_statistics']['samples_per_client'].append(stats['total_samples'])
            
            for class_label, count in stats['class_distribution'].items():
                analysis['overall_statistics']['class_distribution'][class_label] += count
                
        except Exception as e:
            logger.error(f"Failed to analyze client {client_id}: {e}")
            continue
    
    # Calculate additional statistics
    samples_per_client = analysis['overall_statistics']['samples_per_client']
    if samples_per_client:
        analysis['overall_statistics']['avg_samples_per_client'] = np.mean(samples_per_client)
        analysis['overall_statistics']['std_samples_per_client'] = np.std(samples_per_client)
        analysis['overall_statistics']['min_samples_per_client'] = min(samples_per_client)
        analysis['overall_statistics']['max_samples_per_client'] = max(samples_per_client)
    
    return analysis