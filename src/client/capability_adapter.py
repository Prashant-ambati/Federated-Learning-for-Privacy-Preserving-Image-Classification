"""
Client capability adaptation for federated learning.
Dynamically adjusts training parameters based on client computational resources.
"""

import torch
import psutil
import time
import threading
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from ..shared.models import ClientCapabilities, ComputePowerLevel
from ..shared.training import FederatedTrainingConfig

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    memory_available: int  # Bytes
    gpu_usage: Optional[float] = None  # Percentage
    gpu_memory_usage: Optional[float] = None  # Percentage
    network_bandwidth: Optional[float] = None  # Mbps
    disk_usage: Optional[float] = None  # Percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'memory_available': self.memory_available,
            'gpu_usage': self.gpu_usage,
            'gpu_memory_usage': self.gpu_memory_usage,
            'network_bandwidth': self.network_bandwidth,
            'disk_usage': self.disk_usage,
            'timestamp': datetime.now().isoformat()
        }


class ResourceMonitor:
    """Monitors system resources in real-time."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        """
        Initialize resource monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
        """
        self.monitoring_interval = monitoring_interval
        
        # Resource tracking
        self.current_metrics: Optional[ResourceMetrics] = None
        self.metrics_history = []
        self.max_history = 100
        
        # GPU availability
        self.has_gpu = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
        
        # Monitoring control
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        logger.info(f"Resource monitor initialized - GPU available: {self.has_gpu}")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        try:
            with self.lock:
                if self.running:
                    return
                
                self.running = True
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self.monitor_thread.start()
                
                logger.info("Resource monitoring started")
                
        except Exception as e:
            logger.error(f"Failed to start resource monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        try:
            with self.lock:
                if not self.running:
                    return
                
                self.running = False
                
                if self.monitor_thread and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=5.0)
                
                logger.info("Resource monitoring stopped")
                
        except Exception as e:
            logger.error(f"Error stopping resource monitoring: {e}")
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current resource metrics."""
        with self.lock:
            return self.current_metrics
    
    def get_average_metrics(self, window_minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over time window."""
        with self.lock:
            if not self.metrics_history:
                return self.current_metrics
            
            # Filter metrics within time window
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            if not recent_metrics:
                return self.current_metrics
            
            # Calculate averages
            avg_cpu = sum(m['cpu_usage'] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m['memory_usage'] for m in recent_metrics) / len(recent_metrics)
            avg_memory_available = sum(m['memory_available'] for m in recent_metrics) / len(recent_metrics)
            
            avg_gpu = None
            avg_gpu_memory = None
            if any(m['gpu_usage'] is not None for m in recent_metrics):
                gpu_metrics = [m['gpu_usage'] for m in recent_metrics if m['gpu_usage'] is not None]
                avg_gpu = sum(gpu_metrics) / len(gpu_metrics) if gpu_metrics else None
                
                gpu_mem_metrics = [m['gpu_memory_usage'] for m in recent_metrics if m['gpu_memory_usage'] is not None]
                avg_gpu_memory = sum(gpu_mem_metrics) / len(gpu_mem_metrics) if gpu_mem_metrics else None
            
            return ResourceMetrics(
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                memory_available=int(avg_memory_available),
                gpu_usage=avg_gpu,
                gpu_memory_usage=avg_gpu_memory
            )
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                with self.lock:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics.to_dict())
                    
                    # Limit history size
                    if len(self.metrics_history) > self.max_history:
                        self.metrics_history = self.metrics_history[-self.max_history:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # CPU and memory metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_usage = None
        gpu_memory_usage = None
        
        if self.has_gpu:
            try:
                # Get GPU utilization
                gpu_usage = self._get_gpu_utilization()
                gpu_memory_usage = self._get_gpu_memory_usage()
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {e}")
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            memory_available=memory.available,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage
        )
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        if not self.has_gpu:
            return None
        
        try:
            # Use nvidia-ml-py if available, otherwise estimate
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except ImportError:
            # Fallback: estimate based on memory usage
            return self._estimate_gpu_utilization()
        except Exception:
            return None
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage percentage."""
        if not self.has_gpu:
            return None
        
        try:
            memory_info = torch.cuda.memory_stats()
            allocated = memory_info.get('allocated_bytes.all.current', 0)
            reserved = memory_info.get('reserved_bytes.all.current', 0)
            
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            usage_percent = (allocated / total_memory) * 100
            return float(usage_percent)
        except Exception:
            return None
    
    def _estimate_gpu_utilization(self) -> Optional[float]:
        """Estimate GPU utilization based on memory usage."""
        memory_usage = self._get_gpu_memory_usage()
        if memory_usage is not None:
            # Rough estimation: assume utilization correlates with memory usage
            return min(memory_usage * 1.2, 100.0)
        return None


class CapabilityAdapter:
    """Adapts training parameters based on client capabilities and resources."""
    
    def __init__(self, 
                 base_capabilities: ClientCapabilities,
                 resource_monitor: Optional[ResourceMonitor] = None):
        """
        Initialize capability adapter.
        
        Args:
            base_capabilities: Base client capabilities
            resource_monitor: Resource monitoring service
        """
        self.base_capabilities = base_capabilities
        self.resource_monitor = resource_monitor or ResourceMonitor()
        
        # Adaptation history
        self.adaptation_history = []
        self.current_config: Optional[FederatedTrainingConfig] = None
        
        # Thresholds for adaptation
        self.cpu_threshold_high = 80.0  # %
        self.cpu_threshold_low = 30.0   # %
        self.memory_threshold_high = 85.0  # %
        self.memory_threshold_low = 50.0   # %
        self.gpu_threshold_high = 90.0  # %
        
        logger.info("Capability adapter initialized")
    
    def start(self):
        """Start capability adaptation."""
        self.resource_monitor.start_monitoring()
        logger.info("Capability adapter started")
    
    def stop(self):
        """Stop capability adaptation."""
        self.resource_monitor.stop_monitoring()
        logger.info("Capability adapter stopped")
    
    def adapt_training_config(self, 
                            base_config: FederatedTrainingConfig,
                            round_requirements: Optional[Dict[str, Any]] = None) -> FederatedTrainingConfig:
        """
        Adapt training configuration based on current resources.
        
        Args:
            base_config: Base training configuration
            round_requirements: Round-specific requirements
            
        Returns:
            FederatedTrainingConfig: Adapted configuration
        """
        try:
            # Get current resource metrics
            current_metrics = self.resource_monitor.get_current_metrics()
            avg_metrics = self.resource_monitor.get_average_metrics()
            
            if not current_metrics:
                logger.warning("No resource metrics available, using base config")
                return base_config
            
            # Create adapted config
            adapted_config = FederatedTrainingConfig(
                local_epochs=base_config.local_epochs,
                batch_size=base_config.batch_size,
                learning_rate=base_config.learning_rate,
                optimizer_type=base_config.optimizer_type,
                early_stopping_patience=base_config.early_stopping_patience,
                save_checkpoints=base_config.save_checkpoints,
                validation_split=base_config.validation_split
            )
            
            # Adapt based on CPU usage
            adapted_config = self._adapt_for_cpu(adapted_config, current_metrics)
            
            # Adapt based on memory usage
            adapted_config = self._adapt_for_memory(adapted_config, current_metrics)
            
            # Adapt based on GPU usage (if available)
            if current_metrics.gpu_usage is not None:
                adapted_config = self._adapt_for_gpu(adapted_config, current_metrics)
            
            # Apply round requirements
            if round_requirements:
                adapted_config = self._apply_round_requirements(adapted_config, round_requirements)
            
            # Record adaptation
            self._record_adaptation(base_config, adapted_config, current_metrics)
            
            self.current_config = adapted_config
            return adapted_config
            
        except Exception as e:
            logger.error(f"Failed to adapt training config: {e}")
            return base_config
    
    def _adapt_for_cpu(self, 
                      config: FederatedTrainingConfig, 
                      metrics: ResourceMetrics) -> FederatedTrainingConfig:
        """Adapt configuration based on CPU usage."""
        if metrics.cpu_usage > self.cpu_threshold_high:
            # High CPU usage - reduce computational load
            config.batch_size = max(8, config.batch_size // 2)
            config.local_epochs = max(1, config.local_epochs - 1)
            logger.debug(f"Reduced load due to high CPU usage: {metrics.cpu_usage:.1f}%")
            
        elif metrics.cpu_usage < self.cpu_threshold_low:
            # Low CPU usage - can increase load
            if self.base_capabilities.compute_power == ComputePowerLevel.HIGH:
                config.batch_size = min(128, config.batch_size * 2)
                config.local_epochs = min(10, config.local_epochs + 1)
                logger.debug(f"Increased load due to low CPU usage: {metrics.cpu_usage:.1f}%")
        
        return config
    
    def _adapt_for_memory(self, 
                         config: FederatedTrainingConfig, 
                         metrics: ResourceMetrics) -> FederatedTrainingConfig:
        """Adapt configuration based on memory usage."""
        if metrics.memory_usage > self.memory_threshold_high:
            # High memory usage - reduce batch size
            config.batch_size = max(4, config.batch_size // 2)
            logger.debug(f"Reduced batch size due to high memory usage: {metrics.memory_usage:.1f}%")
            
        elif metrics.memory_usage < self.memory_threshold_low:
            # Low memory usage - can increase batch size
            available_gb = metrics.memory_available / (1024**3)
            if available_gb > 2.0:  # At least 2GB available
                config.batch_size = min(256, int(config.batch_size * 1.5))
                logger.debug(f"Increased batch size due to low memory usage: {metrics.memory_usage:.1f}%")
        
        return config
    
    def _adapt_for_gpu(self, 
                      config: FederatedTrainingConfig, 
                      metrics: ResourceMetrics) -> FederatedTrainingConfig:
        """Adapt configuration based on GPU usage."""
        if metrics.gpu_usage and metrics.gpu_usage > self.gpu_threshold_high:
            # High GPU usage - reduce load
            config.batch_size = max(8, config.batch_size // 2)
            logger.debug(f"Reduced batch size due to high GPU usage: {metrics.gpu_usage:.1f}%")
            
        elif metrics.gpu_memory_usage and metrics.gpu_memory_usage > 80.0:
            # High GPU memory usage - reduce batch size
            config.batch_size = max(4, config.batch_size // 2)
            logger.debug(f"Reduced batch size due to high GPU memory: {metrics.gpu_memory_usage:.1f}%")
        
        return config
    
    def _apply_round_requirements(self, 
                                config: FederatedTrainingConfig,
                                requirements: Dict[str, Any]) -> FederatedTrainingConfig:
        """Apply round-specific requirements."""
        # Apply minimum/maximum constraints
        if 'min_batch_size' in requirements:
            config.batch_size = max(requirements['min_batch_size'], config.batch_size)
        
        if 'max_batch_size' in requirements:
            config.batch_size = min(requirements['max_batch_size'], config.batch_size)
        
        if 'min_epochs' in requirements:
            config.local_epochs = max(requirements['min_epochs'], config.local_epochs)
        
        if 'max_epochs' in requirements:
            config.local_epochs = min(requirements['max_epochs'], config.local_epochs)
        
        # Apply specific learning rate if required
        if 'learning_rate' in requirements:
            config.learning_rate = requirements['learning_rate']
        
        return config
    
    def _record_adaptation(self, 
                          base_config: FederatedTrainingConfig,
                          adapted_config: FederatedTrainingConfig,
                          metrics: ResourceMetrics):
        """Record adaptation for analysis."""
        adaptation_record = {
            'timestamp': datetime.now().isoformat(),
            'base_config': base_config.to_dict(),
            'adapted_config': adapted_config.to_dict(),
            'resource_metrics': metrics.to_dict(),
            'adaptations': self._calculate_adaptations(base_config, adapted_config)
        }
        
        self.adaptation_history.append(adaptation_record)
        
        # Limit history size
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
    
    def _calculate_adaptations(self, 
                             base_config: FederatedTrainingConfig,
                             adapted_config: FederatedTrainingConfig) -> Dict[str, Any]:
        """Calculate what adaptations were made."""
        adaptations = {}
        
        if base_config.batch_size != adapted_config.batch_size:
            adaptations['batch_size'] = {
                'from': base_config.batch_size,
                'to': adapted_config.batch_size,
                'change': adapted_config.batch_size - base_config.batch_size
            }
        
        if base_config.local_epochs != adapted_config.local_epochs:
            adaptations['local_epochs'] = {
                'from': base_config.local_epochs,
                'to': adapted_config.local_epochs,
                'change': adapted_config.local_epochs - base_config.local_epochs
            }
        
        if base_config.learning_rate != adapted_config.learning_rate:
            adaptations['learning_rate'] = {
                'from': base_config.learning_rate,
                'to': adapted_config.learning_rate,
                'change': adapted_config.learning_rate - base_config.learning_rate
            }
        
        return adaptations
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptations made."""
        if not self.adaptation_history:
            return {'message': 'No adaptations recorded'}
        
        recent_adaptations = self.adaptation_history[-10:]  # Last 10 adaptations
        
        # Count adaptation types
        adaptation_counts = {
            'batch_size_changes': 0,
            'epoch_changes': 0,
            'learning_rate_changes': 0
        }
        
        for record in recent_adaptations:
            adaptations = record['adaptations']
            if 'batch_size' in adaptations:
                adaptation_counts['batch_size_changes'] += 1
            if 'local_epochs' in adaptations:
                adaptation_counts['epoch_changes'] += 1
            if 'learning_rate' in adaptations:
                adaptation_counts['learning_rate_changes'] += 1
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'recent_adaptations': len(recent_adaptations),
            'adaptation_counts': adaptation_counts,
            'current_config': self.current_config.to_dict() if self.current_config else None,
            'current_metrics': self.resource_monitor.get_current_metrics().to_dict() if self.resource_monitor.get_current_metrics() else None
        }
    
    def update_capabilities(self) -> ClientCapabilities:
        """Update client capabilities based on current performance."""
        try:
            # Get average metrics over longer period
            avg_metrics = self.resource_monitor.get_average_metrics(window_minutes=30)
            
            if not avg_metrics:
                return self.base_capabilities
            
            # Determine compute power level based on performance
            new_compute_power = self._assess_compute_power(avg_metrics)
            
            # Estimate available samples (this would typically be known)
            available_samples = self.base_capabilities.available_samples
            
            # Update network bandwidth estimate (placeholder)
            network_bandwidth = self.base_capabilities.network_bandwidth
            
            # Create updated capabilities
            updated_capabilities = ClientCapabilities(
                compute_power=new_compute_power,
                network_bandwidth=network_bandwidth,
                available_samples=available_samples,
                supported_models=self.base_capabilities.supported_models,
                privacy_requirements=self.base_capabilities.privacy_requirements
            )
            
            logger.info(f"Updated capabilities: compute_power={new_compute_power.value}")
            return updated_capabilities
            
        except Exception as e:
            logger.error(f"Failed to update capabilities: {e}")
            return self.base_capabilities
    
    def _assess_compute_power(self, metrics: ResourceMetrics) -> ComputePowerLevel:
        """Assess compute power level based on resource metrics."""
        # Simple heuristic based on CPU usage and available memory
        if metrics.cpu_usage < 50 and metrics.memory_usage < 60:
            if metrics.gpu_usage is not None and metrics.gpu_usage < 50:
                return ComputePowerLevel.HIGH
            elif metrics.gpu_usage is None:
                return ComputePowerLevel.MEDIUM
        
        if metrics.cpu_usage < 70 and metrics.memory_usage < 80:
            return ComputePowerLevel.MEDIUM
        
        return ComputePowerLevel.LOW