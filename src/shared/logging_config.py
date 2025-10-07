"""
Centralized logging configuration for federated learning system.
Provides structured logging with different levels and output formats.
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'client_id'):
            log_entry['client_id'] = record.client_id
        if hasattr(record, 'round_number'):
            log_entry['round_number'] = record.round_number
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class FederatedLearningFilter(logging.Filter):
    """Custom filter to add federated learning context to log records."""
    
    def __init__(self, component: str):
        """Initialize filter with component name."""
        super().__init__()
        self.component = component
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add component information to log record."""
        record.component = self.component
        return True


def setup_logging(
    component: str,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_json: bool = True,
    enable_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration for federated learning components.
    
    Args:
        component: Component name (coordinator, client, aggregator)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (optional)
        enable_json: Whether to use JSON formatting
        enable_console: Whether to log to console
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(f"federated_learning.{component}")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add component filter
    component_filter = FederatedLearningFilter(component)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if enable_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(component)s] - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        
        console_handler.addFilter(component_filter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        log_file = log_dir_path / f"{component}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        
        if enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(component)s] - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
        
        file_handler.addFilter(component_filter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_dir_path / f"{component}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter() if enable_json else file_formatter)
        error_handler.addFilter(component_filter)
        logger.addHandler(error_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for specific module."""
    return logging.getLogger(name)


def log_federated_event(
    logger: logging.Logger,
    level: str,
    message: str,
    client_id: Optional[str] = None,
    round_number: Optional[int] = None,
    request_id: Optional[str] = None,
    **kwargs
):
    """
    Log federated learning specific events with context.
    
    Args:
        logger: Logger instance
        level: Log level (info, warning, error, debug)
        message: Log message
        client_id: Client identifier (optional)
        round_number: Training round number (optional)
        request_id: Request identifier (optional)
        **kwargs: Additional context fields
    """
    # Create log record with extra context
    extra = {}
    if client_id:
        extra['client_id'] = client_id
    if round_number is not None:
        extra['round_number'] = round_number
    if request_id:
        extra['request_id'] = request_id
    
    # Add any additional context
    extra.update(kwargs)
    
    # Log with appropriate level
    log_method = getattr(logger, level.lower())
    log_method(message, extra=extra)


class MetricsLogger:
    """Logger specifically for metrics and performance data."""
    
    def __init__(self, component: str, log_dir: Optional[str] = None):
        """Initialize metrics logger."""
        self.component = component
        self.logger = logging.getLogger(f"federated_learning.{component}.metrics")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # JSON formatter for metrics
        formatter = JSONFormatter()
        
        # Console handler for metrics
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for metrics
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            
            metrics_file = log_dir_path / f"{component}_metrics.log"
            file_handler = logging.handlers.RotatingFileHandler(
                metrics_file,
                maxBytes=50 * 1024 * 1024,  # 50MB for metrics
                backupCount=10
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.propagate = False
    
    def log_training_metrics(
        self,
        round_number: int,
        client_id: Optional[str] = None,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        training_time: Optional[float] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """Log training metrics."""
        metrics = {
            'metric_type': 'training',
            'round_number': round_number,
            'timestamp': datetime.now().isoformat()
        }
        
        if client_id:
            metrics['client_id'] = client_id
        if accuracy is not None:
            metrics['accuracy'] = accuracy
        if loss is not None:
            metrics['loss'] = loss
        if training_time is not None:
            metrics['training_time_seconds'] = training_time
        if num_samples is not None:
            metrics['num_samples'] = num_samples
        
        metrics.update(kwargs)
        
        self.logger.info("Training metrics", extra=metrics)
    
    def log_system_metrics(
        self,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        network_io: Optional[Dict[str, float]] = None,
        disk_io: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Log system performance metrics."""
        metrics = {
            'metric_type': 'system',
            'timestamp': datetime.now().isoformat()
        }
        
        if cpu_usage is not None:
            metrics['cpu_usage_percent'] = cpu_usage
        if memory_usage is not None:
            metrics['memory_usage_percent'] = memory_usage
        if network_io:
            metrics['network_io'] = network_io
        if disk_io:
            metrics['disk_io'] = disk_io
        
        metrics.update(kwargs)
        
        self.logger.info("System metrics", extra=metrics)
    
    def log_aggregation_metrics(
        self,
        round_number: int,
        num_clients: int,
        total_samples: int,
        aggregation_time: float,
        convergence_score: Optional[float] = None,
        **kwargs
    ):
        """Log aggregation metrics."""
        metrics = {
            'metric_type': 'aggregation',
            'round_number': round_number,
            'num_clients': num_clients,
            'total_samples': total_samples,
            'aggregation_time_seconds': aggregation_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if convergence_score is not None:
            metrics['convergence_score'] = convergence_score
        
        metrics.update(kwargs)
        
        self.logger.info("Aggregation metrics", extra=metrics)
    
    def log_communication_metrics(
        self,
        client_id: str,
        message_type: str,
        message_size_bytes: int,
        latency_ms: float,
        success: bool = True,
        **kwargs
    ):
        """Log communication metrics."""
        metrics = {
            'metric_type': 'communication',
            'client_id': client_id,
            'message_type': message_type,
            'message_size_bytes': message_size_bytes,
            'latency_ms': latency_ms,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        metrics.update(kwargs)
        
        self.logger.info("Communication metrics", extra=metrics)


class AuditLogger:
    """Logger for security and audit events."""
    
    def __init__(self, component: str, log_dir: Optional[str] = None):
        """Initialize audit logger."""
        self.component = component
        self.logger = logging.getLogger(f"federated_learning.{component}.audit")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # JSON formatter for audit logs
        formatter = JSONFormatter()
        
        # File handler for audit logs (always enabled)
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            
            audit_file = log_dir_path / f"{component}_audit.log"
            file_handler = logging.handlers.RotatingFileHandler(
                audit_file,
                maxBytes=100 * 1024 * 1024,  # 100MB for audit logs
                backupCount=20
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler for critical audit events
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.propagate = False
    
    def log_client_registration(self, client_id: str, success: bool, reason: str = ""):
        """Log client registration event."""
        self.logger.info("Client registration", extra={
            'event_type': 'client_registration',
            'client_id': client_id,
            'success': success,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_model_update(self, client_id: str, round_number: int, success: bool, validation_result: str = ""):
        """Log model update event."""
        self.logger.info("Model update", extra={
            'event_type': 'model_update',
            'client_id': client_id,
            'round_number': round_number,
            'success': success,
            'validation_result': validation_result,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_privacy_event(self, event_type: str, client_id: str, privacy_params: Dict[str, Any]):
        """Log privacy-related event."""
        self.logger.info("Privacy event", extra={
            'event_type': f'privacy_{event_type}',
            'client_id': client_id,
            'privacy_params': privacy_params,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security event."""
        log_level = logging.WARNING if severity == 'high' else logging.INFO
        self.logger.log(log_level, "Security event", extra={
            'event_type': f'security_{event_type}',
            'severity': severity,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })


def configure_logging_from_config(config: Dict[str, Any], component: str) -> logging.Logger:
    """
    Configure logging from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        component: Component name
        
    Returns:
        Configured logger
    """
    logging_config = config.get('logging', {})
    
    return setup_logging(
        component=component,
        log_level=logging_config.get('level', 'INFO'),
        log_dir=logging_config.get('log_dir'),
        enable_json=logging_config.get('enable_json', True),
        enable_console=logging_config.get('enable_console', True),
        max_file_size=logging_config.get('max_file_size', 10 * 1024 * 1024),
        backup_count=logging_config.get('backup_count', 5)
    )


# Example usage and testing
if __name__ == "__main__":
    # Test logging configuration
    logger = setup_logging("test_component", log_level="DEBUG", log_dir="./logs")
    
    # Test basic logging
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Test federated learning specific logging
    log_federated_event(
        logger, "info", "Client registered successfully",
        client_id="client_001", round_number=1
    )
    
    # Test metrics logging
    metrics_logger = MetricsLogger("test_component", log_dir="./logs")
    metrics_logger.log_training_metrics(
        round_number=1,
        client_id="client_001",
        accuracy=0.95,
        loss=0.05,
        training_time=120.5,
        num_samples=1000
    )
    
    # Test audit logging
    audit_logger = AuditLogger("test_component", log_dir="./logs")
    audit_logger.log_client_registration("client_001", True, "Valid credentials")
    
    print("Logging test completed. Check ./logs directory for output files.")