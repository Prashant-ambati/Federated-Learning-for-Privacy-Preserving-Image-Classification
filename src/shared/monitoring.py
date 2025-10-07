"""
Monitoring and metrics collection for federated learning system.
Provides system metrics, performance monitoring, and alerting capabilities.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_io_bytes_sent: int
    network_io_bytes_recv: int
    active_connections: int = 0
    process_count: int = 0


@dataclass
class TrainingMetrics:
    """Training-specific metrics."""
    timestamp: datetime
    round_number: int
    client_id: Optional[str] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    training_time_seconds: Optional[float] = None
    num_samples: Optional[int] = None
    convergence_score: Optional[float] = None
    privacy_budget_used: Optional[float] = None


@dataclass
class CommunicationMetrics:
    """Communication performance metrics."""
    timestamp: datetime
    client_id: str
    message_type: str
    message_size_bytes: int
    latency_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class Alert:
    """System alert definition."""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    component: str
    metric_name: str
    metric_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """Collects and stores various system and application metrics."""
    
    def __init__(self, collection_interval: float = 30.0, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Interval between metric collections in seconds
            max_history: Maximum number of metric points to keep in memory
        """
        self.collection_interval = collection_interval
        self.max_history = max_history
        
        # Metric storage
        self.system_metrics: deque = deque(maxlen=max_history)
        self.training_metrics: deque = deque(maxlen=max_history)
        self.communication_metrics: deque = deque(maxlen=max_history)
        
        # Collection state
        self.collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # Baseline metrics for comparison
        self.baseline_metrics: Dict[str, float] = {}
        
        logger.info(f"MetricsCollector initialized with {collection_interval}s interval")
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if self.collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while self.collecting:
            try:
                self.collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
                time.sleep(self.collection_interval)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process information
            process_count = len(psutil.pids())
            
            # Network connections (approximate)
            try:
                connections = psutil.net_connections()
                active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                active_connections = 0
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                network_io_bytes_sent=network.bytes_sent,
                network_io_bytes_recv=network.bytes_recv,
                active_connections=active_connections,
                process_count=process_count
            )\n            \n            with self.lock:\n                self.system_metrics.append(metrics)\n            \n            return metrics\n            \n        except Exception as e:\n            logger.error(f\"Failed to collect system metrics: {str(e)}\")\n            return None\n    \n    def record_training_metrics(self, metrics: TrainingMetrics):\n        \"\"\"Record training metrics.\"\"\"\n        with self.lock:\n            self.training_metrics.append(metrics)\n        logger.debug(f\"Recorded training metrics for round {metrics.round_number}\")\n    \n    def record_communication_metrics(self, metrics: CommunicationMetrics):\n        \"\"\"Record communication metrics.\"\"\"\n        with self.lock:\n            self.communication_metrics.append(metrics)\n        logger.debug(f\"Recorded communication metrics for {metrics.client_id}\")\n    \n    def get_system_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:\n        \"\"\"Get summary of system metrics over specified duration.\"\"\"\n        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)\n        \n        with self.lock:\n            recent_metrics = [\n                m for m in self.system_metrics \n                if m.timestamp >= cutoff_time\n            ]\n        \n        if not recent_metrics:\n            return {'message': 'No metrics available for specified duration'}\n        \n        # Calculate statistics\n        cpu_values = [m.cpu_usage_percent for m in recent_metrics]\n        memory_values = [m.memory_usage_percent for m in recent_metrics]\n        \n        return {\n            'duration_minutes': duration_minutes,\n            'sample_count': len(recent_metrics),\n            'cpu_usage': {\n                'current': cpu_values[-1] if cpu_values else 0,\n                'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,\n                'max': max(cpu_values) if cpu_values else 0,\n                'min': min(cpu_values) if cpu_values else 0\n            },\n            'memory_usage': {\n                'current': memory_values[-1] if memory_values else 0,\n                'average': sum(memory_values) / len(memory_values) if memory_values else 0,\n                'max': max(memory_values) if memory_values else 0,\n                'min': min(memory_values) if memory_values else 0\n            },\n            'network_connections': recent_metrics[-1].active_connections if recent_metrics else 0,\n            'process_count': recent_metrics[-1].process_count if recent_metrics else 0\n        }\n    \n    def get_training_metrics_summary(self, round_count: int = 10) -> Dict[str, Any]:\n        \"\"\"Get summary of recent training metrics.\"\"\"\n        with self.lock:\n            recent_metrics = list(self.training_metrics)[-round_count:]\n        \n        if not recent_metrics:\n            return {'message': 'No training metrics available'}\n        \n        # Group by round\n        rounds_data = defaultdict(list)\n        for metric in recent_metrics:\n            rounds_data[metric.round_number].append(metric)\n        \n        summary = {\n            'total_rounds': len(rounds_data),\n            'total_samples': len(recent_metrics),\n            'rounds': {}\n        }\n        \n        for round_num, round_metrics in rounds_data.items():\n            accuracies = [m.accuracy for m in round_metrics if m.accuracy is not None]\n            losses = [m.loss for m in round_metrics if m.loss is not None]\n            \n            summary['rounds'][round_num] = {\n                'client_count': len(round_metrics),\n                'average_accuracy': sum(accuracies) / len(accuracies) if accuracies else None,\n                'average_loss': sum(losses) / len(losses) if losses else None,\n                'total_samples': sum(m.num_samples for m in round_metrics if m.num_samples),\n                'total_training_time': sum(m.training_time_seconds for m in round_metrics if m.training_time_seconds)\n            }\n        \n        return summary\n    \n    def get_communication_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:\n        \"\"\"Get summary of communication metrics.\"\"\"\n        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)\n        \n        with self.lock:\n            recent_metrics = [\n                m for m in self.communication_metrics \n                if m.timestamp >= cutoff_time\n            ]\n        \n        if not recent_metrics:\n            return {'message': 'No communication metrics available'}\n        \n        # Calculate statistics\n        successful_requests = [m for m in recent_metrics if m.success]\n        failed_requests = [m for m in recent_metrics if not m.success]\n        \n        latencies = [m.latency_ms for m in successful_requests]\n        message_sizes = [m.message_size_bytes for m in recent_metrics]\n        \n        # Group by client\n        client_stats = defaultdict(lambda: {'requests': 0, 'failures': 0, 'total_bytes': 0})\n        for metric in recent_metrics:\n            client_stats[metric.client_id]['requests'] += 1\n            if not metric.success:\n                client_stats[metric.client_id]['failures'] += 1\n            client_stats[metric.client_id]['total_bytes'] += metric.message_size_bytes\n        \n        return {\n            'duration_minutes': duration_minutes,\n            'total_requests': len(recent_metrics),\n            'successful_requests': len(successful_requests),\n            'failed_requests': len(failed_requests),\n            'success_rate': len(successful_requests) / len(recent_metrics) if recent_metrics else 0,\n            'average_latency_ms': sum(latencies) / len(latencies) if latencies else 0,\n            'max_latency_ms': max(latencies) if latencies else 0,\n            'total_bytes_transferred': sum(message_sizes),\n            'unique_clients': len(client_stats),\n            'client_statistics': dict(client_stats)\n        }\n\n\nclass AlertManager:\n    \"\"\"Manages alerts and notifications for system monitoring.\"\"\"\n    \n    def __init__(self):\n        \"\"\"Initialize alert manager.\"\"\"\n        self.alerts: List[Alert] = []\n        self.alert_rules: List[Dict[str, Any]] = []\n        self.alert_callbacks: List[Callable[[Alert], None]] = []\n        self.lock = threading.RLock()\n        \n        # Default alert rules\n        self._setup_default_rules()\n        \n        logger.info(\"AlertManager initialized\")\n    \n    def _setup_default_rules(self):\n        \"\"\"Set up default alerting rules.\"\"\"\n        default_rules = [\n            {\n                'name': 'high_cpu_usage',\n                'metric': 'cpu_usage_percent',\n                'threshold': 80.0,\n                'operator': '>',\n                'severity': 'medium',\n                'message': 'High CPU usage detected'\n            },\n            {\n                'name': 'high_memory_usage',\n                'metric': 'memory_usage_percent',\n                'threshold': 85.0,\n                'operator': '>',\n                'severity': 'medium',\n                'message': 'High memory usage detected'\n            },\n            {\n                'name': 'low_disk_space',\n                'metric': 'disk_usage_percent',\n                'threshold': 90.0,\n                'operator': '>',\n                'severity': 'high',\n                'message': 'Low disk space warning'\n            },\n            {\n                'name': 'training_accuracy_drop',\n                'metric': 'accuracy',\n                'threshold': 0.1,\n                'operator': '<',\n                'severity': 'medium',\n                'message': 'Training accuracy below threshold'\n            },\n            {\n                'name': 'high_communication_latency',\n                'metric': 'latency_ms',\n                'threshold': 5000.0,\n                'operator': '>',\n                'severity': 'medium',\n                'message': 'High communication latency detected'\n            }\n        ]\n        \n        self.alert_rules.extend(default_rules)\n    \n    def add_alert_rule(self, rule: Dict[str, Any]):\n        \"\"\"Add custom alert rule.\"\"\"\n        required_fields = ['name', 'metric', 'threshold', 'operator', 'severity', 'message']\n        if not all(field in rule for field in required_fields):\n            raise ValueError(f\"Alert rule must contain: {required_fields}\")\n        \n        with self.lock:\n            self.alert_rules.append(rule)\n        \n        logger.info(f\"Added alert rule: {rule['name']}\")\n    \n    def add_alert_callback(self, callback: Callable[[Alert], None]):\n        \"\"\"Add callback function to be called when alerts are triggered.\"\"\"\n        self.alert_callbacks.append(callback)\n        logger.info(\"Added alert callback\")\n    \n    def check_system_metrics(self, metrics: SystemMetrics):\n        \"\"\"Check system metrics against alert rules.\"\"\"\n        metric_values = {\n            'cpu_usage_percent': metrics.cpu_usage_percent,\n            'memory_usage_percent': metrics.memory_usage_percent,\n            'disk_usage_percent': metrics.disk_usage_percent,\n            'active_connections': metrics.active_connections\n        }\n        \n        self._check_metrics_against_rules(metric_values, 'system')\n    \n    def check_training_metrics(self, metrics: TrainingMetrics):\n        \"\"\"Check training metrics against alert rules.\"\"\"\n        metric_values = {}\n        if metrics.accuracy is not None:\n            metric_values['accuracy'] = metrics.accuracy\n        if metrics.loss is not None:\n            metric_values['loss'] = metrics.loss\n        if metrics.training_time_seconds is not None:\n            metric_values['training_time_seconds'] = metrics.training_time_seconds\n        \n        self._check_metrics_against_rules(metric_values, 'training', metrics.client_id)\n    \n    def check_communication_metrics(self, metrics: CommunicationMetrics):\n        \"\"\"Check communication metrics against alert rules.\"\"\"\n        metric_values = {\n            'latency_ms': metrics.latency_ms,\n            'message_size_bytes': metrics.message_size_bytes\n        }\n        \n        self._check_metrics_against_rules(metric_values, 'communication', metrics.client_id)\n    \n    def _check_metrics_against_rules(self, metric_values: Dict[str, float], component: str, client_id: Optional[str] = None):\n        \"\"\"Check metric values against alert rules.\"\"\"\n        for rule in self.alert_rules:\n            metric_name = rule['metric']\n            if metric_name not in metric_values:\n                continue\n            \n            metric_value = metric_values[metric_name]\n            threshold = rule['threshold']\n            operator = rule['operator']\n            \n            # Check condition\n            triggered = False\n            if operator == '>' and metric_value > threshold:\n                triggered = True\n            elif operator == '<' and metric_value < threshold:\n                triggered = True\n            elif operator == '==' and metric_value == threshold:\n                triggered = True\n            elif operator == '>=' and metric_value >= threshold:\n                triggered = True\n            elif operator == '<=' and metric_value <= threshold:\n                triggered = True\n            \n            if triggered:\n                self._trigger_alert(rule, metric_value, component, client_id)\n    \n    def _trigger_alert(self, rule: Dict[str, Any], metric_value: float, component: str, client_id: Optional[str] = None):\n        \"\"\"Trigger an alert.\"\"\"\n        alert_id = f\"{rule['name']}_{component}_{int(time.time())}\"\n        if client_id:\n            alert_id += f\"_{client_id}\"\n        \n        alert = Alert(\n            alert_id=alert_id,\n            severity=rule['severity'],\n            message=f\"{rule['message']}: {rule['metric']} = {metric_value} (threshold: {rule['threshold']})\",\n            timestamp=datetime.now(),\n            component=component,\n            metric_name=rule['metric'],\n            metric_value=metric_value,\n            threshold=rule['threshold']\n        )\n        \n        with self.lock:\n            self.alerts.append(alert)\n        \n        logger.warning(f\"Alert triggered: {alert.message}\")\n        \n        # Call alert callbacks\n        for callback in self.alert_callbacks:\n            try:\n                callback(alert)\n            except Exception as e:\n                logger.error(f\"Alert callback failed: {str(e)}\")\n    \n    def get_active_alerts(self) -> List[Alert]:\n        \"\"\"Get list of active (unresolved) alerts.\"\"\"\n        with self.lock:\n            return [alert for alert in self.alerts if not alert.resolved]\n    \n    def resolve_alert(self, alert_id: str):\n        \"\"\"Mark an alert as resolved.\"\"\"\n        with self.lock:\n            for alert in self.alerts:\n                if alert.alert_id == alert_id and not alert.resolved:\n                    alert.resolved = True\n                    alert.resolved_at = datetime.now()\n                    logger.info(f\"Alert resolved: {alert_id}\")\n                    return True\n        return False\n    \n    def get_alert_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of alerts.\"\"\"\n        with self.lock:\n            active_alerts = [a for a in self.alerts if not a.resolved]\n            resolved_alerts = [a for a in self.alerts if a.resolved]\n        \n        # Count by severity\n        severity_counts = defaultdict(int)\n        for alert in active_alerts:\n            severity_counts[alert.severity] += 1\n        \n        return {\n            'total_alerts': len(self.alerts),\n            'active_alerts': len(active_alerts),\n            'resolved_alerts': len(resolved_alerts),\n            'severity_breakdown': dict(severity_counts),\n            'recent_alerts': [\n                {\n                    'alert_id': a.alert_id,\n                    'severity': a.severity,\n                    'message': a.message,\n                    'timestamp': a.timestamp.isoformat(),\n                    'component': a.component\n                }\n                for a in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:10]\n            ]\n        }\n\n\nclass PerformanceMonitor:\n    \"\"\"Comprehensive performance monitoring system.\"\"\"\n    \n    def __init__(self, component: str, config: Dict[str, Any] = None):\n        \"\"\"Initialize performance monitor.\"\"\"\n        self.component = component\n        self.config = config or {}\n        \n        # Initialize components\n        self.metrics_collector = MetricsCollector(\n            collection_interval=self.config.get('collection_interval', 30.0),\n            max_history=self.config.get('max_history', 1000)\n        )\n        \n        self.alert_manager = AlertManager()\n        \n        # Set up alert callbacks\n        self.alert_manager.add_alert_callback(self._handle_alert)\n        \n        # Start monitoring\n        self.metrics_collector.start_collection()\n        \n        logger.info(f\"PerformanceMonitor initialized for {component}\")\n    \n    def _handle_alert(self, alert: Alert):\n        \"\"\"Handle triggered alerts.\"\"\"\n        # Log alert\n        logger.warning(f\"ALERT [{alert.severity.upper()}]: {alert.message}\")\n        \n        # Additional alert handling can be added here\n        # e.g., send notifications, trigger auto-scaling, etc.\n    \n    def record_training_metrics(self, **kwargs):\n        \"\"\"Record training metrics and check for alerts.\"\"\"\n        metrics = TrainingMetrics(\n            timestamp=datetime.now(),\n            **kwargs\n        )\n        \n        self.metrics_collector.record_training_metrics(metrics)\n        self.alert_manager.check_training_metrics(metrics)\n    \n    def record_communication_metrics(self, **kwargs):\n        \"\"\"Record communication metrics and check for alerts.\"\"\"\n        metrics = CommunicationMetrics(\n            timestamp=datetime.now(),\n            **kwargs\n        )\n        \n        self.metrics_collector.record_communication_metrics(metrics)\n        self.alert_manager.check_communication_metrics(metrics)\n    \n    def get_health_status(self) -> Dict[str, Any]:\n        \"\"\"Get overall system health status.\"\"\"\n        system_summary = self.metrics_collector.get_system_metrics_summary(duration_minutes=10)\n        alert_summary = self.alert_manager.get_alert_summary()\n        \n        # Determine overall health\n        critical_alerts = alert_summary['severity_breakdown'].get('critical', 0)\n        high_alerts = alert_summary['severity_breakdown'].get('high', 0)\n        \n        if critical_alerts > 0:\n            health_status = 'critical'\n        elif high_alerts > 0:\n            health_status = 'degraded'\n        elif alert_summary['active_alerts'] > 0:\n            health_status = 'warning'\n        else:\n            health_status = 'healthy'\n        \n        return {\n            'component': self.component,\n            'health_status': health_status,\n            'timestamp': datetime.now().isoformat(),\n            'system_metrics': system_summary,\n            'alerts': alert_summary\n        }\n    \n    def shutdown(self):\n        \"\"\"Shutdown monitoring.\"\"\"\n        self.metrics_collector.stop_collection()\n        logger.info(f\"PerformanceMonitor shutdown for {self.component}\")\n\n\n# Example usage\nif __name__ == \"__main__\":\n    # Test monitoring system\n    monitor = PerformanceMonitor(\"test_component\")\n    \n    # Record some test metrics\n    monitor.record_training_metrics(\n        round_number=1,\n        client_id=\"test_client\",\n        accuracy=0.95,\n        loss=0.05,\n        training_time_seconds=120.0,\n        num_samples=1000\n    )\n    \n    monitor.record_communication_metrics(\n        client_id=\"test_client\",\n        message_type=\"model_update\",\n        message_size_bytes=1024000,\n        latency_ms=150.0,\n        success=True\n    )\n    \n    # Get health status\n    health = monitor.get_health_status()\n    print(json.dumps(health, indent=2))\n    \n    # Shutdown\n    monitor.shutdown()