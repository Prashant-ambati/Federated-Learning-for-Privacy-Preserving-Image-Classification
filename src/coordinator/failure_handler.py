"""
Client failure handling for federated learning coordinator.
Manages client disconnections, timeouts, and recovery strategies.
"""

import threading
import time
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of client failures."""
    TIMEOUT = "timeout"
    DISCONNECTION = "disconnection"
    INVALID_UPDATE = "invalid_update"
    CAPABILITY_MISMATCH = "capability_mismatch"
    PRIVACY_VIOLATION = "privacy_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class FailureAction(Enum):
    """Actions to take on client failure."""
    RETRY = "retry"
    EXCLUDE_ROUND = "exclude_round"
    EXCLUDE_TEMPORARY = "exclude_temporary"
    EXCLUDE_PERMANENT = "exclude_permanent"
    REDUCE_LOAD = "reduce_load"
    NONE = "none"


class ClientFailure:
    """Represents a client failure event."""
    
    def __init__(self, 
                 client_id: str,
                 failure_type: FailureType,
                 round_number: int,
                 timestamp: datetime,
                 details: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize client failure.
        
        Args:
            client_id: Client identifier
            failure_type: Type of failure
            round_number: Round number when failure occurred
            timestamp: Failure timestamp
            details: Failure details
            context: Additional context information
        """
        self.client_id = client_id
        self.failure_type = failure_type
        self.round_number = round_number
        self.timestamp = timestamp
        self.details = details or ""
        self.context = context or {}
        
        # Recovery tracking
        self.recovery_attempts = 0
        self.recovered = False
        self.recovery_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert failure to dictionary."""
        return {
            'client_id': self.client_id,
            'failure_type': self.failure_type.value,
            'round_number': self.round_number,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'context': self.context,
            'recovery_attempts': self.recovery_attempts,
            'recovered': self.recovered,
            'recovery_time': self.recovery_time.isoformat() if self.recovery_time else None
        }


class ClientHealthTracker:
    """Tracks client health and reliability metrics."""
    
    def __init__(self, client_id: str):
        """
        Initialize client health tracker.
        
        Args:
            client_id: Client identifier
        """
        self.client_id = client_id
        
        # Health metrics
        self.total_rounds = 0
        self.successful_rounds = 0
        self.failed_rounds = 0
        self.timeout_count = 0
        self.disconnection_count = 0
        
        # Performance metrics
        self.avg_response_time = 0.0
        self.last_seen = datetime.now()
        self.registration_time = datetime.now()
        
        # Failure history
        self.failure_history = deque(maxlen=50)  # Keep last 50 failures
        self.recent_failures = deque(maxlen=10)  # Last 10 failures for pattern detection
        
        # Health score (0.0 to 1.0)
        self.health_score = 1.0
        
        # Status
        self.is_active = True
        self.is_excluded = False
        self.exclusion_until = None
    
    def record_success(self, response_time: float = 0.0):
        """Record successful round participation."""
        self.total_rounds += 1
        self.successful_rounds += 1
        self.last_seen = datetime.now()
        
        # Update average response time
        if response_time > 0:
            if self.avg_response_time == 0:
                self.avg_response_time = response_time
            else:
                self.avg_response_time = (self.avg_response_time * 0.9) + (response_time * 0.1)
        
        # Update health score
        self._update_health_score()
    
    def record_failure(self, failure: ClientFailure):
        """Record client failure."""
        self.total_rounds += 1
        self.failed_rounds += 1
        self.last_seen = datetime.now()
        
        # Update specific failure counters
        if failure.failure_type == FailureType.TIMEOUT:
            self.timeout_count += 1
        elif failure.failure_type == FailureType.DISCONNECTION:
            self.disconnection_count += 1
        
        # Add to failure history
        self.failure_history.append(failure)
        self.recent_failures.append(failure)
        
        # Update health score
        self._update_health_score()
    
    def get_reliability_score(self) -> float:
        """Calculate client reliability score (0.0 to 1.0)."""
        if self.total_rounds == 0:
            return 1.0
        
        return self.successful_rounds / self.total_rounds
    
    def get_recent_failure_rate(self, window_minutes: int = 60) -> float:
        """Get failure rate in recent time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_failures = [
            f for f in self.recent_failures
            if f.timestamp > cutoff_time
        ]
        
        if not recent_failures:
            return 0.0
        
        # Calculate failure rate per hour
        window_hours = window_minutes / 60.0
        return len(recent_failures) / window_hours
    
    def detect_failure_patterns(self) -> List[str]:
        """Detect patterns in client failures."""
        patterns = []
        
        if len(self.recent_failures) < 3:
            return patterns
        
        # Check for repeated failure types
        failure_types = [f.failure_type for f in self.recent_failures]
        type_counts = defaultdict(int)
        for ft in failure_types:
            type_counts[ft] += 1
        
        for failure_type, count in type_counts.items():
            if count >= 3:
                patterns.append(f"repeated_{failure_type.value}")
        
        # Check for rapid failures
        if len(self.recent_failures) >= 5:
            recent_5 = list(self.recent_failures)[-5:]
            time_span = recent_5[-1].timestamp - recent_5[0].timestamp
            if time_span < timedelta(minutes=10):
                patterns.append("rapid_failures")
        
        # Check for timeout pattern
        timeout_failures = [f for f in self.recent_failures if f.failure_type == FailureType.TIMEOUT]
        if len(timeout_failures) >= 3:
            patterns.append("frequent_timeouts")
        
        return patterns
    
    def _update_health_score(self):
        """Update overall health score."""
        # Base score from reliability
        reliability = self.get_reliability_score()
        
        # Penalty for recent failures
        recent_failure_rate = self.get_recent_failure_rate(30)  # Last 30 minutes
        failure_penalty = min(0.5, recent_failure_rate * 0.1)
        
        # Penalty for specific failure types
        type_penalty = 0.0
        if self.timeout_count > 5:
            type_penalty += 0.1
        if self.disconnection_count > 3:
            type_penalty += 0.1
        
        # Calculate final score
        self.health_score = max(0.0, reliability - failure_penalty - type_penalty)
    
    def is_healthy(self, min_score: float = 0.7) -> bool:
        """Check if client is considered healthy."""
        return self.health_score >= min_score and self.is_active and not self.is_excluded
    
    def get_summary(self) -> Dict[str, Any]:
        """Get client health summary."""
        return {
            'client_id': self.client_id,
            'health_score': self.health_score,
            'reliability_score': self.get_reliability_score(),
            'total_rounds': self.total_rounds,
            'successful_rounds': self.successful_rounds,
            'failed_rounds': self.failed_rounds,
            'timeout_count': self.timeout_count,
            'disconnection_count': self.disconnection_count,
            'recent_failure_rate': self.get_recent_failure_rate(),
            'failure_patterns': self.detect_failure_patterns(),
            'avg_response_time': self.avg_response_time,
            'last_seen': self.last_seen.isoformat(),
            'is_active': self.is_active,
            'is_excluded': self.is_excluded,
            'exclusion_until': self.exclusion_until.isoformat() if self.exclusion_until else None
        }


class FailureHandler:
    """Handles client failures and implements recovery strategies."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 timeout_threshold: int = 300,
                 exclusion_duration: int = 3600):
        """
        Initialize failure handler.
        
        Args:
            max_retries: Maximum retry attempts per failure
            timeout_threshold: Timeout threshold in seconds
            exclusion_duration: Temporary exclusion duration in seconds
        """
        self.max_retries = max_retries
        self.timeout_threshold = timeout_threshold
        self.exclusion_duration = exclusion_duration
        
        # Client tracking
        self.client_health: Dict[str, ClientHealthTracker] = {}
        self.excluded_clients: Set[str] = set()
        self.failure_history = deque(maxlen=1000)
        
        # Failure policies
        self.failure_policies = self._create_default_policies()
        
        # Callbacks
        self.on_client_failed: Optional[Callable] = None
        self.on_client_recovered: Optional[Callable] = None
        self.on_client_excluded: Optional[Callable] = None
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Background monitoring
        self.running = False
        self.monitor_thread = None
        
        logger.info("Failure handler initialized")
    
    def start(self):
        """Start failure monitoring."""
        try:
            with self.lock:
                if self.running:
                    return
                
                self.running = True
                
                # Start monitoring thread
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self.monitor_thread.start()
                
                logger.info("Failure handler started")
                
        except Exception as e:
            logger.error(f"Failed to start failure handler: {e}")
            raise
    
    def stop(self):
        """Stop failure monitoring."""
        try:
            with self.lock:
                if not self.running:
                    return
                
                logger.info("Stopping failure handler")
                self.running = False
                
                # Wait for monitoring thread
                if self.monitor_thread and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=5.0)
                
                logger.info("Failure handler stopped")
                
        except Exception as e:
            logger.error(f"Error stopping failure handler: {e}")
    
    def register_client(self, client_id: str):
        """Register client for health tracking."""
        with self.lock:
            if client_id not in self.client_health:
                self.client_health[client_id] = ClientHealthTracker(client_id)
                logger.info(f"Registered client {client_id} for health tracking")
    
    def unregister_client(self, client_id: str):
        """Unregister client from health tracking."""
        with self.lock:
            self.client_health.pop(client_id, None)
            self.excluded_clients.discard(client_id)
            logger.info(f"Unregistered client {client_id} from health tracking")
    
    def handle_failure(self, 
                      client_id: str,
                      failure_type: FailureType,
                      round_number: int,
                      details: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> FailureAction:
        """
        Handle client failure and determine action.
        
        Args:
            client_id: Client identifier
            failure_type: Type of failure
            round_number: Round number
            details: Failure details
            context: Additional context
            
        Returns:
            FailureAction: Action to take
        """
        try:
            with self.lock:
                # Ensure client is registered
                if client_id not in self.client_health:
                    self.register_client(client_id)
                
                # Create failure record
                failure = ClientFailure(
                    client_id=client_id,
                    failure_type=failure_type,
                    round_number=round_number,
                    timestamp=datetime.now(),
                    details=details,
                    context=context
                )
                
                # Record failure
                self.client_health[client_id].record_failure(failure)
                self.failure_history.append(failure)
                
                # Determine action based on policy
                action = self._determine_action(client_id, failure)
                
                # Execute action
                self._execute_action(client_id, action, failure)
                
                logger.warning(f"Client {client_id} failure: {failure_type.value} -> {action.value}")
                
                # Trigger callback
                if self.on_client_failed:
                    self.on_client_failed(client_id, failure, action)
                
                return action
                
        except Exception as e:
            logger.error(f"Failed to handle client failure: {e}")
            return FailureAction.NONE
    
    def handle_timeout(self, client_id: str, round_number: int) -> FailureAction:
        """Handle client timeout."""
        return self.handle_failure(
            client_id=client_id,
            failure_type=FailureType.TIMEOUT,
            round_number=round_number,
            details=f"Client timed out after {self.timeout_threshold} seconds"
        )
    
    def handle_disconnection(self, client_id: str, round_number: int) -> FailureAction:
        """Handle client disconnection."""
        return self.handle_failure(
            client_id=client_id,
            failure_type=FailureType.DISCONNECTION,
            round_number=round_number,
            details="Client disconnected unexpectedly"
        )
    
    def handle_invalid_update(self, client_id: str, round_number: int, error: str) -> FailureAction:
        """Handle invalid model update."""
        return self.handle_failure(
            client_id=client_id,
            failure_type=FailureType.INVALID_UPDATE,
            round_number=round_number,
            details=f"Invalid model update: {error}"
        )
    
    def record_success(self, client_id: str, response_time: float = 0.0):
        """Record successful client interaction."""
        with self.lock:
            if client_id not in self.client_health:
                self.register_client(client_id)
            
            self.client_health[client_id].record_success(response_time)
            
            # Check if client was excluded and can be recovered
            if client_id in self.excluded_clients:
                health_tracker = self.client_health[client_id]
                if health_tracker.is_healthy():
                    self._recover_client(client_id)
    
    def is_client_healthy(self, client_id: str) -> bool:
        """Check if client is healthy and available."""
        with self.lock:
            if client_id not in self.client_health:
                return True  # New clients are considered healthy
            
            return self.client_health[client_id].is_healthy()
    
    def is_client_excluded(self, client_id: str) -> bool:
        """Check if client is currently excluded."""
        with self.lock:
            return client_id in self.excluded_clients
    
    def get_healthy_clients(self, client_list: List[str]) -> List[str]:
        """Filter list to only healthy clients."""
        with self.lock:
            healthy_clients = []
            
            for client_id in client_list:
                if self.is_client_healthy(client_id) and not self.is_client_excluded(client_id):
                    healthy_clients.append(client_id)
            
            return healthy_clients
    
    def get_client_health_summary(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get client health summary."""
        with self.lock:
            if client_id not in self.client_health:
                return None
            
            return self.client_health[client_id].get_summary()
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get overall failure statistics."""
        with self.lock:
            total_failures = len(self.failure_history)
            
            if total_failures == 0:
                return {
                    'total_failures': 0,
                    'failure_types': {},
                    'excluded_clients': len(self.excluded_clients),
                    'healthy_clients': len([c for c in self.client_health.values() if c.is_healthy()])
                }
            
            # Count failure types
            failure_types = defaultdict(int)
            for failure in self.failure_history:
                failure_types[failure.failure_type.value] += 1
            
            # Recent failures (last hour)
            recent_cutoff = datetime.now() - timedelta(hours=1)
            recent_failures = [f for f in self.failure_history if f.timestamp > recent_cutoff]
            
            return {
                'total_failures': total_failures,
                'recent_failures': len(recent_failures),
                'failure_types': dict(failure_types),
                'excluded_clients': len(self.excluded_clients),
                'healthy_clients': len([c for c in self.client_health.values() if c.is_healthy()]),
                'total_tracked_clients': len(self.client_health)
            }
    
    def _determine_action(self, client_id: str, failure: ClientFailure) -> FailureAction:
        """Determine action to take for a failure."""
        health_tracker = self.client_health[client_id]
        
        # Get failure policy
        policy = self.failure_policies.get(failure.failure_type, {})
        
        # Check failure patterns
        patterns = health_tracker.detect_failure_patterns()
        
        # Determine action based on failure type and patterns
        if failure.failure_type == FailureType.TIMEOUT:
            if health_tracker.timeout_count >= 5:
                return FailureAction.EXCLUDE_TEMPORARY
            elif "frequent_timeouts" in patterns:
                return FailureAction.EXCLUDE_ROUND
            else:
                return FailureAction.RETRY
        
        elif failure.failure_type == FailureType.DISCONNECTION:
            if health_tracker.disconnection_count >= 3:
                return FailureAction.EXCLUDE_TEMPORARY
            elif "rapid_failures" in patterns:
                return FailureAction.EXCLUDE_ROUND
            else:
                return FailureAction.RETRY
        
        elif failure.failure_type == FailureType.INVALID_UPDATE:
            return FailureAction.EXCLUDE_ROUND
        
        elif failure.failure_type == FailureType.PRIVACY_VIOLATION:
            return FailureAction.EXCLUDE_PERMANENT
        
        elif failure.failure_type == FailureType.RESOURCE_EXHAUSTION:
            return FailureAction.REDUCE_LOAD
        
        else:
            # Default action based on health score
            if health_tracker.health_score < 0.3:
                return FailureAction.EXCLUDE_TEMPORARY
            elif health_tracker.health_score < 0.7:
                return FailureAction.EXCLUDE_ROUND
            else:
                return FailureAction.RETRY
    
    def _execute_action(self, client_id: str, action: FailureAction, failure: ClientFailure):
        """Execute the determined action."""
        health_tracker = self.client_health[client_id]
        
        if action == FailureAction.EXCLUDE_ROUND:
            # Exclude from current round only
            health_tracker.is_excluded = True
            health_tracker.exclusion_until = datetime.now() + timedelta(minutes=30)
            self.excluded_clients.add(client_id)
            
        elif action == FailureAction.EXCLUDE_TEMPORARY:
            # Temporary exclusion
            health_tracker.is_excluded = True
            health_tracker.exclusion_until = datetime.now() + timedelta(seconds=self.exclusion_duration)
            self.excluded_clients.add(client_id)
            
        elif action == FailureAction.EXCLUDE_PERMANENT:
            # Permanent exclusion
            health_tracker.is_excluded = True
            health_tracker.is_active = False
            health_tracker.exclusion_until = None
            self.excluded_clients.add(client_id)
            
        elif action == FailureAction.REDUCE_LOAD:
            # Reduce client load (could adjust training parameters)
            pass  # Implementation depends on specific requirements
        
        # Trigger callback for exclusions
        if action in [FailureAction.EXCLUDE_ROUND, FailureAction.EXCLUDE_TEMPORARY, FailureAction.EXCLUDE_PERMANENT]:
            if self.on_client_excluded:
                self.on_client_excluded(client_id, action, failure)
    
    def _recover_client(self, client_id: str):
        """Recover excluded client."""
        health_tracker = self.client_health[client_id]
        
        health_tracker.is_excluded = False
        health_tracker.exclusion_until = None
        self.excluded_clients.discard(client_id)
        
        logger.info(f"Recovered client {client_id}")
        
        # Trigger callback
        if self.on_client_recovered:
            self.on_client_recovered(client_id)
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                time.sleep(60)  # Check every minute
                
                with self.lock:
                    current_time = datetime.now()
                    
                    # Check for clients to recover from temporary exclusion
                    for client_id in list(self.excluded_clients):
                        if client_id in self.client_health:
                            health_tracker = self.client_health[client_id]
                            
                            if (health_tracker.exclusion_until and 
                                current_time > health_tracker.exclusion_until):
                                
                                self._recover_client(client_id)
                
            except Exception as e:
                logger.error(f"Error in failure monitoring loop: {e}")
    
    def _create_default_policies(self) -> Dict[FailureType, Dict[str, Any]]:
        """Create default failure handling policies."""
        return {
            FailureType.TIMEOUT: {
                'max_retries': 3,
                'exclusion_threshold': 5,
                'exclusion_duration': 1800  # 30 minutes
            },
            FailureType.DISCONNECTION: {
                'max_retries': 2,
                'exclusion_threshold': 3,
                'exclusion_duration': 3600  # 1 hour
            },
            FailureType.INVALID_UPDATE: {
                'max_retries': 0,
                'exclusion_threshold': 1,
                'exclusion_duration': 1800  # 30 minutes
            },
            FailureType.PRIVACY_VIOLATION: {
                'max_retries': 0,
                'exclusion_threshold': 1,
                'exclusion_duration': None  # Permanent
            }
        }
    
    def set_callbacks(self,
                     on_client_failed: Optional[Callable] = None,
                     on_client_recovered: Optional[Callable] = None,
                     on_client_excluded: Optional[Callable] = None):
        """Set callback functions for failure events."""
        self.on_client_failed = on_client_failed
        self.on_client_recovered = on_client_recovered
        self.on_client_excluded = on_client_excluded
        
        logger.info("Failure handler callbacks configured")