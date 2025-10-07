"""
REST API endpoints for federated learning coordinator management.
Provides HTTP endpoints for monitoring, configuration, and control.
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

from ..shared.models import TrainingStatus
from .round_manager import RoundManager
from .metrics_tracker import MetricsTracker
from .failure_handler import FailureHandler

logger = logging.getLogger(__name__)


class CoordinatorAPI:
    """REST API for coordinator management."""
    
    def __init__(self, 
                 round_manager: Optional[RoundManager] = None,
                 metrics_tracker: Optional[MetricsTracker] = None,
                 failure_handler: Optional[FailureHandler] = None,
                 host: str = "0.0.0.0",
                 port: int = 8080):
        """
        Initialize coordinator API.
        
        Args:
            round_manager: Round management service
            metrics_tracker: Metrics tracking service
            failure_handler: Failure handling service
            host: Host to bind to
            port: Port to listen on
        """
        self.round_manager = round_manager
        self.metrics_tracker = metrics_tracker
        self.failure_handler = failure_handler
        self.host = host
        self.port = port
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web interfaces
        
        # Configure logging
        self.app.logger.setLevel(logging.INFO)
        
        # Register routes
        self._register_routes()
        
        # Server control
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
        
        logger.info(f"Coordinator API initialized on {host}:{port}")
    
    def _register_routes(self):
        """Register all API routes."""
        
        # Health and status endpoints
        self.app.route('/health', methods=['GET'])(self.health_check)
        self.app.route('/status', methods=['GET'])(self.get_status)
        self.app.route('/info', methods=['GET'])(self.get_info)
        
        # Training management endpoints
        self.app.route('/training/status', methods=['GET'])(self.get_training_status)
        self.app.route('/training/start', methods=['POST'])(self.start_training)
        self.app.route('/training/stop', methods=['POST'])(self.stop_training)
        self.app.route('/training/rounds', methods=['GET'])(self.get_rounds)
        self.app.route('/training/rounds/<int:round_number>', methods=['GET'])(self.get_round_details)
        
        # Client management endpoints
        self.app.route('/clients', methods=['GET'])(self.get_clients)
        self.app.route('/clients/<client_id>', methods=['GET'])(self.get_client_details)
        self.app.route('/clients/<client_id>/health', methods=['GET'])(self.get_client_health)
        self.app.route('/clients/<client_id>/exclude', methods=['POST'])(self.exclude_client)
        self.app.route('/clients/<client_id>/include', methods=['POST'])(self.include_client)
        
        # Metrics endpoints
        self.app.route('/metrics', methods=['GET'])(self.get_metrics)
        self.app.route('/metrics/system', methods=['GET'])(self.get_system_metrics)
        self.app.route('/metrics/training', methods=['GET'])(self.get_training_metrics)
        self.app.route('/metrics/clients', methods=['GET'])(self.get_client_metrics)
        self.app.route('/metrics/export', methods=['GET'])(self.export_metrics)
        
        # Configuration endpoints
        self.app.route('/config', methods=['GET'])(self.get_config)
        self.app.route('/config', methods=['PUT'])(self.update_config)
        self.app.route('/config/privacy', methods=['GET'])(self.get_privacy_config)
        self.app.route('/config/privacy', methods=['PUT'])(self.update_privacy_config)
        
        # Failure management endpoints
        self.app.route('/failures', methods=['GET'])(self.get_failures)
        self.app.route('/failures/statistics', methods=['GET'])(self.get_failure_statistics)
        
        # Administrative endpoints
        self.app.route('/admin/reset', methods=['POST'])(self.reset_system)
        self.app.route('/admin/logs', methods=['GET'])(self.get_logs)
    
    def start_server(self):
        """Start the REST API server."""
        try:
            if self.running:
                logger.warning("API server already running")
                return
            
            self.running = True
            
            # Start server in separate thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            logger.info(f"Coordinator API server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            self.running = False
            raise
    
    def stop_server(self):
        """Stop the REST API server."""
        try:
            if not self.running:
                return
            
            logger.info("Stopping coordinator API server")
            self.running = False
            
            # Wait for server thread
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5.0)
            
            logger.info("Coordinator API server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping API server: {e}")
    
    def _run_server(self):
        """Run the Flask server."""
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"API server error: {e}")
            self.running = False
    
    # Health and status endpoints
    
    def health_check(self):
        """Health check endpoint."""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'round_manager': self.round_manager is not None,
                    'metrics_tracker': self.metrics_tracker is not None,
                    'failure_handler': self.failure_handler is not None
                }
            }
            
            return jsonify(health_status), 200
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
    
    def get_status(self):
        """Get overall system status."""
        try:
            status = {
                'coordinator': {
                    'running': True,
                    'uptime': (datetime.now() - datetime.now()).total_seconds(),  # Placeholder
                    'version': '1.0.0'
                },
                'training': self._get_training_status_dict() if self.round_manager else None,
                'clients': self._get_clients_summary() if self.round_manager else None,
                'metrics': self._get_metrics_summary() if self.metrics_tracker else None
            }
            
            return jsonify(status), 200
            
        except Exception as e:
            logger.error(f"Get status failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_info(self):
        """Get coordinator information."""
        try:
            info = {
                'name': 'Federated Learning Coordinator',
                'version': '1.0.0',
                'description': 'Privacy-preserving federated learning coordinator',
                'features': [
                    'FedAvg aggregation',
                    'Differential privacy',
                    'Client failure handling',
                    'Real-time metrics',
                    'Adaptive training'
                ],
                'endpoints': {
                    'health': '/health',
                    'status': '/status',
                    'training': '/training/*',
                    'clients': '/clients/*',
                    'metrics': '/metrics/*',
                    'config': '/config/*'
                }
            }
            
            return jsonify(info), 200
            
        except Exception as e:
            logger.error(f"Get info failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    # Training management endpoints
    
    def get_training_status(self):
        """Get current training status."""
        try:
            if not self.round_manager:
                return jsonify({'error': 'Round manager not available'}), 503
            
            status = self.round_manager.get_training_status()
            
            return jsonify({
                'current_round': status.current_round,
                'active_clients': status.active_clients,
                'round_progress': status.round_progress,
                'global_accuracy': status.global_accuracy,
                'convergence_score': status.convergence_score,
                'estimated_completion': status.estimated_completion.isoformat() if status.estimated_completion else None
            }), 200
            
        except Exception as e:
            logger.error(f"Get training status failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def start_training(self):
        """Start training process."""
        try:
            if not self.round_manager:
                return jsonify({'error': 'Round manager not available'}), 503
            
            # Get configuration from request
            config = request.get_json() or {}
            
            # Start round manager if not already running
            if not hasattr(self.round_manager, 'running') or not self.round_manager.running:
                self.round_manager.start()
            
            return jsonify({'message': 'Training started', 'timestamp': datetime.now().isoformat()}), 200
            
        except Exception as e:
            logger.error(f"Start training failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def stop_training(self):
        """Stop training process."""
        try:
            if not self.round_manager:
                return jsonify({'error': 'Round manager not available'}), 503
            
            self.round_manager.stop()
            
            return jsonify({'message': 'Training stopped', 'timestamp': datetime.now().isoformat()}), 200
            
        except Exception as e:
            logger.error(f"Stop training failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_rounds(self):
        """Get training rounds information."""
        try:
            if not self.round_manager:
                return jsonify({'error': 'Round manager not available'}), 503
            
            # Get query parameters
            limit = request.args.get('limit', 10, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            # Get rounds (placeholder implementation)
            rounds = []
            for i in range(max(0, self.round_manager.current_round_number - limit), 
                          self.round_manager.current_round_number):
                round_status = self.round_manager.get_round_status(i)
                if round_status:
                    rounds.append(round_status)
            
            return jsonify({
                'rounds': rounds,
                'total': self.round_manager.current_round_number,
                'limit': limit,
                'offset': offset
            }), 200
            
        except Exception as e:
            logger.error(f"Get rounds failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_round_details(self, round_number: int):
        """Get details for a specific round."""
        try:
            if not self.round_manager:
                return jsonify({'error': 'Round manager not available'}), 503
            
            round_status = self.round_manager.get_round_status(round_number)
            
            if not round_status:
                return jsonify({'error': f'Round {round_number} not found'}), 404
            
            return jsonify(round_status), 200
            
        except Exception as e:
            logger.error(f"Get round details failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    # Client management endpoints
    
    def get_clients(self):
        """Get registered clients."""
        try:
            if not self.round_manager:
                return jsonify({'error': 'Round manager not available'}), 503
            
            clients = []
            for client_id, capabilities in self.round_manager.registered_clients.items():
                client_info = {
                    'client_id': client_id,
                    'compute_power': capabilities.compute_power.value,
                    'network_bandwidth': capabilities.network_bandwidth,
                    'available_samples': capabilities.available_samples,
                    'current_round': self.round_manager.get_client_assignment(client_id)
                }
                
                # Add health information if available
                if self.failure_handler:
                    health_summary = self.failure_handler.get_client_health_summary(client_id)
                    if health_summary:
                        client_info['health'] = health_summary
                
                clients.append(client_info)
            
            return jsonify({
                'clients': clients,
                'total': len(clients)
            }), 200
            
        except Exception as e:
            logger.error(f"Get clients failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_client_details(self, client_id: str):
        """Get details for a specific client."""
        try:
            if not self.round_manager:
                return jsonify({'error': 'Round manager not available'}), 503
            
            if client_id not in self.round_manager.registered_clients:
                return jsonify({'error': f'Client {client_id} not found'}), 404
            
            capabilities = self.round_manager.registered_clients[client_id]
            
            client_details = {
                'client_id': client_id,
                'capabilities': {
                    'compute_power': capabilities.compute_power.value,
                    'network_bandwidth': capabilities.network_bandwidth,
                    'available_samples': capabilities.available_samples,
                    'supported_models': capabilities.supported_models
                },
                'privacy_config': {
                    'epsilon': capabilities.privacy_requirements.epsilon,
                    'delta': capabilities.privacy_requirements.delta,
                    'max_grad_norm': capabilities.privacy_requirements.max_grad_norm,
                    'noise_multiplier': capabilities.privacy_requirements.noise_multiplier
                },
                'current_round': self.round_manager.get_client_assignment(client_id)
            }
            
            # Add health information if available
            if self.failure_handler:
                health_summary = self.failure_handler.get_client_health_summary(client_id)
                if health_summary:
                    client_details['health'] = health_summary
            
            # Add metrics if available
            if self.metrics_tracker:
                client_metrics = self.metrics_tracker.get_collector().get_client_metrics(client_id)
                if client_metrics:
                    client_details['metrics'] = client_metrics.to_dict()
            
            return jsonify(client_details), 200
            
        except Exception as e:
            logger.error(f"Get client details failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_client_health(self, client_id: str):
        """Get client health status."""
        try:
            if not self.failure_handler:
                return jsonify({'error': 'Failure handler not available'}), 503
            
            health_summary = self.failure_handler.get_client_health_summary(client_id)
            
            if not health_summary:
                return jsonify({'error': f'Client {client_id} not found'}), 404
            
            return jsonify(health_summary), 200
            
        except Exception as e:
            logger.error(f"Get client health failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def exclude_client(self, client_id: str):
        """Exclude client from training."""
        try:
            if not self.failure_handler:
                return jsonify({'error': 'Failure handler not available'}), 503
            
            # Get exclusion parameters from request
            data = request.get_json() or {}
            duration = data.get('duration', 3600)  # Default 1 hour
            reason = data.get('reason', 'Manual exclusion')
            
            # Handle exclusion (placeholder implementation)
            # This would typically call failure_handler.handle_failure with EXCLUDE_TEMPORARY
            
            return jsonify({
                'message': f'Client {client_id} excluded',
                'duration': duration,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            logger.error(f"Exclude client failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def include_client(self, client_id: str):
        """Include previously excluded client."""
        try:
            if not self.failure_handler:
                return jsonify({'error': 'Failure handler not available'}), 503
            
            # Handle inclusion (placeholder implementation)
            # This would typically call failure_handler recovery methods
            
            return jsonify({
                'message': f'Client {client_id} included',
                'timestamp': datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            logger.error(f"Include client failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    # Metrics endpoints
    
    def get_metrics(self):
        """Get all metrics."""
        try:
            if not self.metrics_tracker:
                return jsonify({'error': 'Metrics tracker not available'}), 503
            
            collector = self.metrics_tracker.get_collector()
            
            metrics = {
                'system': collector.get_system_metrics().to_dict(),
                'training': collector.get_training_progress(),
                'clients': collector.get_client_participation_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(metrics), 200
            
        except Exception as e:
            logger.error(f"Get metrics failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_system_metrics(self):
        """Get system metrics."""
        try:
            if not self.metrics_tracker:
                return jsonify({'error': 'Metrics tracker not available'}), 503
            
            system_metrics = self.metrics_tracker.get_collector().get_system_metrics()
            
            return jsonify(system_metrics.to_dict()), 200
            
        except Exception as e:
            logger.error(f"Get system metrics failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_training_metrics(self):
        """Get training progress metrics."""
        try:
            if not self.metrics_tracker:
                return jsonify({'error': 'Metrics tracker not available'}), 503
            
            training_progress = self.metrics_tracker.get_collector().get_training_progress()
            
            return jsonify(training_progress), 200
            
        except Exception as e:
            logger.error(f"Get training metrics failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_client_metrics(self):
        """Get client participation metrics."""
        try:
            if not self.metrics_tracker:
                return jsonify({'error': 'Metrics tracker not available'}), 503
            
            client_stats = self.metrics_tracker.get_collector().get_client_participation_stats()
            
            return jsonify(client_stats), 200
            
        except Exception as e:
            logger.error(f"Get client metrics failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def export_metrics(self):
        """Export metrics to file."""
        try:
            if not self.metrics_tracker:
                return jsonify({'error': 'Metrics tracker not available'}), 503
            
            # Get export format from query parameters
            format_type = request.args.get('format', 'json')
            
            if format_type == 'json':
                collector = self.metrics_tracker.get_collector()
                
                export_data = {
                    'system_metrics': collector.get_system_metrics().to_dict(),
                    'training_progress': collector.get_training_progress(),
                    'client_stats': collector.get_client_participation_stats(),
                    'recent_rounds': [r.to_dict() for r in collector.get_recent_rounds()],
                    'export_timestamp': datetime.now().isoformat()
                }
                
                response = Response(
                    json.dumps(export_data, indent=2),
                    mimetype='application/json',
                    headers={'Content-Disposition': f'attachment; filename=metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'}
                )
                
                return response
            else:
                return jsonify({'error': f'Unsupported format: {format_type}'}), 400
            
        except Exception as e:
            logger.error(f"Export metrics failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    # Configuration endpoints
    
    def get_config(self):
        """Get current configuration."""
        try:
            # Placeholder configuration
            config = {
                'coordinator': {
                    'host': self.host,
                    'port': self.port,
                    'version': '1.0.0'
                },
                'training': {
                    'min_clients': 2,
                    'max_clients': 50,
                    'round_timeout': 300,
                    'aggregation_algorithm': 'fedavg'
                },
                'privacy': {
                    'default_epsilon': 1.0,
                    'default_delta': 1e-5,
                    'max_grad_norm': 1.0
                }
            }
            
            return jsonify(config), 200
            
        except Exception as e:
            logger.error(f"Get config failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def update_config(self):
        """Update configuration."""
        try:
            new_config = request.get_json()
            
            if not new_config:
                return jsonify({'error': 'No configuration provided'}), 400
            
            # Validate and apply configuration (placeholder)
            # This would typically update the actual configuration
            
            return jsonify({
                'message': 'Configuration updated',
                'timestamp': datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            logger.error(f"Update config failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_privacy_config(self):
        """Get privacy configuration."""
        try:
            privacy_config = {
                'default_epsilon': 1.0,
                'default_delta': 1e-5,
                'max_grad_norm': 1.0,
                'noise_multiplier': 1.0,
                'privacy_levels': {
                    'high': {'epsilon': 0.5, 'delta': 1e-6},
                    'medium': {'epsilon': 1.0, 'delta': 1e-5},
                    'low': {'epsilon': 3.0, 'delta': 1e-4}
                }
            }
            
            return jsonify(privacy_config), 200
            
        except Exception as e:
            logger.error(f"Get privacy config failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def update_privacy_config(self):
        """Update privacy configuration."""
        try:
            new_config = request.get_json()
            
            if not new_config:
                return jsonify({'error': 'No privacy configuration provided'}), 400
            
            # Validate privacy parameters
            epsilon = new_config.get('epsilon')
            delta = new_config.get('delta')
            
            if epsilon is not None and epsilon <= 0:
                return jsonify({'error': 'Epsilon must be positive'}), 400
            
            if delta is not None and (delta <= 0 or delta >= 1):
                return jsonify({'error': 'Delta must be in (0, 1)'}), 400
            
            # Apply configuration (placeholder)
            
            return jsonify({
                'message': 'Privacy configuration updated',
                'timestamp': datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            logger.error(f"Update privacy config failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    # Failure management endpoints
    
    def get_failures(self):
        """Get failure information."""
        try:
            if not self.failure_handler:
                return jsonify({'error': 'Failure handler not available'}), 503
            
            failure_stats = self.failure_handler.get_failure_statistics()
            
            return jsonify(failure_stats), 200
            
        except Exception as e:
            logger.error(f"Get failures failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_failure_statistics(self):
        """Get detailed failure statistics."""
        try:
            if not self.failure_handler:
                return jsonify({'error': 'Failure handler not available'}), 503
            
            stats = self.failure_handler.get_failure_statistics()
            
            return jsonify(stats), 200
            
        except Exception as e:
            logger.error(f"Get failure statistics failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    # Administrative endpoints
    
    def reset_system(self):
        """Reset system state."""
        try:
            # Get reset parameters
            data = request.get_json() or {}
            reset_type = data.get('type', 'soft')  # 'soft' or 'hard'
            
            if reset_type == 'hard':
                # Hard reset - clear all data
                if self.round_manager:
                    self.round_manager.stop()
                    # Clear round data (placeholder)
                
                if self.metrics_tracker:
                    # Clear metrics (placeholder)
                    pass
                
                if self.failure_handler:
                    # Clear failure data (placeholder)
                    pass
            
            return jsonify({
                'message': f'{reset_type.title()} reset completed',
                'timestamp': datetime.now().isoformat()
            }), 200
            
        except Exception as e:
            logger.error(f"Reset system failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_logs(self):
        """Get system logs."""
        try:
            # Get log parameters
            lines = request.args.get('lines', 100, type=int)
            level = request.args.get('level', 'INFO')
            
            # Placeholder log data
            logs = [
                {
                    'timestamp': datetime.now().isoformat(),
                    'level': 'INFO',
                    'message': 'Coordinator API endpoint accessed',
                    'module': 'rest_api'
                }
            ]
            
            return jsonify({
                'logs': logs,
                'total': len(logs),
                'level': level,
                'lines': lines
            }), 200
            
        except Exception as e:
            logger.error(f"Get logs failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    # Helper methods
    
    def _get_training_status_dict(self) -> Dict[str, Any]:
        """Get training status as dictionary."""
        if not self.round_manager:
            return None
        
        status = self.round_manager.get_training_status()
        return {
            'current_round': status.current_round,
            'active_clients': status.active_clients,
            'round_progress': status.round_progress,
            'global_accuracy': status.global_accuracy,
            'convergence_score': status.convergence_score
        }
    
    def _get_clients_summary(self) -> Dict[str, Any]:
        """Get clients summary."""
        if not self.round_manager:
            return None
        
        return {
            'total_registered': len(self.round_manager.registered_clients),
            'active_clients': len([
                cid for cid, assignment in self.round_manager.client_round_assignments.items()
                if assignment is not None
            ])
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.metrics_tracker:
            return None
        
        collector = self.metrics_tracker.get_collector()
        system_metrics = collector.get_system_metrics()
        
        return {
            'total_rounds': system_metrics.total_rounds,
            'current_accuracy': system_metrics.current_global_accuracy,
            'best_accuracy': system_metrics.best_global_accuracy,
            'avg_round_duration': system_metrics.avg_round_duration
        }