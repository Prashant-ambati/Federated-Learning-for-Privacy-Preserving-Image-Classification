"""
Coordinator service main entry point.
Starts the federated learning coordinator with gRPC and HTTP servers.
"""

import argparse
import asyncio
import logging
import signal
import sys
import yaml
import os
from pathlib import Path
from typing import Dict, Any

# Setup basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import coordinator components
from .grpc_server import CoordinatorGRPCServer
from .round_manager import RoundManager
from .metrics_tracker import MetricsTracker
from .failure_handler import FailureHandler
from .rest_api import CoordinatorAPI
from ..shared.database import create_database_manager


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/coordinator.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


class CoordinatorService:
    """Main coordinator service class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize coordinator service."""
        self.config = config
        self.running = False
        
        # Initialize components
        self.grpc_server = None
        self.rest_api = None
        self.round_manager = None
        self.metrics_tracker = None
        self.failure_handler = None
        self.db_manager = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Coordinator service initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def start(self):
        """Start the coordinator service."""
        try:
            self.running = True
            logger.info("Starting federated learning coordinator...")
            
            # Print configuration summary
            server_config = self.config.get('server', {})
            fl_config = self.config.get('federated_learning', {})
            
            logger.info(f"gRPC server will listen on port {server_config.get('grpc_port', 50051)}")
            logger.info(f"HTTP server will listen on port {server_config.get('http_port', 8080)}")
            logger.info(f"Supporting {fl_config.get('min_clients', 2)}-{fl_config.get('max_clients', 50)} clients")
            
            # Initialize database
            try:
                database_url = os.getenv('DATABASE_URL', self.config.get('database', {}).get('url'))
                if database_url:
                    self.db_manager = create_database_manager(database_url)
                    self.db_manager.create_tables()
                    logger.info("Database initialized")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
            
            # Initialize core services
            self.round_manager = RoundManager()
            self.metrics_tracker = MetricsTracker()
            self.failure_handler = FailureHandler()
            
            # Start services
            self.round_manager.start()
            self.metrics_tracker.start()
            self.failure_handler.start()
            
            # Start gRPC server
            grpc_port = server_config.get('grpc_port', 50051)
            self.grpc_server = CoordinatorGRPCServer(port=grpc_port)
            self.grpc_server.start()
            
            # Start REST API
            http_port = server_config.get('http_port', 8080)
            self.rest_api = CoordinatorAPI(
                round_manager=self.round_manager,
                metrics_tracker=self.metrics_tracker,
                failure_handler=self.failure_handler,
                port=http_port
            )
            self.rest_api.start_server()
            
            logger.info("All coordinator services started successfully")
            
            # Main service loop
            while self.running:
                await asyncio.sleep(1)
            
            logger.info("Coordinator service stopped")
            
        except Exception as e:
            logger.error(f"Coordinator service error: {e}")
            raise
    
    async def stop(self):
        """Stop the coordinator service."""
        logger.info("Stopping coordinator service...")
        self.running = False
        
        # Stop services in reverse order
        if self.rest_api:
            self.rest_api.stop_server()
        
        if self.grpc_server:
            self.grpc_server.stop()
        
        if self.failure_handler:
            self.failure_handler.stop()
        
        if self.metrics_tracker:
            self.metrics_tracker.stop()
        
        if self.round_manager:
            self.round_manager.stop()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Federated Learning Coordinator')
    parser.add_argument('--config', '-c', 
                       default='config/coordinator.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Create and start coordinator service
    coordinator = CoordinatorService(config)
    
    try:
        await coordinator.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Coordinator failed: {e}")
        sys.exit(1)
    finally:
        await coordinator.stop()


if __name__ == '__main__':
    # Create event loop and run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Coordinator shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)