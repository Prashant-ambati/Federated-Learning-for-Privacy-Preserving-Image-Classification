"""
Client service main entry point.
Starts a federated learning client that connects to the coordinator.
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

# Import client components
from .federated_trainer import create_federated_trainer


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
    log_file = log_config.get('file', 'logs/client.log')
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


class ClientService:
    """Main client service class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize client service."""
        self.config = config
        self.running = False
        
        # Get client ID from config or environment
        self.client_id = (
            os.environ.get('CLIENT_ID') or 
            config.get('client', {}).get('client_id', 'client-001')
        )
        
        # Initialize trainer
        self.trainer = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Client service initialized with ID: {self.client_id}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def start(self):
        """Start the client service."""
        try:
            self.running = True
            logger.info(f"Starting federated learning client: {self.client_id}")
            
            # Print configuration summary
            client_config = self.config.get('client', {})
            training_config = self.config.get('training', {})
            privacy_config = self.config.get('privacy', {})
            
            coordinator_host = os.environ.get('COORDINATOR_HOST') or client_config.get('coordinator_host', 'localhost')
            coordinator_port = os.environ.get('COORDINATOR_PORT') or client_config.get('coordinator_port', 50051)
            
            logger.info(f"Coordinator: {coordinator_host}:{coordinator_port}")
            logger.info(f"Model: {training_config.get('model_type', 'simple_cnn')}")
            logger.info(f"Dataset: {training_config.get('dataset', 'mnist')}")
            logger.info(f"Privacy: ε={privacy_config.get('epsilon', 1.0)}, δ={privacy_config.get('delta', 1e-5)}")
            
            # Create necessary directories
            data_dir = self.config.get('data', {}).get('data_dir', 'data/')
            checkpoint_dir = self.config.get('checkpoints', {}).get('checkpoint_dir', 'checkpoints/')
            
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            # Create federated trainer
            coordinator_address = f"{coordinator_host}:{coordinator_port}"
            
            self.trainer = create_federated_trainer(
                client_id=self.client_id,
                coordinator_address=coordinator_address,
                compute_power=training_config.get('compute_power', 'medium'),
                network_bandwidth=training_config.get('network_bandwidth', 10),
                available_samples=training_config.get('available_samples', 1000),
                model_type=training_config.get('model_type', 'simple_cnn'),
                dataset_name=training_config.get('dataset', 'mnist'),
                privacy_epsilon=privacy_config.get('epsilon', 1.0),
                privacy_delta=privacy_config.get('delta', 1e-5)
            )
            
            # Initialize and start training
            if self.trainer.initialize():
                self.trainer.start_training()
                
                # Wait for training to complete or service to stop
                while self.running and self.trainer.running:
                    await asyncio.sleep(5)
                    
                    # Log status periodically
                    status = self.trainer.get_status()
                    logger.debug(f"Client status: {status['state']}, Round: {status['current_round']}")
            else:
                logger.error("Failed to initialize federated trainer")
            
            logger.info(f"Client {self.client_id} stopped")
            
        except Exception as e:
            logger.error(f"Client service error: {e}")
            raise
    
    async def stop(self):
        """Stop the client service."""
        logger.info(f"Stopping client {self.client_id}...")
        self.running = False
        
        if self.trainer:
            self.trainer.stop_training()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--config', '-c', 
                       default='config/client.yaml',
                       help='Configuration file path')
    parser.add_argument('--client-id', 
                       help='Client ID (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override client ID if provided
    if args.client_id:
        config.setdefault('client', {})['client_id'] = args.client_id
    
    # Setup logging
    setup_logging(config)
    
    # Create and start client service
    client = ClientService(config)
    
    try:
        await client.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Client failed: {e}")
        sys.exit(1)
    finally:
        await client.stop()


if __name__ == '__main__':
    # Create event loop and run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)