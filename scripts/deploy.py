#!/usr/bin/env python3
"""
Deployment script for the federated learning system.
Handles deployment to various environments.
"""

import argparse
import subprocess
import sys
import os
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_grpc_files():
    """Generate gRPC Python files from protobuf definitions."""
    try:
        logger.info("Generating gRPC files...")
        
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            "--proto_path=proto",
            "--python_out=proto",
            "--grpc_python_out=proto",
            "proto/federated_learning.proto"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ gRPC files generated successfully")
            return True
        else:
            logger.error(f"✗ gRPC generation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"✗ gRPC generation failed: {e}")
        return False


def setup_environment():
    """Setup deployment environment."""
    try:
        logger.info("Setting up environment...")
        
        # Create necessary directories
        directories = [
            'data',
            'logs',
            'checkpoints',
            'models',
            'exports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        
        logger.info("✓ Environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment setup failed: {e}")
        return False


def deploy_local():
    """Deploy system locally with Docker Compose."""
    try:
        logger.info("Deploying system locally...")
        
        # Generate gRPC files
        if not generate_grpc_files():
            return False
        
        # Setup environment
        if not setup_environment():
            return False
        
        # Build Docker images
        logger.info("Building Docker images...")
        result = subprocess.run(
            ["docker-compose", "build"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Docker build failed: {result.stderr}")
            return False
        
        # Start services
        logger.info("Starting services...")
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Docker compose up failed: {result.stderr}")
            return False
        
        logger.info("✓ Local deployment completed successfully")
        logger.info("\n📋 Services started:")
        logger.info("  • PostgreSQL: localhost:5432")
        logger.info("  • Redis: localhost:6379")
        logger.info("  • Coordinator gRPC: localhost:50051")
        logger.info("  • Coordinator HTTP: localhost:8080")
        logger.info("  • Clients: 2 instances running")
        
        logger.info("\n🔍 Monitoring:")
        logger.info("  • Health check: curl http://localhost:8080/health")
        logger.info("  • Training status: curl http://localhost:8080/training/status")
        logger.info("  • Metrics: curl http://localhost:8080/metrics")
        logger.info("  • Logs: docker-compose logs -f")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Local deployment failed: {e}")
        return False


def deploy_development():
    """Deploy system for development (without Docker)."""
    try:
        logger.info("Deploying system for development...")
        
        # Generate gRPC files
        if not generate_grpc_files():
            return False
        
        # Setup environment
        if not setup_environment():
            return False
        
        # Install dependencies
        logger.info("Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Dependency installation failed: {result.stderr}")
            return False
        
        # Install package in development mode
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Package installation failed: {result.stderr}")
            return False
        
        logger.info("✓ Development deployment completed successfully")
        logger.info("\n🚀 To start the system:")
        logger.info("  1. Start infrastructure: docker-compose up -d postgres redis")
        logger.info("  2. Run coordinator: python -m src.coordinator.main")
        logger.info("  3. Run client: python -m src.client.main")
        logger.info("  4. Run tests: python scripts/test_complete_system.py")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Development deployment failed: {e}")
        return False


def validate_deployment():
    """Validate deployment by running tests."""
    try:
        logger.info("Validating deployment...")
        
        # Run comprehensive tests
        result = subprocess.run(
            [sys.executable, "scripts/test_complete_system.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✓ Deployment validation passed")
            return True
        else:
            logger.error(f"✗ Deployment validation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Deployment validation failed: {e}")
        return False


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy Federated Learning System')
    parser.add_argument('--mode', '-m', 
                       choices=['local', 'development', 'validate'],
                       default='development',
                       help='Deployment mode')
    parser.add_argument('--validate', '-v',
                       action='store_true',
                       help='Run validation after deployment')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Deploying Federated Learning System in {args.mode} mode")
    
    success = False
    
    if args.mode == 'local':
        success = deploy_local()
    elif args.mode == 'development':
        success = deploy_development()
    elif args.mode == 'validate':
        success = validate_deployment()
    
    if success and args.validate and args.mode != 'validate':
        logger.info("\n🔍 Running post-deployment validation...")
        success = validate_deployment()
    
    if success:
        logger.info(f"\n🎉 Deployment completed successfully!")
        logger.info("\n📚 Documentation:")
        logger.info("  • README.md - Getting started guide")
        logger.info("  • DEVELOPMENT_STATUS.md - Current implementation status")
        logger.info("  • .kiro/specs/federated-learning-system/ - Complete specifications")
        
        return 0
    else:
        logger.error(f"\n❌ Deployment failed. Please check the logs and fix any issues.")
        return 1


if __name__ == '__main__':
    sys.exit(main())