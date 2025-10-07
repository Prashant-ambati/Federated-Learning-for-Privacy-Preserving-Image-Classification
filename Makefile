# Federated Learning System Makefile

.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install the package and dependencies"
	@echo "  dev-install  - Install in development mode with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start services with Docker Compose"
	@echo "  docker-down  - Stop services with Docker Compose"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Testing and quality
test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Docker operations
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Development helpers
setup-dev: dev-install
	mkdir -p logs data checkpoints
	@echo "Development environment setup complete"

run-coordinator:
	python -m src.coordinator.main --config config/coordinator.yaml

run-client:
	python -m src.client.main --config config/client.yaml

# Database operations
db-init:
	python -c "from src.shared.database import init_database; init_database()"

db-migrate:
	alembic upgrade head

# Monitoring
monitor:
	@echo "Starting monitoring dashboard..."
	@echo "Coordinator metrics: http://localhost:9090"
	@echo "Health check: http://localhost:8080/health"