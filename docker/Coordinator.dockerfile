FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY src/ ./src/
COPY proto/ ./proto/
COPY config/ ./config/
COPY setup.py .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs checkpoints data models

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_ENV=production

# Expose ports (gRPC, HTTP, Metrics)
EXPOSE 50051 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Create entrypoint script
COPY docker/coordinator-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run coordinator service
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "src.coordinator.main", "--config", "config/coordinator.yaml"]