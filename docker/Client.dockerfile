FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
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
RUN mkdir -p logs checkpoints data

# Set environment variables
ENV PYTHONPATH=/app

# Create entrypoint script
COPY docker/client-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run client service
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "src.client.main", "--config", "config/client.yaml"]