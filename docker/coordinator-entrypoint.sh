#!/bin/bash
set -e

# Wait for database to be ready
echo "Waiting for database to be ready..."
until pg_isready -h ${DATABASE_HOST:-postgres} -p ${DATABASE_PORT:-5432} -U ${DATABASE_USER:-federated}; do
  echo "Database is unavailable - sleeping"
  sleep 2
done

echo "Database is ready!"

# Initialize database if needed
echo "Initializing database..."
python -c "
from src.shared.database import init_database
try:
    init_database()
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization failed: {e}')
    exit(1)
"

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
until redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping; do
  echo "Redis is unavailable - sleeping"
  sleep 2
done

echo "Redis is ready!"

# Start the application
echo "Starting coordinator service..."
exec "$@"