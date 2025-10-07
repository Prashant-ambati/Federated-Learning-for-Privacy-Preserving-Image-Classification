#!/bin/bash
set -e

# Wait for coordinator to be ready
echo "Waiting for coordinator to be ready..."
until curl -f http://${COORDINATOR_HOST:-coordinator}:${COORDINATOR_HTTP_PORT:-8080}/health; do
  echo "Coordinator is unavailable - sleeping"
  sleep 5
done

echo "Coordinator is ready!"

# Generate unique client ID if not provided
if [ -z "$CLIENT_ID" ]; then
    export CLIENT_ID="client-$(hostname)-$(date +%s)"
    echo "Generated client ID: $CLIENT_ID"
fi

# Start the application
echo "Starting client service with ID: $CLIENT_ID"
exec "$@"