#!/bin/bash

set -e

# Get the project root directory (two levels up from scripts/docker/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONTAINER_NAME="ml-dev"
IMAGE_NAME="ml-dev-image"

echo "Building ML development container..."
cd "$PROJECT_ROOT"
docker build -f Dockerfile.dev -t $IMAGE_NAME .

echo "Starting ML development container..."
docker run -d \
    --name $CONTAINER_NAME \
    -v "$PROJECT_ROOT":/infra \
    $IMAGE_NAME tail -f /dev/null

echo "Container '$CONTAINER_NAME' is now running."
