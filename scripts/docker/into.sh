#!/bin/bash

CONTAINER_NAME="ml-dev"

if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "Container '$CONTAINER_NAME' is not running."
    echo "Run 'scripts/docker/start.sh' first to start the container."
    exit 1
fi

echo "Entering ML development container..."
docker exec -it $CONTAINER_NAME zsh
exit_code=$?

# Exit code 0 = normal exit, exit code 130 = Ctrl+C, both are normal
if [ $exit_code -eq 0 ] || [ $exit_code -eq 130 ]; then
    exit 0
else
    exit $exit_code
fi
