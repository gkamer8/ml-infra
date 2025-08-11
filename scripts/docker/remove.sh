#!/bin/bash

CONTAINER_NAME="ml-dev"

echo "Stopping and removing ML development container..."

if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    docker stop $CONTAINER_NAME
    echo "Container '$CONTAINER_NAME' stopped."
fi

if docker ps -aq -f name=$CONTAINER_NAME | grep -q .; then
    docker rm $CONTAINER_NAME
    echo "Container '$CONTAINER_NAME' removed."
fi

echo "ML development container cleanup complete."
