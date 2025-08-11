#!/bin/bash

echo "Restarting ML development container..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/remove.sh"
"$SCRIPT_DIR/start.sh"

echo "ML development container restarted."
