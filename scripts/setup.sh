#!/bin/bash

set -e

# Get the project root directory (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "ML Infrastructure Docker Development Setup"
echo "=========================================="
echo ""
echo "This script will add the following aliases to your shell:"
echo "  dstart   - Build and start the ML dev container"
echo "  dinto    - Enter the running ML dev container"
echo "  drestart - Restart the ML dev container"
echo "  dremove  - Stop and remove the ML dev container"
echo ""
echo "These aliases will be added to your shell configuration file."
echo ""

# Detect shell and config file
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
    SHELL_NAME="zsh"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
    SHELL_NAME="bash"
else
    echo "Unsupported shell: $SHELL"
    echo "Please manually add aliases to your shell configuration."
    exit 1
fi

echo "Detected shell: $SHELL_NAME"
echo "Config file: $SHELL_CONFIG"
echo ""

read -p "Do you want to proceed? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Create aliases
ALIASES="
# ML Infrastructure Docker Development Aliases
alias dstart='\"$PROJECT_ROOT/scripts/docker/start.sh\"'
alias dinto='\"$PROJECT_ROOT/scripts/docker/into.sh\"'
alias drestart='\"$PROJECT_ROOT/scripts/docker/restart.sh\"'
alias dremove='\"$PROJECT_ROOT/scripts/docker/remove.sh\"'
"

# Check if aliases already exist
if grep -q "ML Infrastructure Docker Development Aliases" "$SHELL_CONFIG" 2>/dev/null; then
    echo "Aliases already exist in $SHELL_CONFIG"
    echo "Please remove them manually if you want to reinstall."
    exit 1
fi

# Add aliases to shell config
echo "$ALIASES" >> "$SHELL_CONFIG"

echo ""
echo "✅ Aliases added to $SHELL_CONFIG"
echo ""
echo "Please restart your terminal or run:"
echo "  source $SHELL_CONFIG"
