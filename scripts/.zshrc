# Path to your oh-my-zsh installation
export ZSH="/root/.oh-my-zsh"

# Set theme
ZSH_THEME="agnoster"

# Plugins
plugins=(
    git
    python
    extract
    z
    zsh-autosuggestions
    fzf
)

# Load oh-my-zsh
source $ZSH/oh-my-zsh.sh

# User configuration
export EDITOR='vim'

# Zsh autosuggestions configuration
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE="fg=#666666"
ZSH_AUTOSUGGEST_STRATEGY=(history completion)

# Python settings
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Load environment variables from .env if it exists
if [ -f /infra/secrets/.env ]; then
    set -a
    source /infra/secrets/.env
    set +a
    echo "✓ Loaded environment variables from .env"
fi


# UV command protection and aliases
# Save the original uv command path
UV_ORIGINAL_PATH=$(which uv 2>/dev/null)

# Function to intercept uv commands
uv() {
    # Check if this is the allowed system install command
    if [[ "$*" == "pip install --system -e /infra" ]]; then
        # Execute the allowed command using the original uv
        $UV_ORIGINAL_PATH "$@"
    else
        echo "❌ You shouldn't use uv directly in this environment." >&2
        echo "⚠️  This project uses system-wide Python packages installed in the Docker container." >&2
        echo "" >&2
        echo "📋 Instructions:" >&2
        echo "  • To run scripts: python3 -m ml.path.to.script" >&2
        echo "  • To use the CLI: submit [command]" >&2
        echo "  • To reinstall the project: reinstall" >&2
        return 1
    fi
}

# Alias for reinstalling the project
alias reinstall="$UV_ORIGINAL_PATH pip install --system -e /infra && echo '✅ Project reinstalled successfully!'"

# Remove first segment of the prompt
prompt_context() {
  # Empty function to hide username@hostname segment
}
prompt_status() {
  # Empty function to hide status segment (lightning bolt icon)
}

# Welcome message
echo "🚀 ML Development Environment Ready!"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(which python) ($(python --version 2>&1))"
echo "🔧 Use the 'reinstall' alias to reinstall the project packages"
