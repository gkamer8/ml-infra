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
echo ""
