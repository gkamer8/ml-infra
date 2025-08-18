# Path to your oh-my-zsh installation
export ZSH="/root/.oh-my-zsh"

# Set theme
ZSH_THEME="robbyrussell"

# Plugins
plugins=(
    git
    pip
    python
    docker
    docker-compose
    colored-man-pages
    command-not-found
    extract
    z
)

# Load oh-my-zsh
source $ZSH/oh-my-zsh.sh

# User configuration
export LANG=en_US.UTF-8
export EDITOR='vim'

# Python settings
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# Python/ML aliases
alias py='python'
alias ipy='ipython'
alias jn='jupyter notebook'
alias jl='jupyter lab'
alias pypath='which python && python -c "import sys; print(\"Python path:\", sys.executable); print(\"Site packages:\", next((p for p in sys.path if \"site-packages\" in p), None))"'

# Docker aliases
alias dps='docker ps'
alias dpsa='docker ps -a'
alias dimg='docker images'
alias dexec='docker exec -it'
alias dlogs='docker logs -f'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit -m'
alias gp='git push'
alias gl='git pull'
alias glog='git log --oneline --graph --decorate'
alias gb='git branch'
alias gco='git checkout'
alias gd='git diff'

# UV package manager aliases
alias uvi='uv pip install'
alias uvs='uv sync'
alias uvr='uv run'
alias uvl='uv pip list'

# Custom functions
# Quick navigate to common directories
function cdml() {
    cd /infra/ml/$1
}

function cdscripts() {
    cd /infra/scripts/$1
}

# Load environment variables from .env if it exists
if [ -f /infra/secrets/.env ]; then
    set -a
    source /infra/secrets/.env
    set +a
    echo "✓ Loaded environment variables from .env"
fi

# Set up prompt with current directory and git branch
PROMPT='%{$fg[cyan]%}[ml-dev]%{$reset_color%} %{$fg[green]%}%~%{$reset_color%} $(git_prompt_info)
%{$fg[yellow]%}→%{$reset_color%} '

# Git prompt configuration
ZSH_THEME_GIT_PROMPT_PREFIX="%{$fg[blue]%}git:(%{$fg[red]%}"
ZSH_THEME_GIT_PROMPT_SUFFIX="%{$reset_color%} "
ZSH_THEME_GIT_PROMPT_DIRTY="%{$fg[blue]%}) %{$fg[yellow]%}✗"
ZSH_THEME_GIT_PROMPT_CLEAN="%{$fg[blue]%})"

# History settings
HISTSIZE=10000
SAVEHIST=10000
setopt SHARE_HISTORY
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_SPACE

# Enable auto-completion
autoload -Uz compinit && compinit

# Welcome message
echo "🚀 ML Development Environment Ready!"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(which python) ($(python --version 2>&1))"
echo ""
