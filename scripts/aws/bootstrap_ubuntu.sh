#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  git \
  curl \
  ca-certificates \
  gnupg \
  lsb-release \
  build-essential \
  cmake \
  python3-pip \
  python3-venv \
  tmux \
  htop \
  unzip \
  ripgrep

mkdir -p "$HOME/workspace"
cd "$HOME/workspace"

if [ ! -d "$HOME/workspace/jumping_robot" ]; then
  git clone https://github.com/Quantumtato/jumping_robot.git "$HOME/workspace/jumping_robot"
fi

if ! command -v conda >/dev/null 2>&1; then
  wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  "$HOME/miniconda3/bin/conda" init bash
fi

cat <<'EOF' >> "$HOME/.bashrc"
# Added by jumping_robot bootstrap
export PATH="$HOME/miniconda3/bin:$PATH"
EOF

printf '\nBootstrap complete. Open a new shell and run:\n'
printf '  cd ~/workspace/jumping_robot\n'
printf '  python3 -m venv .venv\n'
printf '  source .venv/bin/activate\n'
printf '  python -m pip install --upgrade pip\n'
