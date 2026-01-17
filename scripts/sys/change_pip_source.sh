#!/bin/bash

# ==============================================================================
# Pip/Conda Source Switcher Script
# Usage: ./change_pip_source.sh
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# PIP configuration
set_pip_source() {
    local url=$1
    local name=$2
    
    if ! command -v pip &> /dev/null; then
        echo "pip not found, skipping..."
        return
    fi
    
    if [ "$url" == "default" ]; then
        pip config unset global.index-url
    else
        pip config set global.index-url "$url"
    fi
    
    pip config list
}

# Conda configuration
set_conda_source() {
    local mirror=$1
    
    if ! command -v conda &> /dev/null; then
        echo "conda not found, skipping..."
        return
    fi
    
    local CONDARC="$HOME/.condarc"
    
    case "$mirror" in
        tsinghua)
            cat > "$CONDARC" << EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
            ;;
        ustc)
            cat > "$CONDARC" << EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main
custom_channels:
  conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud
EOF
            ;;
        default)
            conda config --remove-key channels
            conda config --remove-key default_channels
            ;;
    esac
    
    cat "$CONDARC"
}

# Main menu
main() {
    echo -e "${BLUE}=== Package Source Switcher ===${NC}"
    echo "1) Pip - Tsinghua"
    echo "2) Pip - USTC"
    echo "3) Pip - Restore Default"
    echo "4) Conda - Tsinghua"
    echo "5) Conda - USTC"
    echo "6) Conda - Restore Default"
    echo "0) Exit"
    
    read -p "Enter choice: " choice
    
    case $choice in
        1) set_pip_source "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" "Tsinghua" ;;
        2) set_pip_source "https://mirrors.ustc.edu.cn/pypi/simple" "USTC" ;;
        3) set_pip_source "default" "Official" ;;
        4) set_conda_source "tsinghua" ;;
        5) set_conda_source "ustc" ;;
        6) set_conda_source "default" ;;
        0) exit 0 ;;
        *) echo "Invalid choice" ;;
    esac
}

main
