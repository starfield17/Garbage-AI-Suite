#!/bin/bash

# ==============================================================================
# Systemd Service Installation Script
# Usage: 
#   Install: sudo ./install_service.sh <conda_env> <script_path>
#   Remove:  sudo ./remove_service.sh <script_path>
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

ACTION=$1

install_service() {
    local CONDA_ENV=$2
    local SCRIPT_PATH=$3
    
    if [ -z "$CONDA_ENV" ] || [ -z "$SCRIPT_PATH" ]; then
        echo "Usage: $0 install <conda_env> <script_path>"
        exit 1
    fi
    
    SCRIPT_PATH=$(realpath "$SCRIPT_PATH")
    
    if [ ! -f "$SCRIPT_PATH" ]; then
        log_error "Script file '$SCRIPT_PATH' does not exist."
        exit 1
    fi
    
    local USER_NAME=$(whoami)
    local SERVICE_NAME=$(basename "$SCRIPT_PATH" .py).service
    
    local CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -z "$CONDA_BASE" ]; then
        log_error "Conda is not installed."
        exit 1
    fi
    
    local EXEC_START_CMD="/bin/bash -c 'source \"$CONDA_BASE/etc/profile.d/conda.sh\"; conda activate \"$CONDA_ENV\"; python \"$SCRIPT_PATH\"'"
    
    local SERVICE_FILE="[Unit]
Description=Auto-start $SERVICE_NAME
After=network.target

[Service]
Type=simple
User=$USER_NAME
Environment=PATH=$CONDA_BASE/bin:/usr/bin:/bin
WorkingDirectory=$(dirname "$SCRIPT_PATH")
ExecStart=$EXEC_START_CMD
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"
    
    echo "$SERVICE_FILE" | sudo tee /etc/systemd/system/$SERVICE_NAME > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    sudo systemctl start $SERVICE_NAME
    
    log_success "Service '$SERVICE_NAME' installed and started."
}

remove_service() {
    local SCRIPT_PATH=$1
    
    if [ -z "$SCRIPT_PATH" ]; then
        echo "Usage: $0 remove <script_path>"
        exit 1
    fi
    
    SCRIPT_PATH=$(realpath "$SCRIPT_PATH")
    local SERVICE_NAME=$(basename "$SCRIPT_PATH" .py).service
    
    sudo systemctl stop $SERVICE_NAME 2>/dev/null || true
    sudo systemctl disable $SERVICE_NAME
    sudo rm -f /etc/systemd/system/$SERVICE_NAME
    sudo systemctl daemon-reload
    
    log_success "Service '$SERVICE_NAME' removed."
}

case "$ACTION" in
    install)
        install_service "$@"
        ;;
    remove)
        remove_service "$@"
        ;;
    *)
        echo "Usage: $0 <install|remove> ..."
        exit 1
        ;;
esac
