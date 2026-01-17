#!/bin/bash

# ==============================================================================
# Systemd Service Uninstallation Script
# Usage: sudo ./uninstall_service.sh <script_path>
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

remove_service() {
    local SCRIPT_PATH=$1
    
    if [ -z "$SCRIPT_PATH" ]; then
        echo "Usage: $0 <script_path>"
        exit 1
    fi
    
    SCRIPT_PATH=$(realpath "$SCRIPT_PATH")
    local SERVICE_NAME=$(basename "$SCRIPT_PATH" .py).service
    
    if ! systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
        log_error "Service '$SERVICE_NAME' does not exist."
        exit 1
    fi
    
    sudo systemctl stop $SERVICE_NAME 2>/dev/null || true
    sudo systemctl disable $SERVICE_NAME
    sudo rm -f /etc/systemd/system/$SERVICE_NAME
    sudo systemctl daemon-reload
    
    log_success "Service '$SERVICE_NAME' uninstalled."
}

remove_service "$@"
