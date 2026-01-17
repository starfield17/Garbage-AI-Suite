#!/bin/bash

# ==============================================================================
# NPU Configuration Script for RK3588
# Usage: sudo ./npu_config.sh
# ==============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run this script as root"
        exit 1
    fi
}

install_dependencies() {
    log_info "Installing dependencies..."
    apt-get update
    apt-get install -y cmake build-essential git python3-pip
}

configure_proxy() {
    read -p "Enter proxy address (IP:Port) or press Enter to skip: " proxy
    if [[ ! -z "$proxy" ]]; then
        export http_proxy="http://$proxy"
        export https_proxy="https://$proxy"
        log_info "Proxy configured"
    fi
}

clone_rknpu2() {
    if [ -d "rknpu2" ]; then
        log_info "Updating rknpu2 repository..."
        cd rknpu2 && git pull && cd ..
    else
        log_info "Cloning rknpu2 repository..."
        git clone https://github.com/rockchip-linux/rknpu2
    fi
}

copy_libraries() {
    log_info "Copying library files..."
    
    if [ -f rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so ]; then
        cp rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
    fi
    
    if [ -f rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknn_api.so ]; then
        cp rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/librknn_api.so /usr/lib/
    fi
}

copy_headers() {
    log_info "Copying header files..."
    mkdir -p /usr/include/rknn
    
    if [ -f rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_api.h ]; then
        cp rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_api.h /usr/include/rknn/
    fi
    
    if [ -f rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_matmul_api.h ]; then
        cp rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_matmul_api.h /usr/include/rknn/
    fi
}

copy_server_files() {
    log_info "Copying rknn_server files..."
    
    if [ -f rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/rknn_server ]; then
        cp rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/rknn_server /usr/bin/
    fi
    
    if [ -f rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/start_rknn.sh ]; then
        cp rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/start_rknn.sh /usr/bin/
        chmod +x /usr/bin/start_rknn.sh
    fi
    
    if [ -f rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/restart_rknn.sh ]; then
        cp rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/restart_rknn.sh /usr/bin/
        chmod +x /usr/bin/restart_rknn.sh
    fi
}

update_ldconfig() {
    log_info "Updating linker configuration..."
    ldconfig
}

start_rknn_server() {
    log_info "Starting rknn_server..."
    /usr/bin/start_rknn.sh
}

main() {
    check_root
    
    configure_proxy
    install_dependencies
    clone_rknpu2
    copy_libraries
    copy_headers
    copy_server_files
    update_ldconfig
    start_rknn_server
    
    log_success "NPU configuration complete!"
}

main
