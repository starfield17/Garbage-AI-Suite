#!/usr/bin/env bash

# ==============================================================================
# Proxy Configuration Script
# Usage: ./proxy.sh
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

current_shell=$(ps -p $$ -ocomm=)

if [[ "$current_shell" == "fish" ]]; then
    fish -c "
        read -P 'Enter proxy address (IP:port) or press Enter to skip: ' proxy
        if [ -n \"\$proxy\" ]
            set -x http_proxy \"http://\$proxy\"
            set -x https_proxy \"https://\$proxy\"
            set -x ftp_proxy \"ftp://\$proxy\"
            set -x socks_proxy \"socks://\$proxy\"
            set -x no_proxy \"localhost,127.0.0.1,::1\"
            echo 'Proxy configured successfully'
        else
            echo 'No proxy configured'
        end
    "
else
    read -p "Enter proxy address (IP:port) or press Enter to skip: " proxy
    if [[ -n "$proxy" ]]; then
        export http_proxy="http://$proxy"
        export https_proxy="https://$proxy"
        export ftp_proxy="ftp://$proxy"
        export socks_proxy="socks://$proxy"
        export no_proxy="localhost,127.0.0.1,::1"
        echo -e "${GREEN}[SUCCESS]${NC} Proxy configured successfully"
    else
        echo "No proxy configured"
    fi
fi
