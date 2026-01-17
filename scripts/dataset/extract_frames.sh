#!/bin/bash

# ==============================================================================
# Video Frame Extraction Script
# Usage: ./extract_frames.sh <video_path|directory> [fps]
# ==============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Supported video extensions
VIDEO_EXTENSIONS=("mp4" "avi" "mkv" "mov" "flv" "wmv" "webm")

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if file is a video file
is_video_file() {
    local file="$1"
    local extension="${file##*.}"
    extension="${extension,,}"
    
    for ext in "${VIDEO_EXTENSIONS[@]}"; do
        if [[ "$extension" == "$ext" ]]; then
            return 0
        fi
    done
    return 1
}

# Extract frames from video
extract_frames() {
    local VIDEO_PATH="$1"
    local FPS="${2:-1}"
    
    if [ ! -f "$VIDEO_PATH" ]; then
        log_error "File '$VIDEO_PATH' does not exist."
        return 1
    fi
    
    local VIDEO_DIR
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
    local VIDEO_FILENAME
    VIDEO_FILENAME=$(basename "$VIDEO_PATH")
    local VIDEO_NAME="${VIDEO_FILENAME%.*}"
    
    local OUTPUT_DIR="$VIDEO_DIR/${VIDEO_NAME}_frames"
    mkdir -p "$OUTPUT_DIR"
    chmod 755 "$OUTPUT_DIR"
    
    log_info "Extracting frames from '$VIDEO_PATH'..."
    
    if ffmpeg -hide_banner -i "$VIDEO_PATH" -vf "fps=$FPS" "$OUTPUT_DIR/${VIDEO_NAME}_%04d.png" 2>/dev/null; then
        log_success "Frames saved to: $OUTPUT_DIR"
    else
        log_error "Failed to extract frames from '$VIDEO_PATH'"
        return 1
    fi
}

# Process directory or file
main() {
    local PATH="$1"
    local FPS="${2:-1}"
    
    if [ -z "$PATH" ]; then
        echo "Usage: $0 <video_path|directory> [fps]"
        exit 1
    fi
    
    if [ ! -e "$PATH" ]; then
        log_error "Path '$PATH' does not exist."
        exit 1
    fi
    
    if ! command -v ffmpeg >/dev/null 2>&1; then
        log_error "ffmpeg is not installed."
        exit 1
    fi
    
    if [ -d "$PATH" ]; then
        log_info "Processing directory: $PATH"
        while IFS= read -r -d '' file; do
            if is_video_file "$file"; then
                extract_frames "$file" "$FPS"
            fi
        done < <(find "$PATH" -type f -print0)
    elif [ -f "$PATH" ]; then
        if is_video_file "$PATH"; then
            extract_frames "$PATH" "$FPS"
        else
            log_error "File '$PATH' is not a supported video format."
            exit 1
        fi
    fi
}

main "$@"
