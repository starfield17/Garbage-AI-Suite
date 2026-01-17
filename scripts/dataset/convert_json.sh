#!/bin/bash

# ==============================================================================
# JSON Format Conversion Script
# Converts label data to YOLO/COCO formats
# Usage: ./convert_json.sh <input_json> <output_format> [output_dir]
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Convert to YOLO format
convert_to_yolo() {
    local INPUT_JSON="$1"
    local OUTPUT_DIR="${2:-./yolo_output}"
    
    mkdir -p "$OUTPUT_DIR"
    
    log_info "Converting '$INPUT_JSON' to YOLO format..."
    
    python3 << EOF
import json
import sys
from pathlib import Path

input_file = "$INPUT_JSON"
output_dir = "$OUTPUT_DIR"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Category mapping for YOLO format
category_map = {
    "Kitchen_waste": 0,
    "Recyclable_waste": 1,
    "Hazardous_waste": 2,
    "Other_waste": 3
}

for image in data.get('images', []):
    image_id = image['id']
    file_name = image['file_name']
    
    yolo_labels = []
    
    for annotation in data.get('annotations', []):
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']
            category_id = annotation['category_id']
            
            # Convert COCO bbox (x, y, w, h) to YOLO (x_center, y_center, w, h)
            x_center = (bbox[0] + bbox[2] / 2) / image['width']
            y_center = (bbox[1] + bbox[3] / 2) / image['height']
            w = bbox[2] / image['width']
            h = bbox[3] / image['height']
            
            class_id = category_map.get(category_id, 0)
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    if yolo_labels:
        output_file = Path(output_dir) / f"{Path(file_name).stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_labels))
        print(f"Created: {output_file}")

print("Conversion complete!")
EOF

    log_success "YOLO format labels saved to: $OUTPUT_DIR"
}

# Main
main() {
    if [ $# -lt 2 ]; then
        echo "Usage: $0 <input_json> <yolo|coco> [output_dir]"
        exit 1
    fi
    
    local INPUT_JSON="$1"
    local FORMAT="$2"
    local OUTPUT_DIR="${3:-./output}"
    
    if [ ! -f "$INPUT_JSON" ]; then
        log_error "Input file '$INPUT_JSON' does not exist."
        exit 1
    fi
    
    case "$FORMAT" in
        yolo|YOLO)
            convert_to_yolo "$INPUT_JSON" "$OUTPUT_DIR"
            ;;
        *)
            log_error "Unsupported format: $FORMAT"
            exit 1
            ;;
    esac
}

main "$@"
