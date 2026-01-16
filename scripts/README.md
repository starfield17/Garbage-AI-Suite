# Scripts Documentation

This directory contains utility scripts for system operations, device management, and data conversion.

## Directory Structure

```
scripts/
├── sh/                    # Shell scripts for system operations
├── python_tools/          # Python utility tools
├── sys/                   # System administration scripts
├── devices/               # Device-specific scripts
└── architect/             # Architecture validation
```

## Shell Scripts (`sh/`)

### get_frame.sh

Capture a single frame from camera device.

```bash
./sh/get_frame.sh [device] [output_file]

# Example
./sh/get_frame.sh /dev/video0 frame.jpg
```

**Parameters**:
- `device`: Camera device path (default: /dev/video0)
- `output_file`: Output image path (default: frame.jpg)

### uart-manager.sh

Manage UART serial connections.

```bash
./sh/uart-manager.sh [port] [baud_rate]

# Example - start monitoring
./sh/uart-manager.sh /dev/ttyUSB0 115200

# Example - send test command
./sh/uart-manager.sh /dev/ttyUSB0 115200 "TX"
```

**Parameters**:
- `port`: Serial port device path
- `baud_rate`: Baud rate (default: 115200)

**Commands**:
- `TX`: Transmit test pattern
- `RX`: Receive mode
- `status`: Check connection status

## Python Tools (`python_tools/`)

### json-converter.py

Convert between different JSON label formats (COCO, YOLO, bbox).

```bash
python python_tools/json-converter.py --input input.json --output output.json --from-format coco --to-format yolo

# Convert COCO to YOLO
python python_tools/json-converter.py -i labels_coco.json -o labels_yolo --from coco --to yolo

# Convert YOLO to bbox
python python_tools/json-converter.py -i labels_yolo.json -o labels_bbox --from yolo --to bbox

# Convert bbox to COCO
python python_tools/json-converter.py -i labels_bbox.json -o labels_coco --from bbox --to coco
```

**Options**:
- `-i, --input`: Input file path
- `-o, --output`: Output file path
- `--from`: Input format (coco, yolo, bbox)
- `--to`: Output format (coco, yolo, bbox)
- `--class-map`: Path to class mapping YAML (optional)

### convert_mapping.py

Convert and validate class mapping configurations.

```bash
# Convert YAML mapping to JSON
python python_tools/convert_mapping.py --input class_map.yaml --output class_map.json

# Validate existing mapping
python python_tools/convert_mapping.py --validate class_map.yaml

# Generate mapping from dataset
python python_tools/convert_mapping.py --from-dataset /path/to/images --output mapping.yaml
```

**Options**:
- `-i, --input`: Input file (YAML/JSON)
- `-o, --output`: Output file
- `--validate`: Validate existing mapping file
- `--from-dataset`: Auto-generate from dataset folder

## System Scripts (`sys/`)

### add_to_systemd.sh

Register a service with systemd.

```bash
# Register garbage detection service
./sys/add_to_systemd.sh garbage-detect /path/to/garbage-deploy --deploy-id raspi_yolo_v12n

# Register with auto-restart
./sys/add_to_systemd.sh garbage-detect /path/to/garbage-deploy --deploy-id raspi_yolo_v12n --restart always

# Remove service
./sys/add_to_systemd.sh garbage-detect --remove
```

**Options**:
- `name`: Service name
- `command`: Executable path with arguments
- `--user`: Run as user service (no root)
- `--restart`: Restart policy (always, on-failure, never)
- `--remove`: Remove existing service

### add_to_systemd_bin.sh

Install the systemd service wrapper binary.

```bash
# Install wrapper
./sys/add_to_systemd_bin.sh --install

# Verify installation
./sys/add_to_systemd_bin.sh --verify
```

### change_pip_conda_source.sh

Switch between pip/conda package sources.

```bash
# Switch to Tsinghua mirror
./sys/change_pip_conda_source.sh --mirror tsinghua

# Switch to official sources
./sys/change_pip_conda_source.sh --mirror official

# List available mirrors
./sys/change_pip_conda_source.sh --list
```

**Options**:
- `--mirror`: Mirror to use (tsinghua, aliyun, official)
- `--pip-only`: Only change pip source
- `--conda-only`: Only change conda source
- `--list`: Show available mirrors

### proxy.sh

Configure system proxy settings.

```bash
# Set HTTP/HTTPS proxy
./sys/proxy.sh set http://proxy.example.com:8080

# Set SOCKS5 proxy
./sys/proxy.sh set socks5 http://proxy.example.com:1080

# Show current proxy settings
./sys/proxy.sh show

# Clear proxy settings
./sys/proxy.sh clear
```

### rescan.sh

Rescan for connected devices (cameras, serial ports).

```bash
# Scan all devices
./sys/rescan.sh

# Scan specific device type
./sys/rescan.sh --cameras
./sys/rescan.sh --serial

# Verbose output
./sys/rescan.sh -v

# JSON output for scripting
./sys/rescan.sh --json
```

## Device Scripts (`devices/`)

### rk3588/RK3588_npu_config.sh

Configure NPU settings for RK3588 device.

```bash
# Set performance mode
./devices/rk3588/RK3588_npu_config.sh performance

# Set power-saving mode
./devices/rk3588/RK3588_npu_config.sh power

# Query NPU status
./devices/rk3588/RK3588_npu_config.sh status

# Reset NPU
./devices/rk3588/RK3588_npu_config.sh reset
```

### rk3588/npu_monitor.sh

Monitor NPU utilization and temperature.

```bash
# Monitor in real-time
./devices/rk3588/npu_monitor.sh

# Log to file
./devices/rk3588/npu_monitor.sh --log npu.log

# JSON output
./devices/rk3588/npu_monitor.sh --json
```

## Architecture Validation (`architect/`)

### check_architecture.py

Validate that the project follows architectural rules.

```bash
# Run full validation
python architect/check_architecture.py

# Check specific module
python architect/check_architecture.py --module train

# Check import boundaries
python architect/check_architecture.py --imports

# Generate report
python architect/check_architecture.py --report report.txt
```

**What it checks**:
- Import boundary violations
- Unidirectional dependency flow
- Port implementations match interfaces
- Contract schemas are valid JSON
- Configuration files are valid YAML

**Exit codes**:
- 0: All checks passed
- 1: One or more checks failed
- 2: Invalid arguments or internal error

## Usage Tips

### Running from Any Directory

Add to your `.bashrc` or `.zshrc`:

```bash
export PATH="/path/to/Garbage-AI-Suite/scripts/sh:$PATH"
export PATH="/path/to/Garbage-AI-Suite/scripts/python_tools:$PATH"
```

### Making Scripts Executable

```bash
chmod +x scripts/sh/*.sh
chmod +x scripts/sys/*.sh
chmod +x scripts/devices/**/*.sh
```

### Combining with Cron

Schedule regular tasks:

```bash
# Add to crontab
crontab -e

# Run device scan every hour
0 * * * * /path/to/scripts/sys/rescan.sh --json > /var/log/devices.json

# Monitor NPU daily
0 0 * * * /path/to/scripts/devices/rk3588/npu_monitor.sh --log /var/log/npu-$(date +%Y%m%d).log
```

## Script Development Guidelines

When adding new scripts:

1. **Location**: Put scripts in appropriate subdirectory
2. **Documentation**: Add entry to this file
3. **Executable**: Run `chmod +x` on shell scripts
4. **Error Handling**: Include proper error codes and messages
5. **Dependencies**: Document required tools/packages
6. **Testing**: Test on target systems before committing

## Troubleshooting

### Permission Denied

```bash
# Make script executable
chmod +x script.sh

# Or run with bash
bash script.sh
```

### Missing Dependencies

```bash
# Install required packages
pip install -r scripts/requirements.txt

# Or use conda
conda install --file scripts/requirements.txt
```

### Command Not Found

Ensure scripts directory is in your PATH (see above).
