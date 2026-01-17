# 部署文档

## 目录

- [快速开始](#快速开始)
- [Docker 部署](#docker-部署)
- [原生部署](#原生部署)
- [设备配置](#设备配置)
- [系统服务](#系统服务)
- [故障排除](#故障排除)

---

## 快速开始

### 使用 Docker

```bash
# 构建所有镜像
docker build -f docker/train.Dockerfile -t garbage-ai:train .
docker build -f docker/deploy.Dockerfile -t garbage-ai:deploy .
docker build -f docker/autolabel.Dockerfile -t garbage-ai:autolabel .

# 或使用 docker-compose
docker-compose up --build
```

### 使用 Docker Compose

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## Docker 部署

### 训练镜像

```bash
docker build -f docker/train.Dockerfile -t garbage-ai:train .

# 运行训练
docker run --rm \
    -v $(pwd)/models:/data/models \
    -v $(pwd)/datasets:/data/datasets \
    -v $(pwd)/config:/data/config \
    -e CUDA_VISIBLE_DEVICES=0 \
    garbage-ai:train train --model yolo --data datasets/
```

### 部署镜像

```bash
docker build -f docker/deploy.Dockerfile -t garbage-ai:deploy .

# 运行部署服务
docker run --rm \
    -v $(pwd)/models:/data/models \
    -v $(pwd)/config:/data/config \
    -p 8000:8000 \
    --device /dev/ttyUSB0:/dev/ttyUSB0 \
    garbage-ai:deploy deploy run --profile rk3588
```

### 自动标注镜像

```bash
docker build -f docker/autolabel.Dockerfile -t garbage-ai:autolabel .

# 运行自动标注
docker run --rm \
    -v $(pwd)/images:/data/images \
    -v $(pwd)/labels:/data/labels \
    -v $(pwd)/models:/data/models \
    -v $(pwd)/config:/data/config \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    garbage-ai:autolabel autolabel run --engine vlm --images /data/images/
```

---

## 原生部署

### 环境要求

- Python 3.10+
- CUDA 11.8+ (可选，用于 GPU 加速)
- FFmpeg (用于视频帧提取)

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装共享内核
pip install -e shared/

# 安装训练模块
pip install -e train/

# 安装部署模块
pip install -e deploy/

# 安装自动标注模块
pip install -e autolabel/
```

### 配置环境

```bash
# 复制配置模板
cp config/taxonomy/waste_categories.yaml.example config/taxonomy/waste_categories.yaml
cp config/mappings/train_class_map.yaml.example config/mappings/train_class_map.yaml

# 编辑配置
vim config/taxonomy/waste_categories.yaml
```

---

## 设备配置

### RK3588 NPU 配置

```bash
# 运行 NPU 配置脚本
sudo ./scripts/device/rk3588/npu_config.sh

# 监控 NPU 状态
./scripts/device/rk3588/npu_monitor.sh
```

### 串口设备权限

```bash
# 添加用户到 dialout 组
sudo usermod -a -G dialout $USER

# 重启后生效
```

---

## 系统服务

### 安装系统服务

```bash
# 安装训练服务
sudo ./scripts/device/systemd/install_service.sh myenv /path/to/train_script.py

# 卸载服务
sudo ./scripts/device/systemd/uninstall_service.sh /path/to/train_script.py
```

### 管理服务

```bash
# 查看状态
systemctl status <service_name>

# 查看日志
journalctl -u <service_name> -f

# 重启服务
sudo systemctl restart <service_name>

# 停止服务
sudo systemctl stop <service_name>
```

---

## 故障排除

### Docker 构建问题

```bash
# 清理 Docker 缓存
docker system prune -a

# 重新构建
docker build --no-cache -f docker/train.Dockerfile -t garbage-ai:train .
```

### GPU 访问问题

```bash
# 验证 NVIDIA 驱动
nvidia-smi

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 串口权限问题

```bash
# 检查串口设备
ls -la /dev/ttyUSB*

# 临时获取权限
sudo chmod 666 /dev/ttyUSB0

# 永久配置 (udev 规则)
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/99-serial.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 性能优化

```bash
# Docker 内存限制
docker run -m 4g ...

# CPU 核心限制
docker run --cpus=4 ...

# NPU 设备访问
docker run --device /dev/dri:/dev/dri ...
```
