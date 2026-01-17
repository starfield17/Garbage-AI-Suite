#!/usr/bin/env python3
# train/src/train_context/api/cli/main.py
"""Train CLI 入口"""

import argparse
import sys
from pathlib import Path

from ...application.command.start_training_cmd import StartTrainingCmd
from ...application.command.export_artifact_cmd import ExportArtifactCmd
from ...application.command.convert_dataset_cmd import ConvertDatasetCmd
from ...application.handler.start_training_handler import StartTrainingHandler
from ...application.handler.export_artifact_handler import ExportArtifactHandler
from ...application.handler.convert_dataset_handler import ConvertDatasetHandler


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Train - 训练工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    start_parser = subparsers.add_parser("start", help="开始训练")
    start_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="模型类型 (yolo, faster_rcnn)"
    )
    start_parser.add_argument(
        "--variant", "-v",
        type=str,
        default="n",
        help="模型变体 (如 n, s, m, l, x)"
    )
    start_parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="数据集路径"
    )
    start_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="训练轮数"
    )
    start_parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="批次大小"
    )
    start_parser.add_argument(
        "--lr", "-l",
        type=float,
        default=0.01,
        help="学习率"
    )
    start_parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出目录"
    )
    start_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="训练设备 (auto, cpu, cuda)"
    )
    
    export_parser = subparsers.add_parser("export", help="导出模型")
    export_parser.add_argument(
        "--run-id", "-r",
        type=str,
        required=True,
        help="运行 ID"
    )
    export_parser.add_argument(
        "--format", "-f",
        type=str,
        default="onnx",
        choices=["pt", "onnx", "rknn"],
        help="导出格式"
    )
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出路径"
    )
    
    convert_parser = subparsers.add_parser("convert", help="转换数据集格式")
    convert_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入路径"
    )
    convert_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出路径"
    )
    convert_parser.add_argument(
        "--from", "-f",
        type=str,
        choices=["yolo", "coco", "voc"],
        required=True,
        help="源格式"
    )
    convert_parser.add_argument(
        "--to", "-t",
        type=str,
        choices=["yolo", "coco", "voc"],
        required=True,
        help="目标格式"
    )
    
    list_parser = subparsers.add_parser("list", help="列出训练记录")
    
    args = parser.parse_args()
    
    if args.command == "start":
        handle_start(args)
    elif args.command == "export":
        handle_export(args)
    elif args.command == "convert":
        handle_convert(args)
    elif args.command == "list":
        handle_list(args)
    else:
        parser.print_help()
        sys.exit(1)


def handle_start(args):
    """处理开始训练"""
    handler = StartTrainingHandler()
    
    command = StartTrainingCmd(
        model_family=args.model.lower(),
        model_variant=args.variant,
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        output_dir=args.output,
        device=args.device
    )
    
    print(f"Starting training with {args.model}...")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}")
    
    result = handler.handle(command)
    
    print(f"\n{'='*50}")
    print(f"Training Result")
    print(f"{'='*50}")
    print(f"Run ID: {result.run_id}")
    print(f"Status: {result.status}")
    print(f"Success: {result.success}")
    
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Best Model: {result.best_model_path}")
    
    sys.exit(0 if result.success else 1)


def handle_export(args):
    """处理导出"""
    handler = ExportArtifactHandler()
    
    command = ExportArtifactCmd(
        run_id=args.run_id,
        format=args.format,
        output_path=args.output
    )
    
    result = handler.handle(command)
    
    print(f"\n{'='*50}")
    print(f"Export Result")
    print(f"{'='*50}")
    print(f"Success: {result.success}")
    
    if result.success:
        print(f"Original: {result.original_path}")
        print(f"Exported: {result.exported_path}")
    else:
        print(f"Error: {result.error}")
    
    sys.exit(0 if result.success else 1)


def handle_convert(args):
    """处理转换"""
    handler = ConvertDatasetHandler()
    
    command = ConvertDatasetCmd(
        input_path=args.input,
        output_path=args.output,
        source_format=args.from_,
        target_format=args.to
    )
    
    result = handler.handle(command)
    
    print(f"\n{'='*50}")
    print(f"Convert Result")
    print(f"{'='*50}")
    print(f"Success: {result.success}")
    
    if not result.success:
        print(f"Error: {result.error}")
    
    sys.exit(0 if result.success else 1)


def handle_list(args):
    """处理列出记录"""
    print("Listing training records...")
    print("(Feature to be implemented)")


if __name__ == "__main__":
    main()
