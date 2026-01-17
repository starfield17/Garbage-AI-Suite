"""CLI入口点"""

import sys
import argparse
import logging
from pathlib import Path

from shared_kernel.config.loader import ConfigLoader
from shared_kernel.utils.logging_setup import setup_logging

from autolabel_context.application.command.run_autolabel_cmd import RunAutoLabelCmd
from autolabel_context.application.handler.run_autolabel_handler import RunAutoLabelHandler
from autolabel_context.infrastructure.persistence.engine_repository_impl import EngineRepositoryImpl
from autolabel_context.infrastructure.persistence.file_label_store import FileLabelStore


def setup_logging_config():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='Auto-label images using YOLO, Faster R-CNN, or VLM engines'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    run_parser = subparsers.add_parser('run', help='Run auto-labeling')
    run_parser.add_argument(
        '--engine', '-e',
        required=True,
        choices=['yolo', 'faster_rcnn', 'vlm'],
        help='Detection engine type'
    )
    run_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing images'
    )
    run_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for label files'
    )
    run_parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    run_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=4,
        help='Batch size for parallel processing (default: 4)'
    )
    run_parser.add_argument(
        '--model-id', '-m',
        help='Specific model ID to use'
    )
    
    list_parser = subparsers.add_parser('list', help='List available engines')
    
    return parser


def cmd_run(args) -> int:
    """处理run命令"""
    config_loader = ConfigLoader()
    
    engine_repo = EngineRepositoryImpl(config_loader)
    label_store = FileLabelStore(args.output)
    handler = RunAutoLabelHandler(engine_repo, label_store, config_loader)
    
    command = RunAutoLabelCmd(
        engine_type=args.engine,
        image_paths=[args.input],
        output_dir=args.output,
        confidence_threshold=args.confidence,
        batch_size=args.batch_size,
        model_id=args.model_id
    )
    
    try:
        result = handler.handle(command)
        
        print(f"\nAuto-labeling completed:")
        print(f"  Job ID: {result.job_id}")
        print(f"  Status: {result.status}")
        print(f"  Statistics:")
        print(f"    - Total images: {result.total_images}")
        print(f"    - Processed: {result.processed_images}")
        print(f"    - Skipped: {result.skipped_images}")
        print(f"    - Failed: {result.failed_images}")
        print(f"    - Total detections: {result.total_detections}")
        print(f"    - Success rate: {result.success_rate:.2%}")
        
        return 0
        
    except Exception as e:
        logging.exception("Error running auto-labeling")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list(args) -> int:
    """处理list命令"""
    print("Available engines:")
    print("  - yolo: YOLO detection (requires model file)")
    print("  - faster_rcnn: Faster R-CNN detection (requires model file)")
    print("  - vlm: Vision Language Model (requires API key)")
    return 0


def main() -> int:
    """主入口点"""
    setup_logging_config()
    
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'run':
        return cmd_run(args)
    elif args.command == 'list':
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
