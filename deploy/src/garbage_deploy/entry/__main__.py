"""Entry points for deploy module."""

import argparse
import sys
from pathlib import Path

from garbage_deploy.application.usecases import RunRealtimeInferenceUseCase


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time garbage detection and deployment")
    parser.add_argument("--deploy-id", type=str, required=True, help="Deploy profile ID")
    parser.add_argument("--manifest", type=str, required=True, help="Path to model manifest.json")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (no camera/serial)")
    parser.add_argument("--debug-window", action="store_true", help="Enable debug window")
    return parser.parse_args()


def main():
    args = parse_args()

    use_case = RunRealtimeInferenceUseCase()

    result = use_case.execute(
        deploy_id=args.deploy_id,
        manifest_path=Path(args.manifest),
        dry_run=args.dry_run,
        debug_window=args.debug_window,
    )

    if result["success"]:
        print("Deployment completed")
        print(f"Detections: {result['total_detections']}")
        sys.exit(0)
    else:
        print(f"Deployment failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
