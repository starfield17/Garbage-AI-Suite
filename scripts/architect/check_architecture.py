#!/usr/bin/env python3

import sys
import lib.util

from garbage_shared.config_loader import ConfigLoader
from garbage_shared.contracts_models import ModelManifestDTO, ClassInfoDTO, ModelFileDTO
from datetime import datetime

def check_architecture():
    errors = []

    for root, dirs, files in [
        ("shared", ["src/garbage_shared"]),
        ("train", ["src/garbage_train"]),
        ("deploy", ["src/garbage_deploy"]),
        ("autolabel", ["src/garbage_autolabel"]),
    ]:
        allowed_layers = {"entry", "application", "domain", "infrastructure"}
        root_path = f"{dirs}/src/garbage_{dirs}"

        for dirpath in Path(root_path).rglob("**/*.py"):
            relative = str(dirpath.relative_to(root_path))
            parts = relative.split("/")

            if len(parts) > 4:
                layer = parts[2]
                if layer not in allowed_layers:
                    errors.append(
                        f"INVALID: {relative} - Layer '{layer}' not allowed in {dirs} module"
                    )
                continue

            if len(parts) == 4 and parts[1] in allowed_layers:
                layer = parts[1]
                if layer not in allowed_layers:
                    errors.append(
                        f"INVALID: {relative} - Entry layer '{layer}' not allowed in {dirs} module"
                    )

                elif len(parts) >= 4:
                    if parts[1] == "domain":
                        for dep in parts[2:]:
                            if dep in allowed_layers and dep != parts[1]:
                                errors.append(
                                    f"INVALID: {relative} - Domain '{dep}' cannot import infrastructure (violates unidirectional deps)"
                                )

                    elif parts[1] == "infrastructure":
                        for dep in parts[2:]:
                            if dep in allowed_layers:
                                errors.append(
                                    f"INVALID: {relative} - Infrastructure '{dep}' not allowed in {dirs} module"
                                )

        if errors:
            print("Architecture violations found:", file=sys.stderr)
            for error in errors:
                print(f"  ✗ {error}")
            sys.exit(1)
        else:
            print("All architecture checks passed!")
            sys.exit(0)


if __name__ == "__main__":
    check_architecture()
