#!/usr/bin/env python3
"""
Architecture validation script for Garbage-AI-Suite.

Validates that the project follows the unidirectional dependency rule:
- Entry → Application → Domain → Infrastructure
- Infrastructure cannot import Application or Domain
- Cross-module imports are forbidden (shared is allowed)
"""

import sys
from pathlib import Path

from garbage_shared.config_loader import ConfigLoader
from garbage_shared.contracts_models import ModelManifestDTO, ClassInfoDTO, ModelFileDTO


def check_architecture():
    """Check architecture constraints across all modules."""
    errors = []

    modules = [
        ("shared", "src/garbage_shared"),
        ("train", "src/garbage_train"),
        ("deploy", "src/garbage_deploy"),
        ("autolabel", "src/garbage_autolabel"),
    ]

    for module_name, module_path in modules:
        root_path = Path(module_path)

        if not root_path.exists():
            continue

        for file_path in root_path.rglob("**/*.py"):
            relative = file_path.relative_to(root_path)
            parts = relative.parts

            if len(parts) < 2:
                continue

            # Check layer structure
            layer = parts[0]
            allowed_layers = {"entry", "application", "domain", "infrastructure"}

            if layer not in allowed_layers:
                # Check if this is a nested structure issue
                if len(parts) >= 2:
                    potential_layer = parts[1]
                    if potential_layer in allowed_layers:
                        errors.append(
                            f"INVALID: {relative} - File in wrong directory structure"
                        )
                continue

            # Check imports in this file
            file_content = file_path.read_text(encoding="utf-8")
            imports = extract_imports(file_content)

            # Domain cannot import Infrastructure or Application
            if layer == "domain":
                for imp in imports:
                    if "infrastructure" in imp or "application" in imp:
                        if "garbage_shared" not in imp:  # shared is allowed
                            errors.append(
                                f"INVALID: {relative} - Domain cannot import {imp}"
                            )

            # Infrastructure cannot import Application or Domain
            if layer == "infrastructure":
                for imp in imports:
                    if (
                        imp.startswith("garbage_train.application")
                        or imp.startswith("garbage_train.domain")
                        or imp.startswith("garbage_deploy.application")
                        or imp.startswith("garbage_deploy.domain")
                        or imp.startswith("garbage_autolabel.application")
                        or imp.startswith("garbage_autolabel.domain")
                    ):
                        errors.append(
                            f"INVALID: {relative} - Infrastructure cannot import {imp}"
                        )

            # Cross-module imports are forbidden (except shared)
            if layer in {"entry", "application", "domain", "infrastructure"}:
                for imp in imports:
                    # Check for cross-module imports
                    if imp.startswith("garbage_train") and "garbage_shared" not in imp:
                        if "train" not in module_name and "shared" not in imp:
                            errors.append(
                                f"INVALID: {relative} - Cross-module import: {imp}"
                            )
                    if imp.startswith("garbage_deploy") and "garbage_shared" not in imp:
                        if "deploy" not in module_name and "shared" not in imp:
                            errors.append(
                                f"INVALID: {relative} - Cross-module import: {imp}"
                            )
                    if (
                        imp.startswith("garbage_autolabel")
                        and "garbage_shared" not in imp
                    ):
                        if "autolabel" not in module_name and "shared" not in imp:
                            errors.append(
                                f"INVALID: {relative} - Cross-module import: {imp}"
                            )

    if errors:
        print("Architecture violations found:", file=sys.stderr)
        for error in errors[:20]:  # Limit output
            print(f"  ✗ {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors", file=sys.stderr)
        sys.exit(1)
    else:
        print("✓ All architecture checks passed!")
        sys.exit(0)


def extract_imports(content: str) -> list[str]:
    """Extract all import statements from Python content."""
    imports = []
    lines = content.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("import ") and " as " not in line:
            module = line.replace("import ", "").split(",")[0].strip()
            imports.append(module)
        elif line.startswith("from ") and " import " in line:
            parts = line.split(" import ")
            module = parts[0].replace("from ", "").strip()
            imports.append(module)

    return imports


if __name__ == "__main__":
    check_architecture()
