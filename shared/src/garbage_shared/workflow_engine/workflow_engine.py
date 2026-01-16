"""Workflow engine for YAML-based orchestration."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

from garbage_shared.config_loader import ConfigLoader
from garbage_shared.observability import get_logger

log = get_logger(__name__)


@dataclass
class WorkflowStep:
    """A single workflow step definition."""

    name: str
    handler: str
    params: dict[str, Any]
    on_error: str | None = None


class WorkflowEngine:
    """Execute workflows defined in YAML files."""

    def __init__(self, config_base_path: Path | str | None = None):
        self.config_loader = ConfigLoader(config_base_path)
        self.step_registry: dict[str, Callable] = {}
        self.logger = get_logger(__name__)

    def register_step(self, name: str, handler: Callable) -> None:
        """Register a step handler function."""
        self.step_registry[name] = handler
        self.logger.info("Registered workflow step", step=name)

    def load_workflow(self, workflow_path: str | Path) -> dict[str, Any]:
        """Load workflow definition from YAML."""
        workflow = self.config_loader.load_yaml(workflow_path)

        required_keys = ["name", "description", "steps"]
        for key in required_keys:
            if key not in workflow:
                raise ValueError(f"Workflow missing required key: {key}")

        return workflow

    def parse_steps(self, steps_data: list[dict]) -> list[WorkflowStep]:
        """Parse step definitions from YAML data."""
        steps = []
        for step_data in steps_data:
            if "name" not in step_data or "handler" not in step_data:
                raise ValueError(f"Step missing required keys: {step_data}")

            step = WorkflowStep(
                name=step_data["name"],
                handler=step_data["handler"],
                params=step_data.get("params", {}),
                on_error=step_data.get("on_error"),
            )
            steps.append(step)

        return steps

    async def execute_workflow(
        self,
        workflow_path: str | Path,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a workflow from YAML definition."""
        context = context or {}
        workflow = self.load_workflow(workflow_path)
        steps = self.parse_steps(workflow["steps"])

        self.logger.info(
            "Starting workflow",
            workflow=workflow["name"],
            steps_count=len(steps),
        )

        for step in steps:
            await self._execute_step(step, context)

        self.logger.info("Workflow completed", workflow=workflow["name"])
        return context

    async def _execute_step(
        self, step: WorkflowStep, context: dict[str, Any]
    ) -> None:
        """Execute a single workflow step with error handling."""
        self.logger.info(
            "Executing step",
            step=step.name,
            handler=step.handler,
        )

        if step.handler not in self.step_registry:
            raise ValueError(f"Step handler not registered: {step.handler}")

        handler = self.step_registry[step.handler]
        step_context = copy.deepcopy(context)

        try:
            if asyncio and asyncio.iscoroutinefunction(handler):
                result = await handler(**step.params, **step_context)
            else:
                result = handler(**step_params, **step_context)

            if isinstance(result, dict):
                context.update(result)

            self.logger.info("Step completed", step=step.name)

        except Exception as e:
            self.logger.error("Step failed", step=step.name, error=str(e))

            if step.on_error:
                error_handler = self.step_registry.get(step.on_error)
                if error_handler:
                    await error_handler(error=e, context=context, step=step)

            raise


import asyncio
