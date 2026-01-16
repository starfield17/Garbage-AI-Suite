"""Export adapters."""

from typing import Dict
from pathlib import Path

import torch
from ultralytics import YOLO
from garbage_train.application.ports import ExporterPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class ONNXExporter(ExporterPort):
    def __init__(self, config: Dict):
        self.imgsz = config.get("imgsz", 640)
        self.half = config.get("half", False)
        self.simplify = config.get("simplify", True)
        self.dynamic = config.get("dynamic", False)
        self.opset = config.get("opset", 12)

    def export_to_onnx(self, model_path: Path, config: Dict) -> Dict:
        model = YOLO(str(model_path))

        export_path = model_path.with_suffix(".onnx")

        log.info("Exporting to ONNX", input=str(model_path), output=str(export_path))

        model.export(
            format="onnx",
            imgsz=self.imgsz,
            half=self.half,
            simplify=self.simplify,
            dynamic=self.dynamic,
            opset=self.opset,
        )

        if export_path.exists():
            log.info("ONNX export completed", path=str(export_path))
            return {"success": True, "output_path": str(export_path)}
        else:
            return {"success": False, "error": "ONNX export failed"}


class TorchScriptExporter(ExporterPort):
    def export_to_torchscript(self, model_path: Path, config: Dict) -> Dict:
        model = YOLO(str(model_path))

        export_path = model_path.with_suffix(".torchscript")

        log.info("Exporting to TorchScript", input=str(model_path), output=str(export_path))

        try:
            model.export(
                format="torchscript",
                imgsz=config.get("imgsz", 640),
            )
        except Exception as e:
            log.error("TorchScript export failed", error=str(e))
            return {"success": False, "error": str(e)}

        if export_path.exists():
            log.info("TorchScript export completed", path=str(export_path))
            return {"success": True, "output_path": str(export_path)}
        else:
            return {"success": False, "error": "TorchScript export failed"}


class RKNNExporter(ExporterPort):
    def export_to_rknn(self, model_path: Path, config: Dict) -> Dict:
        onnx_path = model_path.with_suffix(".onnx")

        log.info("Exporting to RKNN", input=str(model_path), output=str(model_path))

        try:
            from rknnlite.api import RKNNLite

            rknn = RKNNLite(verbose=True)
            ret = rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=config.get("target_platform", "rk3588"),
                quantized_dtype="asymmetric_quantized-u8",
                optimization_level=3,
            )

            ret = rknn.load_onnx(model=str(onnx_path))

            ret = rknn.build(do_quantization=True, dataset="./dataset.txt")

            rknn_path = model_path.with_suffix(".rknn")
            ret = rknn.export_rknn(rknn_path)

            if ret == 0:
                log.info("RKNN export completed", path=str(rknn_path))
                return {"success": True, "output_path": str(rknn_path)}
            else:
                return {"success": False, "error": f"RKNN export failed: {ret}"}

        except ImportError:
            log.warning("RKNN not available, skipping RKNN export")
            return {"success": False, "error": "RKNN runtime not installed"}
