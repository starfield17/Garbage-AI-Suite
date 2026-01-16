"""Application use cases."""

from garbage_autolabel.application.ports import ModelAdapterPort, LabelWriterPort
from garbage_autolabel.domain.atoms import filter_by_confidence, clip_bbox_to_image, map_class_names
from garbage_autolabel.infrastructure.model_adapters import YOLOAdapter, FasterRCNNAdapter, QwenVLAdapter
from garbage_autolabel.infrastructure.label_formats import BBoxWriter, COCOWriter, YOLOWriter
from garbage_autolabel.infrastructure.storage import DatasetScanner
from garbage_shared.config_loader import ConfigLoader
from garbage_shared.observability import get_logger
from garbage_shared.contracts_models import BBoxLabelDTO

log = get_logger(__name__)


class AutoLabelDatasetUseCase:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.scanner = DatasetScanner()

    def execute(
        self,
        config_path,
        model_type,
        input_dir,
        output_dir,
        viz_dir=None,
        output_format="bbox",
        confidence_threshold=0.5,
    ) -> dict:
        try:
            log.info(
                "Starting auto-labeling",
                model_type=model_type,
                input_dir=str(input_dir),
                output_dir=str(output_dir),
            )

            config = self.config_loader.load_yaml(config_path)

            adapter = self._get_adapter(model_type, config)
            writer = self._get_writer(output_format)

            images = self.scanner.list_images(input_dir)

            stats = {"total": len(images), "processed": 0, "failed": 0}

            for image_path in images:
                try:
                    self._process_image(image_path, adapter, writer, confidence_threshold, viz_dir, stats)
                    stats["processed"] += 1
                except Exception as e:
                    log.error("Processing failed", image=str(image_path), error=str(e))
                    stats["failed"] += 1

            result = {
                "success": True,
                "stats": stats,
            }
            log.info("Auto-labeling completed", result=result)
            return result

        except Exception as e:
            log.error("Auto-labeling failed", error=str(e))
            return {"success": False, "error": str(e)}

    def _get_adapter(self, model_type: str, config: dict) -> ModelAdapterPort:
        adapters = {
            "yolo": YOLOAdapter(config),
            "faster_rcnn": FasterRCNNAdapter(config),
            "qwen_vl": QwenVLAdapter(config),
        }
        if model_type not in adapters:
            raise ValueError(f"Unknown model type: {model_type}")
        return adapters[model_type]

    def _get_writer(self, output_format: str) -> LabelWriterPort:
        writers = {
            "bbox": BBoxWriter(),
            "coco": COCOWriter(),
            "yolo": YOLOWriter(),
        }
        return writers[output_format]

    def _process_image(
        self, image_path, adapter, writer, confidence_threshold, viz_dir, stats
    ) -> None:
        detections = adapter.predict(image_path)

        detections = filter_by_confidence(detections, confidence_threshold)
        detections = clip_bbox_to_image(detections, *adapter.get_image_size(image_path))
        detections = map_class_names(detections, adapter.get_class_map())

        label_dto = BBoxLabelDTO(
            labels=[
                {
                    "name": det["class_name"],
                    "x1": det["bbox"]["x1"],
                    "y1": det["bbox"]["y1"],
                    "x2": det["bbox"]["x2"],
                    "y2": det["bbox"]["y2"],
                    "confidence": det["confidence"],
                }
                for det in detections
            ]
        )

        writer.write(image_path, label_dto)

        if viz_dir:
            self._save_viz(image_path, detections, viz_dir)

    def _save_viz(self, image_path, detections, viz_dir):
        import cv2

        image = cv2.imread(str(image_path))
        for det in detections:
            x1, y1, x2, y2 = int(det["bbox"]["x1"]), int(det["bbox"]["y1"]), int(det["bbox"]["x2"]), int(det["bbox"]["y2"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{det['class_name']}: {det['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        output_path = viz_dir / image_path.name
        cv2.imwrite(str(output_path), image)
