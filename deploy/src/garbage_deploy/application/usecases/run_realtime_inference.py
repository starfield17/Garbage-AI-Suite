"""Application use cases."""

from garbage_deploy.application.ports import RuntimePort, CameraPort, SerialPort, ClockPort
from garbage_deploy.domain.atoms import (
    DetectionState,
    check_detection_stability,
    can_count_new_garbage,
    map_serial_payload,
)
from garbage_deploy.infrastructure.runtime_adapters import TorchRuntimeAdapter
from garbage_deploy.infrastructure.io import OpenCVCameraAdapter, PySerialAdapter, SystemClockAdapter
from garbage_shared.config_loader import ConfigLoader
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class RunRealtimeInferenceUseCase:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.clock = SystemClockAdapter()
        self.runtime = None
        self.camera = None
        self.serial = None

    def execute(
        self,
        deploy_id: str,
        manifest_path,
        dry_run: bool = False,
        debug_window: bool = False,
    ) -> dict:
        try:
            log.info(
                "Starting deployment",
                deploy_id=deploy_id,
                dry_run=dry_run,
            )

            deploy_profile = self.config_loader.load_yaml(f"configs/registry/deploy_profiles.yaml")
            profile = deploy_profile["deploy_profiles"][deploy_id]

            manifest = self.config_loader.load_yaml(manifest_path)

            self._initialize_components(profile, manifest, dry_run)
            result = self._run_inference_loop(profile, dry_run, debug_window)

            self._cleanup()

            return {"success": True, "total_detections": result}

        except Exception as e:
            log.error("Deployment failed", error=str(e))
            self._cleanup()
            return {"success": False, "error": str(e)}

    def _initialize_components(self, profile, manifest, dry_run):
        runtime_config = {
            "model_path": profile.get("model_path", manifest.get("files", [{}])[0].get("path", "")),
            "device": profile.get("device_type", "cpu"),
            "confidence_threshold": profile.get("confidence_threshold", 0.7),
            "iou_threshold": profile.get("iou_threshold", 0.45),
        }

        self.runtime = TorchRuntimeAdapter(runtime_config)
        self.runtime.load_model(manifest_path, runtime_config["device"])

        if not dry_run:
            camera_config = {
                "device_index": profile["camera"].get("device_index", 0),
                "width": profile["camera"].get("width", 640),
                "height": profile["camera"].get("height", 480),
                "crop_region": profile["camera"].get("crop_region"),
            }
            self.camera = OpenCVCameraAdapter(camera_config)
            self.camera.read()

            if profile["serial"].get("enabled", True):
                serial_config = {
                    "port": profile["serial"].get("port", "/dev/ttyUSB0"),
                    "baud": profile["serial"].get("baud", 115200),
                }
                self.serial = PySerialAdapter(serial_config)
                self.serial.open()

        log.info("Components initialized")

    def _run_inference_loop(self, profile, dry_run, debug_window):
        state = DetectionState()
        total_detections = 0
        frame_count = 0

        stability_config = profile.get("stability", {})
        min_position_frames = stability_config.get("min_position_frames", 5)
        min_confidence_frames = stability_config.get("min_confidence_frames", 3)
        cooldown_ms = stability_config.get("cooldown_ms", 2000)

        try:
            while True:
                if dry_run:
                    log.info("Dry run mode - would process frame", frame=frame_count)
                    frame_count += 1
                    break

                frame = self.camera.read()
                if frame is None:
                    log.warning("Frame read failed")
                    break

                detections = self.runtime.infer(frame)

                for det in detections:
                    stable, _ = check_detection_stability(
                        state,
                        det["class_id"],
                        self.clock.now(),
                        min_position_frames,
                        min_confidence_frames,
                    )

                    if stable:
                        can_count, _ = can_count_new_garbage(
                            state,
                            det["class_id"],
                            self.clock.now(),
                            cooldown_ms,
                        )

                        if can_count:
                            class_id, x, y = map_serial_payload(
                                det, self.camera.width, self.camera.height
                            )

                            payload = bytes([class_id, x, y])
                            self.serial.send(payload)
                            total_detections += 1
                            log.info(
                                "Detection sent",
                                class_id=class_id,
                                x=x,
                                y=y,
                            )

                frame_count += 1

                if debug_window:
                    self._render_debug_frame(frame, detections)

                if frame_count % 100 == 0:
                    log.info("Processed frames", count=frame_count, detections=total_detections)

        except KeyboardInterrupt:
            log.info("Interrupted by user")
            return total_detections

    def _render_debug_frame(self, frame, detections):
        import cv2

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{det['class_id']}: {det['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.imshow("Debug", frame)
        cv2.waitKey(1)

    def _cleanup(self):
        if self.camera:
            self.camera.release()
        if self.serial:
            self.serial.close()
        cv2.destroyAllWindows()
        log.info("Cleanup completed")
