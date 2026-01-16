"""Qwen-VL VLM adapter."""

from pathlib import Path
from typing import List, Dict
import os
import time

import cv2
from openai import OpenAI

from garbage_autolabel.application.ports import ModelAdapterPort
from garbage_shared.observability import get_logger

log = get_logger(__name__)


class QwenVLAdapter(ModelAdapterPort):
    def __init__(self, config: dict):
        self.api_key = os.environ.get("VLM_API_KEY", "")
        self.model = config.get("model", "qwen-vl-max")
        self.base_url = config.get("base_url", "")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

        if not self.api_key:
            raise ValueError("VLM_API_KEY environment variable not set")

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def predict(self, image_path: Path) -> List[Dict]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        _, buffer = cv2.imencode(".jpg", image)
        import base64

        base64_image = base64.b64encode(buffer).decode("utf-8")

        prompt = """Identify all waste objects in this image and provide bounding boxes.
Format your response as JSON with this structure:
{
  "detections": [
    {
      "class_name": "object_name",
      "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
      "confidence": 0.95
    }
  ]
}
Classes: Kitchen_waste (potato, daikon, carrot), Recyclable_waste (bottle, can), Hazardous_waste (battery, drug), Other_waste (tile, stone, brick)."""

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            ],
                        }
                    ],
                    )

                result = self._parse_response(response)
                return result

            except Exception as e:
                if attempt < self.max_retries - 1:
                    log.warning(
                        "API call failed, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    log.error("API call failed after retries", error=str(e))
                    raise

    def _parse_response(self, response) -> List[Dict]:
        content = response.choices[0].message.content
        import json

        try:
            data = json.loads(content)
            return data.get("detections", [])
        except json.JSONDecodeError:
            log.error("Failed to parse VLM response", content=content)
            return []

    def get_class_map(self) -> Dict[str, int]:
        return {
            "Kitchen_waste": 0,
            "Recyclable_waste": 1,
            "Hazardous_waste": 2,
            "Other_waste": 3,
        }

    def get_image_size(self, image_path: Path) -> tuple[int, int]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        return width, height
