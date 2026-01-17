"""VLM检测引擎"""

import base64
import cv2
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

from .i_detection_engine import IDetectionEngine
from autolabel_context.domain.model.value_object.engine_type import EngineType


class VlmEngine(IDetectionEngine):
    """VLM检测引擎实现（使用OpenAI兼容API）"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化VLM引擎
        
        Args:
            config: 引擎配置，包含api_key, base_url, model等
        """
        self._config = config
        self._api_key = config.get("api_key")
        self._base_url = config.get("base_url", "https://api.openai.com/v1")
        self._model = config.get("model", "gpt-4o")
        self._confidence_threshold = config.get("confidence_threshold", 0.5)
        self._max_retries = config.get("max_retries", 3)
        
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url
        )
        
        self._prompt = self._build_prompt()
    
    def _build_prompt(self) -> str:
        """构建VLM提示词"""
        return (
            "Please identify objects in the image and classify them into one of the four main waste categories: "
            "**Kitchen_waste**, **Recyclable_waste**, **Hazardous_waste**, or **Other_waste**.\n\n"
            "Provide the following information for each detection:\n"
            "1. name: Must be exactly one of 'Kitchen_waste', 'Recyclable_waste', 'Hazardous_waste', or 'Other_waste'.\n"
            "2. Bounding box coordinates (x1, y1, x2, y2).\n"
            "3. Detection confidence score.\n\n"
            "Return the results strictly in JSON format:\n"
            "```json\n"
            "{\n"
            "  \"labels\": [\n"
            "    {\"name\": \"Kitchen_waste\", \"x1\": 100, \"y1\": 200, \"x2\": 300, \"y2\": 400, \"confidence\": 0.95}\n"
            "  ]\n"
            "}\n"
            "```"
        )
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.VLM
    
    def validate(self) -> bool:
        """验证API是否可用"""
        try:
            return self._api_key is not None and len(self._api_key) > 0
        except Exception:
            return False
    
    def detect(self, image_path: str) -> List[dict]:
        """检测图像中的对象"""
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        img_height, img_width = img.shape[:2]
        
        _, buffer = cv2.imencode('.jpg', img)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        detections = []
        retries = 0
        
        while retries <= self._max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }]
                )
                
                response_text = response.choices[0].message.content
                detections = self._parse_response(response_text)
                break
                
            except Exception:
                retries += 1
        
        valid_detections = []
        for det in detections:
            confidence = det.get("confidence", 0)
            if confidence >= self._confidence_threshold:
                x1 = max(0, min(det.get("x1", 0), img_width))
                y1 = max(0, min(det.get("y1", 0), img_height))
                x2 = max(0, min(det.get("x2", 0), img_width))
                y2 = max(0, min(det.get("y2", 0), img_height))
                
                if x2 > x1 and y2 > y1:
                    valid_detections.append({
                        "name": det.get("name"),
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
        
        return valid_detections
    
    def _parse_response(self, response_text: str) -> List[dict]:
        """解析API响应"""
        detections = []
        
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                data = json.loads(json_str)
                labels = data.get("labels", [])
                for label in labels:
                    detections.append({
                        "name": label.get("name"),
                        "x1": label.get("x1", 0),
                        "y1": label.get("y1", 0),
                        "x2": label.get("x2", 0),
                        "y2": label.get("y2", 0),
                        "confidence": label.get("confidence", 0.0)
                    })
            except json.JSONDecodeError:
                pass
        
        return detections
