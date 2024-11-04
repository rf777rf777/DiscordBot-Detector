from typing import List

class Yolov8DetectionItem:
    def __init__(self, className: str, confidence: float, box: List[List[float]]):
        self.className = className
        self.confidence = confidence
        self.box = box

class Yolov8DetectionResult:
    def __init__(self, detections: List[Yolov8DetectionItem]):
        self.detections = detections
