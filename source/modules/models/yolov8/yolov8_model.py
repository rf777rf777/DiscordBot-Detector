import io
import onnxruntime as ort
import numpy as np
from PIL import Image
from modules.models.yolov8.Yolov8DetectionResult import Yolov8DetectionItem, Yolov8DetectionResult

session = ort.InferenceSession("modules/models/yolov8/yolov8n.onnx")
input_name = session.get_inputs()[0].name

def __detect(contents: io.BytesIO, conf: float):
    image = Image.open(contents).convert('RGB')
    image = image.resize((640, 640))
    img_np = np.array(image).astype(np.float32)
    img_np = img_np / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    outputs = session.run(None, {input_name: img_np})
    result = outputs[0][0]
    
    detections = []
    for box in result:
        if box[4] >= conf:  #Confidence threshold
            x_center, y_center, width, height = box[:4]
            class_id = int(box[5])
            confidence = float(box[4])
            detections.append(
                Yolov8DetectionItem(
                    className=str(class_id),  
                    confidence=confidence,
                    box=[[x_center, y_center, width, height]]
                )
            )
    return detections

def get_detect_result_info(contents: io.BytesIO, conf=0.5) -> Yolov8DetectionResult:
    detections = __detect(contents, conf)
    return Yolov8DetectionResult(detections=detections)
