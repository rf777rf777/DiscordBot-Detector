import torch, io
from modules.models.yolov8.Yolov8DetectionResult import Yolov8DetectionItem, Yolov8DetectionResult
from ultralytics import YOLO
from PIL import Image

# check if you can use gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def __detect(contents: io.BytesIO, conf: float):
    image = Image.open(contents).convert('RGB')
    
    # Load the YOLO model
    model = YOLO("modules/models/yolov8/yolov8n.pt").to(device)
    results = model.predict(image, conf=conf) 

    if device == 'cuda':
        # 確保數據被從 GPU 移動到 CPU
        result = results[0].cpu()

        # 刪除模型以釋放資源
        del model    
        # 清理 GPU 緩存
        torch.cuda.empty_cache()
    else:
        result = results[0]

    return  result 

def get_detect_result_info(contents: io.BytesIO, conf=0.5) -> Yolov8DetectionResult:
    result = __detect(contents, conf)
    detections = []
    for box in result.boxes:
        detections.append(
            Yolov8DetectionItem(
                className=result.names[int(box.cls)], 
                confidence=float(box.conf), 
                box=box.xywh.tolist()))
    return Yolov8DetectionResult(detections=detections)