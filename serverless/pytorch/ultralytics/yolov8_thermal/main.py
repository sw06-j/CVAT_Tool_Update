import json
import base64
import io
import os
import time
from PIL import Image
from ultralytics import YOLO

# ==============================================================================
# 🔥 열화상 YOLOv8 전용 자동 라벨러 (Dynamic Hot-Reloading 지원)
# ==============================================================================

def get_or_reload_model(context):
    """model_weights.pt 파일의 수정 시간을 감지하여 실시간 핫리로드"""
    weights_candidates = [
        "/opt/nuclio/model_weights.pt",
        "/model_weights.pt",
        "model_weights.pt",
        "/opt/nuclio/yolov8n.pt",
        "yolov8n.pt"
    ]
    
    target_weights = "yolov8n.pt"
    for wpath in weights_candidates:
        if os.path.exists(wpath):
            target_weights = wpath
            break

    mtime = os.path.getmtime(target_weights) if os.path.exists(target_weights) else 0
    current_mtime = getattr(context.user_data, "model_mtime", -1)

    if not hasattr(context.user_data, "model") or current_mtime != mtime:
        context.logger.info(f"🔄 [가중치 로드] 최신 모델 적용: {target_weights} (수정시간: {mtime})")
        context.user_data.model = YOLO(target_weights)
        context.user_data.model_mtime = mtime

    return context.user_data.model


def init_context(context):
    context.logger.info("🔥 YOLOv8 열화상 전용 모델 초기화 중...")
    get_or_reload_model(context)
    context.logger.info("✅ YOLOv8 모델 준비 완료!")


def handler(context, event):
    context.logger.info("==========================================")
    context.logger.info("📸 [CVAT 🪄] 자동 사전 라벨링 추론 요청 수신!")
    data = event.body

    # 최신 가중치 모델 가져오기 (실시간 재학습 반영)
    model = get_or_reload_model(context)

    # Decode image from CVAT request
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    w, h = image.size

    threshold = float(data.get("threshold", 0.25))
    results = model.predict(image, conf=threshold, verbose=False)
    
    encoded_results = []
    if len(results) > 0:
        res = results[0]
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # CVAT spec label: class_0, class_1, etc.
            cls_name = f"class_{cls_id}"
            
            # CVAT Pixel coordinates [xtl, ytl, xbr, ybr] bounded by width/height
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            xtl = max(0.0, float(xyxy[0]))
            ytl = max(0.0, float(xyxy[1]))
            xbr = min(float(w), float(xyxy[2]))
            ybr = min(float(h), float(xyxy[3]))
            
            encoded_results.append({
                "confidence": str(round(conf, 3)),
                "label": cls_name,
                "points": [xtl, ytl, xbr, ybr],
                "type": "rectangle"
            })
            
    context.logger.info(f"🎯 추론 완료! 감지 객체 개수: {len(encoded_results)}")
    context.logger.info("==========================================")
            
    return context.Response(
        body=json.dumps(encoded_results),
        headers={},
        content_type="application/json",
        status_code=200
    )
