import json
import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ==============================================================================
# 🎯 열화상 시계열 자동 추적 - CVAT Tracker 타입
#
# CVAT AI Tools > Trackers 탭에서 사용합니다.
# N1 프레임에서 사각형을 그린 뒤 Track 버튼을 누르면
# N2 프레임까지 프레임마다 이 핸들러가 호출되어 사각형을 자동 추적합니다.
#
# 동작 원리:
#   - 이전 프레임 사각형(shapes) → Lucas-Kanade Optical Flow → 현재 프레임 위치 추정
#   - YOLO IOU 보정으로 정밀도 향상
#   - CLAHE 열화상 전처리로 감지 성능 극대화
# ==============================================================================

# 상태 저장: job_id별로 이전 프레임 캐시 (worker 재사용 대비)
_state_cache = {}  # { job_id: { "prev_gray": ..., "prev_boxes": [...] } }


def normalize_shapes(shapes):
    """
    CVAT Tracker는 shapes를 두 가지 형태로 전달할 수 있음:
      - dict: {"type": "rectangle", "points": [x1,y1,x2,y2], "label": "...", ...}
      - list: [x1, y1, x2, y2]  ← 좌표만 바로 전달
    두 형태를 모두 dict 형태로 통일합니다.
    """
    normalized = []
    for s in shapes:
        if isinstance(s, list):
            normalized.append({
                "type": "rectangle",
                "label": "class_0",
                "points": s,
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "attributes": []
            })
        elif isinstance(s, dict):
            # points가 없으면 건너뜀
            if "points" not in s and len(s) == 0:
                continue
            normalized.append(s)
        else:
            continue
    return normalized


def apply_clahe_thermal(img_np_rgb):
    """열화상 CLAHE 전처리"""
    lab = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)


def optical_flow_track(prev_gray, curr_gray, boxes):
    """
    Lucas-Kanade Optical Flow로 이전 프레임 사각형 → 현재 프레임 위치 추적
    """
    if prev_gray is None or len(boxes) == 0:
        return boxes

    tracked = []
    lk_params = dict(
        winSize=(25, 25), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    for box in boxes:
        pts_in = box.get("points", [])
        if len(pts_in) < 4:
            tracked.append(box)
            continue

        xtl, ytl, xbr, ybr = pts_in[0], pts_in[1], pts_in[2], pts_in[3]
        bw, bh = xbr - xtl, ybr - ytl
        cx, cy = (xtl + xbr) / 2.0, (ytl + ybr) / 2.0

        # 9개 추적 포인트 (중심 + 코너 + 엣지 중간)
        track_pts = np.array([
            [cx, cy],
            [xtl, ytl], [xbr, ytl], [xtl, ybr], [xbr, ybr],
            [cx, ytl], [cx, ybr], [xtl, cy], [xbr, cy]
        ], dtype=np.float32).reshape(-1, 1, 2)

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, track_pts, None, **lk_params
        )

        good_new = new_pts[status.ravel() == 1].reshape(-1, 2)
        good_old = track_pts[status.ravel() == 1].reshape(-1, 2)

        if len(good_new) < 2:
            tracked.append(box)
            continue

        deltas = good_new - good_old
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))

        # 크기 변화 추정 (±15% 제한)
        if len(good_new) >= 4:
            sx = float(np.clip(np.std(good_new[:, 0]) / (np.std(good_old[:, 0]) + 1e-6), 0.85, 1.15))
            sy = float(np.clip(np.std(good_new[:, 1]) / (np.std(good_old[:, 1]) + 1e-6), 0.85, 1.15))
        else:
            sx, sy = 1.0, 1.0

        new_cx, new_cy = cx + dx, cy + dy
        new_bw, new_bh = bw * sx, bh * sy

        h, w = curr_gray.shape
        new_xtl = max(0.0, new_cx - new_bw / 2)
        new_ytl = max(0.0, new_cy - new_bh / 2)
        new_xbr = min(float(w), new_xtl + new_bw)
        new_ybr = min(float(h), new_ytl + new_bh)

        tracked.append({
            **box,
            "points": [round(new_xtl, 2), round(new_ytl, 2),
                       round(new_xbr, 2), round(new_ybr, 2)]
        })

    return tracked


def yolo_iou_refine(model, curr_np, tracked_boxes, threshold):
    """YOLO 감지 결과로 Optical Flow 결과 보정 (IOU 매칭)"""
    results = model.predict(curr_np, conf=threshold, verbose=False)
    if not results or not results[0].boxes:
        return tracked_boxes

    yolo_boxes = []
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        yolo_boxes.append({"xyxy": xyxy, "conf": float(box.conf[0].item())})

    refined = []
    for tbox in tracked_boxes:
        tp = tbox["points"]
        best_iou, best_yb = 0.0, None
        for yb in yolo_boxes:
            yp = yb["xyxy"]
            ix1, iy1 = max(tp[0], yp[0]), max(tp[1], yp[1])
            ix2, iy2 = min(tp[2], yp[2]), min(tp[3], yp[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (tp[2]-tp[0])*(tp[3]-tp[1]) + (yp[2]-yp[0])*(yp[3]-yp[1]) - inter
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou, best_yb = iou, yb

        if best_yb and best_iou > 0.25:
            h, w = curr_np.shape[:2]
            yp = best_yb["xyxy"]
            refined.append({
                **tbox,
                "points": [
                    round(max(0.0, yp[0]), 2), round(max(0.0, yp[1]), 2),
                    round(min(float(w), yp[2]), 2), round(min(float(h), yp[3]), 2)
                ]
            })
        else:
            refined.append(tbox)

    return refined


def init_context(context):
    context.logger.info("🎯 열화상 시계열 Tracker 모델 초기화 중...")

    weights_path = "/opt/nuclio/model_weights.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolov8m.pt"
        context.logger.info(f"커스텀 가중치 없음 → 기본 yolov8m 사용")

    context.user_data.model = YOLO(weights_path)
    context.logger.info(f"✅ Tracker 모델 준비 완료! (가중치: {weights_path})")


def handler(context, event):
    context.logger.info("🎯 [Tracker] 프레임 추적 요청 수신")

    data = event.body
    threshold = float(data.get("threshold", 0.3))
    job_id = str(data.get("job_id", data.get("job", "default")))

    # 현재 프레임 이미지 디코딩 + CLAHE 전처리
    buf = io.BytesIO(base64.b64decode(data["image"]))
    pil_img = Image.open(buf).convert("RGB")
    img_np = np.array(pil_img)
    img_np = apply_clahe_thermal(img_np)
    curr_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = curr_gray.shape

    # CVAT tracker 프로토콜:
    #   - 초기화: shapes에 실제 shape 데이터 전달
    #   - 추적:   shapes에 [null, null, ...] 전달 (states만 의미 있음)
    # → None을 제거해서 실제 shape 여부 판단
    raw_shapes = data.get("shapes", [])
    raw_shapes_valid = [s for s in raw_shapes if s is not None]

    if raw_shapes_valid:
        prev_shapes = normalize_shapes(raw_shapes_valid)
        context.logger.info(f"📌 [초기화] {len(prev_shapes)}개 shapes 수신")
    else:
        cached = _state_cache.get(job_id, {})
        prev_shapes = cached.get("prev_shapes", [])
        context.logger.info(f"🔄 [추적] 캐시에서 {len(prev_shapes)}개 shapes 로드")

    # job별 이전 프레임 상태 불러오기
    state = _state_cache.get(job_id, {"prev_gray": None})
    prev_gray = state.get("prev_gray")

    if not prev_shapes:
        _state_cache[job_id] = {"prev_gray": curr_gray, "prev_shapes": []}
        return context.Response(
            body=json.dumps({"shapes": [], "states": []}),
            headers={},
            content_type="application/json",
            status_code=200
        )

    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        context.logger.info("📌 [첫 프레임] YOLO 초기 감지 수행")
        refined = yolo_iou_refine(context.user_data.model, img_np, prev_shapes, threshold)
    else:
        context.logger.info(f"🔄 [연속 프레임] Optical Flow + YOLO 보정")
        tracked = optical_flow_track(prev_gray, curr_gray, prev_shapes)
        refined = yolo_iou_refine(context.user_data.model, img_np, tracked, threshold)

    # 상태 업데이트
    _state_cache[job_id] = {"prev_gray": curr_gray, "prev_shapes": refined}

    output = []
    for s in refined:
        pts = s.get("points", [])
        if len(pts) < 4:
            output.append({
                "type": "rectangle",
                "label": s.get("label", "class_0"),
                "points": [0.0, 0.0, 1.0, 1.0],
                "occluded": s.get("occluded", False),
                "outside": False,
                "z_order": 0,
                "attributes": []
            })
            continue
        output.append({
            "type": "rectangle",
            "label": s.get("label", "class_0"),
            "points": [
                max(0.0, pts[0]), max(0.0, pts[1]),
                min(float(w), pts[2]), min(float(h), pts[3])
            ],
            "occluded": s.get("occluded", False),
            "outside": False,
            "z_order": 0,
            "attributes": []
        })

    # 각 shape마다 빈 state 생성 (CVAT tracker 프로토콜 요구사항)
    states = [{} for _ in output]

    context.logger.info(f"✅ {len(output)}개 사각형 추적 완료")

    return context.Response(
        body=json.dumps({"shapes": output, "states": states}),
        headers={},
        content_type="application/json",
        status_code=200
    )

