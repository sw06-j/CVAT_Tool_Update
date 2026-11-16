import os
import sys
import base64
import json
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import requests
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 🎯 CVAT 전용 N1 ~ N2 시계열 범위 자동 라벨러
# ============================================================
def _load_dotenv_if_needed():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv_if_needed()

CVAT_SERVER = os.environ.get("CVAT_SERVER", "http://localhost:8080")
CVAT_USER = os.environ.get("CVAT_USER", "")
CVAT_PASS = os.environ.get("CVAT_PASS", "")

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "serverless", "pytorch", "ultralytics", "yolov8_thermal", "model_weights.pt")


def get_cvat_session():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    res = session.post(f"{CVAT_SERVER}/api/auth/login", json={"username": CVAT_USER, "password": CVAT_PASS})
    if res.status_code == 200:
        token = res.json().get("key", "")
        if token:
            session.headers["Authorization"] = f"Token {token}"
        csrf = session.cookies.get("csrftoken", "")
        if csrf:
            session.headers["X-CSRFToken"] = csrf
        return session
    return None


def get_frame_np(session, task_id, frame_idx):
    res = session.get(
        f"{CVAT_SERVER}/api/tasks/{task_id}/data",
        params={"type": "frame", "quality": "original", "number": frame_idx}
    )
    if res.status_code == 200:
        pil_img = Image.open(BytesIO(res.content)).convert("RGB")
        return np.array(pil_img)
    return None


def optical_flow_track(prev_gray, curr_gray, boxes):
    if prev_gray is None or len(boxes) == 0:
        return boxes

    tracked = []
    lk_params = dict(winSize=(25, 25), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    for box in boxes:
        pts = box["points"]
        xtl, ytl, xbr, ybr = pts[0], pts[1], pts[2], pts[3]
        bw, bh = xbr - xtl, ybr - ytl
        cx, cy = (xtl + xbr) / 2.0, (ytl + ybr) / 2.0

        track_pts = np.array([
            [cx, cy], [xtl, ytl], [xbr, ytl], [xtl, ybr], [xbr, ybr],
            [cx, ytl], [cx, ybr], [xtl, cy], [xbr, cy]
        ], dtype=np.float32).reshape(-1, 1, 2)

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, track_pts, None, **lk_params)
        good_new = new_pts[status.ravel() == 1].reshape(-1, 2)
        good_old = track_pts[status.ravel() == 1].reshape(-1, 2)

        if len(good_new) < 2:
            tracked.append(box)
            continue

        deltas = good_new - good_old
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))

        new_cx, new_cy = cx + dx, cy + dy
        h, w = curr_gray.shape

        new_xtl = max(0.0, new_cx - bw / 2.0)
        new_ytl = max(0.0, new_cy - bh / 2.0)
        new_xbr = min(float(w), new_xtl + bw)
        new_ybr = min(float(h), new_ytl + bh)

        tracked.append({
            "label_id": box["label_id"],
            "points": [round(new_xtl, 2), round(new_ytl, 2), round(new_xbr, 2), round(new_ybr, 2)]
        })

    return tracked


def run_range_annotation(task_id, n1, n2, conf_thresh=0.25):
    session = get_cvat_session()
    if not session:
        print("❌ CVAT 로그인 실패")
        return

    # Job ID 조회
    job_res = session.get(f"{CVAT_SERVER}/api/jobs?task_id={task_id}&page_size=10")
    jobs = job_res.json().get("results", [])
    if not jobs:
        print("❌ 해당 Task에 등록된 Job을 찾을 수 없습니다.")
        return
    job_id = jobs[0]["id"]

    # 라벨 목록
    lbl_res = session.get(f"{CVAT_SERVER}/api/labels?task_id={task_id}&page_size=100")
    task_labels = lbl_res.json().get("results", []) if lbl_res.status_code == 200 else []
    first_label_id = task_labels[0]["id"] if task_labels else 1

    # 기존 n1 프레임 라벨이 있는지 확인
    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations")
    existing_shapes = anno_res.json().get("shapes", []) if anno_res.status_code == 200 else []
    
    n1_shapes = [s for s in existing_shapes if s.get("frame") == n1 and s.get("type") == "rectangle"]

    weights = WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else "yolov8n.pt"
    print(f"📦 사용 가중치: {weights}")
    model = YOLO(weights)

    current_boxes = []
    if n1_shapes:
        print(f"📌 [N{n1} 프레임] 사용자가 그린 기존 박스 {len(n1_shapes)}개 발견 → 시계열 추적 시작점으로 사용")
        for s in n1_shapes:
            current_boxes.append({"label_id": s["label_id"], "points": s["points"]})
    else:
        print(f"🤖 [N{n1} 프레임] 기존 박스 없음 → YOLO 모델로 초기 객체 자동 감지")
        n1_img = get_frame_np(session, task_id, n1)
        if n1_img is not None:
            results = model.predict(n1_img, conf=conf_thresh, verbose=False)
            if results and len(results[0].boxes) > 0:
                for b in results[0].boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    current_boxes.append({"label_id": first_label_id, "points": [round(x, 2) for x in xyxy]})
                print(f"  ✅ N{n1}에서 {len(current_boxes)}개 객체 자동 감지됨")

    if not current_boxes:
        print(f"⚠️ N{n1} 프레임에 추적할 객체가 없습니다.")
        return

    # N1 ~ N2 시계열 추적 루프
    all_new_shapes = []
    prev_gray = None

    print(f"\n🚀 N{n1} ~ N{n2} 프레임 시계열 연속 라벨링 진행 중 (총 {n2-n1+1}프레임)...")

    for f_idx in range(n1, n2 + 1):
        curr_np = get_frame_np(session, task_id, f_idx)
        if curr_np is None:
            continue
        curr_gray = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)

        if f_idx == n1:
            # N1 프레임 박스 저장
            tracked_boxes = current_boxes
        else:
            # Optical flow 추적
            tracked_boxes = optical_flow_track(prev_gray, curr_gray, current_boxes)

        prev_gray = curr_gray
        current_boxes = tracked_boxes

        for b in tracked_boxes:
            all_new_shapes.append({
                "type": "rectangle",
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "frame": f_idx,
                "label_id": b["label_id"],
                "group": 0,
                "source": "auto",
                "attributes": [],
                "points": b["points"]
            })

    # CVAT 서버에 일괄 주입 (PATCH)
    print(f"\n📦 총 {len(all_new_shapes)}개 시계열 BBox를 CVAT Job #{job_id}에 저장 중...")
    patch_res = session.patch(
        f"{CVAT_SERVER}/api/jobs/{job_id}/annotations",
        params={"action": "create"},
        json={"version": 0, "tags": [], "shapes": all_new_shapes, "tracks": []}
    )

    if patch_res.status_code in (200, 201):
        print(f"🎉 [성공] N{n1} ~ N{n2} 프레임 시계열 자동 라벨링 완료! CVAT에서 새로고침하여 확인하세요.\n")
    else:
        print(f"❌ 저장 실패: {patch_res.status_code}\n{patch_res.text[:300]}")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        task_id = int(sys.argv[1])
        n1 = int(sys.argv[2])
        n2 = int(sys.argv[3])
        run_range_annotation(task_id, n1, n2)
    else:
        print("사용법: python temporal_range_labeler.py <task_id> <n1> <n2>")
