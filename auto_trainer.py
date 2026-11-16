import os
import re
import sys
import json
import time
import glob
import base64
import shutil
import threading
import requests
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import urllib.parse
from ultralytics import YOLO

# Unbuffered stdout for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 🔥 CVAT 다중 모델 & 클래스 병합/분리 제어 엔진
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "active_learning_dataset")
EVAL_GT_DIR = os.path.join(BASE_DIR, "eval_gt_datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_DIR = os.path.join(BASE_DIR, "serverless", "pytorch", "ultralytics", "yolov8_thermal")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model_weights.pt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(EVAL_GT_DIR, exist_ok=True)

# 초기 모델 파일 보장
if os.path.exists(WEIGHTS_PATH) and not os.path.exists(os.path.join(MODELS_DIR, "IR8_model.pt")):
    shutil.copy2(WEIGHTS_PATH, os.path.join(MODELS_DIR, "IR8_model.pt"))

base_yolo = os.path.join(os.path.dirname(BASE_DIR), "yolov8n.pt")
if os.path.exists(base_yolo) and not os.path.exists(os.path.join(MODELS_DIR, "yolov8n_base.pt")):
    shutil.copy2(base_yolo, os.path.join(MODELS_DIR, "yolov8n_base.pt"))

ACTIVE_CONFIG_FILE = os.path.join(MODELS_DIR, "active_model.txt")
training_lock = threading.Lock()
is_training = False
training_progress_msg = ""


def get_active_model_name():
    if os.path.exists(ACTIVE_CONFIG_FILE):
        try:
            with open(ACTIVE_CONFIG_FILE, "r", encoding="utf-8") as f:
                name = f.read().strip()
                if name and os.path.exists(os.path.join(MODELS_DIR, f"{name}.pt")):
                    return name
        except Exception:
            pass
    if os.path.exists(os.path.join(MODELS_DIR, "IR8_model.pt")):
        return "IR8_model"
    models = list_available_models()
    return models[0]["name"] if models else "IR8_model"


def set_active_model_name(name):
    name = name.strip()
    target_path = os.path.join(MODELS_DIR, f"{name}.pt")
    if not os.path.exists(target_path):
        return False, f"모델 파일이 존재하지 않습니다: {name}.pt"
    with open(ACTIVE_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(name)
    shutil.copy2(target_path, WEIGHTS_PATH)
    print(f"🎯 [활성 모델 변경] -> {name} (경로: {target_path})", flush=True)
    return True, f"활성 모델이 '{name}'(으)로 변경되었습니다."


def list_available_models():
    models = []
    active_name = get_active_model_name()
    for pt in glob.glob(os.path.join(MODELS_DIR, "*.pt")):
        bname = os.path.splitext(os.path.basename(pt))[0]
        size_mb = round(os.path.getsize(pt) / (1024 * 1024), 2)
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(pt)))
        models.append({
            "name": bname,
            "filename": os.path.basename(pt),
            "size_mb": size_mb,
            "modified": mtime,
            "is_active": (bname == active_name)
        })
    models.sort(key=lambda x: x["name"])
    return models


def create_new_model(new_name, base_model_name="IR8_model"):
    new_name = new_name.strip().replace(" ", "_")
    if not new_name:
        return False, "모델 이름을 입력해주세요."
    
    target_file = os.path.join(MODELS_DIR, f"{new_name}.pt")
    if os.path.exists(target_file):
        return False, f"이미 존재하는 모델 이름입니다: {new_name}"

    base_file = os.path.join(MODELS_DIR, f"{base_model_name}.pt")
    if not os.path.exists(base_file):
        base_file = WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else "yolov8n.pt"

    shutil.copy2(base_file, target_file)
    set_active_model_name(new_name)
    print(f"✨ [새 모델 생성] '{new_name}.pt' (기반: {base_model_name}) 생성 및 활성화 완료!", flush=True)
    return True, f"새 모델 '{new_name}'이(가) 성공적으로 생성 및 활성화되었습니다."


def get_active_weights_path():
    active_name = get_active_model_name()
    target_path = os.path.join(MODELS_DIR, f"{active_name}.pt")
    if os.path.exists(target_path):
        return target_path
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
    return "yolov8n.pt"


def get_cvat_session():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    try:
        res = session.post(f"{CVAT_SERVER}/api/auth/login", json={"username": CVAT_USER, "password": CVAT_PASS}, timeout=10)
        if res.status_code == 200:
            token = res.json().get("key", "")
            if token:
                session.headers["Authorization"] = f"Token {token}"
            csrf = session.cookies.get("csrftoken", "")
            if csrf:
                session.headers["X-CSRFToken"] = csrf
            return session
    except Exception as e:
        print(f"⚠️ [CVAT 세션 오류] {e}", flush=True)
    return None


def get_frame_np(session, task_id, frame_idx):
    res = session.get(
        f"{CVAT_SERVER}/api/tasks/{task_id}/data",
        params={"type": "frame", "quality": "original", "number": frame_idx},
        timeout=15
    )
    if res.status_code == 200:
        arr = np.frombuffer(res.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None


def sanitize_annotations_for_put(anno_data):
    """CVAT PUT /annotations는 전면 교체(새로 생성)로 처리되므로 id 필드가 없어야 유효함"""
    cleaned = {
        "version": anno_data.get("version", 0),
        "tags": [],
        "shapes": [],
        "tracks": []
    }
    for tag in anno_data.get("tags", []):
        t = dict(tag)
        t.pop("id", None)
        cleaned["tags"].append(t)

    for shape in anno_data.get("shapes", []):
        s = dict(shape)
        s.pop("id", None)
        cleaned["shapes"].append(s)

    for track in anno_data.get("tracks", []):
        t = dict(track)
        t.pop("id", None)
        if "shapes" in t:
            clean_trk_shapes = []
            for ts in t["shapes"]:
                ts_dict = dict(ts)
                ts_dict.pop("id", None)
                clean_trk_shapes.append(ts_dict)
            t["shapes"] = clean_trk_shapes
        cleaned["tracks"].append(t)

    return cleaned


# ============================================================
# 🏷️ 클래스 일괄 병합 & 분리 / 재배치 엔진
# ============================================================
def merge_job_labels(job_id, source_label_id, target_label_id, start_frame=None, stop_frame=None):
    """Job 내의 모든 어노테이션에서 source_label_id를 가진 박스들을 target_label_id로 일괄 치환 (범위 지정 지원)"""
    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    source_label_id = int(source_label_id)
    target_label_id = int(target_label_id)

    if source_label_id == target_label_id:
        return {"error": "출발 라벨과 도착 라벨이 동일합니다."}

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=15)
    if anno_res.status_code != 200:
        return {"error": f"어노테이션 조회 실패: {anno_res.text}"}

    anno_data = anno_res.json()
    shapes = anno_data.get("shapes", [])
    tracks = anno_data.get("tracks", [])

    has_range = (start_frame is not None and stop_frame is not None)
    s_frame = int(start_frame) if has_range else -1
    e_frame = int(stop_frame) if has_range else 999999

    modified_count = 0
    for s in shapes:
        if s.get("label_id") == source_label_id:
            f = s.get("frame", 0)
            if not has_range or (s_frame <= f <= e_frame):
                s["label_id"] = target_label_id
                modified_count += 1

    for t in tracks:
        if t.get("label_id") == source_label_id:
            f = t.get("frame", 0)
            if not has_range or (s_frame <= f <= e_frame):
                t["label_id"] = target_label_id
                modified_count += 1

    if modified_count == 0:
        return {"status": "success", "modified_count": 0, "message": "해당 조건에 일치하는 박스가 없습니다."}

    put_res = session.put(
        f"{CVAT_SERVER}/api/jobs/{job_id}/annotations",
        json=sanitize_annotations_for_put(anno_data),
        timeout=30
    )
    if put_res.status_code not in (200, 201):
        return {"error": f"어노테이션 갱신 실패: {put_res.text}"}

    range_desc = f"프레임 {s_frame}~{e_frame}" if has_range else "전체 프레임"
    print(f"🔄 [클래스 병합 완료] Job #{job_id} ({range_desc}) | 라벨 #{source_label_id} ➔ #{target_label_id} ({modified_count}개 박스 일괄 치환)", flush=True)
    return {
        "status": "success",
        "job_id": job_id,
        "modified_count": modified_count,
        "source_label_id": source_label_id,
        "target_label_id": target_label_id,
        "range": range_desc,
        "message": f"({range_desc}) 총 {modified_count}개의 박스가 새로운 클래스로 성공적으로 병합되었습니다."
    }


def delete_job_label_annotations(job_id, label_id, delete_label_entity=False, start_frame=None, stop_frame=None):
    """Job 내에서 특정 label_id에 속한 모든 사각형 박스를 일괄 삭제하고, 옵션에 따라 라벨 자체도 삭제 (범위 지정 지원)"""
    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    label_id = int(label_id)

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=15)
    if anno_res.status_code != 200:
        return {"error": f"어노테이션 조회 실패: {anno_res.text}"}

    anno_data = anno_res.json()
    shapes = anno_data.get("shapes", [])
    tracks = anno_data.get("tracks", [])

    has_range = (start_frame is not None and stop_frame is not None)
    s_frame = int(start_frame) if has_range else -1
    e_frame = int(stop_frame) if has_range else 999999

    filtered_shapes = []
    deleted_shapes = 0
    for s in shapes:
        if s.get("label_id") == label_id:
            f = s.get("frame", 0)
            if not has_range or (s_frame <= f <= e_frame):
                deleted_shapes += 1
                continue
        filtered_shapes.append(s)

    filtered_tracks = []
    deleted_tracks = 0
    for t in tracks:
        if t.get("label_id") == label_id:
            f = t.get("frame", 0)
            if not has_range or (s_frame <= f <= e_frame):
                deleted_tracks += 1
                continue
        filtered_tracks.append(t)

    deleted_count = deleted_shapes + deleted_tracks

    anno_data["shapes"] = filtered_shapes
    anno_data["tracks"] = filtered_tracks

    put_res = session.put(
        f"{CVAT_SERVER}/api/jobs/{job_id}/annotations",
        json=sanitize_annotations_for_put(anno_data),
        timeout=30
    )
    if put_res.status_code not in (200, 201):
        return {"error": f"어노테이션 갱신 실패: {put_res.text}"}

    label_deleted = False
    if delete_label_entity and not has_range:
        try:
            del_lbl_res = session.delete(f"{CVAT_SERVER}/api/labels/{label_id}", timeout=10)
            if del_lbl_res.status_code in (200, 204):
                label_deleted = True
        except Exception:
            pass

    range_desc = f"프레임 {s_frame}~{e_frame}" if has_range else "전체 프레임"
    print(f"🗑️ [클래스 일괄 삭제 완료] Job #{job_id} ({range_desc}) | 라벨 #{label_id} ({deleted_count}개 박스 삭제, 라벨엔티티삭제: {label_deleted})", flush=True)
    return {
        "status": "success",
        "job_id": job_id,
        "deleted_count": deleted_count,
        "label_id": label_id,
        "label_deleted": label_deleted,
        "range": range_desc,
        "message": f"({range_desc}) 총 {deleted_count}개의 박스가 데이터셋에서 성공적으로 삭제되었습니다."
    }


def split_or_reassign_shapes(job_id, shape_ids, target_label_id=None, new_label_name=None):
    """선택된 shape_ids 박스들을 특정 라벨(또는 새로 생성된 라벨)로 일괄 분리/이동"""
    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    job_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}", timeout=10)
    if job_res.status_code != 200:
        return {"error": f"Job #{job_id} 조회 실패"}
    task_id = job_res.json().get("task_id")
    project_id = job_res.json().get("project_id")

    final_label_id = target_label_id

    # 만약 새 라벨 이름이 주어졌다면 먼저 프로젝트/태스크에 새 라벨 생성
    if new_label_name and new_label_name.strip():
        new_name = new_label_name.strip()
        lbl_query = f"project_id={project_id}" if project_id else f"task_id={task_id}"
        lbl_res = session.get(f"{CVAT_SERVER}/api/labels?{lbl_query}&page_size=100", timeout=10)
        task_labels = lbl_res.json().get("results", []) if lbl_res.status_code == 200 else []
        
        existing_lbl = next((l for l in task_labels if l["name"] == new_name), None)
        if existing_lbl:
            final_label_id = existing_lbl["id"]
        else:
            import random
            rand_color = f"#{random.randint(0, 0xFFFFFF):06x}"
            patch_labels = []
            for l in task_labels:
                patch_labels.append({"id": l["id"], "name": l["name"], "color": l.get("color", "#000000")})
            patch_labels.append({"name": new_name, "color": rand_color, "type": "any"})

            if project_id:
                session.patch(f"{CVAT_SERVER}/api/projects/{project_id}", json={"labels": patch_labels}, timeout=20)
            else:
                session.patch(f"{CVAT_SERVER}/api/tasks/{task_id}", json={"labels": patch_labels}, timeout=20)

            lbl_res2 = session.get(f"{CVAT_SERVER}/api/labels?{lbl_query}&page_size=100", timeout=10)
            if lbl_res2.status_code == 200:
                for l in lbl_res2.json().get("results", []):
                    if l["name"] == new_name:
                        final_label_id = l["id"]
                        break

    if not final_label_id:
        return {"error": "이동할 대상 클래스를 생성하거나 찾지 못했습니다."}


    final_label_id = int(final_label_id)
    shape_id_set = set(int(sid) for sid in shape_ids)

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=15)
    if anno_res.status_code != 200:
        return {"error": f"어노테이션 조회 실패: {anno_res.text}"}

    anno_data = anno_res.json()
    shapes = anno_data.get("shapes", [])

    modified_count = 0
    for s in shapes:
        if s.get("id") in shape_id_set:
            s["label_id"] = final_label_id
            modified_count += 1

    if modified_count == 0:
        return {"error": "일치하는 박스를 찾지 못했습니다."}

    put_res = session.put(
        f"{CVAT_SERVER}/api/jobs/{job_id}/annotations",
        json=sanitize_annotations_for_put(anno_data),
        timeout=30
    )
    if put_res.status_code not in (200, 201):
        return {"error": f"어노테이션 갱신 실패: {put_res.text}"}

    print(f"✂️ [클래스 분리/이동 완료] Job #{job_id} | {modified_count}개 박스 ➔ 라벨 #{final_label_id}", flush=True)
    return {
        "status": "success",
        "job_id": job_id,
        "modified_count": modified_count,
        "target_label_id": final_label_id,
        "message": f"선택된 {modified_count}개의 박스가 새 클래스로 분리 이동되었습니다."
    }



def get_label_shapes_gallery(job_id, label_id=None, start_frame=None, stop_frame=None):
    """특정 라벨에 속한 박스 목록 및 정보 조회 (분리 갤러리용, 범위 필터 지원)"""
    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    job_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}", timeout=10)
    if job_res.status_code != 200:
        return {"error": f"Job #{job_id} 조회 실패"}
    task_id = job_res.json().get("task_id")

    lbl_res = session.get(f"{CVAT_SERVER}/api/labels?task_id={task_id}&page_size=100", timeout=10)
    labels_map = {l["id"]: l["name"] for l in lbl_res.json().get("results", [])} if lbl_res.status_code == 200 else {}

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=15)
    if anno_res.status_code != 200:
        return {"error": "어노테이션 조회 실패"}

    shapes = anno_res.json().get("shapes", [])
    target_lid = int(label_id) if label_id is not None and str(label_id).isdigit() else None

    has_range = (start_frame is not None and stop_frame is not None and str(start_frame).isdigit() and str(stop_frame).isdigit())
    s_frame = int(start_frame) if has_range else -1
    e_frame = int(stop_frame) if has_range else 999999

    results = []
    for s in shapes:
        if s.get("type") != "rectangle":
            continue
        lid = s.get("label_id")
        if target_lid is not None and lid != target_lid:
            continue
        f = s.get("frame", 0)
        if has_range and not (s_frame <= f <= e_frame):
            continue
        
        results.append({
            "id": s.get("id"),
            "frame": f,
            "label_id": lid,
            "label_name": labels_map.get(lid, f"Label_{lid}"),
            "points": [round(float(p), 1) for p in s.get("points", [])]
        })

    results.sort(key=lambda x: (x["frame"], x["id"]))

    return {
        "job_id": job_id,
        "total_count": len(results),
        "shapes": results
    }



# ============================================================
# 🧠 명시적 수동 학습 실행 (On-Demand Training Engine)
# ============================================================
def train_selected_model(target_model_name=None, epochs=5):
    """지정된 모델을 대상으로 CVAT 라벨 데이터 누적 학습 수행"""
    global is_training, training_progress_msg

    if not target_model_name:
        target_model_name = get_active_model_name()

    with training_lock:
        if is_training:
            return {"error": "이미 다른 학습 작업이 진행 중입니다."}
        is_training = True
        training_progress_msg = f"'{target_model_name}' 모델 데이터 수집 및 학습 준비 중..."

    try:
        session = get_cvat_session()
        if not session:
            return {"error": "CVAT 로그인 실패"}

        res = session.get(f"{CVAT_SERVER}/api/tasks?page_size=100", timeout=10)
        if res.status_code != 200:
            return {"error": "CVAT 태스크 목록 조회 실패"}

        tasks = res.json().get("results", [])
        if not tasks:
            return {"error": "학습할 태스크가 존재하지 않습니다."}

        train_img_dir = os.path.join(DATASET_DIR, "train", "images")
        train_lbl_dir = os.path.join(DATASET_DIR, "train", "labels")
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)

        total_labeled_frames = 0
        all_labels_map = {}

        for task in tasks:
            task_id = task["id"]
            lbl_res = session.get(f"{CVAT_SERVER}/api/labels?task_id={task_id}&page_size=100", timeout=10)
            if lbl_res.status_code == 200:
                for l in lbl_res.json().get("results", []):
                    all_labels_map[l["id"]] = l["name"]

            job_res = session.get(f"{CVAT_SERVER}/api/jobs?task_id={task_id}&page_size=100", timeout=10)
            if job_res.status_code != 200:
                continue

            for job in job_res.json().get("results", []):
                job_id = job["id"]
                anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=10)
                if anno_res.status_code != 200:
                    continue

                shapes = anno_res.json().get("shapes", [])
                frame_boxes = {}
                for s in shapes:
                    if s.get("type") != "rectangle" or s.get("outside", False):
                        continue
                    frame = s.get("frame", 0)
                    pts = s.get("points", [])
                    if len(pts) < 4:
                        continue
                    lid = s.get("label_id", 0)
                    frame_boxes.setdefault(frame, []).append((lid, pts))

                for frame, bboxes in frame_boxes.items():
                    img_name = f"task{task_id}_frame{frame:06d}.jpg"
                    img_path = os.path.join(train_img_dir, img_name)
                    lbl_path = os.path.join(train_lbl_dir, f"task{task_id}_frame{frame:06d}.txt")

                    if not os.path.exists(img_path):
                        img_res = session.get(
                            f"{CVAT_SERVER}/api/tasks/{task_id}/data",
                            params={"type": "frame", "quality": "original", "number": frame},
                            timeout=15
                        )
                        if img_res.status_code == 200:
                            with open(img_path, "wb") as f:
                                f.write(img_res.content)
                        else:
                            continue

                    try:
                        with Image.open(img_path) as im:
                            img_w, img_h = im.size
                    except Exception:
                        img_w, img_h = 640, 512

                    yolo_lines = []
                    for lid, pts in bboxes:
                        xtl, ytl, xbr, ybr = pts[0], pts[1], pts[2], pts[3]
                        bw = max(1.0, xbr - xtl)
                        bh = max(1.0, ybr - ytl)
                        cx = xtl + bw / 2.0
                        cy = ytl + bh / 2.0
                        cls_idx = list(sorted(all_labels_map.keys())).index(lid) if lid in all_labels_map else 0
                        yolo_lines.append(f"{cls_idx} {cx/img_w:.6f} {cy/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}")

                    with open(lbl_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_lines))
                    total_labeled_frames += 1

        if total_labeled_frames == 0:
            return {"error": "라벨링된 데이터가 없어 학습을 진행할 수 없습니다."}

        print(f"\n🧠 [수동 학습 실행] 모델: '{target_model_name}' | 에포크: {epochs} | 총 {total_labeled_frames}개 프레임 기반 학습 시작...", flush=True)

        yaml_path = os.path.join(DATASET_DIR, "data.yaml")
        abs_ds_dir = os.path.abspath(DATASET_DIR).replace('\\', '/')
        sorted_labels = [all_labels_map[k] for k in sorted(all_labels_map.keys())] if all_labels_map else ["object"]
        yaml_content = f"path: {abs_ds_dir}\ntrain: train/images\nval: train/images\nnames:\n"
        for idx, lname in enumerate(sorted_labels):
            yaml_content += f"  {idx}: '{lname}'\n"

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        base_weights_path = os.path.join(MODELS_DIR, f"{target_model_name}.pt")
        if not os.path.exists(base_weights_path):
            base_weights_path = get_active_weights_path()

        runs_dir = os.path.join(DATASET_DIR, "runs")
        model = YOLO(base_weights_path)
        
        effective_epochs = max(1, int(epochs))
        model.train(data=yaml_path, epochs=effective_epochs, imgsz=640, batch=4, project=runs_dir, name="cvat_manual_train", exist_ok=True, verbose=False)

        best_weights = os.path.join(runs_dir, "cvat_manual_train", "weights", "best.pt")
        if not os.path.exists(best_weights):
            best_weights = os.path.join(runs_dir, "cvat_manual_train", "weights", "last.pt")

        if os.path.exists(best_weights):
            target_save_path = os.path.join(MODELS_DIR, f"{target_model_name}.pt")
            shutil.copy2(best_weights, target_save_path)
            shutil.copy2(best_weights, WEIGHTS_PATH)
            print(f"🎉 [학습 완료] '{target_model_name}.pt' 모델 가중치가 성공적으로 갱신되었습니다!\n", flush=True)
            return {
                "status": "success",
                "model_name": target_model_name,
                "epochs": effective_epochs,
                "labeled_frames": total_labeled_frames,
                "message": f"'{target_model_name}' 모델이 {total_labeled_frames}개 프레임으로 성공적으로 학습되었습니다."
            }

        return {"error": "학습 가중치 파일 생성 실패"}

    except Exception as e:
        print(f"❌ [학습 오류] {e}", flush=True)
        return {"error": str(e)}
    finally:
        with training_lock:
            is_training = False
            training_progress_msg = ""


# ============================================================
# 🎯 AI 단일 프레임 감지
# ============================================================
def run_ai_prediction(job_id, frame_idx, conf_thresh=0.25):
    """현재 프레임에 대해 활성 모델 기반 자동 사전 라벨링 수행 (수작업 라벨 보호)"""
    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    job_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}", timeout=10)
    if job_res.status_code != 200:
        return {"error": f"Job #{job_id} 조회 실패"}
    
    job_data = job_res.json()
    task_id = job_data.get("task_id")

    lbl_res = session.get(f"{CVAT_SERVER}/api/labels?task_id={task_id}&page_size=100", timeout=10)
    task_labels = lbl_res.json().get("results", []) if lbl_res.status_code == 200 else []
    first_label_id = task_labels[0]["id"] if task_labels else 1

    active_name = get_active_model_name()
    weights = get_active_weights_path()
    model = YOLO(weights)

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=10)
    existing_shapes = anno_res.json().get("shapes", []) if anno_res.status_code == 200 else []
    frame_existing = [s for s in existing_shapes if s.get("frame") == frame_idx and s.get("type") == "rectangle"]
    
    if frame_existing:
        print(f"🛡️ [기존 라벨 보호] 프레임 #{frame_idx}에는 이미 수작업 라벨({len(frame_existing)}개)이 존재합니다. (AI 주입 스킵)", flush=True)
        return {
            "status": "skipped",
            "job_id": job_id,
            "frame": frame_idx,
            "message": f"프레임 #{frame_idx}에 이미 {len(frame_existing)}개의 라벨이 있어 기존 라벨을 보호했습니다.",
            "detected_count": 0,
            "weights_used": active_name
        }

    frame_np = get_frame_np(session, task_id, frame_idx)
    if frame_np is None:
        print(f"❌ [AI 감지 실패] 프레임 {frame_idx} 이미지 다운로드 실패", flush=True)
        return {"error": f"프레임 {frame_idx} 이미지를 가져올 수 없습니다."}

    effective_thresh = max(0.05, float(conf_thresh)) if conf_thresh else 0.15
    results = model.predict(frame_np, conf=effective_thresh, verbose=False)
    new_shapes = []

    detected_confs = []
    if results and len(results[0].boxes) > 0:
        for b in results[0].boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            detected_confs.append(round(conf, 3))
            xyxy = b.xyxy[0].cpu().numpy().tolist()

            target_label_id = first_label_id
            if cls_id < len(task_labels):
                target_label_id = task_labels[cls_id]["id"]

            new_shapes.append({
                "type": "rectangle",
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "frame": frame_idx,
                "label_id": target_label_id,
                "group": 0,
                "source": "auto",
                "attributes": [],
                "points": [round(float(x), 2) for x in xyxy]
            })

    print(f"🤖 [AI 자동 감지] 프레임 #{frame_idx} | 모델: {active_name} | 임계값: {effective_thresh} | 감지수: {len(new_shapes)}개 (신뢰도: {detected_confs})", flush=True)

    if new_shapes:
        patch_res = session.patch(
            f"{CVAT_SERVER}/api/jobs/{job_id}/annotations",
            params={"action": "create"},
            json={"version": 0, "tags": [], "shapes": new_shapes, "tracks": []},
            timeout=15
        )
        if patch_res.status_code not in (200, 201):
            print(f"❌ [어노테이션 주입 실패] {patch_res.text[:200]}", flush=True)
            return {"error": f"어노테이션 주입 실패: {patch_res.text[:200]}"}

    return {
        "status": "success",
        "job_id": job_id,
        "frame": frame_idx,
        "detected_count": len(new_shapes),
        "weights_used": active_name,
        "confidences": detected_confs
    }


# ============================================================
# 🚀 전체 미라벨링 빈 프레임 일괄 감지
# ============================================================
def run_ai_prediction_all_empty(job_id, conf_thresh=0.15):
    """라벨링되지 않은 모든 빈 프레임들을 찾아 일괄 AI 자동 라벨링 수행 (수작업 라벨 100% 보존)"""
    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    job_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}", timeout=10)
    if job_res.status_code != 200:
        return {"error": f"Job #{job_id} 조회 실패"}
    
    job_data = job_res.json()
    task_id = job_data.get("task_id")
    start_frame = job_data.get("start_frame", 0)
    stop_frame = job_data.get("stop_frame", 100)

    lbl_res = session.get(f"{CVAT_SERVER}/api/labels?task_id={task_id}&page_size=100", timeout=10)
    task_labels = lbl_res.json().get("results", []) if lbl_res.status_code == 200 else []
    first_label_id = task_labels[0]["id"] if task_labels else 1

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=10)
    existing_shapes = anno_res.json().get("shapes", []) if anno_res.status_code == 200 else []
    labeled_frames = set(s.get("frame") for s in existing_shapes if s.get("type") == "rectangle")

    empty_frames = [f for f in range(start_frame, stop_frame + 1) if f not in labeled_frames]

    if not empty_frames:
        print(f"ℹ️ [일괄 감지] Job #{job_id}의 모든 프레임이 이미 라벨링되어 있습니다.", flush=True)
        return {
            "status": "success",
            "message": "모든 프레임이 이미 라벨링되어 있어 건너뛰었습니다.",
            "empty_frames_count": 0,
            "created_boxes": 0
        }

    active_name = get_active_model_name()
    weights = get_active_weights_path()
    model = YOLO(weights)

    effective_thresh = max(0.05, float(conf_thresh)) if conf_thresh else 0.15
    print(f"🚀 [전체 빈 프레임 일괄 감지] 총 {len(empty_frames)}개 미라벨링 프레임 자동 예측 시작 (모델: {active_name}, 임계값: {effective_thresh})...", flush=True)

    all_new_shapes = []
    processed_count = 0

    for f_idx in empty_frames:
        frame_np = get_frame_np(session, task_id, f_idx)
        if frame_np is None:
            continue
        
        results = model.predict(frame_np, conf=effective_thresh, verbose=False)
        if results and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                cls_id = int(b.cls[0].item())
                xyxy = b.xyxy[0].cpu().numpy().tolist()

                target_label_id = first_label_id
                if cls_id < len(task_labels):
                    target_label_id = task_labels[cls_id]["id"]

                all_new_shapes.append({
                    "type": "rectangle",
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "frame": f_idx,
                    "label_id": target_label_id,
                    "group": 0,
                    "source": "auto",
                    "attributes": [],
                    "points": [round(float(x), 2) for x in xyxy]
                })
        processed_count += 1
        if processed_count % 20 == 0 or processed_count == len(empty_frames):
            print(f"⏳ [일괄 감지 진행 중] {processed_count}/{len(empty_frames)} 프레임 처리 완료... (누적 감지 박스: {len(all_new_shapes)}개)", flush=True)

    if all_new_shapes:
        patch_res = session.patch(
            f"{CVAT_SERVER}/api/jobs/{job_id}/annotations",
            params={"action": "create"},
            json={"version": 0, "tags": [], "shapes": all_new_shapes, "tracks": []},
            timeout=45
        )
        if patch_res.status_code not in (200, 201):
            print(f"❌ [일괄 주입 실패] {patch_res.text[:200]}", flush=True)
            return {"error": f"어노테이션 일괄 저장 실패: {patch_res.text[:200]}"}

    print(f"🎉 [전체 일괄 감지 완료] {processed_count}개 빈 프레임 중 {len(all_new_shapes)}개 박스 주입 성공!\n", flush=True)
    return {
        "status": "success",
        "job_id": job_id,
        "empty_frames_count": processed_count,
        "created_boxes": len(all_new_shapes),
        "weights_used": active_name
    }


sam_model_instance = None


def get_sam_model():
    """MobileSAM 기반 초경량/고속 시각 객체 분할 추적 모델 로드"""
    global sam_model_instance
    if sam_model_instance is None:
        from ultralytics import SAM
        model_path = os.path.join(BASE_DIR, "mobile_sam.pt")
        if not os.path.exists(model_path):
            model_path = "mobile_sam.pt"
        print(f"🎯 [SAM 모델 로딩] {model_path} 초기화 중...", flush=True)
        sam_model_instance = SAM(model_path)
    return sam_model_instance


def track_with_sam_boxes(sam_model, curr_frame_np, prev_boxes):
    """SAM 프롬프트 기반으로 다음 프레임의 객체 분할 마스크와 사각형 BBox 추출"""
    if not prev_boxes or curr_frame_np is None:
        return []
    try:
        formatted_boxes = [[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in prev_boxes]
        results = sam_model(curr_frame_np, bboxes=formatted_boxes, verbose=False)
        if not results or not results[0].masks or len(results[0].masks) == 0:
            return prev_boxes

        tracked = []
        masks = results[0].masks
        for i, mask_xy in enumerate(masks.xy):
            if mask_xy is not None and len(mask_xy) > 2:
                x1 = float(mask_xy[:, 0].min())
                y1 = float(mask_xy[:, 1].min())
                x2 = float(mask_xy[:, 0].max())
                y2 = float(mask_xy[:, 1].max())
                if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                    tracked.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                    continue
            if i < len(prev_boxes):
                tracked.append(prev_boxes[i])
        return tracked if tracked else prev_boxes
    except Exception as e:
        print(f"⚠️ [SAM 추적 예외] {e}", flush=True)
        return prev_boxes


# ============================================================
# ⚡ 시계열 자동 라벨링 & 보간
# ============================================================
def run_temporal_tracking(job_id, n1, n2, conf_thresh=0.25, mode="forward", ref_start=None, ref_end=None, target_start=None, target_end=None, engine="hybrid"):
    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    job_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}", timeout=10)
    if job_res.status_code != 200:
        return {"error": f"Job #{job_id} 조회 실패"}
    
    job_data = job_res.json()
    task_id = job_data.get("task_id")

    lbl_res = session.get(f"{CVAT_SERVER}/api/labels?task_id={task_id}&page_size=100", timeout=10)
    task_labels = lbl_res.json().get("results", []) if lbl_res.status_code == 200 else []
    first_label_id = task_labels[0]["id"] if task_labels else 1

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=10)
    if anno_res.status_code != 200:
        return {"error": f"어노테이션 조회 실패: {anno_res.text}"}

    existing_shapes = anno_res.json().get("shapes", [])
    existing_labeled_frames = set(s.get("frame") for s in existing_shapes if s.get("type") == "rectangle")

    active_name = get_active_model_name()
    weights = get_active_weights_path()
    model = YOLO(weights)
    new_shapes = []

    sam_m = get_sam_model() if engine == "sam" else None

    # 모드 1: 구간 패턴 참조 -> 미래 예측
    if mode == "range_project" and ref_start is not None and ref_end is not None and target_start is not None and target_end is not None:
        ref_boxes = {}
        for s in existing_shapes:
            if s.get("type") == "rectangle" and ref_start <= s.get("frame", -1) <= ref_end:
                ref_boxes.setdefault(s.get("frame"), []).append(s)

        current_boxes = []
        if ref_end in ref_boxes:
            current_boxes = [list(b.get("points")) for b in ref_boxes[ref_end]]
        elif ref_boxes:
            last_ref_f = max(ref_boxes.keys())
            current_boxes = [list(b.get("points")) for b in ref_boxes[last_ref_f]]

        prev_frame_np = get_frame_np(session, task_id, ref_end)

        for f_idx in range(target_start, target_end + 1):
            curr_frame_np = get_frame_np(session, task_id, f_idx)
            if curr_frame_np is None:
                continue

            frame_has_user_label = (f_idx in existing_labeled_frames)

            if engine == "sam" and sam_m is not None and current_boxes:
                tracked_boxes = track_with_sam_boxes(sam_m, curr_frame_np, current_boxes)
                final_boxes = tracked_boxes if tracked_boxes else current_boxes
            else:
                yolo_results = model.predict(curr_frame_np, conf=conf_thresh, verbose=False)
                yolo_boxes = []
                if yolo_results and len(yolo_results[0].boxes) > 0:
                    for b in yolo_results[0].boxes:
                        yolo_boxes.append(b.xyxy[0].cpu().numpy().tolist())

                tracked_boxes = []
                if prev_frame_np is not None and current_boxes:
                    prev_gray = cv2.cvtColor(prev_frame_np, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(curr_frame_np, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    for box in current_boxes:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        h, w = curr_gray.shape
                        x1_c, x2_c = max(0, min(w-1, x1)), max(0, min(w-1, x2))
                        y1_c, y2_c = max(0, min(h-1, y1)), max(0, min(h-1, y2))
                        
                        if x2_c > x1_c and y2_c > y1_c:
                            box_flow = flow[y1_c:y2_c, x1_c:x2_c]
                            dx = np.median(box_flow[..., 0])
                            dy = np.median(box_flow[..., 1])
                        else:
                            dx, dy = 0, 0

                        tracked_boxes.append([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy])

                final_boxes = []
                if yolo_boxes:
                    for yb in yolo_boxes:
                        final_boxes.append(yb)
                elif tracked_boxes:
                    final_boxes = tracked_boxes
                else:
                    final_boxes = current_boxes

            if frame_has_user_label:
                user_boxes_on_frame = [list(s.get("points")) for s in existing_shapes if s.get("frame") == f_idx and s.get("type") == "rectangle"]
                if user_boxes_on_frame:
                    current_boxes = user_boxes_on_frame
                else:
                    current_boxes = final_boxes
            else:
                for box in final_boxes:
                    new_shapes.append({
                        "type": "rectangle",
                        "occluded": False,
                        "outside": False,
                        "z_order": 0,
                        "frame": f_idx,
                        "label_id": first_label_id,
                        "group": 0,
                        "source": "auto",
                        "attributes": [],
                        "points": [round(float(x), 2) for x in box]
                    })
                current_boxes = final_boxes

            prev_frame_np = curr_frame_np

    # 모드 2: 양방향 보간 (Interpolation)
    elif mode == "interpolate":
        boxes_n1 = [s.get("points") for s in existing_shapes if s.get("frame") == n1 and s.get("type") == "rectangle"]
        boxes_n2 = [s.get("points") for s in existing_shapes if s.get("frame") == n2 and s.get("type") == "rectangle"]

        if not boxes_n1 or not boxes_n2:
            return {"error": f"보간을 위해서는 프레임 #{n1}과 #{n2} 양쪽에 모두 사각형 라벨이 있어야 합니다."}

        total_steps = n2 - n1
        for f_idx in range(n1 + 1, n2):
            if f_idx in existing_labeled_frames:
                continue

            alpha = (f_idx - n1) / float(total_steps)
            for b1 in boxes_n1:
                best_b2 = min(boxes_n2, key=lambda b2: (b1[0]-b2[0])**2 + (b1[1]-b2[1])**2)
                inter_pts = [
                    round(b1[i] * (1.0 - alpha) + best_b2[i] * alpha, 2)
                    for i in range(4)
                ]
                new_shapes.append({
                    "type": "rectangle",
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "frame": f_idx,
                    "label_id": first_label_id,
                    "group": 0,
                    "source": "auto",
                    "attributes": [],
                    "points": inter_pts
                })

    # 모드 3: 단방향 연속 추적
    else:
        init_shapes = [s for s in existing_shapes if s.get("frame") == n1 and s.get("type") == "rectangle"]
        current_boxes = [list(s.get("points")) for s in init_shapes]
        prev_frame_np = get_frame_np(session, task_id, n1)

        for f_idx in range(n1 + 1, n2 + 1):
            curr_frame_np = get_frame_np(session, task_id, f_idx)
            if curr_frame_np is None:
                continue

            frame_has_user_label = (f_idx in existing_labeled_frames)
            
            if engine == "sam" and sam_m is not None and current_boxes:
                tracked_boxes = track_with_sam_boxes(sam_m, curr_frame_np, current_boxes)
            else:
                tracked_boxes = []
                if prev_frame_np is not None and current_boxes:
                    prev_gray = cv2.cvtColor(prev_frame_np, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(curr_frame_np, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    for box in current_boxes:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        h, w = curr_gray.shape
                        x1_c, x2_c = max(0, min(w-1, x1)), max(0, min(w-1, x2))
                        y1_c, y2_c = max(0, min(h-1, y1)), max(0, min(h-1, y2))
                        
                        if x2_c > x1_c and y2_c > y1_c:
                            box_flow = flow[y1_c:y2_c, x1_c:x2_c]
                            dx = np.median(box_flow[..., 0])
                            dy = np.median(box_flow[..., 1])
                        else:
                            dx, dy = 0, 0

                        tracked_boxes.append([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy])

            if frame_has_user_label:
                user_boxes = [list(s.get("points")) for s in existing_shapes if s.get("frame") == f_idx and s.get("type") == "rectangle"]
                if user_boxes:
                    current_boxes = user_boxes
                else:
                    current_boxes = tracked_boxes
            else:
                for box in tracked_boxes:
                    new_shapes.append({
                        "type": "rectangle",
                        "occluded": False,
                        "outside": False,
                        "z_order": 0,
                        "frame": f_idx,
                        "label_id": first_label_id,
                        "group": 0,
                        "source": "auto",
                        "attributes": [],
                        "points": [round(float(x), 2) for x in box]
                    })
                current_boxes = tracked_boxes

            prev_frame_np = curr_frame_np


    if new_shapes:
        patch_res = session.patch(
            f"{CVAT_SERVER}/api/jobs/{job_id}/annotations",
            params={"action": "create"},
            json={"version": 0, "tags": [], "shapes": new_shapes, "tracks": []},
            timeout=30
        )
        if patch_res.status_code not in (200, 201):
            return {"error": f"어노테이션 저장 실패: {patch_res.text}"}

    return {
        "status": "success",
        "job_id": job_id,
        "created_boxes": len(new_shapes),
        "weights_used": active_name
    }


# ============================================================
# 📊 모델 정확도 및 IoU 정량 평가 엔진 (Quantitative Evaluation)
# ============================================================
def compute_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def calculate_ap50(recalls, precisions):
    if not recalls or not precisions:
        return 0.0
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = [p for r, p in zip(recalls, precisions) if r >= t]
        p_max = max(prec_at_rec) if prec_at_rec else 0.0
        ap += p_max / 11.0
    return ap


def _compute_eval_report(frames_records, model_name, eval_source, eval_mode, effective_iou_thresh, effective_conf_thresh, log_label, job_id=0):
    """공통 정량 평가(Precision/Recall/F1/mIoU/mAP@0.5/오분류) 집계 헬퍼"""
    class_detections = {}
    class_stats = {}
    confusion_pairs = {}
    all_loc_ious = []

    for rec in frames_records:
        current_gts = rec["gt_boxes"]
        gt_boxes_by_class = {}
        for g in current_gts:
            gt_boxes_by_class.setdefault(g["class_name"], []).append(g["box"])

        pred_boxes_by_class = rec["pred_boxes_by_class"]
        all_preds_frame = rec["all_preds_frame"]

        all_classes = set(list(gt_boxes_by_class.keys()) + list(pred_boxes_by_class.keys()))
        for cname in all_classes:
            stats = class_stats.setdefault(cname, {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []})
            gts = gt_boxes_by_class.get(cname, [])
            p_list = sorted(pred_boxes_by_class.get(cname, []), key=lambda x: x[0], reverse=True)

            stats["gt"] += len(gts)
            stats["pred"] += len(p_list)

            matched_gt = [False] * len(gts)
            for conf, pbox in p_list:
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, gbox in enumerate(gts):
                    if matched_gt[g_idx]:
                        continue
                    iou = compute_box_iou(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                if best_iou >= effective_iou_thresh:
                    matched_gt[best_gt_idx] = True
                    stats["tp"] += 1
                    stats["ious"].append(best_iou)
                    class_detections.setdefault(cname, []).append((conf, 1, best_iou))
                else:
                    stats["fp"] += 1
                    class_detections.setdefault(cname, []).append((conf, 0, best_iou))
            stats["fn"] += matched_gt.count(False)

        for conf, pbox, pcname in all_preds_frame:
            best_loc_iou = 0.0
            best_gt_class = None
            for g in current_gts:
                iou = compute_box_iou(pbox, g["box"])
                if iou > best_loc_iou:
                    best_loc_iou = iou
                    best_gt_class = g["class_name"]
            if best_loc_iou > 0:
                all_loc_ious.append(best_loc_iou)
                if best_loc_iou >= effective_iou_thresh and best_gt_class != pcname:
                    pair_key = f"{best_gt_class} ➔ {pcname}"
                    confusion_pairs.setdefault(pair_key, []).append(best_loc_iou)

    total_gt = sum(v["gt"] for v in class_stats.values())
    total_pred = sum(v["pred"] for v in class_stats.values())
    total_tp = sum(v["tp"] for v in class_stats.values())
    total_fp = sum(v["fp"] for v in class_stats.values())
    total_fn = sum(v["fn"] for v in class_stats.values())

    overall_p = round(total_tp / (total_tp + total_fp) * 100, 2) if (total_tp + total_fp) > 0 else 0.0
    overall_r = round(total_tp / (total_tp + total_fn) * 100, 2) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = round(2 * (overall_p / 100) * (overall_r / 100) / ((overall_p / 100) + (overall_r / 100)), 3) if (overall_p + overall_r) > 0 else 0.0

    all_tp_ious = [iou for v in class_stats.values() for iou in v["ious"]]
    overall_class_miou = round(float(np.mean(all_tp_ious)) * 100, 2) if all_tp_ious else 0.0
    overall_loc_miou = round(float(np.mean(all_loc_ious)) * 100, 2) if all_loc_ious else 0.0

    per_class_results = []
    ap_list = []
    for cname, v in sorted(class_stats.items(), key=lambda x: -x[1]["gt"]):
        gt_cnt = v["gt"]
        pred_cnt = v["pred"]
        tp = v["tp"]
        fp = v["fp"]
        fn = v["fn"]
        p = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
        r = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
        f1 = round(2 * (p / 100) * (r / 100) / ((p / 100) + (r / 100)), 3) if (p + r) > 0 else 0.0
        c_miou = round(float(np.mean(v["ious"])) * 100, 2) if v["ious"] else 0.0

        dets = sorted(class_detections.get(cname, []), key=lambda x: x[0], reverse=True)
        acc_tp = 0
        acc_fp = 0
        recalls = []
        precisions = []
        for d_conf, d_tp, _ in dets:
            if d_tp == 1:
                acc_tp += 1
            else:
                acc_fp += 1
            rec = acc_tp / gt_cnt if gt_cnt > 0 else 0.0
            prec = acc_tp / (acc_tp + acc_fp) if (acc_tp + acc_fp) > 0 else 0.0
            recalls.append(rec)
            precisions.append(prec)
        c_ap = round(calculate_ap50(recalls, precisions) * 100, 2)
        if gt_cnt > 0:
            ap_list.append(c_ap)

        per_class_results.append({
            "class_name": cname,
            "gt": gt_cnt,
            "pred": pred_cnt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1_score": f1,
            "miou": c_miou,
            "ap50": c_ap
        })

    overall_map50 = round(float(np.mean(ap_list)), 2) if ap_list else 0.0

    confusion_list = []
    for pair, ious in sorted(confusion_pairs.items(), key=lambda x: -len(x[1])):
        gt_c, pred_c = pair.split(" ➔ ")
        confusion_list.append({
            "gt_class": gt_c,
            "pred_class": pred_c,
            "count": len(ious),
            "avg_iou": round(float(np.mean(ious)) * 100, 2)
        })

    md_lines = [
        "| 부품 클래스 | GT 수량 | 감지 수량 | TP | FP | FN | Precision(%) | Recall(%) | F1-Score | mIoU(%) | AP@0.5(%) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    for r_item in per_class_results:
        md_lines.append(
            f"| {r_item['class_name']} | {r_item['gt']} | {r_item['pred']} | {r_item['tp']} | {r_item['fp']} | {r_item['fn']} | {r_item['precision']}% | {r_item['recall']}% | {r_item['f1_score']} | {r_item['miou']}% | {r_item['ap50']}% |"
        )
    md_lines.append(
        f"| **전체 종합 (Overall)** | **{total_gt}** | **{total_pred}** | **{total_tp}** | **{total_fp}** | **{total_fn}** | **{overall_p}%** | **{overall_r}%** | **{overall_f1}** | **{overall_class_miou}%** | **{overall_map50}%** |"
    )
    md_table = "\n".join(md_lines)

    print(f"📊 [정확도 정량 평가 완료] {log_label} ({len(frames_records)}개 프레임) | mIoU: {overall_class_miou}% | 위치IoU: {overall_loc_miou}% | mAP@0.5: {overall_map50}% | F1: {overall_f1}", flush=True)

    return {
        "status": "success",
        "job_id": job_id,
        "model_used": model_name if eval_source == "model" else "CVAT Auto Annotations",
        "eval_source": eval_source,
        "eval_mode": eval_mode,
        "evaluated_frames_count": len(frames_records),
        "iou_thresh": effective_iou_thresh,
        "conf_thresh": effective_conf_thresh,
        "overall": {
            "total_gt": total_gt,
            "total_pred": total_pred,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": overall_p,
            "recall": overall_r,
            "f1_score": overall_f1,
            "class_miou": overall_class_miou,
            "loc_miou": overall_loc_miou,
            "map50": overall_map50
        },
        "per_class": per_class_results,
        "confusion_pairs": confusion_list[:5],
        "markdown_table": md_table
    }


def _collect_yolo_predictions(model, frame_np, conf_thresh):
    """단일 프레임에 대한 YOLO 추론 결과 수집 헬퍼"""
    pred_boxes_by_class = {}
    all_preds_frame = []
    if model is not None and frame_np is not None:
        results = model.predict(frame_np, conf=conf_thresh, verbose=False)
        if results and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                cid = int(b.cls[0].item())
                cname = model.names.get(cid, str(cid))
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                pred_boxes_by_class.setdefault(cname, []).append((conf, xyxy))
                all_preds_frame.append((conf, xyxy, cname))
    return pred_boxes_by_class, all_preds_frame


def evaluate_job_accuracy(job_id, model_name=None, iou_thresh=0.5, conf_thresh=0.25, eval_mode="manual", start_frame=None, stop_frame=None, eval_source="model", max_frames=50):
    """Job 내 수작업 Ground Truth(정답지) 기반 모델 mIoU, mAP, 정밀도/재현율 정량 평가"""
    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    job_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}", timeout=10)
    if job_res.status_code != 200:
        return {"error": f"Job #{job_id} 조회 실패: {job_res.text}"}

    job_data = job_res.json()
    task_id = job_data.get("task_id")
    job_start = job_data.get("start_frame", 0)
    job_stop = job_data.get("stop_frame", 100)

    lbl_res = session.get(f"{CVAT_SERVER}/api/labels?task_id={task_id}&page_size=100", timeout=10)
    task_labels = lbl_res.json().get("results", []) if lbl_res.status_code == 200 else []
    labels_map = {l["id"]: l["name"] for l in task_labels}

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=15)
    if anno_res.status_code != 200:
        return {"error": f"어노테이션 조회 실패: {anno_res.text}"}

    shapes = anno_res.json().get("shapes", [])
    manual_shapes = [s for s in shapes if s.get("source") != "auto" and s.get("type") == "rectangle"]
    auto_shapes = [s for s in shapes if s.get("source") == "auto" and s.get("type") == "rectangle"]

    gt_by_frame = {}
    for s in manual_shapes:
        f = s.get("frame", 0)
        cname = labels_map.get(s.get("label_id"), f"Label_{s.get('label_id')}")
        gt_by_frame.setdefault(f, []).append({
            "class_name": cname,
            "box": s.get("points", [])
        })

    if eval_mode == "manual":
        eval_frames = sorted(list(gt_by_frame.keys()))
    elif eval_mode == "range" and start_frame is not None and stop_frame is not None:
        eval_frames = [f for f in range(int(start_frame), int(stop_frame) + 1)]
    else:
        eval_frames = [f for f in range(job_start, job_stop + 1)]

    limit = max(1, min(100, int(max_frames)))
    eval_frames = eval_frames[:limit]

    if not eval_frames:
        return {"error": "평가할 프레임이 존재하지 않습니다."}

    if not model_name:
        model_name = get_active_model_name()

    model = None
    if eval_source == "model":
        weights_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
        if not os.path.exists(weights_path):
            weights_path = get_active_weights_path()
        model = YOLO(weights_path)

    effective_iou_thresh = max(0.1, min(0.95, float(iou_thresh)))
    effective_conf_thresh = max(0.01, min(0.95, float(conf_thresh)))

    frames_records = []
    for f_idx in eval_frames:
        current_gts = gt_by_frame.get(f_idx, [])
        if eval_source == "model" and model is not None:
            frame_np = get_frame_np(session, task_id, f_idx)
            pred_boxes_by_class, all_preds_frame = _collect_yolo_predictions(model, frame_np, effective_conf_thresh)
        else:
            pred_boxes_by_class = {}
            all_preds_frame = []
            for a in [s for s in auto_shapes if s.get("frame") == f_idx]:
                cname = labels_map.get(a.get("label_id"), f"Label_{a.get('label_id')}")
                xyxy = a.get("points", [])
                pred_boxes_by_class.setdefault(cname, []).append((1.0, xyxy))
                all_preds_frame.append((1.0, xyxy, cname))

        frames_records.append({
            "gt_boxes": current_gts,
            "pred_boxes_by_class": pred_boxes_by_class,
            "all_preds_frame": all_preds_frame,
        })

    return _compute_eval_report(
        frames_records=frames_records,
        model_name=model_name,
        eval_source=eval_source,
        eval_mode=eval_mode,
        effective_iou_thresh=effective_iou_thresh,
        effective_conf_thresh=effective_conf_thresh,
        log_label=f"Job #{job_id}",
        job_id=job_id,
    )


# ============================================================
# 📁 평가 전용 정답지(GT) 이미지셋 독립 관리 (학습셋 active_learning_dataset과 완전 분리)
# ============================================================
def _sanitize_gt_set_name(name):
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', str(name or '').strip())
    cleaned = cleaned.strip('. ')
    return cleaned[:60] if cleaned else ""


def _load_gt_set_meta(set_dir):
    meta_path = os.path.join(set_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_gt_set_meta(set_dir, meta):
    meta_path = os.path.join(set_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def list_eval_gt_sets():
    gt_sets = []
    if os.path.exists(EVAL_GT_DIR):
        for entry in sorted(os.listdir(EVAL_GT_DIR)):
            set_dir = os.path.join(EVAL_GT_DIR, entry)
            if not os.path.isdir(set_dir):
                continue
            meta = _load_gt_set_meta(set_dir)
            if not meta:
                continue
            items = meta.get("items", [])
            class_counts = {}
            total_boxes = 0
            for item in items:
                for b in item.get("boxes", []):
                    cname = b.get("class_name", "unknown")
                    class_counts[cname] = class_counts.get(cname, 0) + 1
                    total_boxes += 1
            gt_sets.append({
                "name": meta.get("name", entry),
                "image_count": len(items),
                "box_count": total_boxes,
                "class_counts": class_counts,
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", ""),
            })
    return {"status": "success", "gt_sets": gt_sets}


def save_job_manual_to_gt_set(job_id, set_name, frame_mode="manual", start_frame=None, stop_frame=None, current_frame=None):
    safe_name = _sanitize_gt_set_name(set_name)
    if not safe_name:
        return {"error": "유효한 정답지(GT) 셋 이름을 입력하세요."}

    session = get_cvat_session()
    if not session:
        return {"error": "CVAT 로그인 실패"}

    job_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}", timeout=10)
    if job_res.status_code != 200:
        return {"error": f"Job #{job_id} 조회 실패: {job_res.text}"}

    job_data = job_res.json()
    task_id = job_data.get("task_id")

    lbl_res = session.get(f"{CVAT_SERVER}/api/labels?task_id={task_id}&page_size=100", timeout=10)
    task_labels = lbl_res.json().get("results", []) if lbl_res.status_code == 200 else []
    labels_map = {l["id"]: l["name"] for l in task_labels}

    anno_res = session.get(f"{CVAT_SERVER}/api/jobs/{job_id}/annotations", timeout=15)
    if anno_res.status_code != 200:
        return {"error": f"어노테이션 조회 실패: {anno_res.text}"}

    shapes = anno_res.json().get("shapes", [])
    manual_shapes = [s for s in shapes if s.get("source") != "auto" and s.get("type") == "rectangle"]

    gt_by_frame = {}
    for s in manual_shapes:
        f = int(s.get("frame", 0))
        cname = labels_map.get(s.get("label_id"), f"Label_{s.get('label_id')}")
        gt_by_frame.setdefault(f, []).append({
            "class_name": cname,
            "box": [round(float(x), 2) for x in s.get("points", [])]
        })

    if frame_mode == "current" and current_frame is not None:
        cf = int(current_frame)
        target_frames = [cf] if cf in gt_by_frame else []
    elif frame_mode == "range" and start_frame is not None and stop_frame is not None:
        sf, ef = int(start_frame), int(stop_frame)
        target_frames = sorted([f for f in gt_by_frame.keys() if sf <= f <= ef])
    else:
        target_frames = sorted(gt_by_frame.keys())

    if not target_frames:
        return {"error": "선택한 범위에 저장할 수작업(Manual) GT 라벨이 없습니다."}

    set_dir = os.path.join(EVAL_GT_DIR, safe_name)
    images_dir = os.path.join(set_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    meta = _load_gt_set_meta(set_dir) or {
        "name": safe_name,
        "created_at": now_str,
        "updated_at": now_str,
        "items": [],
    }

    existing_by_id = {item["id"]: item for item in meta.get("items", [])}
    saved_count = 0

    for f_idx in target_frames[:200]:
        frame_np = get_frame_np(session, task_id, f_idx)
        if frame_np is None:
            continue
        item_id = f"job{job_id}_f{f_idx}"
        img_filename = f"{item_id}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        Image.fromarray(frame_np).save(img_path, format="JPEG", quality=95)
        h, w = frame_np.shape[:2]
        existing_by_id[item_id] = {
            "id": item_id,
            "image_file": img_filename,
            "source_job_id": int(job_id),
            "source_task_id": int(task_id),
            "source_frame": int(f_idx),
            "width": int(w),
            "height": int(h),
            "boxes": gt_by_frame[f_idx],
            "saved_at": now_str,
        }
        saved_count += 1

    meta["items"] = sorted(existing_by_id.values(), key=lambda x: (x.get("source_job_id", 0), x.get("source_frame", 0)))
    meta["updated_at"] = now_str
    _save_gt_set_meta(set_dir, meta)

    total_boxes = sum(len(it.get("boxes", [])) for it in meta["items"])
    print(f"📁 [정답지(GT) 셋 저장] '{safe_name}': {saved_count}장 반영 (누적 {len(meta['items'])}장, 박스 {total_boxes}개)", flush=True)

    return {
        "status": "success",
        "set_name": safe_name,
        "saved_images": saved_count,
        "total_images": len(meta["items"]),
        "total_boxes": total_boxes,
    }


def get_eval_gt_set_detail(set_name):
    safe_name = _sanitize_gt_set_name(set_name)
    set_dir = os.path.join(EVAL_GT_DIR, safe_name)
    meta = _load_gt_set_meta(set_dir)
    if not meta:
        return {"error": f"정답지 셋 '{safe_name}'을 찾을 수 없습니다."}

    images_dir = os.path.join(set_dir, "images")
    detailed_items = []
    for item in meta.get("items", []):
        img_path = os.path.join(images_dir, item.get("image_file", ""))
        thumb_b64 = ""
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as im:
                    im.thumbnail((280, 210))
                    buf = BytesIO()
                    im.save(buf, format="JPEG", quality=80)
                    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception:
                thumb_b64 = ""
        class_summary = {}
        for b in item.get("boxes", []):
            cn = b.get("class_name", "unknown")
            class_summary[cn] = class_summary.get(cn, 0) + 1
        detailed_items.append({
            "id": item.get("id"),
            "source_job_id": item.get("source_job_id"),
            "source_frame": item.get("source_frame"),
            "box_count": len(item.get("boxes", [])),
            "class_summary": class_summary,
            "saved_at": item.get("saved_at", ""),
            "thumbnail": thumb_b64,
        })

    return {
        "status": "success",
        "name": meta.get("name", safe_name),
        "created_at": meta.get("created_at", ""),
        "updated_at": meta.get("updated_at", ""),
        "items": detailed_items,
    }


def delete_eval_gt_item(set_name, item_id):
    safe_name = _sanitize_gt_set_name(set_name)
    set_dir = os.path.join(EVAL_GT_DIR, safe_name)
    meta = _load_gt_set_meta(set_dir)
    if not meta:
        return {"error": f"정답지 셋 '{safe_name}'을 찾을 수 없습니다."}

    new_items = []
    removed = None
    for item in meta.get("items", []):
        if item.get("id") == item_id:
            removed = item
        else:
            new_items.append(item)

    if not removed:
        return {"error": f"이미지 '{item_id}'를 찾을 수 없습니다."}

    img_path = os.path.join(set_dir, "images", removed.get("image_file", ""))
    if os.path.exists(img_path):
        os.remove(img_path)

    meta["items"] = new_items
    meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_gt_set_meta(set_dir, meta)

    return {"status": "success", "set_name": safe_name, "deleted_id": item_id, "remaining_images": len(new_items)}


def delete_eval_gt_set(set_name):
    safe_name = _sanitize_gt_set_name(set_name)
    set_dir = os.path.join(EVAL_GT_DIR, safe_name)
    if not safe_name or not os.path.exists(set_dir):
        return {"error": f"정답지 셋 '{safe_name}'을 찾을 수 없습니다."}
    shutil.rmtree(set_dir)
    return {"status": "success", "deleted_set": safe_name}


def evaluate_gt_set_accuracy(set_name, model_name=None, iou_thresh=0.5, conf_thresh=0.25, max_frames=50):
    """저장된 독립 정답지(GT) 이미지셋 기반 모델 정량 평가"""
    safe_name = _sanitize_gt_set_name(set_name)
    set_dir = os.path.join(EVAL_GT_DIR, safe_name)
    meta = _load_gt_set_meta(set_dir)
    if not meta:
        return {"error": f"정답지 셋 '{safe_name}'을 찾을 수 없습니다."}

    items = meta.get("items", [])
    limit = max(1, min(200, int(max_frames)))
    items = items[:limit]
    if not items:
        return {"error": f"정답지 셋 '{safe_name}'에 평가할 이미지가 없습니다."}

    if not model_name:
        model_name = get_active_model_name()

    weights_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    if not os.path.exists(weights_path):
        weights_path = get_active_weights_path()
    model = YOLO(weights_path)

    effective_iou_thresh = max(0.1, min(0.95, float(iou_thresh)))
    effective_conf_thresh = max(0.01, min(0.95, float(conf_thresh)))

    images_dir = os.path.join(set_dir, "images")
    frames_records = []
    for item in items:
        img_path = os.path.join(images_dir, item.get("image_file", ""))
        if not os.path.exists(img_path):
            continue
        frame_np = np.array(Image.open(img_path).convert("RGB"))
        pred_boxes_by_class, all_preds_frame = _collect_yolo_predictions(model, frame_np, effective_conf_thresh)
        frames_records.append({
            "gt_boxes": item.get("boxes", []),
            "pred_boxes_by_class": pred_boxes_by_class,
            "all_preds_frame": all_preds_frame,
        })

    if not frames_records:
        return {"error": "정답지 이미지 파일을 불러올 수 없습니다."}

    return _compute_eval_report(
        frames_records=frames_records,
        model_name=model_name,
        eval_source="model",
        eval_mode=f"gt_set:{safe_name}",
        effective_iou_thresh=effective_iou_thresh,
        effective_conf_thresh=effective_conf_thresh,
        log_label=f"GT 셋 '{safe_name}'",
        job_id=0,
    )


# ============================================================
# 🌐 REST API 핸들러
# ============================================================
class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/api/models":
            active_model = get_active_model_name()
            models = list_available_models()
            res = {
                "active_model": active_model,
                "models": models,
                "is_training": is_training,
                "training_status": training_progress_msg
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))
            return

        elif path == "/api/labels-gallery":
            job_id = int(qs.get("job_id", [0])[0])
            label_id = qs.get("label_id", [None])[0]
            start_frame = qs.get("start_frame", [None])[0]
            stop_frame = qs.get("stop_frame", [None])[0]
            res = get_label_shapes_gallery(job_id, label_id, start_frame, stop_frame)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))
            return

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "active", "service": "CVAT Multi-Model & Class Management Engine"}).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
        
        try:
            payload = json.loads(post_data.decode('utf-8')) if post_data else {}
            path = self.path.split('?')[0]

            if path == "/api/models/select":
                model_name = payload.get("model_name", "")
                ok, msg = set_active_model_name(model_name)
                status_code = 200 if ok else 400
                res = {"status": "success" if ok else "error", "message": msg, "active_model": get_active_model_name()}
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/models/create":
                model_name = payload.get("model_name", "")
                base_model = payload.get("base_model", "IR8_model")
                ok, msg = create_new_model(model_name, base_model)
                status_code = 200 if ok else 400
                res = {"status": "success" if ok else "error", "message": msg, "active_model": get_active_model_name()}
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/models/train":
                model_name = payload.get("model_name", get_active_model_name())
                epochs = int(payload.get("epochs", 5))

                if is_training:
                    res = {"error": "현재 다른 학습 작업이 진행 중입니다."}
                    self.send_response(400)
                else:
                    threading.Thread(target=train_selected_model, args=(model_name, epochs), daemon=True).start()
                    res = {
                        "status": "training_started",
                        "model_name": model_name,
                        "epochs": epochs,
                        "message": f"'{model_name}' 모델에 대한 {epochs} Epochs 학습이 백그라운드에서 시작되었습니다."
                    }
                    self.send_response(200)

                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/labels-merge":
                # 클래스 일괄 병합 (범위 지원)
                job_id = int(payload.get("job_id", 0))
                source_label_id = int(payload.get("source_label_id", 0))
                target_label_id = int(payload.get("target_label_id", 0))
                start_frame = payload.get("start_frame")
                stop_frame = payload.get("stop_frame")

                res = merge_job_labels(job_id, source_label_id, target_label_id, start_frame, stop_frame)
                status_code = 200 if "status" in res else 400
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/labels-split":
                # 클래스 분리 및 이동
                job_id = int(payload.get("job_id", 0))
                shape_ids = payload.get("shape_ids", [])
                target_label_id = payload.get("target_label_id", None)
                new_label_name = payload.get("new_label_name", None)

                res = split_or_reassign_shapes(job_id, shape_ids, target_label_id, new_label_name)
                status_code = 200 if "status" in res else 400
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/labels-delete":
                # 클래스 일괄 삭제 (범위 지원)
                job_id = int(payload.get("job_id", 0))
                label_id = int(payload.get("label_id", 0))
                delete_label_entity = bool(payload.get("delete_label_entity", False))
                start_frame = payload.get("start_frame")
                stop_frame = payload.get("stop_frame")

                res = delete_job_label_annotations(job_id, label_id, delete_label_entity, start_frame, stop_frame)
                status_code = 200 if "status" in res else 400
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))



            elif path == "/api/temporal-track":
                job_id = int(payload.get("job_id", 0))
                n1 = int(payload.get("n1", 0))
                n2 = int(payload.get("n2", 0))
                thresh = float(payload.get("threshold", 0.25))
                mode = payload.get("mode", "forward")
                ref_start = payload.get("ref_start")
                ref_end = payload.get("ref_end")
                target_start = payload.get("target_start")
                target_end = payload.get("target_end")

                if ref_start is not None: ref_start = int(ref_start)
                if ref_end is not None: ref_end = int(ref_end)
                if target_start is not None: target_start = int(target_start)
                if target_end is not None: target_end = int(target_end)

                engine = payload.get("engine", "hybrid")

                res = run_temporal_tracking(
                    job_id, n1, n2,
                    conf_thresh=thresh,
                    mode=mode,
                    ref_start=ref_start,
                    ref_end=ref_end,
                    target_start=target_start,
                    target_end=target_end,
                    engine=engine
                )

                
                status_code = 200 if "status" in res else 400
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/ai-predict":
                job_id = int(payload.get("job_id", 0))
                frame = int(payload.get("frame", 0))
                thresh = float(payload.get("threshold", 0.15))

                res = run_ai_prediction(job_id, frame, thresh)
                status_code = 200 if "status" in res else 400
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/ai-predict-all":
                job_id = int(payload.get("job_id", 0))
                thresh = float(payload.get("threshold", 0.15))

                res = run_ai_prediction_all_empty(job_id, thresh)
                status_code = 200 if "status" in res else 400
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/unsupervised-discover":
                job_id = int(payload.get("job_id", 0))
                start_frame = int(payload.get("start_frame", 0))
                stop_frame = int(payload.get("stop_frame", 50))
                num_clusters = int(payload.get("num_clusters", 4))

                session = get_cvat_session()
                if not session:
                    res = {"error": "CVAT 로그인 실패"}
                else:
                    try:
                        from unsupervised_engine import discover_patterns_in_job
                        res = discover_patterns_in_job(session, CVAT_SERVER, job_id, start_frame, stop_frame, num_clusters)
                    except Exception as e:
                        res = {"error": f"비지도 패턴 발견 오류: {str(e)}"}

                status_code = 200 if "status" in res else 400
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            elif path == "/api/evaluate-accuracy":
                gt_action = payload.get("gt_action", None)
                job_id = int(payload.get("job_id", 0))

                if gt_action == "list_gt_sets":
                    res = list_eval_gt_sets()
                elif gt_action == "save_gt_set":
                    res = save_job_manual_to_gt_set(
                        job_id=job_id,
                        set_name=payload.get("set_name", ""),
                        frame_mode=payload.get("frame_mode", "manual"),
                        start_frame=payload.get("start_frame", None),
                        stop_frame=payload.get("stop_frame", None),
                        current_frame=payload.get("current_frame", None),
                    )
                elif gt_action == "get_gt_set_detail":
                    res = get_eval_gt_set_detail(payload.get("set_name", ""))
                elif gt_action == "delete_gt_item":
                    res = delete_eval_gt_item(payload.get("set_name", ""), payload.get("item_id", ""))
                elif gt_action == "delete_gt_set":
                    res = delete_eval_gt_set(payload.get("set_name", ""))
                elif gt_action == "evaluate_gt_set":
                    res = evaluate_gt_set_accuracy(
                        set_name=payload.get("set_name", ""),
                        model_name=payload.get("model_name", None),
                        iou_thresh=float(payload.get("iou_threshold", 0.5)),
                        conf_thresh=float(payload.get("conf_threshold", 0.25)),
                        max_frames=int(payload.get("max_frames", 50)),
                    )
                else:
                    model_name = payload.get("model_name", None)
                    iou_thresh = float(payload.get("iou_threshold", 0.5))
                    conf_thresh = float(payload.get("conf_threshold", 0.25))
                    eval_mode = payload.get("eval_mode", "manual")
                    start_frame = payload.get("start_frame", None)
                    stop_frame = payload.get("stop_frame", None)
                    eval_source = payload.get("eval_source", "model")
                    max_frames = int(payload.get("max_frames", 50))

                    res = evaluate_job_accuracy(
                        job_id=job_id,
                        model_name=model_name,
                        iou_thresh=iou_thresh,
                        conf_thresh=conf_thresh,
                        eval_mode=eval_mode,
                        start_frame=start_frame,
                        stop_frame=stop_frame,
                        eval_source=eval_source,
                        max_frames=max_frames
                    )
                status_code = 200 if "status" in res else 400
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))

            else:
                event = payload.get('event', 'update:job')
                print(f"📩 [CVAT 저장 감지] 웹훅: {event} (수동 학습 모드이므로 자동 학습은 생략됨)", flush=True)

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "event": event, "mode": "manual_training"}).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

    def log_message(self, format, *args):
        return


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def start_server(port=5000):
    server_address = ('', port)
    httpd = ReusableTCPServer(server_address, APIHandler)
    active_m = get_active_model_name()
    print(f"🔥 [CVAT AI 다중 모델 & 클래스 관리 엔진 가동] 포트 {port}에서 대기 중... (현재 활성 모델: {active_m})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    start_server(5000)
