import os
import sys
import time
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import KMeans

# ============================================================
# 🔮 비지도 열화상 객체 발견 및 군집화 엔진 (Unsupervised Discovery)
# ============================================================

class ThermalFeatureExtractor:
    """사전 학습된 ResNet18 백본 기반 고속 시각 특징 추출기"""
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        except Exception:
            self.model = models.resnet18(pretrained=True)
            
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, patches):
        """패치 이미지 리스트(numpy BGR or PIL) ➔ 512차원 임베딩 행렬"""
        if not patches:
            return np.empty((0, 512), dtype=np.float32)

        tensors = []
        for p in patches:
            if isinstance(p, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
            else:
                pil_img = p
            tensors.append(self.transform(pil_img))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model(batch)
            if self.device == "cuda":
                features = features.cpu()
            embeddings = features.numpy()

        # L2 Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7
        return embeddings / norms


def extract_thermal_proposals(frame_bgr, min_area=350, max_area=90000):
    """
    열화상 프레임에서 온도 차이, 경계선, MSER 기반 부품 후보 영역(BBox) 자동 추출
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    candidates = []

    # 1. 멀티스케일 Otsu & 적응형 이진화 (고온/저온 부품 분리)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh_otsu)
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

    # 2. 열화상 경계 구배 (Morphological Gradient)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, thresh_grad = cv2.threshold(gradient, 25, 255, cv2.THRESH_BINARY)

    masks = [thresh_otsu, thresh_inv, thresh_adapt, thresh_grad]

    for mask in masks:
        # 모폴로지 닫기 연산으로 부품 내부 구멍 메우기
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / float(bh)
                # 극단적인 선 형태(전선, 노이즈) 제외 (가로세로 비율 0.15~6.5 허용)
                if 0.15 <= aspect_ratio <= 6.5:
                    candidates.append([x, y, x + bw, y + bh, area])

    # 3. MSER (Maximally Stable Extremal Regions) 추가 추출
    try:
        mser = cv2.MSER_create(delta=5, min_area=int(min_area), max_area=int(max_area))
        regions, _ = mser.detectRegions(gray)
        for p in regions:
            x, y, bw, bh = cv2.boundingRect(p)
            area = bw * bh
            if min_area <= area <= max_area:
                candidates.append([x, y, x + bw, y + bh, area])
    except Exception:
        pass

    if not candidates:
        return []

    # 4. Non-Maximum Suppression (NMS) 중복 제거
    boxes = np.array(candidates)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]  # 면적을 기준으로 정렬

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_inter = np.maximum(0.0, xx2 - xx1)
        h_inter = np.maximum(0.0, yy2 - yy1)
        inter = w_inter * h_inter

        area1 = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area2 = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        union = area1 + area2 - inter

        iou = inter / (union + 1e-7)
        inds = np.where(iou <= 0.4)[0]
        order = order[inds + 1]

    final_boxes = boxes[keep][:18]  # 프레임당 최대 18개 핵심 후보
    return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in final_boxes]


def discover_patterns_in_job(session, cvat_server, job_id, start_frame=0, stop_frame=50, num_clusters=4):
    """
    Job 내의 프레임들을 스캔하여 비지도 패턴을 발견하고 CVAT에 가명 라벨로 자동 주입
    """
    job_res = session.get(f"{cvat_server}/api/jobs/{job_id}", timeout=10)
    if job_res.status_code != 200:
        return {"error": f"Job #{job_id} 조회 실패"}

    job_data = job_res.json()
    task_id = job_data.get("task_id")
    project_id = job_data.get("project_id")
    job_start = job_data.get("start_frame", 0)
    job_stop = job_data.get("stop_frame", 100)

    start_f = max(job_start, int(start_frame))
    stop_f = min(job_stop, int(stop_frame))
    num_k = max(2, min(10, int(num_clusters)))

    print(f"\n🔮 [비지도 패턴 발견 시작] Job #{job_id} (Task #{task_id}, Project #{project_id}) | 프레임 {start_f}~{stop_f} | 목표 패턴 수: {num_k}개...", flush=True)

    # 1. 기존 라벨링된 프레임 확인 (수작업 라벨 보호)
    anno_res = session.get(f"{cvat_server}/api/jobs/{job_id}/annotations", timeout=15)
    existing_shapes = anno_res.json().get("shapes", []) if anno_res.status_code == 200 else []
    user_labeled_frames = set(s.get("frame") for s in existing_shapes if s.get("type") == "rectangle" and s.get("source") != "auto")

    # 2. 프레임별 관심 후보 패치 수집
    patches = []
    patch_meta = []  # (frame, [x1, y1, x2, y2])

    for f_idx in range(start_f, stop_f + 1):
        if f_idx in user_labeled_frames:
            # 수작업 라벨링된 프레임은 패턴 추출용으로만 읽고, 덮어쓰지 않음
            pass

        img_res = session.get(
            f"{cvat_server}/api/tasks/{task_id}/data",
            params={"type": "frame", "quality": "original", "number": f_idx},
            timeout=15
        )
        if img_res.status_code != 200:
            continue

        frame_np = cv2.imdecode(np.frombuffer(img_res.content, np.uint8), cv2.IMREAD_COLOR)
        if frame_np is None:
            continue

        proposals = extract_thermal_proposals(frame_np)
        for p in proposals:
            x1, y1, x2, y2 = p
            crop = frame_np[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                patches.append(crop)
                patch_meta.append((f_idx, p))

    if len(patches) < num_k:
        return {"error": f"추출된 부품 후보 영역({len(patches)}개)이 너무 적어 군집화를 수행할 수 없습니다."}

    print(f"📦 [후보 영역 추출 완료] 총 {len(patches)}개 열화상 부품 패치 수집됨. AI 특징 임베딩 추출 중...", flush=True)

    # 3. AI 시각 특징 추출 (Feature Extraction)
    extractor = ThermalFeatureExtractor()
    embeddings = extractor.extract_features(patches)

    # 4. K-Means 군집화 (Clustering)
    print(f"🧠 [K-Means 군집화 실행] {len(embeddings)}개 벡터 ➔ {num_k}개 클러스터 분류...", flush=True)
    kmeans = KMeans(n_clusters=num_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 클러스터 크기 순 정렬
    cluster_counts = {c: int(np.sum(cluster_labels == c)) for c in range(num_k)}
    sorted_clusters = sorted(cluster_counts.keys(), key=lambda c: cluster_counts[c], reverse=True)

    print(f"📊 [군집화 결과] 클러스터 분포: { {f'Cluster_{c}': cluster_counts[c] for c in sorted_clusters} }", flush=True)

    # 5. CVAT에 가명 라벨(발견패턴_1~N) 생성 (PATCH project or task)
    PALETTE_COLORS = ["#eb2f96", "#fa541c", "#faad14", "#a0d911", "#13c2c2", "#722ed1", "#1890ff", "#eb2f96"]

    lbl_query = f"project_id={project_id}" if project_id else f"task_id={task_id}"
    lbl_res = session.get(f"{cvat_server}/api/labels?{lbl_query}&page_size=100", timeout=10)
    current_labels = lbl_res.json().get("results", []) if lbl_res.status_code == 200 else []
    existing_lbl_map = {l["name"]: l["id"] for l in current_labels}

    needed_new_labels = []
    for rank in range(num_k):
        p_name = f"발견패턴_{rank + 1}"
        if p_name not in existing_lbl_map:
            color = PALETTE_COLORS[rank % len(PALETTE_COLORS)]
            needed_new_labels.append({"name": p_name, "color": color, "type": "any"})

    if needed_new_labels:
        patch_labels_payload = []
        for l in current_labels:
            patch_labels_payload.append({"id": l["id"], "name": l["name"], "color": l.get("color", "#000000")})
        patch_labels_payload.extend(needed_new_labels)

        if project_id:
            session.patch(f"{cvat_server}/api/projects/{project_id}", json={"labels": patch_labels_payload}, timeout=20)
        else:
            session.patch(f"{cvat_server}/api/tasks/{task_id}", json={"labels": patch_labels_payload}, timeout=20)

        # 갱신된 라벨 목록 재조회
        lbl_res_after = session.get(f"{cvat_server}/api/labels?{lbl_query}&page_size=100", timeout=10)
        if lbl_res_after.status_code == 200:
            existing_lbl_map = {l["name"]: l["id"] for l in lbl_res_after.json().get("results", [])}

    cluster_to_label_id = {}
    for rank, c_idx in enumerate(sorted_clusters):
        p_name = f"발견패턴_{rank + 1}"
        cluster_to_label_id[c_idx] = existing_lbl_map.get(p_name, current_labels[0]["id"] if current_labels else 1)


    # 6. CVAT에 Bounding Box 주입 (수작업 라벨 프레임 보호)
    new_shapes = []
    for idx, (f_idx, bbox) in enumerate(patch_meta):
        if f_idx in user_labeled_frames:
            continue  # 수작업 프레임은 건드리지 않음

        c_idx = cluster_labels[idx]
        target_lid = cluster_to_label_id.get(c_idx)

        new_shapes.append({
            "type": "rectangle",
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "frame": f_idx,
            "label_id": target_lid,
            "group": 0,
            "source": "auto",
            "attributes": [],
            "points": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        })

    if new_shapes:
        patch_res = session.patch(
            f"{cvat_server}/api/jobs/{job_id}/annotations",
            params={"action": "create"},
            json={"version": 0, "tags": [], "shapes": new_shapes, "tracks": []},
            timeout=45
        )
        if patch_res.status_code not in (200, 201):
            return {"error": f"어노테이션 주입 실패: {patch_res.text[:200]}"}

    print(f"🎉 [비지도 발견 완료] 총 {len(new_shapes)}개의 발견 객체가 {num_k}개 패턴 클래스로 자동 주입되었습니다!\n", flush=True)

    return {
        "status": "success",
        "job_id": job_id,
        "frames_scanned": stop_f - start_f + 1,
        "total_proposals": len(patches),
        "num_clusters": num_k,
        "created_boxes": len(new_shapes),
        "cluster_distribution": {f"발견패턴_{rank+1}": cluster_counts[c] for rank, c in enumerate(sorted_clusters)},
        "message": f"총 {len(new_shapes)}개의 반복 패턴이 발견되어 '발견패턴_1~{num_k}' 라벨로 자동 주입되었습니다."
    }
