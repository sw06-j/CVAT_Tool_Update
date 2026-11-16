import json
import requests
from rest_framework import status
from rest_framework.response import Response

# ==============================================================================
# CVAT 백엔드 ↔ 로컬 고속 AI 다중 모델 & 클래스 관리 연동 모듈
# ==============================================================================
LOCAL_TRACK_URL = "http://host.docker.internal:5000/api/temporal-track"
LOCAL_PREDICT_URL = "http://host.docker.internal:5000/api/ai-predict"
LOCAL_PREDICT_ALL_URL = "http://host.docker.internal:5000/api/ai-predict-all"
LOCAL_MODELS_URL = "http://host.docker.internal:5000/api/models"
LOCAL_MODELS_SELECT_URL = "http://host.docker.internal:5000/api/models/select"
LOCAL_MODELS_CREATE_URL = "http://host.docker.internal:5000/api/models/create"
LOCAL_MODELS_TRAIN_URL = "http://host.docker.internal:5000/api/models/train"
LOCAL_LABELS_MERGE_URL = "http://host.docker.internal:5000/api/labels-merge"
LOCAL_LABELS_SPLIT_URL = "http://host.docker.internal:5000/api/labels-split"
LOCAL_LABELS_DELETE_URL = "http://host.docker.internal:5000/api/labels-delete"
LOCAL_LABELS_GALLERY_URL = "http://host.docker.internal:5000/api/labels-gallery"


def run_temporal_tracking_for_job(job_id, data_or_n1=None, n2=None, threshold=0.25, *args, **kwargs):
    """로컬 고속 AI 엔진으로 시계열 추적/보간 요청 위임"""
    try:
        if isinstance(data_or_n1, dict):
            payload = dict(data_or_n1)
        else:
            payload = {
                "n1": data_or_n1 if data_or_n1 is not None else 0,
                "n2": n2 if n2 is not None else 0,
                "threshold": threshold,
                "mode": kwargs.get("mode", "forward")
            }
        payload["job_id"] = job_id
        res = requests.post(LOCAL_TRACK_URL, json=payload, timeout=60)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"시계열 추적 처리 오류: {str(e)}"}


def run_ai_predict_for_frame(job_id, frame_idx=0, threshold=0.25, *args, **kwargs):
    """로컬 고속 AI 엔진으로 현재 프레임 AI 사전 라벨링 요청 위임"""
    try:
        if isinstance(frame_idx, dict):
            payload = dict(frame_idx)
        else:
            payload = {
                "frame": frame_idx,
                "threshold": threshold
            }
        payload["job_id"] = job_id
        res = requests.post(LOCAL_PREDICT_URL, json=payload, timeout=15)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"AI 사전 라벨링 처리 오류: {str(e)}"}


def run_ai_predict_all_empty_for_job(job_id, threshold=0.15, *args, **kwargs):
    """로컬 고속 AI 엔진으로 라벨링 안 된 전체 빈 프레임 일괄 자동 라벨링 요청 위임"""
    try:
        payload = {
            "job_id": job_id,
            "threshold": threshold
        }
        res = requests.post(LOCAL_PREDICT_ALL_URL, json=payload, timeout=120)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"전체 일괄 라벨링 처리 오류: {str(e)}"}


def get_models_list():
    """모델 목록 및 활성 모델 정보 조회"""
    try:
        res = requests.get(LOCAL_MODELS_URL, timeout=10)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"모델 목록 조회 실패: {str(e)}"}


def select_active_model(model_name):
    """활성 모델 변경"""
    try:
        res = requests.post(LOCAL_MODELS_SELECT_URL, json={"model_name": model_name}, timeout=10)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"모델 변경 실패: {str(e)}"}


def create_custom_model(model_name, base_model="IR8_model"):
    """새 모델 생성"""
    try:
        res = requests.post(LOCAL_MODELS_CREATE_URL, json={"model_name": model_name, "base_model": base_model}, timeout=10)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"새 모델 생성 실패: {str(e)}"}


def train_custom_model(model_name=None, epochs=5):
    """선택된 모델 수동 학습 시작"""
    try:
        res = requests.post(LOCAL_MODELS_TRAIN_URL, json={"model_name": model_name, "epochs": epochs}, timeout=10)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"학습 시작 실패: {str(e)}"}


def merge_job_labels_proxy(job_id, source_label_id, target_label_id, start_frame=None, stop_frame=None):
    """클래스 일괄 병합 프록시 (범위 지원)"""
    try:
        payload = {
            "job_id": job_id,
            "source_label_id": source_label_id,
            "target_label_id": target_label_id,
            "start_frame": start_frame,
            "stop_frame": stop_frame
        }
        res = requests.post(LOCAL_LABELS_MERGE_URL, json=payload, timeout=30)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"클래스 병합 처리 오류: {str(e)}"}


def split_job_labels_proxy(job_id, shape_ids, target_label_id=None, new_label_name=None):
    """클래스 분리 및 이동 프록시"""
    try:
        payload = {
            "job_id": job_id,
            "shape_ids": shape_ids,
            "target_label_id": target_label_id,
            "new_label_name": new_label_name
        }
        res = requests.post(LOCAL_LABELS_SPLIT_URL, json=payload, timeout=30)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"클래스 분리 처리 오류: {str(e)}"}


def delete_job_labels_proxy(job_id, label_id, delete_label_entity=False, start_frame=None, stop_frame=None):
    """클래스 일괄 삭제 프록시 (범위 지원)"""
    try:
        payload = {
            "job_id": job_id,
            "label_id": label_id,
            "delete_label_entity": delete_label_entity,
            "start_frame": start_frame,
            "stop_frame": stop_frame
        }
        res = requests.post(LOCAL_LABELS_DELETE_URL, json=payload, timeout=30)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"클래스 삭제 처리 오류: {str(e)}"}


def get_job_labels_gallery_proxy(job_id, label_id=None, start_frame=None, stop_frame=None):
    """특정 라벨 박스 목록 조회 프록시 (범위 지원)"""
    try:
        params = {"job_id": job_id}
        if label_id is not None:
            params["label_id"] = label_id
        if start_frame is not None:
            params["start_frame"] = start_frame
        if stop_frame is not None:
            params["stop_frame"] = stop_frame
        res = requests.get(LOCAL_LABELS_GALLERY_URL, params=params, timeout=15)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"갤러리 목록 조회 실패: {str(e)}"}



LOCAL_UNSUPERVISED_URL = "http://host.docker.internal:5000/api/unsupervised-discover"
LOCAL_EVALUATE_ACCURACY_URL = "http://host.docker.internal:5000/api/evaluate-accuracy"


def run_unsupervised_discover_for_job(job_id, start_frame=0, stop_frame=50, num_clusters=4):
    """비지도 패턴 발견 및 자동 라벨링 요청 위임"""
    try:
        payload = {
            "job_id": job_id,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "num_clusters": num_clusters
        }
        res = requests.post(LOCAL_UNSUPERVISED_URL, json=payload, timeout=180)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"비지도 패턴 발견 처리 오류: {str(e)}"}


def run_evaluate_accuracy_proxy(job_id, data=None):
    """모델 정확도 및 IoU 정량 평가 요청 위임"""
    try:
        payload = dict(data) if isinstance(data, dict) else {}
        payload["job_id"] = job_id
        res = requests.post(LOCAL_EVALUATE_ACCURACY_URL, json=payload, timeout=180)
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "로컬 AI 엔진(auto_trainer.py)이 실행되지 않았습니다."}
    except Exception as e:
        return {"error": f"정확도 평가 처리 오류: {str(e)}"}


