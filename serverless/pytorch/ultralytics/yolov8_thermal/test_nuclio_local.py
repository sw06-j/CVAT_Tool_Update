import base64
import json
import os
import sys
from PIL import Image

# Import main handler
import main

class DummyContext:
    def __init__(self):
        class Logger:
            def info(self, msg): print(f"[Nuclio Log] {msg}")
        self.logger = Logger()
        class UserData:
            pass
        self.user_data = UserData()

class DummyResponse:
    def build_response(self, provider, headers, body, content_type, status_code):
        return json.loads(body)

class DummyEvent:
    def __init__(self, img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        self.body = {"image": encoded, "threshold": 0.25}

def run_local_test():
    print("=" * 60)
    print("=== [1] Nuclio 컨텍스트 초기화 (YOLOv8 모델 로드) ===")
    ctx = DummyContext()
    main.init_context(ctx)

    # Look for a test image in project directory
    test_img = None
    possible_imgs = [
        "../../../../../3-가공/A1_B1_P2_2022-05-12_IR1_00001.jpg",
        "../../../../../test_LBL2/A1_B1_P2_2022-05-12_IR1_00001.jpg"
    ]
    for p in possible_imgs:
        if os.path.exists(p):
            test_img = p
            break

    if not test_img:
        # Create a dummy image for testing
        print("테스트용 샘플 이미지를 생성합니다...")
        test_img = "sample_test.jpg"
        img = Image.new("RGB", (640, 512), color=(100, 100, 100))
        img.save(test_img)

    print(f"\n=== [2] 이미지 추론 요청 테스트 ({os.path.basename(test_img)}) ===")
    evt = DummyEvent(test_img)
    ctx.response = DummyResponse()

    result_json = main.handler(ctx, evt)

    print("\n=== [3] CVAT 규격 반환 좌표 (JSON) ===")
    print(json.dumps(result_json, indent=2, ensure_ascii=False))
    print("=" * 60)
    print("✅ CVAT Nuclio 모델 핸들러 로컬 테스트 성공!")
    print("=" * 60)

if __name__ == "__main__":
    run_local_test()
