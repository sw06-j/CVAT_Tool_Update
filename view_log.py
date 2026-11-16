import sys
import time
import os

# Windows 기본 cp949 환경에서도 이모지 및 특수문자가 100% 안전하게 출력되도록 재설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

log_file = r"C:\Users\Jeongsiwoo\.gemini\antigravity\brain\8e6532c3-7ba5-4f60-9949-54f92a948094\.system_generated\tasks\task-5276.log"

print("=" * 75)
print("  [CVAT] Realtime AI Engine Log Monitor")
print("  - Save 클릭 시: 실시간 YOLOv8 재학습 과정 출력")
print("  - AI 자동 감지 / 시계열 추적 클릭 시: 실시간 감지 결과 출력")
print("=" * 75)
print()

if not os.path.exists(log_file):
    print("[INFO] 로그 파일을 연결하는 중입니다...")

while not os.path.exists(log_file):
    time.sleep(1)

try:
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        # 최근 30줄 출력
        lines = f.readlines()
        for l in lines[-30:]:
            sys.stdout.write(l)
        sys.stdout.flush()

        # 실시간 모니터링 루프
        while True:
            line = f.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            else:
                time.sleep(0.5)
except KeyboardInterrupt:
    print("\n[INFO] 로그 모니터를 종료합니다.")
except Exception as e:
    print(f"\n[ERROR] 로그 읽기 오류: {e}")
