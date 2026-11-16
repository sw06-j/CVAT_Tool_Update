#!/bin/sh
set -e
apk add --no-cache curl
curl -sL https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64 -o /usr/local/bin/nuctl
chmod +x /usr/local/bin/nuctl

nuctl create project cvat --platform local || true

echo "Deploying official ONNX YOLO v7 model into CVAT..."
nuctl deploy --project-name cvat \
  --path /opt/nuclio/serverless/onnx/WongKinYiu/yolov7/nuclio \
  --file /opt/nuclio/serverless/onnx/WongKinYiu/yolov7/nuclio/function.yaml \
  --platform local \
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'

echo "Function deployment completed!"
nuctl get function --platform local
