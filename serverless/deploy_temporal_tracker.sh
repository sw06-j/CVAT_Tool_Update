#!/bin/sh
set -e
apk add --no-cache curl
curl -sL https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64 -o /usr/local/bin/nuctl
chmod +x /usr/local/bin/nuctl

nuctl create project cvat --platform local || true

echo "Deploying Temporal Tracker (N1->N2 Time-Series Tracker) model..."
nuctl deploy --project-name cvat \
  --path /opt/nuclio/serverless/pytorch/ultralytics/yolov8_temporal_tracker/nuclio \
  --file /opt/nuclio/serverless/pytorch/ultralytics/yolov8_temporal_tracker/nuclio/function.yaml \
  --platform local \
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'

echo "Temporal Tracker deployment complete!"
nuctl get function --platform local
