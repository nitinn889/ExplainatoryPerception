#!/usr/bin/env bash
# Fetch the COCO-pretrained SSD-MobileNet-V2 detector used by nano/detector.py.
#
# Two files are needed by cv2.dnn.readNetFromTensorflow:
#   frozen_inference_graph.pb                      - the weights (~70 MB)
#   ssd_mobilenet_v2_coco_2018_03_29.pbtxt         - the text graph OpenCV needs
#                                                    to rebuild SSD's postprocessing
#
# No YOLO anywhere in this pipeline - SSD-MobileNet-V2 only, per the build spec.
set -euo pipefail

DEST="${SSD_MODEL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/models/ssd_mobilenet_v2_coco}"
mkdir -p "$DEST"

PB="$DEST/frozen_inference_graph.pb"
PBTXT="$DEST/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

# Primary source: the official TensorFlow 1 Detection Model Zoo archive.
TF_ARCHIVE="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
# Mirror: the same frozen graph committed to a public GitHub repo, for networks
# that can reach raw.githubusercontent.com but not download.tensorflow.org.
MIRROR_PB="https://raw.githubusercontent.com/rdeepc/ExploreOpencvDnn/master/models/frozen_inference_graph.pb"
MIRROR_PBTXT="https://raw.githubusercontent.com/rdeepc/ExploreOpencvDnn/master/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

# The official frozen graph's SHA-256. Both sources are verified against it, so
# a mirror that has drifted or been tampered with is rejected rather than used.
EXPECTED_PB_SHA256="2a8d8a89d695842e60d8c6d144181100555563e21acf2fa1e8f561fec5c3c6ad"

verify_pb() {
  [ -f "$PB" ] || return 1
  local actual
  actual="$(sha256sum "$PB" | cut -d' ' -f1)"
  if [ "$actual" != "$EXPECTED_PB_SHA256" ]; then
    echo "  checksum mismatch (got $actual)" >&2
    return 1
  fi
  return 0
}

if verify_pb && [ -f "$PBTXT" ]; then
  echo "SSD-MobileNet-V2 already present and verified in $DEST"
  exit 0
fi

echo "Fetching SSD-MobileNet-V2 (COCO) into $DEST ..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if curl -fsSL --max-time 900 -o "$TMP/model.tar.gz" "$TF_ARCHIVE" 2>/dev/null; then
  echo "  got the official TensorFlow archive"
  tar -xzf "$TMP/model.tar.gz" -C "$TMP"
  cp "$TMP"/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb "$PB"
else
  echo "  download.tensorflow.org unreachable, trying the GitHub mirror"
  curl -fsSL --max-time 900 -o "$PB" "$MIRROR_PB"
fi

if ! verify_pb; then
  echo "ERROR: downloaded weights failed SHA-256 verification; refusing to use them." >&2
  rm -f "$PB"
  exit 1
fi

# The .pbtxt is not inside the TF archive - it lives in OpenCV's test data.
if [ ! -f "$PBTXT" ]; then
  curl -fsSL --max-time 300 -o "$PBTXT" \
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt" \
    || curl -fsSL --max-time 300 -o "$PBTXT" "$MIRROR_PBTXT"
fi

echo "Done:"
ls -la "$PB" "$PBTXT"
