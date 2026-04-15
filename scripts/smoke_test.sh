#!/usr/bin/env bash
# End-to-end smoke test for the EcoMarineAI HTTP service.
#
# Assumes docker compose is already running. Exits non-zero on any failure
# so it can be wired into CI directly.

set -euo pipefail

BACKEND="${BACKEND:-http://localhost:8000}"
TMP_DIR="$(mktemp -d)"
export TMP_DIR
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "→ Generating fixture images in ${TMP_DIR}..."
python3 - <<'PY'
import os
from PIL import Image, ImageDraw

tmp = os.environ["TMP_DIR"]
# A "non-cetacean" image: pure red 100×100 (matches the StubPipeline rule
# AND clearly fails CLIP positives — both the unit-test stub and the real
# model should reject this).
img_red = Image.new("RGB", (200, 200), color=(220, 20, 20))
img_red.save(os.path.join(tmp, "noise_red.png"))

# A "cetacean-like" image: greyscale gradient placeholder. Real CLIP will
# probably still reject this as not-a-whale, so the smoke test only asserts
# round-trip JSON shape for the negative case in lieu of bundling a real photo.
img_gradient = Image.new("L", (400, 200))
draw = ImageDraw.Draw(img_gradient)
for x in range(400):
    draw.line([(x, 0), (x, 200)], fill=int(x * 255 / 400))
img_gradient.convert("RGB").save(os.path.join(tmp, "gradient.png"))
print("Fixtures ready.")
PY

echo "→ Health check..."
curl -fsS "${BACKEND}/health" | tee /dev/null

echo
echo "→ POST /v1/predict-single (red noise → expect either rejected:true OR is_cetacean:false)"
RESP=$(curl -fsS -X POST -F "file=@${TMP_DIR}/noise_red.png;type=image/png" "${BACKEND}/v1/predict-single")
echo "$RESP" | python3 -m json.tool

if echo "$RESP" | python3 -c "
import json, sys
data = json.load(sys.stdin)
required = ['image_ind', 'bbox', 'class_animal', 'id_animal', 'probability', 'is_cetacean', 'cetacean_score', 'rejected', 'rejection_reason', 'model_version']
missing = [k for k in required if k not in data]
if missing:
    sys.exit(f'Missing fields in response: {missing}')
print('All Detection fields present.')
"; then
    echo "✓ Detection schema validated."
else
    echo "✗ Detection schema validation failed." >&2
    exit 1
fi

echo
echo "→ POST /v1/predict-batch (zip with 2 images)"
ZIP_PATH="${TMP_DIR}/batch.zip"
( cd "${TMP_DIR}" && zip -q "${ZIP_PATH}" noise_red.png gradient.png )
BATCH=$(curl -fsS -X POST -F "archive=@${ZIP_PATH};type=application/zip" "${BACKEND}/v1/predict-batch")
echo "$BATCH" | python3 -m json.tool | head -40

if echo "$BATCH" | python3 -c "
import json, sys
data = json.load(sys.stdin)
assert isinstance(data, list), f'expected list, got {type(data).__name__}'
assert len(data) == 2, f'expected 2 results, got {len(data)}'
print(f'Batch returned {len(data)} results.')
"; then
    echo "✓ Batch endpoint validated."
else
    echo "✗ Batch validation failed." >&2
    exit 1
fi

echo
echo "→ /metrics endpoint"
curl -fsS "${BACKEND}/metrics" | head -20

echo
echo "→ /v1/drift-stats endpoint"
curl -fsS "${BACKEND}/v1/drift-stats" | python3 -m json.tool

echo
echo "✅ Smoke test passed."
