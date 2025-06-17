from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from zipfile import ZipFile, BadZipFile
from pydantic import BaseModel
import io, random, base64, yaml
from PIL import Image, UnidentifiedImageError

from .response_models import Detection

app = FastAPI(title="Whales Identification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- загружаем маппинг id → человекочитаемое имя ---
BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
ID_TO_NAME = cfg.get("id_to_name", {})


def detection_id(filename: str, img_bytes: bytes) -> dict:
    bbox = [random.randint(0, 50) for _ in range(4)]
    class_id = "whale"
    prob = round(random.uniform(0.8, 1.0), 3)

    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image_ind": filename,
        "bbox": bbox,
        "class_animal": class_id,
        "id_animal": ID_TO_NAME.get(class_id, class_id),
        "probability": prob,
        "mask": mask_b64
    }


@app.post("/predict-single", response_model=Detection, summary="Фото → JSON с bbox+mask")
async def predict_single(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Только изображения.")
    data = await file.read()
    det = detection_id(file.filename, data)
    return JSONResponse(content=det)


@app.post("/predict-batch", summary="ZIP → JSON[]")
async def predict_batch(archive: UploadFile = File(...)):
    if archive.content_type not in ("application/zip", "application/x-zip-compressed"):
        raise HTTPException(415, "Ожидается ZIP-архив.")

    raw = await archive.read()
    try:
        zf = ZipFile(io.BytesIO(raw))
    except BadZipFile:
        raise HTTPException(400, "Не удаётся распаковать архив.")

    results: list[dict] = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue

        try:
            img_bytes = zf.read(name)

            with Image.open(io.BytesIO(img_bytes)) as img:
                img.verify()

            det = detection_id(name, img_bytes)
            results.append(det)

        except (KeyError, UnidentifiedImageError):
            continue
        except Exception:
            continue

    zf.close()
    return JSONResponse(content=results)