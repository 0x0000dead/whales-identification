from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from starlette import status
from zipfile import ZipFile, BadZipFile
import csv, io, uuid, random

from starlette.middleware.cors import CORSMiddleware

app = FastAPI(title="Whales Identification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def mock_single_prediction() -> dict:
    """Возвращает предикт"""
    return {
        "whale_id": f"W-{random.randint(1, 999):03}",
        "confidence": round(random.uniform(0.8, 1.0), 3),
        "length_cm": random.randint(80, 200),
    }


@app.post("/predict-single", summary="Фото → JSON")
async def predict_single(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Только изображения.")

    _ = await file.read()  # передаём в ML-модель
    result = mock_single_prediction()
    return JSONResponse(result)


@app.post("/predict-batch", summary="ZIP → CSV")
async def predict_batch(archive: UploadFile = File(...)):
    if archive.content_type not in {"application/zip", "application/x-zip-compressed"}:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Ожидается ZIP-архив.")

    try:
        buf = io.BytesIO(await archive.read())
        with ZipFile(buf) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
    except BadZipFile:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Не удаётся распаковать архив.")

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["image", "whale_id", "confidence"])

    for name in names:
        pred = mock_single_prediction()
        writer.writerow([name, pred["whale_id"], pred["confidence"]])

    csv_buf.seek(0)
    headers = {
        "Content-Disposition": 'attachment; filename="predictions.csv"'
    }
    return StreamingResponse(iter([csv_buf.getvalue()]),
                             media_type="text/csv",
                             headers=headers)
