# whale_infer.py -----------------------------------------------------------
import io, base64, cv2, torch, albumentations as A, numpy as np, pandas as pd
from albumentations.pytorch import ToTensorV2
from rembg import remove
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ---------- параметры ----------
ROOT = Path(__file__).resolve().parent          # папка с кодом
CSV  = ROOT/"resources/database.csv"
CKPT = ROOT/"models/model-e15.pt"
DEVICE, IMG_SIZE, PATCH = (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                           448, 32)

# ---------- трансформ ----------
_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE),
                 A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                 ToTensorV2()], p=1)

class _DS(Dataset):
    def __init__(self, imgs): self.imgs = imgs
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = cv2.cvtColor(self.imgs[i], cv2.COLOR_BGR2RGB)
        return {"image": _tf(image=img)["image"]}

# ---------- CSV → id-список + словарь ----------
_df      = pd.read_csv(CSV).drop_duplicates("individual_id")
ID_LIST  = _df["individual_id"].astype(str).tolist()            # 15 587
ID2NAME  = dict(zip(_df["individual_id"].astype(str), _df["species"]))

# ---------- модель (из твоего кода, без изменений) ----------
from vision_transformer import VisionTransformer        # импортируй реализацию

_MODEL = VisionTransformer(
        embed_dim=784, hidden_dim=1568,
        num_heads=8, num_layers=6,
        patch_size=PATCH, num_channels=3,
        num_patches=196, num_classes=len(ID_LIST),
        dropout=0.2).to(DEVICE).eval()

_STATE = torch.load(CKPT, map_location=DEVICE)
_MODEL.load_state_dict(_STATE["model_state_dict"], strict=False)

# ---------- helpers ----------
def _mask_b64(img_b: bytes) -> str:
    return base64.b64encode(remove(img_b)).decode()

def infer(filename: str, img_bytes: bytes) -> dict:
    np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    dl     = DataLoader(_DS([np_img]), batch_size=1)
    with torch.no_grad():
        logits = _MODEL(next(iter(dl))["image"].to(DEVICE))
        prob, idx = torch.softmax(logits,1)[0].max(0)

    class_id = ID_LIST[int(idx)]
    return {
        "image_ind":   filename,
        "bbox":        [0,0,np_img.shape[1], np_img.shape[0]],
        "class_animal":class_id,
        "id_animal":   ID2NAME.get(class_id, class_id),
        "probability": round(float(prob),4),
        "mask":        _mask_b64(img_bytes),
    }
