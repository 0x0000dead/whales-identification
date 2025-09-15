from pydantic import BaseModel
from rembg import remove
from PIL import Image
import base64
import io

import io, random, yaml, cv2, numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import pandas as pd

class Detection(BaseModel):
    image_ind: str
    bbox: list[int]
    class_animal: str
    id_animal: str
    probability: float
    mask: str | None = None # base64 PNG с удалённым фоном


# MASK MODULE

def generate_base64_mask_with_removed_background(img_bytes: bytes) -> str:
    """
    Принимает байты изображения, удаляет фон и возвращает base64 PNG без фона.
    """
    # Удаление фона
    output = remove(img_bytes)
    
    # Преобразование результата в PNG и кодирование в base64
    processed_image = Image.open(io.BytesIO(output)).convert("RGBA")
    buf = io.BytesIO()
    processed_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()



# BEST TRAINED MODEL for probability

def get_precitions(img_bytes):
    CSV_PATH = "/Users/savandanov/Documents/Github/whales-identification/research/demo-ui/resources/database.csv"
    df = pd.read_csv(CSV_PATH)
    uniq_df = df.drop_duplicates("individual_id")        # ← только 15 587 строк
    CLASS_ID_LIST = uniq_df["individual_id"].astype(str).tolist()

    # ② Словарь id → species
    ID_TO_NAME = dict(zip(uniq_df["individual_id"].astype(str), uniq_df["species"]))

    # ③ Быстрая sanity-проверка
    assert len(CLASS_ID_LIST) == 15_587, f"Ожидалось 15 587, а получили {len(CLASS_ID_LIST)}"
    # — сама сеть
    _model = VisionTransformer(
        embed_dim=784,
        hidden_dim=1568,
        num_heads=8,
        num_layers=6,
        patch_size=32,
        num_channels=3,
        num_patches=196,
        num_classes=len(CLASS_ID_LIST),
        dropout=0.2,
    ).to(CONFIG["device"]).eval()

    ckpt = torch.load("/Users/savandanov/Documents/Github/whales-identification/research/demo-ui/models/model-e15.pt", map_location=CONFIG["device"])
    _model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    loader = DataLoader(
        HappyWhaleTestDataset([np_img], transforms=data_transforms),
        batch_size=1,
        shuffle=False,
    )

    with torch.no_grad():
        batch = next(iter(loader))["image"].to(CONFIG["device"])
        logits = _model(batch)
        probs = torch.softmax(logits, 1)
        top_prob, top_idx = probs[0].max(0)

    class_idx = int(top_idx.item())
    class_id = CLASS_ID_LIST[class_idx]          # hex-id из соревки
    name = ID_TO_NAME.get(class_id, class_id)    # читаемое имя / fallback

    # --- простая заглушка bbox (если ещё не детектируешь) ---
    bbox = [0, 0, np_img.shape[1], np_img.shape[0]]


CONFIG = {
    "img_size": 448,
    "test_batch_size": 1,
    "num_classes": 15_587,
    "patch_size": 32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

data_transforms = A.Compose(
    [
        A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)


class HappyWhaleTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transforms=None):
        self.img_list = img_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return {"image": img}



class AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def img_to_patch(x, patch, flat=True):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch, patch, W // patch, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).flatten(1, 2)
    if flat:
        x = x.flatten(2, 4)
    return x


class VisionTransformer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.inp = torch.nn.Linear(num_channels * patch_size ** 2, embed_dim)
        self.blocks = torch.nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim), torch.nn.Linear(embed_dim, num_classes)
        )
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = torch.nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.inp(x)

        cls = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls, x], 1) + self.pos_emb[:, : T + 1]
        x = self.drop(x).transpose(0, 1)  # [T+1, B, D]
        x = self.blocks(x)
        return self.mlp_head(x[0])

