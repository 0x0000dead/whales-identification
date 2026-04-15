"""Cetacean individual identification model — lazy-loaded singleton.

This module replaces ``whale_infer.py`` and the inference half of
``response_models.py``. Key fix: model loading is **lazy** (deferred until first
``predict()`` call), not a module-level side effect. Importing this module is
free; loading torch + ViT happens only when the pipeline is actually built at
FastAPI startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import PredictionResult

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CSV = _BASE_DIR / "resources" / "database.csv"
_DEFAULT_CKPT = _BASE_DIR / "models" / "model-e15.pt"
_DEFAULT_IMG_SIZE = 448
_DEFAULT_PATCH = 32


class IdentificationModel:
    """Lazy-loaded ViT identifier for cetacean individuals.

    Why lazy: the previous ``whale_infer.py`` did ``torch.load()`` at import
    time, which crashed under pytest, broke cold start, and made the API
    impossible to mock. Now construction is cheap and the heavy work happens
    inside ``_load()``, gated by ``_loaded``.
    """

    def __init__(
        self,
        csv_path: Path = _DEFAULT_CSV,
        ckpt_path: Path = _DEFAULT_CKPT,
        img_size: int = _DEFAULT_IMG_SIZE,
        patch_size: int = _DEFAULT_PATCH,
        model_version: str = "vit_l32-v1",
    ) -> None:
        self.csv_path = csv_path
        self.ckpt_path = ckpt_path
        self.img_size = img_size
        self.patch_size = patch_size
        self.model_version = model_version

        self._loaded = False
        self._model = None
        self._device = None
        self._transform = None
        self._id_list: list[str] = []
        self._id_to_name: dict[str, str] = {}

    def _load(self) -> None:
        if self._loaded:
            return

        # Cheap pre-flight checks first — fail fast before pulling in torch.
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Identification database not found at {self.csv_path}. "
                "Run scripts/download_models.sh or set IDENTIFICATION_CSV env var."
            )
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.ckpt_path}. "
                "Run scripts/download_models.sh to fetch weights."
            )

        import albumentations as A  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from albumentations.pytorch import ToTensorV2  # noqa: PLC0415

        df = pd.read_csv(self.csv_path).drop_duplicates("individual_id")
        self._id_list = df["individual_id"].astype(str).tolist()
        self._id_to_name = dict(
            zip(df["individual_id"].astype(str), df["species"], strict=False)
        )

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._transform = A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        )

        model = _VisionTransformer(
            embed_dim=784,
            hidden_dim=1568,
            num_heads=8,
            num_layers=6,
            patch_size=self.patch_size,
            num_channels=3,
            num_patches=196,
            num_classes=len(self._id_list),
            dropout=0.2,
        ).to(self._device)
        model.eval()

        state = torch.load(  # nosec B614 - trusted local checkpoint
            self.ckpt_path,
            map_location=self._device,
            weights_only=False,
        )
        model.load_state_dict(state["model_state_dict"], strict=False)

        self._model = model
        self._loaded = True
        logger.info(
            "Loaded IdentificationModel: %d classes, device=%s, ckpt=%s",
            len(self._id_list),
            self._device,
            self.ckpt_path.name,
        )

    def predict(self, pil_img: "Image.Image") -> PredictionResult:
        """Run inference on a PIL image and return ``PredictionResult``.

        First call lazily loads the model. Subsequent calls reuse it.
        """
        self._load()

        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        # PIL → BGR numpy (cv2 convention) → RGB
        np_img = np.array(pil_img.convert("RGB"))
        h, w = np_img.shape[:2]

        tensor = self._transform(image=np_img)["image"].unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_prob, top_idx = probs.max(0)

        class_id = self._id_list[int(top_idx)]
        species = self._id_to_name.get(class_id, class_id)
        return PredictionResult(
            class_id=class_id,
            species=species,
            probability=round(float(top_prob), 4),
            bbox=[0, 0, w, h],
        )

    def predict_array(self, np_img: "np.ndarray") -> PredictionResult:
        """Convenience wrapper for code paths that already have a numpy image."""
        from PIL import Image  # noqa: PLC0415

        return self.predict(Image.fromarray(np_img))

    def background_mask(self, img_bytes: bytes) -> str:
        """Return base64-encoded PNG with background removed (rembg)."""
        import base64  # noqa: PLC0415

        from rembg import remove  # noqa: PLC0415

        return base64.b64encode(remove(img_bytes)).decode()


# ---------------------------------------------------------------------------
# Private ViT implementation (lifted verbatim from response_models.py).
# Kept module-private because the only consumer is IdentificationModel above.
# ---------------------------------------------------------------------------


def _img_to_patch(x, patch: int, flat: bool = True):
    import torch  # noqa: F401, PLC0415

    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch, patch, W // patch, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).flatten(1, 2)
    if flat:
        x = x.flatten(2, 4)
    return x


def _make_attention_block(embed_dim: int, hidden_dim: int, num_heads: int, dropout: float):
    import torch.nn as nn  # noqa: PLC0415

    class AttentionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            y = self.norm1(x)
            x = x + self.attn(y, y, y)[0]
            x = x + self.mlp(self.norm2(x))
            return x

    return AttentionBlock()


class _VisionTransformer:
    """Thin wrapper that defers torch.nn import to instantiation time."""

    def __new__(cls, **kwargs):
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        embed_dim = kwargs["embed_dim"]
        hidden_dim = kwargs["hidden_dim"]
        num_channels = kwargs["num_channels"]
        num_heads = kwargs["num_heads"]
        num_layers = kwargs["num_layers"]
        num_classes = kwargs["num_classes"]
        patch_size = kwargs["patch_size"]
        num_patches = kwargs["num_patches"]
        dropout = kwargs.get("dropout", 0.0)

        class VisionTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_size = patch_size
                self.inp = nn.Linear(num_channels * patch_size**2, embed_dim)
                self.blocks = nn.Sequential(
                    *[
                        _make_attention_block(embed_dim, hidden_dim, num_heads, dropout)
                        for _ in range(num_layers)
                    ]
                )
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, num_classes),
                )
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
                self.pos_emb = nn.Parameter(
                    torch.randn(1, 1 + num_patches, embed_dim)
                )
                self.drop = nn.Dropout(dropout)

            def forward(self, x):
                x = _img_to_patch(x, self.patch_size)
                B, T, _ = x.shape
                x = self.inp(x)
                cls = self.cls_token.repeat(B, 1, 1)
                x = torch.cat([cls, x], 1) + self.pos_emb[:, : T + 1]
                x = self.drop(x).transpose(0, 1)
                x = self.blocks(x)
                return self.mlp_head(x[0])

        return VisionTransformer()
