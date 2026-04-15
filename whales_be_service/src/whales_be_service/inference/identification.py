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
_FALLBACK_CKPT = _BASE_DIR / "models" / "resnet101.pth"
# EfficientNet-B4 checkpoint from ktakita/happywhale-exp004-effb4-trainall
# mirrored to the EcoMarineAI HF org.
_EFFB4_CKPT = _BASE_DIR / "models" / "efficientnet_b4_512_fold0.ckpt"
_EFFB4_CLASSES = _BASE_DIR / "models" / "encoder_classes.npy"
_EFFB4_SPECIES_MAP = _BASE_DIR / "resources" / "species_map.csv"
_EFFB4_IMG_SIZE = 512
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
        fallback_ckpt: Path = _FALLBACK_CKPT,
        effb4_ckpt: Path = _EFFB4_CKPT,
        effb4_classes: Path = _EFFB4_CLASSES,
        effb4_species_map: Path = _EFFB4_SPECIES_MAP,
    ) -> None:
        self.csv_path = csv_path
        self.ckpt_path = ckpt_path
        self.fallback_ckpt = fallback_ckpt
        self.effb4_ckpt = effb4_ckpt
        self.effb4_classes = effb4_classes
        self.effb4_species_map = effb4_species_map
        self.img_size = img_size
        self.patch_size = patch_size
        self.model_version = model_version

        self._loaded = False
        # "vit_full" | "effb4_15k" | "resnet_fallback" | "uninitialised"
        self._mode: str = "uninitialised"
        self._model = None
        self._device = None
        self._transform = None
        self._torchvision_transform = None
        self._id_list: list[str] = []
        self._id_to_name: dict[str, str] = {}

    def _load(self) -> None:
        if self._loaded:
            return

        # Preferred: EfficientNet-B4 trained on 13837 individual_ids (ArcFace).
        # This is the canonical production checkpoint mirrored on our HF.
        if (
            self.effb4_ckpt.exists()
            and self.effb4_classes.exists()
            and self.effb4_species_map.exists()
        ):
            self._load_effb4_arcface()
            self.model_version = "effb4-arcface-v1"
            return

        # Secondary: original ViT + individual_id database ------------------
        if self.csv_path.exists() and self.ckpt_path.exists():
            self._load_vit_full()
            return

        # Fallback: a simple pickled torchvision classifier (e.g. resnet101) --
        if self.fallback_ckpt.exists():
            try:
                self._load_resnet_fallback()
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "ResNet fallback failed (%s); gate-only mode will be used.", e
                )

        raise FileNotFoundError(
            f"No identification weights found. Tried: {self.effb4_ckpt} (effb4), "
            f"{self.ckpt_path} (ViT), {self.fallback_ckpt} (ResNet). "
            "Run scripts/download_models.sh to populate models/."
        )

    def _load_effb4_arcface(self) -> None:
        """Load the EfficientNet-B4 + ArcFace checkpoint (13837 individuals).

        The checkpoint keys are::

            model.*             — timm efficientnet_b4 backbone
            embedding.{weight,bias} — Linear(1792 → 512)
            arc.weight          — [num_classes, 512] ArcFace row-wise cosine head

        Inference: image → backbone → adaptive_avg_pool → embedding → L2 normalise
        → cosine similarity vs normalised arc.weight → softmax → argmax →
        encoder_classes[idx] → species_map lookup.
        """
        import csv as _csv  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import timm  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from torchvision import transforms  # noqa: PLC0415

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class _EffB4Arcface(nn.Module):
            def __init__(self, num_classes: int = 15587) -> None:
                super().__init__()
                self.backbone = timm.create_model(
                    "efficientnet_b4", pretrained=False, num_classes=0, global_pool=""
                )
                self.embedding = nn.Linear(1792, 512)
                self.arc_weight = nn.Parameter(torch.zeros(num_classes, 512))

            def forward(self, x):
                feat = self.backbone(x)
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                emb = self.embedding(feat)
                emb_n = F.normalize(emb, dim=1)
                w_n = F.normalize(self.arc_weight, dim=1)
                return emb_n @ w_n.T

        classes = np.load(self.effb4_classes, allow_pickle=True)
        self._id_list = [str(c) for c in classes]

        with self.effb4_species_map.open() as f:
            self._id_to_name = {
                row["individual_id"]: row["species"] for row in _csv.DictReader(f)
            }

        model = _EffB4Arcface(num_classes=15587)
        ckpt = torch.load(  # nosec B614
            self.effb4_ckpt, map_location=self._device, weights_only=False
        )
        sd = ckpt["state_dict"]
        remap: dict = {}
        for k, v in sd.items():
            if k.startswith("model."):
                remap["backbone." + k[len("model."):]] = v
            elif k.startswith("embedding."):
                remap[k] = v
            elif k == "arc.weight":
                remap["arc_weight"] = v
        model.load_state_dict(remap, strict=False)
        model.eval().to(self._device)

        self._torchvision_transform = transforms.Compose(
            [
                transforms.Resize((_EFFB4_IMG_SIZE, _EFFB4_IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._model = model
        self._mode = "effb4_15k"
        self.model_version = "effb4-arcface-v1"
        self._loaded = True
        logger.info(
            "Loaded IdentificationModel (effb4_15k): %d classes, device=%s, ckpt=%s",
            len(self._id_list),
            self._device,
            self.effb4_ckpt.name,
        )

    def _load_vit_full(self) -> None:
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
        self._mode = "vit_full"
        self._loaded = True
        logger.info(
            "Loaded IdentificationModel (vit_full): %d classes, device=%s, ckpt=%s",
            len(self._id_list),
            self._device,
            self.ckpt_path.name,
        )

    def _load_resnet_fallback(self) -> None:
        """Minimal fallback: pickled torchvision ResNet used as a coarse species
        classifier. Without a proper class-to-name mapping, predictions are
        surfaced as ``class_idx:<N>`` strings. Better than nothing for
        demo / sanity-check purposes.
        """
        import torch  # noqa: PLC0415
        from torchvision import transforms  # noqa: PLC0415

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(  # nosec B614
            self.fallback_ckpt,
            map_location=self._device,
            weights_only=False,
        )
        model.eval()
        # Pre-built torchvision transforms — inferred from standard 224×224
        # ImageNet-style eval pipelines used by ResNet/EfficientNet checkpoints.
        self._torchvision_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._model = model
        self._mode = "resnet_fallback"
        self._loaded = True
        n_out = getattr(getattr(model, "fc", None), "out_features", None)
        logger.info(
            "Loaded IdentificationModel (resnet_fallback): out_features=%s, ckpt=%s",
            n_out,
            self.fallback_ckpt.name,
        )

    def predict(self, pil_img: "Image.Image") -> PredictionResult:
        """Run inference on a PIL image and return ``PredictionResult``.

        First call lazily loads the model. Subsequent calls reuse whichever
        backend was chosen (``vit_full`` or ``resnet_fallback``).
        """
        self._load()

        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        np_img = np.array(pil_img.convert("RGB"))
        h, w = np_img.shape[:2]

        if self._mode == "resnet_fallback":
            tensor = self._torchvision_transform(pil_img.convert("RGB")).unsqueeze(0).to(self._device)
            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                top_prob, top_idx = probs.max(0)
            class_id = f"class_idx:{int(top_idx)}"
            return PredictionResult(
                class_id=class_id,
                species="cetacean_coarse_fallback",
                probability=round(float(top_prob), 4),
                bbox=[0, 0, w, h],
            )

        if self._mode == "effb4_15k":
            tensor = self._torchvision_transform(pil_img.convert("RGB")).unsqueeze(0).to(self._device)
            n_active = len(self._id_list)
            with torch.no_grad():
                logits = self._model(tensor)  # [1, 15587] cosine similarities
                # Temperature-scaled softmax over just the active classes.
                scaled = logits[:, :n_active] * 30.0
                probs = torch.softmax(scaled, dim=1)[0]
                top_prob, top_idx = probs.max(0)
            class_id = self._id_list[int(top_idx)]
            species = self._id_to_name.get(class_id, class_id)
            return PredictionResult(
                class_id=class_id,
                species=species,
                probability=round(float(top_prob), 4),
                bbox=[0, 0, w, h],
            )

        # vit_full path
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

    def predict_topk(self, pil_img: "Image.Image", k: int = 5) -> list[tuple[str, str, float]]:
        """Return top-k predictions as [(class_id, species, probability), ...].

        Only supported by the ``effb4_15k`` backend; returns a single-element
        list for the other backends. Used by scripts/compute_metrics.py to
        compute top-5 accuracy honestly.
        """
        self._load()

        import torch  # noqa: PLC0415

        if self._mode == "effb4_15k":
            tensor = (
                self._torchvision_transform(pil_img.convert("RGB"))
                .unsqueeze(0)
                .to(self._device)
            )
            n_active = len(self._id_list)
            with torch.no_grad():
                logits = self._model(tensor)
                scaled = logits[:, :n_active] * 30.0
                probs = torch.softmax(scaled, dim=1)[0]
                top = probs.topk(min(k, n_active))
            results: list[tuple[str, str, float]] = []
            for prob, idx in zip(top.values.tolist(), top.indices.tolist(), strict=False):
                class_id = self._id_list[int(idx)]
                species = self._id_to_name.get(class_id, class_id)
                results.append((class_id, species, round(float(prob), 4)))
            return results

        # Fallback: single prediction wrapped in a list
        single = self.predict(pil_img)
        return [(single.class_id, single.species, single.probability)]

    def background_mask(self, img_bytes: bytes) -> str | None:
        """Return base64-encoded PNG with background removed (rembg).

        Falls back to ``None`` (no mask) if rembg isn't importable — some
        older Python versions have rembg bugs that terminate on import.
        """
        import base64  # noqa: PLC0415

        try:
            from rembg import remove  # noqa: PLC0415
        except (ImportError, SystemExit) as e:
            logger.warning("rembg unavailable (%s); skipping background mask.", e)
            return None
        try:
            return base64.b64encode(remove(img_bytes)).decode()
        except Exception as e:  # noqa: BLE001
            logger.warning("rembg.remove failed (%s); skipping background mask.", e)
            return None


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
