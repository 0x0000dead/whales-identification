"""INT8 dynamic quantization of the production EfficientNet-B4 ArcFace checkpoint.

Part of Stage 3 · §3.5 Оптимизация параметров моделей. Applies
``torch.quantization.quantize_dynamic`` to every ``nn.Linear`` in the ArcFace
backbone + embedding head and saves the resulting state dict to a new file.

Dynamic quantization is the *safe* choice here: it quantises weights once at
save-time and activations on-the-fly during inference. No calibration dataset
is required, and the accuracy drop on image classifiers with a linear head is
typically < 1 pp top-1 (see PyTorch blog, "Dynamic Quantization on BERT").

USAGE
-----

    poetry run python scripts/quantize_effb4.py \\
        --ckpt whales_be_service/src/whales_be_service/models/efficientnet_b4_512_fold0.ckpt \\
        --out models/effb4_int8.pt \\
        --benchmark

The ``--benchmark`` flag additionally runs 20 forward passes on a dummy
(1, 3, 512, 512) tensor for fp32 and int8 variants and prints their mean
latency — enough to feed the "INT8 results" section of ``reports/OPTIMIZATION.md``.

CAVEATS
-------

* Peak RAM during quantisation is ~1.2 GB (full EffB4 backbone + ArcFace head
  held twice, briefly). **Do NOT run this in CI** — it will OOM the GitHub
  Actions runner. Run locally on a dev box or on a CPU-only inference VM.
* The dynamic-quantised model can ONLY run on CPU (torch restriction). If you
  need GPU inference, use static PTQ or QAT instead — out of scope for Stage 3.
* encoder_classes.npy and species_map.csv are NOT touched — they're just index
  maps, not part of the neural net.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("quantize_effb4")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)


# ---------------------------------------------------------------------------
# Model definition — copy of the inner class in
# whales_be_service.inference.identification._load_effb4_arcface. Duplicated
# intentionally so this script has zero dependency on the backend package and
# can be shipped standalone.
# ---------------------------------------------------------------------------


def _build_effb4_arcface(num_classes: int = 15587):
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _EffB4Arcface(nn.Module):
        def __init__(self) -> None:
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

    return _EffB4Arcface()


def _load_checkpoint(model, ckpt_path: Path) -> None:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # nosec B614
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    remap: dict = {}
    for k, v in sd.items():
        if k.startswith("model."):
            remap["backbone." + k[len("model.") :]] = v
        elif k.startswith("embedding."):
            remap[k] = v
        elif k == "arc.weight":
            remap["arc_weight"] = v
    missing, unexpected = model.load_state_dict(remap, strict=False)
    logger.info(
        "Loaded %d tensors (missing=%d unexpected=%d)",
        len(remap),
        len(missing),
        len(unexpected),
    )


def _apply_dynamic_int8(model):
    """Quantise every ``nn.Linear`` → int8 dynamically."""
    import torch
    import torch.nn as nn

    logger.info("Applying torch.quantization.quantize_dynamic (Linear → int8)")
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return qmodel


def _benchmark(model, *, runs: int, label: str) -> dict:
    import numpy as np
    import torch

    model.eval()
    dummy = torch.randn(1, 3, 512, 512)
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(dummy)

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    result = {
        "label": label,
        "runs": runs,
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }
    logger.info(
        "%-6s  mean=%6.2f ms  p50=%6.2f ms  p95=%6.2f ms",
        label,
        result["mean_ms"],
        result["p50_ms"],
        result["p95_ms"],
    )
    return result


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to efficientnet_b4_512_fold0.ckpt (PyTorch Lightning format).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/effb4_int8.pt"),
        help="Destination for the quantised state dict.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=15587,
        help="num_classes of the ArcFace head (matches the checkpoint).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a fp32-vs-int8 latency benchmark on a dummy (1, 3, 512, 512) input.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of benchmark runs per variant.",
    )
    args = parser.parse_args(argv)

    if not args.ckpt.exists():
        logger.error("Checkpoint not found: %s", args.ckpt)
        return 2

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "PyTorch is not installed. Install with `poetry install` in whales_be_service."
        )
        return 3

    # ---------- Load fp32 ----------
    logger.info("Building fp32 model…")
    model_fp32 = _build_effb4_arcface(num_classes=args.num_classes)
    _load_checkpoint(model_fp32, args.ckpt)
    model_fp32.eval()

    fp32_results: dict = {}
    int8_results: dict = {}
    if args.benchmark:
        fp32_results = _benchmark(model_fp32, runs=args.runs, label="fp32")

    # ---------- Quantise ----------
    qmodel = _apply_dynamic_int8(model_fp32)

    # ---------- Save ----------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import torch  # noqa: PLC0415

    torch.save(qmodel.state_dict(), args.out)  # nosec B614
    logger.info(
        "Saved quantised model  path=%s  size=%.2f MB  fp32=%.2f MB",
        args.out,
        _file_size_mb(args.out),
        _file_size_mb(args.ckpt),
    )

    if args.benchmark:
        int8_results = _benchmark(qmodel, runs=args.runs, label="int8")
        logger.info(
            "SPEEDUP: fp32=%.2f ms → int8=%.2f ms (%.2f×)",
            fp32_results["mean_ms"],
            int8_results["mean_ms"],
            fp32_results["mean_ms"] / max(int8_results["mean_ms"], 1e-6),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
