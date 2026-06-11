import os

from setuptools import setup

# Runtime dependencies are intentionally minimal: importing
# ``whales_identify`` / ``whales_identify.cli`` only needs the standard
# library, and the CLI lazily imports Pillow to open images. The inference
# stack itself (torch, timm, rembg, ...) is provided by the consuming
# environment (see whales_be_service/pyproject.toml).
#
# The full training stack is opt-in:
#
#     pip install -e ".[train]"

setup(
    name="whales_identification_library",
    version=os.environ.get("VERSION", "0.1.0"),
    packages=[
        "whales_identify",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pillow",
    ],
    extras_require={
        "train": [
            "timm==1.0.9",
            "wandb==0.18.3",
            "opencv-python==4.10.0.84",
            "numpy>=1.26,<3",
            "pandas==2.2.3",
            "torch==2.4.1",
            "tqdm==4.66.5",
            "scikit-learn==1.5.2",
            "colorama==0.4.6",
            "joblib==1.4.2",
            "albumentations==1.4.18",
        ],
    },
)
