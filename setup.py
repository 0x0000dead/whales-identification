import os

from setuptools import setup

setup(
    name="whales_identification_library",
    version=os.environ.get("VERSION", "0.1.0"),
    packages=[
        "whales_identify",
    ],
    install_requires=["timm==1.0.9",
                      "wandb==0.18.3",
                      "opencv-python==4.10.0.84",
                      "numpy==2.1.2",
                      "pandas==2.2.3",
                      "torch==2.4.1",
                      "tqdm==4.66.5",
                      "scikit-learn==1.5.2",
                      "colorama==0.4.6",
                      "joblib==1.4.2",
                      "albumentations==1.4.18",
                      "setuptools"]
)
