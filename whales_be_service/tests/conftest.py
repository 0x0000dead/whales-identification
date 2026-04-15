import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


class _StubGate:
    def __init__(self, is_cetacean: bool = True, score: float = 0.87) -> None:
        self.is_cetacean = is_cetacean
        self.score = score

    def __call__(self, pil_img):  # not used
        from whales_be_service.inference.schemas import GateResult

        return GateResult(
            positive_score=self.score,
            negative_score=1 - self.score,
            is_cetacean=self.is_cetacean,
            margin=2 * self.score - 1,
        )


class StubPipeline:
    """Predictable in-process pipeline used by API tests.

    Behaviour: any image with the red channel > 200 is "rejected as not a
    marine mammal"; everything else is "accepted" with a fixed individual ID.
    Bypasses CLIP and the ViT entirely.
    """

    model_version = "stub-v1"

    def predict(self, pil_img, filename, img_bytes=None, generate_mask=True):
        from whales_be_service.inference.schemas import RejectionReason
        from whales_be_service.response_models import Detection

        red = pil_img.convert("RGB").getpixel((0, 0))[0]
        if red > 200:
            return Detection(
                image_ind=filename,
                bbox=[0, 0, pil_img.width, pil_img.height],
                class_animal="",
                id_animal="unknown",
                probability=0.0,
                mask=None,
                is_cetacean=False,
                cetacean_score=0.12,
                rejected=True,
                rejection_reason=RejectionReason.NOT_A_MARINE_MAMMAL.value,
                model_version=self.model_version,
            )

        return Detection(
            image_ind=filename,
            bbox=[0, 0, pil_img.width, pil_img.height],
            class_animal="1a71fbb72250",
            id_animal="humpback_whale",
            probability=0.92,
            mask=None,
            is_cetacean=True,
            cetacean_score=0.91,
            rejected=False,
            rejection_reason=None,
            model_version=self.model_version,
        )

    def warmup(self) -> None:  # pragma: no cover
        pass


@pytest.fixture(autouse=True)
def _override_pipeline():
    """Replace the real inference pipeline with the stub for every API test."""
    from whales_be_service.main import app, get_pipeline_dep

    stub = StubPipeline()
    app.dependency_overrides[get_pipeline_dep] = lambda: stub
    yield
    app.dependency_overrides.pop(get_pipeline_dep, None)
