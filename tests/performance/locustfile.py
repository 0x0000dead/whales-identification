"""
Performance tests for the Whales Identification API.

Usage:
    pip install locust
    locust -f tests/performance/locustfile.py --host http://localhost:8000

Target metrics (from project requirements):
    - p95 latency: < 8 seconds per 1920x1080 image
    - Processing speed: < 8 seconds per image
"""

import io

from locust import HttpUser, between, task
from PIL import Image


def create_test_image(width=448, height=448):
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


class WhaleAPIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.test_image = create_test_image()

    @task(3)
    def predict_single(self):
        self.client.post(
            "/v1/predict-single",
            files={"file": ("test.png", self.test_image, "image/png")},
        )

    @task(1)
    def predict_batch(self):
        import zipfile

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i in range(3):
                zf.writestr(f"img_{i}.png", self.test_image)
        zip_buf.seek(0)

        self.client.post(
            "/v1/predict-batch",
            files={
                "archive": ("batch.zip", zip_buf.getvalue(), "application/zip")
            },
        )

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def metrics_check(self):
        self.client.get("/metrics")
