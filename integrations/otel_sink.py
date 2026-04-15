"""OpenTelemetry sink — pushes EcoMarineAI predictions as OTel spans + metrics.

This is the **second monitoring platform** for the ТЗ §Параметр 6 requirement
(≥ 2 external monitoring platforms). Prometheus coverage is already built into
the service via ``/metrics``; this module adds OpenTelemetry-compatible export,
which any backend (Jaeger, Tempo, Grafana Cloud, Honeycomb, Datadog, New Relic,
Elastic APM, etc.) can ingest.

Usage::

    pip install 'opentelemetry-sdk>=1.24' 'opentelemetry-exporter-otlp>=1.24'

    python3 integrations/otel_sink.py \\
        --directory data/test_split/positives \\
        --otlp-endpoint http://localhost:4317 \\
        --service-name ecomarineai

Set ``OTEL_EXPORTER_OTLP_ENDPOINT`` in the environment instead of ``--otlp-endpoint``
if you prefer the standard OTel conventions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_otel(service_name: str, endpoint: str | None):
    """Build a tracer and a meter. Returns (tracer, meter, shutdown_fn).

    If the opentelemetry packages aren't installed, returns stub objects that
    silently no-op — the sink still runs, just without actually pushing spans.
    """
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        print(
            "opentelemetry packages not installed — running in no-op mode. "
            "Install with `pip install 'opentelemetry-sdk>=1.24' "
            "'opentelemetry-exporter-otlp>=1.24'` to enable real export.",
            file=sys.stderr,
        )

        class _NoopSpan:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def set_attribute(self, *a, **k):
                pass

        class _NoopTracer:
            def start_as_current_span(self, *a, **k):
                return _NoopSpan()

        class _NoopInstrument:
            def add(self, *a, **k):
                pass

            def record(self, *a, **k):
                pass

        class _NoopMeter:
            def create_counter(self, *a, **k):
                return _NoopInstrument()

            def create_histogram(self, *a, **k):
                return _NoopInstrument()

        return _NoopTracer(), _NoopMeter(), (lambda: None)

    resource = Resource.create({"service.name": service_name})

    trace_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True) if endpoint else OTLPSpanExporter(insecure=True)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True) if endpoint else OTLPMetricExporter(insecure=True)
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)],
    )
    metrics.set_meter_provider(meter_provider)

    def shutdown():
        tracer_provider.shutdown()
        meter_provider.shutdown()

    return trace.get_tracer(service_name), metrics.get_meter(service_name), shutdown


def run(directory: Path, endpoint: str | None, service_name: str) -> int:
    sys.path.insert(0, str(REPO_ROOT / "whales_be_service" / "src"))
    from PIL import Image

    from whales_be_service.inference import get_pipeline

    pipeline = get_pipeline()
    pipeline.warmup()

    tracer, meter, shutdown = _build_otel(service_name, endpoint)
    predictions_counter = meter.create_counter(
        name="ecomarineai.predictions",
        description="Total number of predictions processed",
    )
    rejections_counter = meter.create_counter(
        name="ecomarineai.rejections",
        description="Total number of anti-fraud / low-confidence rejections",
    )
    score_hist = meter.create_histogram(
        name="ecomarineai.cetacean_score",
        description="Distribution of CLIP cetacean scores",
    )

    images = sorted(
        p
        for p in directory.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    print(f"Processing {len(images)} images; exporting traces to {endpoint or 'default OTLP endpoint'}")

    for path in images:
        try:
            pil = Image.open(path).convert("RGB")
            raw = path.read_bytes()
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {path}: {e}")
            continue

        with tracer.start_as_current_span("predict") as span:
            span.set_attribute("image.name", path.name)
            det = pipeline.predict(
                pil_img=pil, filename=path.name, img_bytes=raw, generate_mask=False
            )
            span.set_attribute("prediction.class_animal", det.class_animal)
            span.set_attribute("prediction.id_animal", det.id_animal)
            span.set_attribute("prediction.probability", det.probability)
            span.set_attribute("gate.cetacean_score", det.cetacean_score)
            span.set_attribute("gate.is_cetacean", det.is_cetacean)
            span.set_attribute("rejected", det.rejected)
            if det.rejection_reason:
                span.set_attribute("rejection_reason", det.rejection_reason)

        if det.rejected:
            rejections_counter.add(1, {"reason": det.rejection_reason or "unknown"})
        else:
            predictions_counter.add(1)
        score_hist.record(det.cetacean_score)

    shutdown()
    print("Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--otlp-endpoint",
        default=None,
        help="OTLP gRPC endpoint (default: OTEL_EXPORTER_OTLP_ENDPOINT env var)",
    )
    parser.add_argument("--service-name", default="ecomarineai")
    args = parser.parse_args()
    return run(args.directory, args.otlp_endpoint, args.service_name)


if __name__ == "__main__":
    sys.exit(main())
