"""Unit tests for :mod:`integrations.gbif_sink`."""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from integrations.gbif_sink import GBIFError, GBIFSink


def _transport(handler):
    return httpx.MockTransport(handler)


def test_build_darwin_core_happy_path():
    rec = GBIFSink.build_darwin_core(
        species="humpback_whale",
        latitude=42.0,
        longitude=-70.5,
        observed_on=date(2026, 4, 15),
        count=3,
    )
    assert rec["basisOfRecord"] == "MachineObservation"
    assert rec["scientificName"] == "humpback whale"  # underscore → space
    assert rec["decimalLatitude"] == 42.0
    assert rec["decimalLongitude"] == -70.5
    assert rec["eventDate"] == "2026-04-15"
    assert rec["individualCount"] == 3
    assert rec["geodeticDatum"] == "WGS84"
    assert rec["occurrenceStatus"] == "present"


def test_build_darwin_core_rejects_bad_lat():
    with pytest.raises(ValueError):
        GBIFSink.build_darwin_core(
            species="orca",
            latitude=200.0,
            longitude=0.0,
            observed_on="2026-04-15",
        )


def test_build_darwin_core_rejects_bad_lon():
    with pytest.raises(ValueError):
        GBIFSink.build_darwin_core(
            species="orca",
            latitude=0.0,
            longitude=-999.0,
            observed_on="2026-04-15",
        )


def test_build_darwin_core_rejects_zero_count():
    with pytest.raises(ValueError):
        GBIFSink.build_darwin_core(
            species="orca",
            latitude=0.0,
            longitude=0.0,
            observed_on="2026-04-15",
            count=0,
        )


def test_build_darwin_core_requires_species():
    with pytest.raises(ValueError):
        GBIFSink.build_darwin_core(
            species="",
            latitude=0.0,
            longitude=0.0,
            observed_on="2026-04-15",
        )


def test_push_occurrence_sends_correct_payload():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(201, json={"key": 12345, "status": "created"})

    sink = GBIFSink(
        api_key="gbif-key",
        dataset_key="dataset-uuid",
        transport=_transport(handler),
    )
    result = sink.push_occurrence(
        species="orcinus_orca",
        latitude=50.1,
        longitude=-125.2,
        observed_on=date(2026, 4, 15),
        count=2,
    )
    assert captured["method"] == "POST"
    # Base URL carries the /v1 prefix
    assert captured["path"] == "/v1/occurrence"
    assert captured["auth"] == "Bearer gbif-key"
    assert captured["body"]["scientificName"] == "orcinus orca"
    assert captured["body"]["datasetKey"] == "dataset-uuid"
    assert result == {"key": 12345, "status": "created"}


def test_push_occurrence_401_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    sink = GBIFSink(api_key="bad", transport=_transport(handler))
    with pytest.raises(GBIFError) as ei:
        sink.push_occurrence(
            species="orca",
            latitude=0.0,
            longitude=0.0,
            observed_on="2026-04-15",
        )
    assert ei.value.status_code == 401


def test_push_occurrence_429_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    sink = GBIFSink(api_key="k", transport=_transport(handler))
    with pytest.raises(GBIFError) as ei:
        sink.push_occurrence(
            species="orca",
            latitude=0.0,
            longitude=0.0,
            observed_on="2026-04-15",
        )
    assert ei.value.status_code == 429
