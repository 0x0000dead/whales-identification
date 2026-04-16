"""Unit tests for :mod:`integrations.inat_sink`."""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from integrations.inat_sink import INaturalistError, iNaturalistSink


def _transport(handler):
    return httpx.MockTransport(handler)


def test_build_observation_happy_path():
    body = iNaturalistSink.build_observation(
        taxon_name="Megaptera novaeangliae",
        lat=42.0,
        lon=-70.5,
        observed_on=date(2026, 4, 15),
        description="Breaching humpback",
    )
    obs = body["observation"]
    assert obs["species_guess"] == "Megaptera novaeangliae"
    assert obs["latitude"] == 42.0
    assert obs["longitude"] == -70.5
    assert obs["observed_on_string"] == "2026-04-15"
    assert obs["description"] == "Breaching humpback"
    assert obs["geoprivacy"] == "open"


def test_build_observation_rejects_bad_coords():
    with pytest.raises(ValueError):
        iNaturalistSink.build_observation(
            taxon_name="whale",
            lat=100.0,
            lon=0.0,
            observed_on="2026-04-15",
        )
    with pytest.raises(ValueError):
        iNaturalistSink.build_observation(
            taxon_name="whale",
            lat=0.0,
            lon=200.0,
            observed_on="2026-04-15",
        )


def test_build_observation_requires_taxon():
    with pytest.raises(ValueError):
        iNaturalistSink.build_observation(
            taxon_name="",
            lat=0.0,
            lon=0.0,
            observed_on="2026-04-15",
        )


def test_submit_observation_happy_path():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["method"] = request.method
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            201, json={"id": 999, "species_guess": "whale", "status": "ok"}
        )

    sink = iNaturalistSink(api_token="inat-token", transport=_transport(handler))
    result = sink.submit_observation(
        taxon_name="Megaptera",
        lat=42.0,
        lon=-70.0,
        observed_on=date(2026, 4, 15),
    )
    assert captured["method"] == "POST"
    # Base URL carries the /v1 prefix
    assert captured["path"] == "/v1/observations"
    assert captured["auth"] == "Bearer inat-token"
    assert captured["body"]["observation"]["species_guess"] == "Megaptera"
    assert result["id"] == 999


def test_submit_observation_401_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    sink = iNaturalistSink(api_token="bad", transport=_transport(handler))
    with pytest.raises(INaturalistError) as ei:
        sink.submit_observation(
            taxon_name="whale",
            lat=0.0,
            lon=0.0,
            observed_on="2026-04-15",
        )
    assert ei.value.status_code == 401


def test_submit_observation_429_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    sink = iNaturalistSink(api_token="k", transport=_transport(handler))
    with pytest.raises(INaturalistError) as ei:
        sink.submit_observation(
            taxon_name="whale",
            lat=0.0,
            lon=0.0,
            observed_on="2026-04-15",
        )
    assert ei.value.status_code == 429


def test_extra_fields_merge_into_observation():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["observation"]["custom_field"] == "hello"
        return httpx.Response(201, json={"id": 1})

    sink = iNaturalistSink(api_token="k", transport=_transport(handler))
    sink.submit_observation(
        taxon_name="whale",
        lat=0.0,
        lon=0.0,
        observed_on="2026-04-15",
        extra={"custom_field": "hello"},
    )
