"""Unit tests for :mod:`integrations.happywhale_sink.connector`.

Every test stubs HTTP via ``httpx.MockTransport`` — no real network traffic.
"""

from __future__ import annotations

import json

import httpx
import pytest

from integrations.happywhale_sink import HappyWhaleAPIError, HappyWhaleConnector


def _mk_transport(handler):
    return httpx.MockTransport(handler)


def _ok_handler(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    # base_url carries the /v1 prefix
    assert request.url.path == "/v1/observations"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["content-type"].startswith("application/json")
    body = json.loads(request.content.decode())
    return httpx.Response(
        200, json={"id": "obs-42", "received": body, "status": "accepted"}
    )


def test_submit_observation_happy_path():
    transport = _mk_transport(_ok_handler)
    conn = HappyWhaleConnector(api_key="test-key", transport=transport)
    result = conn.submit_observation(
        {
            "species": "humpback_whale",
            "latitude": 42.0,
            "longitude": -70.5,
            "observed_at": "2026-04-15T12:00:00Z",
        }
    )
    assert result["id"] == "obs-42"
    assert result["status"] == "accepted"
    assert result["received"]["species"] == "humpback_whale"


def test_submit_observation_401_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    conn = HappyWhaleConnector(api_key="wrong-key", transport=_mk_transport(handler))
    with pytest.raises(HappyWhaleAPIError) as ei:
        conn.submit_observation(
            {
                "species": "orca",
                "latitude": 0.0,
                "longitude": 0.0,
                "observed_at": "2026-04-15",
            }
        )
    assert ei.value.status_code == 401
    assert "invalid API key" in ei.value.message


def test_submit_observation_429_rate_limited():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "30"}, json={})

    conn = HappyWhaleConnector(api_key="k", transport=_mk_transport(handler))
    with pytest.raises(HappyWhaleAPIError) as ei:
        conn.submit_observation(
            {
                "species": "orca",
                "latitude": 0.0,
                "longitude": 0.0,
                "observed_at": "2026-04-15",
            }
        )
    assert ei.value.status_code == 429
    assert "retry-after=30" in ei.value.message


def test_submit_observation_validates_required_fields():
    conn = HappyWhaleConnector(api_key="k", transport=_mk_transport(_ok_handler))
    with pytest.raises(ValueError) as ei:
        conn.submit_observation({"species": "orca"})
    assert "Missing required fields" in str(ei.value)


def test_empty_api_key_rejected():
    with pytest.raises(ValueError):
        HappyWhaleConnector(api_key="")


def test_health_returns_true_on_2xx():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/health"
        return httpx.Response(200, json={"ok": True})

    conn = HappyWhaleConnector(api_key="k", transport=_mk_transport(handler))
    import asyncio

    assert asyncio.run(conn.health_async()) is True


def test_health_returns_false_on_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    conn = HappyWhaleConnector(api_key="k", transport=_mk_transport(handler))
    import asyncio

    assert asyncio.run(conn.health_async()) is False
