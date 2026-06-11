"""CORS configuration tests.

The UI must be able to call the API from any machine on the network when
``ALLOWED_ORIGINS="*"`` (the docker-compose dev default), while explicit
origin lists keep working with credentials enabled.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from whales_be_service.main import _setup_cors

LAN_ORIGIN = "http://192.168.1.50:8080"


def _client_with_origins(monkeypatch, origins_value: str) -> TestClient:
    monkeypatch.setenv("ALLOWED_ORIGINS", origins_value)
    app = FastAPI()

    @app.post("/ping")
    def ping() -> dict:
        return {"ok": True}

    _setup_cors(app)
    return TestClient(app)


def _preflight(client: TestClient):
    return client.options(
        "/ping",
        headers={
            "Origin": LAN_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )


def test_wildcard_allows_any_lan_origin(monkeypatch):
    client = _client_with_origins(monkeypatch, "*")
    resp = _preflight(client)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"
    # Wildcard origin must not be combined with credentials (CORS spec).
    assert "access-control-allow-credentials" not in resp.headers


def test_explicit_list_allows_listed_origin(monkeypatch):
    client = _client_with_origins(monkeypatch, f"http://localhost:5173,{LAN_ORIGIN}")
    resp = _preflight(client)
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == LAN_ORIGIN
    assert resp.headers["access-control-allow-credentials"] == "true"


def test_explicit_list_blocks_unlisted_origin(monkeypatch):
    client = _client_with_origins(monkeypatch, "http://localhost:5173")
    resp = _preflight(client)
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers
