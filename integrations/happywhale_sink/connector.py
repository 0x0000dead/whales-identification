"""HappyWhale REST connector.

Submits Detection-shaped observations to the HappyWhale public API. The real
endpoint is protected by a bearer token which is expected to live in
``HAPPYWHALE_API_KEY`` — never commit this value.

Design notes:
- Async-first via ``httpx.AsyncClient`` so the connector can be driven from
  FastAPI BackgroundTasks without blocking the event loop.
- The sync wrapper uses ``asyncio.run`` so existing scripts and CLI tools can
  call the connector without caring about concurrency.
- Tests inject an ``httpx.MockTransport`` to assert request shaping — no real
  HTTP traffic ever leaves the test process.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.happywhale.com/v1"


class HappyWhaleAPIError(RuntimeError):
    """Raised when the HappyWhale API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HappyWhale API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class HappyWhaleConnector:
    """Thin HTTPS client for the HappyWhale observations endpoint.

    Parameters
    ----------
    api_key:
        Bearer token; passed as ``Authorization: Bearer <key>``.
    base_url:
        Override for self-hosted deployments or staging environments.
    transport:
        Optional ``httpx`` transport — tests pass ``httpx.MockTransport`` here
        to intercept requests; production code leaves this as ``None``.
    timeout:
        Per-request timeout in seconds. Default 10s covers the typical API
        latency with room for a handshake retry.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout

    # --- Internal helpers -------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "EcoMarineAI/1.1 (+https://github.com/0x0000dead/whales-identification)",
        }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self._timeout,
            transport=self._transport,
        )

    # --- Public API -------------------------------------------------------

    async def submit_observation_async(
        self, observation: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /observations — returns the decoded JSON payload.

        Raises :class:`HappyWhaleAPIError` on any non-2xx response so callers
        can special-case 401 (bad key) and 429 (rate limit) without parsing
        the HTTP status themselves.
        """
        required_fields = {"species", "latitude", "longitude", "observed_at"}
        missing = required_fields - observation.keys()
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        async with self._client() as client:
            try:
                resp = await client.post("/observations", json=observation)
            except httpx.RequestError as exc:
                logger.warning("HappyWhale network error: %s", exc)
                raise HappyWhaleAPIError(0, f"network error: {exc}") from exc

        if resp.status_code == 401:
            raise HappyWhaleAPIError(401, "invalid API key")
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after", "unknown")
            raise HappyWhaleAPIError(
                429, f"rate limited (retry-after={retry_after})"
            )
        if not resp.is_success:
            raise HappyWhaleAPIError(resp.status_code, resp.text[:200])

        try:
            return resp.json()
        except ValueError as exc:
            raise HappyWhaleAPIError(resp.status_code, "malformed JSON body") from exc

    def submit_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Sync wrapper around :meth:`submit_observation_async`."""
        return asyncio.run(self.submit_observation_async(observation))

    async def health_async(self) -> bool:
        """GET /health — returns True on 2xx, False otherwise."""
        async with self._client() as client:
            try:
                resp = await client.get("/health")
            except httpx.RequestError:
                return False
        return resp.is_success
