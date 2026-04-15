"""iNaturalist (iNat) sink — publishes observations via the v1 API.

Reference: https://api.inaturalist.org/v1/observations. iNat uses OAuth2
bearer tokens — the expected secret lives in ``INAT_API_TOKEN``. Do not commit
tokens. For test / CI this module is always mocked via ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.inaturalist.org/v1"


class INaturalistError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"iNaturalist error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class iNaturalistSink:  # noqa: N801 — name matches upstream product
    """Thin client around iNat v1 ``/observations`` POST.

    Observations must include at minimum: ``taxon_name``, latitude / longitude,
    and an observation date. Optional fields (description, accuracy, etc.)
    can be supplied via the ``extra`` argument of :meth:`submit_observation`.
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.api_token = api_token or os.environ.get("INAT_API_TOKEN", "")
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout

    # --- Internals --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "EcoMarineAI-iNatSink/1.0",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self._timeout,
            transport=self._transport,
        )

    @staticmethod
    def build_observation(
        taxon_name: str,
        lat: float,
        lon: float,
        observed_on: date | datetime | str,
        description: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not taxon_name:
            raise ValueError("taxon_name is required")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"lat out of range: {lat}")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"lon out of range: {lon}")

        if isinstance(observed_on, (datetime, date)):
            observed_str = observed_on.isoformat()
        else:
            observed_str = str(observed_on)

        observation: dict[str, Any] = {
            "species_guess": taxon_name,
            "latitude": lat,
            "longitude": lon,
            "observed_on_string": observed_str,
            "positional_accuracy": 50,
            "geoprivacy": "open",
        }
        if description:
            observation["description"] = description
        if extra:
            observation.update(extra)
        return {"observation": observation}

    # --- Public API -------------------------------------------------------

    async def submit_observation_async(
        self,
        taxon_name: str,
        lat: float,
        lon: float,
        observed_on: date | datetime | str,
        description: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = self.build_observation(
            taxon_name=taxon_name,
            lat=lat,
            lon=lon,
            observed_on=observed_on,
            description=description,
            extra=extra,
        )
        async with self._client() as client:
            try:
                resp = await client.post("/observations", json=body)
            except httpx.RequestError as exc:
                raise INaturalistError(0, f"network error: {exc}") from exc

        if resp.status_code == 401:
            raise INaturalistError(401, "invalid API token")
        if resp.status_code == 429:
            raise INaturalistError(429, "rate limited")
        if not resp.is_success:
            raise INaturalistError(resp.status_code, resp.text[:200])
        try:
            return resp.json()
        except ValueError as exc:
            raise INaturalistError(
                resp.status_code, "malformed JSON body"
            ) from exc

    def submit_observation(
        self,
        taxon_name: str,
        lat: float,
        lon: float,
        observed_on: date | datetime | str,
        description: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.submit_observation_async(
                taxon_name=taxon_name,
                lat=lat,
                lon=lon,
                observed_on=observed_on,
                description=description,
                extra=extra,
            )
        )
