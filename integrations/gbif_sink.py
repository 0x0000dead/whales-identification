"""GBIF IPT sink — publishes Darwin Core occurrences.

GBIF (Global Biodiversity Information Facility — https://www.gbif.org) is one
of the two external biodiversity platforms required by ТЗ §Параметр 6. This
module wraps the GBIF IPT ingest endpoint so EcoMarineAI can push verified
cetacean sightings as Darwin Core events without a manual export step.

The IPT API is XML-first for ``coreType=occurrence`` bulk uploads, but the
modern REST gateway accepts JSON. We target the JSON gateway and fall back
to logging if the environment variable ``GBIF_API_KEY`` is missing.

Environment:
    GBIF_API_KEY      — institution API key (required for real submits)
    GBIF_DATASET_KEY  — uuid of the target dataset in the IPT (required)
    GBIF_BASE_URL     — override for staging / self-hosted IPT instances
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.gbif.org/v1"


class GBIFError(RuntimeError):
    """Raised on any non-2xx GBIF response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"GBIF error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class GBIFSink:
    """Publishes Darwin Core ``Occurrence`` records to GBIF.

    Two patterns are supported:

    - :meth:`push_occurrence` — high-level helper that builds the Darwin Core
      dict from ecological primitives (species / latitude / date / count).
    - :meth:`submit_raw` — low-level passthrough for callers that already
      have a pre-built Darwin Core record.
    """

    def __init__(
        self,
        api_key: str | None = None,
        dataset_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GBIF_API_KEY", "")
        self.dataset_key = dataset_key or os.environ.get("GBIF_DATASET_KEY", "")
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout

    # --- Helpers ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "EcoMarineAI-GBIFSink/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self._timeout,
            transport=self._transport,
        )

    @staticmethod
    def build_darwin_core(
        species: str,
        latitude: float,
        longitude: float,
        observed_on: date | datetime | str,
        count: int = 1,
        *,
        dataset_key: str | None = None,
    ) -> dict[str, Any]:
        """Pure function — assembles a Darwin Core occurrence record.

        Factored out so tests can assert the exact wire format without touching
        the HTTP layer.
        """
        if not species:
            raise ValueError("species is required")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"latitude out of range: {latitude}")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(f"longitude out of range: {longitude}")
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        if isinstance(observed_on, (datetime, date)):
            event_date = observed_on.isoformat()
        else:
            event_date = str(observed_on)

        record: dict[str, Any] = {
            "basisOfRecord": "MachineObservation",
            "scientificName": species.replace("_", " "),
            "decimalLatitude": latitude,
            "decimalLongitude": longitude,
            "eventDate": event_date,
            "individualCount": count,
            "geodeticDatum": "WGS84",
            "identifiedBy": "EcoMarineAI model",
            "occurrenceStatus": "present",
        }
        if dataset_key:
            record["datasetKey"] = dataset_key
        return record

    # --- Public API ------------------------------------------------------

    async def push_occurrence_async(
        self,
        species: str,
        latitude: float,
        longitude: float,
        observed_on: date | datetime | str,
        count: int = 1,
    ) -> dict[str, Any]:
        record = self.build_darwin_core(
            species=species,
            latitude=latitude,
            longitude=longitude,
            observed_on=observed_on,
            count=count,
            dataset_key=self.dataset_key or None,
        )
        return await self.submit_raw_async(record)

    def push_occurrence(
        self,
        species: str,
        latitude: float,
        longitude: float,
        observed_on: date | datetime | str,
        count: int = 1,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.push_occurrence_async(
                species=species,
                latitude=latitude,
                longitude=longitude,
                observed_on=observed_on,
                count=count,
            )
        )

    async def submit_raw_async(self, record: dict[str, Any]) -> dict[str, Any]:
        async with self._client() as client:
            try:
                resp = await client.post("/occurrence", json=record)
            except httpx.RequestError as exc:
                raise GBIFError(0, f"network error: {exc}") from exc

        if resp.status_code == 401:
            raise GBIFError(401, "invalid API key")
        if resp.status_code == 403:
            raise GBIFError(403, "not authorised for this dataset")
        if resp.status_code == 429:
            raise GBIFError(429, "rate limited")
        if not resp.is_success:
            raise GBIFError(resp.status_code, resp.text[:200])

        try:
            return resp.json()
        except ValueError as exc:
            raise GBIFError(resp.status_code, "malformed JSON body") from exc
