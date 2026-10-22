"""Minimal Soniox REST client (httpx only). Errors are safe: the API key is never included."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

STT_BASE_URL = "https://api.soniox.com/v1"
TTS_URL = "https://tts-rt.soniox.com/tts"

TTS_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
API_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
UPLOAD_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

_MAX_ERROR_CHARS = 300


class SonioxError(Exception):
    """A Soniox call failed. The message is safe to show to users and logs."""


class SonioxClient:
    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        poll_timeout: float = 300.0,
        poll_initial: float = 1.0,
        poll_max: float = 5.0,
    ) -> None:
        self._key = api_key
        self._transport = transport
        self._sleep = sleep
        self._monotonic = monotonic
        self.poll_timeout = poll_timeout
        self.poll_initial = poll_initial
        self.poll_max = poll_max

    # -- plumbing ---------------------------------------------------------

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _request(self, op: str, method: str, url: str, *, timeout: httpx.Timeout,
                 **kw: Any) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._key}"}
        try:
            with httpx.Client(transport=self._transport, timeout=timeout) as http:
                resp = http.request(method, url, headers=headers, **kw)
        except httpx.TimeoutException:
            raise SonioxError(f"Soniox {op} request timed out") from None
        except httpx.HTTPError as exc:
            raise SonioxError(f"Soniox {op} request failed ({type(exc).__name__})") from None
        except UnicodeError:  # e.g. non-ASCII key in the header; the exception text would echo it
            raise SonioxError(f"Soniox {op} request failed (invalid API key characters)") from None
        if resp.status_code >= 400:
            raise SonioxError(self._http_error(op, resp))
        return resp

    def _http_error(self, op: str, resp: httpx.Response) -> str:
        etype = emsg = None
        try:
            body = resp.json()
            if isinstance(body, dict):
                etype = body.get("error_type")
                emsg = body.get("error_message") or body.get("message")
        except ValueError:
            pass
        parts = [f"Soniox {op} failed (HTTP {resp.status_code}"]
        if isinstance(etype, str) and etype:
            parts.append(f", {etype}")
        text = "".join(parts) + ")"
        if isinstance(emsg, str) and emsg:
            text += f": {emsg}"
        return self._redact(text)[:_MAX_ERROR_CHARS]

    def _json(self, op: str, resp: httpx.Response) -> dict[str, Any]:
        try:
            body = resp.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise SonioxError(f"Soniox {op} returned a malformed response")
        return body

    # -- TTS --------------------------------------------------------------

    def synthesize(self, payload: dict[str, Any]) -> bytes:
        resp = self._request("TTS", "POST", TTS_URL, timeout=TTS_TIMEOUT, json=payload)
        if not resp.content:
            raise SonioxError("Soniox TTS returned empty audio")
        return resp.content

    # -- async STT --------------------------------------------------------

    def upload_file(self, path: str) -> str:
        with open(path, "rb") as fh:
            files = {"file": (os.path.basename(path) or "audio", fh)}
            resp = self._request("file upload", "POST", f"{STT_BASE_URL}/files",
                                 timeout=UPLOAD_TIMEOUT, files=files)
        return self._require_id("file upload", self._json("file upload", resp))

    def create_transcription(self, body: dict[str, Any]) -> str:
        resp = self._request("transcription create", "POST", f"{STT_BASE_URL}/transcriptions",
                             timeout=API_TIMEOUT, json=body)
        return self._require_id("transcription create", self._json("transcription create", resp))

    @staticmethod
    def _require_id(op: str, body: dict[str, Any]) -> str:
        value = body.get("id")
        if not isinstance(value, str) or not value:
            raise SonioxError(f"Soniox {op} returned a malformed response (no id)")
        return value

    def wait_for_completion(self, transcription_id: str) -> None:
        """Poll until ``completed``; raise on ``error`` or timeout."""
        start = self._monotonic()
        interval = self.poll_initial
        while True:
            resp = self._request("transcription status", "GET",
                                 f"{STT_BASE_URL}/transcriptions/{transcription_id}", timeout=API_TIMEOUT)
            body = self._json("transcription status", resp)
            status = body.get("status")
            if status == "completed":
                return
            if status == "error":
                etype, emsg = body.get("error_type"), body.get("error_message")
                detail = ": ".join(str(x) for x in (etype, emsg) if isinstance(x, str) and x)
                raise SonioxError(self._redact(
                    "Soniox transcription failed" + (f" ({detail})" if detail else "")
                )[:_MAX_ERROR_CHARS])
            # Documented non-terminal statuses are queued/processing; any other value is treated as
            # still in flight so the poll stays bounded by poll_timeout instead of failing early.
            remaining = self.poll_timeout - (self._monotonic() - start)
            if remaining <= 0:
                raise SonioxError(f"Soniox transcription timed out after {self.poll_timeout:g}s")
            self._sleep(min(interval, remaining))
            interval = min(interval * 1.5, self.poll_max)

    def get_transcript(self, transcription_id: str) -> str:
        resp = self._request("transcript fetch", "GET",
                             f"{STT_BASE_URL}/transcriptions/{transcription_id}/transcript",
                             timeout=API_TIMEOUT)
        body = self._json("transcript fetch", resp)
        text = body.get("text")
        if isinstance(text, str):
            return text.strip()
        tokens = body.get("tokens")
        if isinstance(tokens, list) and tokens and all(
            isinstance(t, dict) and isinstance(t.get("text"), str) for t in tokens
        ):
            return "".join(t["text"] for t in tokens).strip()
        raise SonioxError("Soniox transcript response was malformed")

    def delete_transcription(self, transcription_id: str) -> None:
        self._request("transcription delete", "DELETE",
                      f"{STT_BASE_URL}/transcriptions/{transcription_id}", timeout=API_TIMEOUT)

    def delete_file(self, file_id: str) -> None:
        self._request("file delete", "DELETE", f"{STT_BASE_URL}/files/{file_id}", timeout=API_TIMEOUT)

    def best_effort_cleanup(self, transcription_id: str | None, file_id: str | None) -> None:
        """Delete remote resources; failures are logged (redacted) and never raised."""
        steps: list[Any] = []
        if transcription_id:
            steps.append(("transcription", self.delete_transcription, transcription_id))
        if file_id:
            steps.append(("file", self.delete_file, file_id))
        for label, fn, rid in steps:
            try:
                fn(rid)
            except Exception as exc:
                logger.debug("Soniox %s cleanup failed: %s", label, self._redact(str(exc)))
