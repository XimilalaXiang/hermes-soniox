# hermes-soniox

Native **BYOK** [Soniox](https://soniox.com) text-to-speech and speech-to-text for
[Hermes Agent](https://github.com/NousResearch/hermes-agent). Provider id: `soniox` (both TTS and STT).
It calls Soniox directly with your own `SONIOX_API_KEY` — no proxy, no gateway, no third-party middleman.
Only Hermes-bundled `httpx` and the standard library are used.

## Install

```bash
hermes plugins install XimilalaXiang/hermes-soniox
hermes plugins enable hermes-soniox
```

Add your key to `~/.hermes/.env` (never to `config.yaml`):

```bash
SONIOX_API_KEY=your-soniox-key
```

## Data path (what leaves your machine)

| Feature | Destination | Data sent |
|---|---|---|
| TTS | `POST https://tts-rt.soniox.com/tts` | reply text, model, voice, language, format, speed |
| STT | `https://api.soniox.com/v1` (`/files`, `/transcriptions`) | the audio file, language hint, optional prompt text |

The key is sent only as an `Authorization: Bearer` header to those two Soniox hosts. It is read from the
environment on each call, never written to config, returned, logged, or included in error messages.
After each transcription the plugin deletes the remote transcription and the uploaded file (best effort).
Soniox's own retention and billing terms apply to anything you send.

## Configuration

```yaml
tts:
  provider: soniox
  model: tts-rt-v2      # optional; non-"tts-*" values are ignored
  voice: Meilin         # optional
  soniox:
    language: zh        # optional, default zh
stt:
  enabled: true
  provider: soniox
  soniox:
    model: stt-async-v5 # optional
    language: zh        # optional, default zh
```

Defaults: TTS model `tts-rt-v2`, language `zh`, voice `Meilin`; STT model `stt-async-v5`, language `zh`.

How Hermes passes settings (verified against Hermes v0.22.4): TTS receives `tts.voice`, `tts.model`,
`tts.speed` and `tts.output_format` from the dispatcher; `tts.soniox.language` is read by the plugin itself.
STT receives `stt.soniox.model`, and `stt.soniox.language` (falling back to `stt.language`) as the language hint.
`stt.prompt` is sent to Soniox as `context.text`.

- **Languages**: BCP-47 values are reduced to the ISO code (`zh-CN` → `zh`). STT accepts a comma-separated
  list (`zh,en`) which is sent as `language_hints`; hints bias recognition, they do not restrict it.
- **Voices**: Soniox built-in voice names or a cloned voice id. The plugin's catalog is a short static list;
  the authenticated `GET https://api.soniox.com/v1/tts-models` catalog confirms that `tts-rt-v2` supports
  `Meilin` and Chinese (`zh`). Any voice can speak any language supported by the model. `tts.voice`/`tts.model`
  are shared Hermes keys; a stale value from another provider (a non-`tts-*` model is ignored, a voice is not)
  will be sent as-is.
- **Speed**: Hermes allows 0.25–4.0; Soniox accepts 0.7–1.3, so values are clamped into that range.
- **Formats**: `mp3`, `wav`, `flac` natively; `ogg`/`opus` request Soniox `opus` (saved as `.ogg` if the
  response is an Ogg container, otherwise `.opus`). Unknown formats fall back to `mp3`.
  The provider opts into voice bubbles; Hermes converts to Opus with ffmpeg when needed.
- **Limits**: TTS text ≤ 5000 characters (the Soniox limit; Hermes may apply its own cap first). STT polls with backoff
  (1s → 5s) for up to 300 s until `completed`/`error`, then fails with a timeout error.
- Realtime WebSocket TTS/STT is not implemented in 0.1.0.

## Testing

```bash
uv venv --python 3.14 .venv-test && uv pip install --python .venv-test/bin/python pytest ruff httpx
.venv-test/bin/python -m pytest -q
.venv-test/bin/python -m ruff check .
```

Tests use mocked HTTP only and need no credentials. They import Hermes' provider ABCs from
`~/.hermes/hermes-agent` (override with `HERMES_AGENT_PATH`).

## Troubleshooting

- *"SONIOX_API_KEY is not set"* — add it to `~/.hermes/.env` and restart Hermes.
- *HTTP 401* — wrong key. *HTTP 402* — Soniox balance/budget exhausted. *HTTP 429* — rate/concurrency limit.
- *HTTP 400 invalid_request* on TTS — voice or language not valid for the model; check `/v1/tts-models`.
- *"transcription timed out"* — very long audio; retry or split the file.
- Provider not picked up — `hermes plugins list`, ensure it is enabled, and that `tts.provider`/`stt.provider` is exactly `soniox`.

## Uninstall / rollback

```bash
hermes plugins disable hermes-soniox
hermes plugins uninstall hermes-soniox
```

Then set `tts.provider` / `stt.provider` back to your previous provider and remove `SONIOX_API_KEY`
from `~/.hermes/.env` if you no longer need it.

## License

MIT
