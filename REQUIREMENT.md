# Audio Transcription + Summary Pipeline Requirements

## Overview
This pipeline downloads audio from a single video URL, optionally speeds up the audio, splits the audio into fixed-duration chunks, transcribes each chunk using OpenAI’s audio API, and saves per‑chunk outputs plus a combined transcript and metadata. It can also generate a concise summary of the combined transcript using a chat model, with both estimated and actual token/cost reporting.

## Steps

1) Download Audio
- Use `yt-dlp` via Python.
- Extract audio to `.m4a` using the `FFmpegExtractAudio` post‑processor.

2) Optional Speed Processing
- Apply speed change via ffmpeg `atempo` (default 1.5x).
- If processing fails, fall back to original audio without aborting the run.

3) Split into Chunks
- Split by time duration (e.g., every 1000 seconds) using ffmpeg’s segment muxer.
- Orchestrate ffmpeg from Python (no manual shell invocations in the workflow).

4) Transcription
- Send each chunk to OpenAI Audio Transcriptions API.
- Request `response_format="json"` and capture both `text` and `usage` tokens from the response.
- Continue on per‑chunk errors; record the error in metadata and move on.

5) Outputs
- Per‑run folder at `outputs/<YYYYMMDD_HHMMSS>/`.
- Per‑chunk `.txt` files contain the API output as JSON: `{ text, usage }`.
- Combined transcript as `combined.txt` (text only).
- `metadata.json` with run info, chunk list (including usage tokens and per‑chunk cost), and totals.

6) Optional Summary
- Summarize the combined transcript to ~200 words using a chat model (default `gpt-4o-mini`).
- Prompt is configurable by file and enforces “same language as transcript”.
- Estimate input tokens with `tiktoken` (prompt + transcript) before calling the API.
- Record actual input/output tokens and costs after the call.

## Cost and Usage Tracking
- Transcription: bill input tokens only at a configurable price per 1M tokens (default `$3.0`).
- Summary: default pricing is `$0.15` per 1M input tokens and `$0.6` per 1M output tokens (configurable).
- Store per‑chunk `input_tokens`, `output_tokens`, `total_tokens`, and `cost_usd`.
- Store totals under `stats` in `metadata.json`.
- Store summary’s estimated input tokens/cost and actual token usage/cost breakdown.

## Constraints
- Entire workflow runs from Python.
- No manual shell `ffmpeg` calls; invoke via the Python wrapper.
- `ffmpeg` and `ffprobe` binaries must be available in `PATH`.

## Runtime Configuration (CLI)
- Required: `url` (single video URL)
- Optional flags:
  - `--chunk-seconds` (default `1000`)
  - `--outdir` (default `outputs`)
  - `--model` (default `gpt-4o-mini-transcribe`)
  - `--language` (default `auto`)
  - `--price-per-mtoken` (transcription; default `3.0` USD per 1M input tokens)
  - `--speed` (default `1.5`, applied via ffmpeg `atempo`; use `1.0` to disable)
  - `--no-summary` to skip summarization
  - `--summary-model` (default `gpt-4o-mini`)
  - `--summary-prompt-file` (default `summary_prompt.txt`)
  - `--summary-price-in-per-mtoken` (default `0.15` USD per 1M input)
  - `--summary-price-out-per-mtoken` (default `0.6` USD per 1M output)

## Environment
- `OPENAI_API_KEY` must be set.
- Python dependencies: `yt-dlp`, `ffmpeg-python`, `openai`, `tiktoken` (see `requirements.txt`).
