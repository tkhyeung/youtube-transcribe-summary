# YouTube Audio → OpenAI Transcription + Summary

Author: This README was written by the AI assistant in this repository.

## What It Does
- Downloads audio from a single video URL using `yt-dlp`.
- Optionally speeds the audio using ffmpeg `atempo` (default 1.5x).
- Splits audio into fixed-duration chunks via ffmpeg segmenter.
- Transcribes each chunk with OpenAI’s Audio Transcriptions API.
- Saves outputs into a unique per‑run folder with:
  - Per‑chunk JSON files (`.txt`) containing the API output: text + usage tokens
  - Combined transcript (`combined.txt`)
  - Metadata (`metadata.json`) with run details, chunk list, token usage, and cost estimates
- Optionally generates a ~200‑word summary of the combined transcript with a chat model.

## Key Features
- Per‑run folders under `outputs/<YYYYMMDD_HHMMSS>/` keep runs isolated.
- Per‑chunk `.txt` files hold JSON: `{ text, usage }` (JSON‑safe usage object).
- Pricing and usage:
  - Transcription: input tokens only at `--price-per-mtoken` (default `$3.0`/1M input tokens).
  - Summary (chat): input `$0.15`/1M, output `$0.6`/1M (defaults configurable per flag).
- Robust behavior: per‑chunk errors don’t abort; errors are recorded in metadata.
- Summary prompt enforces “same language as transcript”; can be customized by file.

## Requirements
- System: `ffmpeg` and `ffprobe` available in `PATH`.
- Python packages (see `requirements.txt`): `yt-dlp`, `ffmpeg-python`, `openai`, `tiktoken`.
- Environment: `OPENAI_API_KEY` must be set.

## Install
```
pip install -r requirements.txt
```

## Usage
```
python youtube_audio_transcribe.py "<video_url>" \
  --chunk-seconds 1000 \
  --outdir outputs \
  --model gpt-4o-mini-transcribe \
  --language auto \
  --price-per-mtoken 3.0 \
  --speed 1.5 \
  --summary-model gpt-4o-mini \
  --summary-price-in-per-mtoken 0.15 \
  --summary-price-out-per-mtoken 0.6 \
  --summary-prompt-file summary_prompt.txt
```

Notes
- Use `--speed 1.0` to disable speed processing.
- The per‑chunk output files (`*_chunk_XXXX.txt`) contain JSON with both the transcribed text and a JSON‑safe `usage` object, for example:
```
{
  "text": "Imagine the wildest idea...",
  "usage": {
    "type": "tokens",
    "input_tokens": 14,
    "input_token_details": {
      "text_tokens": 10,
      "audio_tokens": 4
    },
    "output_tokens": 101,
    "total_tokens": 115
  }
}
```

## Outputs
- Per‑run directory: `outputs/<run_id>/`
  - `*_chunk_XXXX.m4a` audio chunks
  - `*_chunk_XXXX.txt` per chunk (JSON: text + usage)
  - `combined.txt` concatenated transcript
  - `summary.txt` generated summary (when enabled)
  - `metadata.json` with:
    - Run info: `source_url`, `video_id`, `title`, `downloaded_audio`, `processed_audio`, `created_utc`, `run_id`
    - Settings: `chunk_seconds`, `model`, `language`, transcription pricing
    - Chunks: usage tokens and per‑chunk `cost_usd`
    - Stats: transcription totals and cost
    - Summary: model, prompt file, estimated tokens/cost, actual usage and costs, summary file

## Cost Model
- Transcription: input tokens only at `--price-per-mtoken` (default: `$3.0` per 1M input tokens).
- Summary (chat): defaults are `$0.15` per 1M input and `$0.6` per 1M output tokens.
- The script prints per‑chunk costs, totals, and summary‑related estimates/actuals; all values are stored in `metadata.json`.

## Troubleshooting
- “Required binary not found”: ensure `ffmpeg` and `ffprobe` are installed and in `PATH`.
- API errors on a chunk will be logged into `metadata.json` under that chunk and the run will continue.

## Caveats
- Usage tokens are taken directly from the API. If an SDK update changes return shapes, the script converts usage to a JSON‑safe dict; unknown shapes are stored under `usage.raw` as a string.
