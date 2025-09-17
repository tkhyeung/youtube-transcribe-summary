"""AWS Lambda entrypoint for the YouTube transcription + summarization pipeline."""
import base64
import datetime as dt
import glob
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional

import boto3



DEFAULT_OUTPUT_BASE = "/tmp/youtube-transcribe"
DEFAULT_SUMMARY_PROMPT = (
    "Summarize the transcript in clear prose, capturing key points, "
    "structure, and conclusions. Limit to about 200 words. "
    "Write the summary in the same language as the transcript; do not translate."
)
OPENAI_SECRET_NAME_ENV = "OPENAI_SECRET_NAME"
OPENAI_SECRET_JSON_KEY_ENV = "OPENAI_SECRET_JSON_KEY"
OPENAI_SECRET_JSON_DEFAULT_KEY = "OPENAI_API_KEY"

_secret_cache: Dict[str, str] = {}


def _require(module_name: str, pip_name: str) -> None:
    try:
        __import__(module_name)
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install with: pip install {pip_name}") from exc


def _check_binaries() -> None:
    for bin_name in ("ffmpeg", "ffprobe"):
        if shutil.which(bin_name) is None:
            raise RuntimeError(
                f"Required binary '{bin_name}' not found in PATH. Ensure your layer ships ffmpeg/ffprobe.")


def safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9._-]", "", text)
    return text or "audio"


def download_audio(url: str, outdir: str) -> Dict[str, Any]:
    _require("yt_dlp", "yt-dlp")
    import yt_dlp

    os.makedirs(outdir, exist_ok=True)

    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info.get("id")
    title = info.get("title") or video_id
    audio_path = os.path.join(outdir, f"{video_id}.m4a")
    if not os.path.exists(audio_path):
        matches = sorted(glob.glob(os.path.join(outdir, f"{video_id}*.m4a")))
        if matches:
            audio_path = matches[0]
        else:
            raise FileNotFoundError(
                f"Downloaded audio file not found for video id {video_id} in {outdir}")

    return {"id": video_id, "title": title, "audio_path": audio_path}


def probe_duration_seconds(path: str) -> float:
    _require("ffmpeg", "ffmpeg-python")
    import ffmpeg

    probe = ffmpeg.probe(path)
    duration = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio" and stream.get("duration"):
            duration = float(stream["duration"])  # type: ignore[arg-type]
            break
    if duration is None:
        duration = float(probe.get("format", {}).get("duration", 0.0))
    return float(duration or 0.0)


def split_audio_ffmpeg(input_path: str, outdir: str, base: str, chunk_seconds: int) -> List[str]:
    _require("ffmpeg", "ffmpeg-python")
    import ffmpeg

    os.makedirs(outdir, exist_ok=True)
    pattern = os.path.join(outdir, f"{base}_chunk_%04d.m4a")

    stream = ffmpeg.input(input_path)
    out = (
        ffmpeg
        .output(
            stream,
            pattern,
            f="segment",
            segment_time=str(int(chunk_seconds)),
            reset_timestamps="1",
            **{"c:a": "copy"},
        )
        .overwrite_output()
    )
    try:
        out.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            f"ffmpeg segmenting failed: {exc.stderr.decode('utf-8', errors='ignore')}") from exc

    chunk_paths = sorted(glob.glob(os.path.join(outdir, f"{base}_chunk_*.m4a")))
    if not chunk_paths:
        raise RuntimeError("No audio chunks were produced by ffmpeg segmenting.")
    return chunk_paths


def process_audio_ffmpeg(input_path: str, outdir: str, base: str, speed: float = 1.5) -> Dict[str, Any]:
    _require("ffmpeg", "ffmpeg-python")
    import ffmpeg

    os.makedirs(outdir, exist_ok=True)
    processed_path = os.path.join(outdir, f"{base}__processed.m4a")

    filters: List[str] = []
    if speed and abs(speed - 1.0) > 1e-6:
        filters.append(f"atempo={speed}")

    stream = ffmpeg.input(input_path)
    kwargs: Dict[str, Any] = {"c:a": "aac"}
    if filters:
        kwargs["af"] = ",".join(filters)

    out = ffmpeg.output(stream, processed_path, **kwargs).overwrite_output()
    try:
        out.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            f"ffmpeg processing failed: {exc.stderr.decode('utf-8', errors='ignore')}") from exc

    return {
        "processed_path": processed_path,
        "filters": kwargs.get("af", ""),
        "speed": speed,
    }


def _extract_attr_or_key(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_jsonable_usage(usage_obj: Any) -> Dict[str, Any]:
    if usage_obj is None:
        return {}
    try:
        usage_type = _extract_attr_or_key(usage_obj, "type", "tokens")
        input_tokens = int(
            _extract_attr_or_key(
                usage_obj,
                "input_tokens",
                _extract_attr_or_key(usage_obj, "prompt_tokens", 0),
            )
            or 0
        )
        output_tokens = int(
            _extract_attr_or_key(
                usage_obj,
                "output_tokens",
                _extract_attr_or_key(usage_obj, "completion_tokens", 0),
            )
            or 0
        )
        total_tokens = int(
            _extract_attr_or_key(usage_obj, "total_tokens", input_tokens + output_tokens) or 0
        )
        details_obj = _extract_attr_or_key(usage_obj, "input_token_details", None)
        text_tokens = int(_extract_attr_or_key(details_obj, "text_tokens", 0) or 0) if details_obj else 0
        audio_tokens = int(_extract_attr_or_key(details_obj, "audio_tokens", 0) or 0) if details_obj else 0
        return {
            "type": usage_type,
            "input_tokens": input_tokens,
            "input_token_details": {
                "text_tokens": text_tokens,
                "audio_tokens": audio_tokens,
            },
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    except Exception:
        return {"raw": str(usage_obj)}


def estimate_tokens(text: str, model: str) -> int:
    try:
        _require("tiktoken", "tiktoken")
        import tiktoken  # type: ignore

        try:
            encoder = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                encoder = tiktoken.get_encoding("o200k_base")
            except Exception:
                encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def summarize_transcript(client, model: str, transcript_text: str, prompt_text: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a precise summarization assistant."},
        {
            "role": "user",
            "content": f"{prompt_text}\n\n=== TRANSCRIPT START ===\n{transcript_text}\n=== TRANSCRIPT END ===",
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        content = choice.message.content if choice and getattr(choice, "message", None) else None
        text = content if isinstance(content, str) else ""
        usage_json = to_jsonable_usage(getattr(resp, "usage", None))
        return {"summary": text, "usage": usage_json}
    except Exception as exc:
        return {"error": str(exc)}


def transcribe_file(path: str, client, model: str, language: str) -> Dict[str, Any]:
    kwargs = {
        "model": model,
        "response_format": "json",
    }
    if language and language.lower() != "auto":
        kwargs["language"] = language

    with open(path, "rb") as file_handle:
        result = client.audio.transcriptions.create(file=file_handle, **kwargs)

    text = _extract_attr_or_key(result, "text")
    usage = _extract_attr_or_key(result, "usage", default=None)

    if isinstance(result, str) and not text:
        text = result
        usage = None

    if not isinstance(text, str):
        raise RuntimeError("Unexpected transcription response: missing text")

    return {"text": text, "usage": usage}


def _resolve_openai_api_key() -> str:
    secret_name = os.environ.get(OPENAI_SECRET_NAME_ENV)
    if not secret_name:
        raise RuntimeError(
            "OPENAI_SECRET_NAME environment variable required for Secrets Manager lookup.")

    json_key = os.environ.get(OPENAI_SECRET_JSON_KEY_ENV, OPENAI_SECRET_JSON_DEFAULT_KEY)
    return _fetch_secret_from_secrets_manager(secret_name, json_key)


def _fetch_secret_from_secrets_manager(secret_name: str, json_key: Optional[str]) -> str:
    if secret_name in _secret_cache:
        return _secret_cache[secret_name]

    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)

    if "SecretString" in response:
        raw_payload = response["SecretString"]
    else:
        raw_payload = base64.b64decode(response["SecretBinary"]).decode("utf-8")

    secret_value: Optional[str] = None
    if json_key:
        try:
            payload_json = json.loads(raw_payload)
        except json.JSONDecodeError:
            secret_value = raw_payload
        else:
            if json_key in payload_json:
                secret_value = str(payload_json[json_key])
            elif len(payload_json) == 1:
                # Fallback: single-key JSON, grab the only value
                secret_value = str(next(iter(payload_json.values())))
            else:
                keys = ", ".join(payload_json.keys())
                raise RuntimeError(
                    f"Secret '{secret_name}' missing key '{json_key}'. Available keys: {keys}"
                )
    if secret_value is None:
        secret_value = raw_payload

    _secret_cache[secret_name] = secret_value
    return secret_value


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler.

    Expected event payload keys:
      - url (str, required): YouTube video URL.
      - chunk_seconds (int, optional): defaults to 180.
      - model (str, optional): transcription model; defaults to gpt-4o-mini-transcribe.
      - language (str, optional): language code or "auto".
      - price_per_mtoken (float, optional): cost estimate per 1M input tokens.
      - speed (float, optional): audio speed factor for preprocessing.
      - no_summary (bool, optional): skip summarization step.
      - summary_model (str, optional): chat model for summarization.
      - summary_prompt (str, optional): override prompt text.
      - summary_price_in_per_mtoken (float, optional): price per 1M input tokens for summary.
      - summary_price_out_per_mtoken (float, optional): price per 1M output tokens for summary.
      - summary_prompt_language_hint (str, optional): appended to the prompt if provided.
      - output_base_dir (str, optional): override base output directory (must exist and be writable).
      - s3_bucket (str, optional): when present, upload run artifacts to this bucket.
      - s3_prefix (str, optional): prefix to use when uploading to S3.
    """
    event = event or {}

    if "url" not in event or not event["url"]:
        return _error_response("Missing required field 'url'.")

    try:
        result = _run_pipeline(event)
        return {"status": "ok", **result}
    except Exception as exc:  # pragma: no cover - surfaced via Lambda logs
        return _error_response(str(exc))


def _run_pipeline(event: Dict[str, Any]) -> Dict[str, Any]:
    url = event["url"]
    chunk_seconds = int(event.get("chunk_seconds", 180))
    model = event.get("model", "gpt-4o-mini-transcribe")
    language = event.get("language", "auto")
    price_per_mtoken = float(event.get("price_per_mtoken", 3.0))
    speed = float(event.get("speed", 1.5))
    no_summary = bool(event.get("no_summary", False))
    summary_model = event.get("summary_model", "gpt-4o-mini")
    summary_price_in = float(event.get("summary_price_in_per_mtoken", 0.15))
    summary_price_out = float(event.get("summary_price_out_per_mtoken", 0.6))
    base_dir = event.get("output_base_dir", DEFAULT_OUTPUT_BASE)
    os.makedirs(base_dir, exist_ok=True)

    _check_binaries()
    _require("yt_dlp", "yt-dlp")
    _require("ffmpeg", "ffmpeg-python")
    _require("openai", "openai")

    from openai import OpenAI  # type: ignore

    try:
        utc = dt.UTC
    except AttributeError:  # pragma: no cover - Python <3.11 compatibility
        utc = dt.timezone.utc

    now_utc = dt.datetime.now(utc)
    run_id = now_utc.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    info = download_audio(url, run_dir)
    audio_path = info["audio_path"]
    video_id = info["id"]
    title = info.get("title") or video_id

    base = safe_slug(f"{video_id}_{title}")

    processing_info: Dict[str, Any] = {}
    processed_path = audio_path
    try:
        processing_info = process_audio_ffmpeg(audio_path, run_dir, base, speed=speed)
        processed_path = processing_info.get("processed_path", audio_path)
    except Exception as exc:
        processing_info = {"error": str(exc)}

    duration = probe_duration_seconds(processed_path)
    chunk_paths = split_audio_ffmpeg(processed_path, run_dir, base, chunk_seconds)

    api_key = _resolve_openai_api_key()
    client = OpenAI(api_key=api_key)

    combined_text_parts: List[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_usage_tokens = 0
    total_cost_usd = 0.0
    metadata: Dict[str, Any] = {
        "source_url": url,
        "video_id": video_id,
        "title": title,
        "downloaded_audio": audio_path,
        "processed_audio": processed_path,
        "duration_seconds": duration,
        "chunk_seconds": chunk_seconds,
        "model": model,
        "language": language,
        "created_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "output_dir": run_dir,
        "chunks": [],
    }
    if processing_info:
        metadata["audio_processing"] = processing_info

    for idx, cpath in enumerate(chunk_paths):
        try:
            resp = transcribe_file(cpath, client=client, model=model, language=language)
            text = resp["text"]
            usage_raw = resp.get("usage") if isinstance(resp, dict) else None
            usage_json = to_jsonable_usage(usage_raw)
            combined_text_parts.append(text)

            chunk_txt = cpath.replace(".m4a", ".txt")
            with open(chunk_txt, "w", encoding="utf-8") as f:
                json.dump({"text": text, "usage": usage_json}, f, ensure_ascii=False, indent=2)

            in_tokens = int(usage_json.get("input_tokens", 0) or 0)
            out_tokens = int(usage_json.get("output_tokens", 0) or 0)
            tot_tokens = int(usage_json.get("total_tokens", in_tokens + out_tokens) or 0)
            chunk_cost = (in_tokens / 1_000_000.0) * price_per_mtoken

            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            total_usage_tokens += tot_tokens
            total_cost_usd += chunk_cost

            metadata["chunks"].append({
                "index": idx,
                "file": cpath,
                "transcript_file": chunk_txt,
                "start_seconds": idx * chunk_seconds,
                "usage": usage_json,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": tot_tokens,
                "cost_usd": round(chunk_cost, 6),
            })
        except Exception as exc:
            metadata["chunks"].append({
                "index": idx,
                "file": cpath,
                "start_seconds": idx * chunk_seconds,
                "error": str(exc),
            })

    combined_text = "\n\n".join(combined_text_parts)
    combined_path = os.path.join(run_dir, "combined.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    summary_info: Optional[Dict[str, Any]] = None
    summary_path: Optional[str] = None
    summary_text: Optional[str] = None

    if not no_summary and combined_text.strip():
        prompt_text = str(event.get("summary_prompt") or DEFAULT_SUMMARY_PROMPT).strip()
        language_hint = event.get("summary_prompt_language_hint")
        if language_hint:
            prompt_text += f"\n\n{language_hint.strip()}"
        prompt_text += "\n\nPlease write the summary in the same language as the transcript (do not translate)."

        est_tokens = estimate_tokens(f"{prompt_text}\n\n{combined_text}", model=summary_model)
        est_input_cost = (est_tokens / 1_000_000.0) * summary_price_in

        sum_resp = summarize_transcript(
            client,
            model=summary_model,
            transcript_text=combined_text,
            prompt_text=prompt_text,
        )

        if "summary" in sum_resp:
            summary_text = sum_resp["summary"]
            summary_path = os.path.join(run_dir, "summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
        else:
            summary_text = None

        usage_clean = to_jsonable_usage(sum_resp.get("usage")) if isinstance(sum_resp, dict) else {}
        actual_in = int(usage_clean.get("input_tokens", 0) or 0)
        actual_out = int(usage_clean.get("output_tokens", 0) or 0)
        actual_total = int(usage_clean.get("total_tokens", actual_in + actual_out) or 0)
        actual_cost_in = (actual_in / 1_000_000.0) * summary_price_in
        actual_cost_out = (actual_out / 1_000_000.0) * summary_price_out
        actual_cost_total = actual_cost_in + actual_cost_out

        summary_info = {
            "model": summary_model,
            "prompt_text": prompt_text,
            "estimated_input_tokens": est_tokens,
            "estimated_input_cost_usd": round(est_input_cost, 6),
            "price_per_mtoken_usd": {
                "input": summary_price_in,
                "output": summary_price_out,
            },
            "usage": usage_clean,
            "actual_input_tokens": actual_in,
            "actual_output_tokens": actual_out,
            "actual_total_tokens": actual_total,
            "actual_cost_usd": {
                "input": round(actual_cost_in, 6),
                "output": round(actual_cost_out, 6),
                "total": round(actual_cost_total, 6),
            },
            "summary_file": summary_path,
            "error": sum_resp.get("error") if isinstance(sum_resp, dict) else None,
        }

    metadata["combined_transcript_file"] = combined_path
    if summary_info:
        metadata["summary"] = summary_info
    metadata["stats"] = {
        "chunks": len(chunk_paths),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_usage_tokens,
        "token_counting": "api_usage",
        "price_per_mtoken_usd": price_per_mtoken,
        "estimated_cost_usd": round(total_cost_usd, 6),
    }

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    uploads = None
    bucket = event.get("s3_bucket")
    if bucket:
        prefix = event.get("s3_prefix", f"youtube-transcribe/{run_id}")
        uploads = _upload_directory_to_s3(run_dir, bucket, prefix)

    return {
        "run_id": run_id,
        "output_dir": run_dir,
        "combined_transcript": combined_path,
        "summary_file": summary_path,
        "metadata_file": meta_path,
        "summary_text": summary_text,
        "metadata": metadata,
        "s3_uploads": uploads,
    }


def _upload_directory_to_s3(local_dir: str, bucket: str, prefix: str) -> List[Dict[str, str]]:
    import boto3

    s3 = boto3.client("s3")
    uploaded: List[Dict[str, str]] = []
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel_path = os.path.relpath(local_path, local_dir)
            key = "/".join([p for p in (prefix.rstrip("/"), rel_path) if p])
            s3.upload_file(local_path, bucket, key)
            uploaded.append({"bucket": bucket, "key": key})
    return uploaded


def _error_response(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}
