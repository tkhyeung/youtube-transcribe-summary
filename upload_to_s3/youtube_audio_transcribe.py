import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from typing import List, Dict, Any, Optional


def _extract_attr_or_key(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_jsonable_usage(usage_obj: Any) -> Dict[str, Any]:
    if usage_obj is None:
        return {}
    try:
        utype = _extract_attr_or_key(usage_obj, "type", "tokens")
        # Support multiple naming conventions across endpoints
        input_tokens = int(
            _extract_attr_or_key(usage_obj, "input_tokens",
                                 _extract_attr_or_key(usage_obj, "prompt_tokens", 0)) or 0)
        output_tokens = int(
            _extract_attr_or_key(usage_obj, "output_tokens",
                                 _extract_attr_or_key(usage_obj, "completion_tokens", 0)) or 0)
        total_tokens = int(_extract_attr_or_key(usage_obj, "total_tokens", input_tokens + output_tokens) or 0)
        details_obj = _extract_attr_or_key(usage_obj, "input_token_details", None)
        text_tokens = int(_extract_attr_or_key(details_obj, "text_tokens", 0) or 0) if details_obj is not None else 0
        audio_tokens = int(_extract_attr_or_key(details_obj, "audio_tokens", 0) or 0) if details_obj is not None else 0
        return {
            "type": utype,
            "input_tokens": input_tokens,
            "input_token_details": {
                "text_tokens": text_tokens,
                "audio_tokens": audio_tokens,
            },
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
    except Exception:
        # Last-resort: stringify to avoid JSON errors
        return {"raw": str(usage_obj)}


def estimate_tokens(text: str, model: str) -> int:
    try:
        _require("tiktoken", "tiktoken")
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            # Fall back to a general-purpose encoding for GPT-4o family, then cl100k
            try:
                enc = tiktoken.get_encoding("o200k_base")
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback heuristic if tiktoken unavailable
        return max(1, len(text) // 4)


def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def upload_directory_to_s3(local_dir: str, bucket: str, prefix: Optional[str] = None) -> List[Dict[str, str]]:
    """Upload all files in local_dir to S3 under bucket/prefix."""
    _require("boto3", "boto3")
    import boto3

    s3 = boto3.client("s3")
    uploads: List[Dict[str, str]] = []
    normalized_prefix = (prefix or "").strip("/")

    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel_path = os.path.relpath(local_path, local_dir)
            rel_key = rel_path.replace(os.sep, "/")
            key_parts = [normalized_prefix, rel_key] if normalized_prefix else [rel_key]
            key = "/".join(part for part in key_parts if part)
            s3.upload_file(local_path, bucket, key)
            uploads.append({"bucket": bucket, "key": key})

    return uploads


def summarize_transcript(client, model: str, transcript_text: str, prompt_text: str) -> Dict[str, Any]:
    # Use Chat Completions for summarization
    messages = [
        {"role": "system", "content": "You are a precise summarization assistant."},
        {"role": "user", "content": f"{prompt_text}\n\n=== TRANSCRIPT START ===\n{transcript_text}\n=== TRANSCRIPT END ==="},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        # Extract summary text and usage
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        content = choice.message.content if choice and getattr(choice, "message", None) else None
        text = content if isinstance(content, str) else ""
        usage_json = to_jsonable_usage(getattr(resp, "usage", None))
        return {"summary": text, "usage": usage_json}
    except Exception as e:
        return {"error": str(e)}


def transcribe_file(path: str, client, model: str, language: str) -> Dict[str, Any]:
    # language: 'auto' means let server auto-detect
    kwargs = {
        "model": model,
        # Request structured response so we can read usage tokens
        "response_format": "json",
    }
    if language and language.lower() != "auto":
        kwargs["language"] = language

    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(file=f, **kwargs)

    # Try to extract text and usage regardless of SDK's return type
    text = _extract_attr_or_key(result, "text")
    usage = _extract_attr_or_key(result, "usage", default=None)

    # Fallback if API returned plain text string (no usage available)
    if isinstance(result, str) and not text:
        text = result
        usage = None

    if not isinstance(text, str):
        raise RuntimeError("Unexpected transcription response: missing text")

    return {"text": text, "usage": usage}


def _require(module_name: str, pip_name: str):
    try:
        __import__(module_name)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install with: pip install {pip_name}") from e


def _check_binaries():
    import shutil
    for bin_name in ("ffmpeg", "ffprobe"):
        if shutil.which(bin_name) is None:
            raise RuntimeError(
                f"Required binary '{bin_name}' not found in PATH. Please install ffmpeg.")


def safe_slug(text: str) -> str:
    text = text.strip().lower()
    # Replace whitespace with single underscore
    text = re.sub(r"\s+", "_", text)
    # Keep alnum, dash, underscore, dot
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
        # Fallback: try to locate any m4a matching the id
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
    # Prefer stream duration, fallback to format duration
    duration = None
    for s in probe.get("streams", []):
        if s.get("codec_type") == "audio" and s.get("duration"):
            duration = float(s["duration"])  # type: ignore
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
    # Use segment muxer with stream copy for speed, reset timestamps per segment
    out = (
        ffmpeg
        .output(stream, pattern,
                f="segment",
                segment_time=str(int(chunk_seconds)),
                reset_timestamps="1",
                **{"c:a": "copy"})
        .overwrite_output()
    )
    # Run and capture for clearer error messages
    try:
        out.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:  # type: ignore
        raise RuntimeError(
            f"ffmpeg segmenting failed: {e.stderr.decode('utf-8', errors='ignore')}") from e

    chunk_paths = sorted(glob.glob(os.path.join(outdir, f"{base}_chunk_*.m4a")))
    if not chunk_paths:
        raise RuntimeError("No chunks were produced by ffmpeg segmenting.")
    return chunk_paths


def process_audio_ffmpeg(input_path: str, outdir: str, base: str, speed: float = 1.5) -> Dict[str, Any]:
    _require("ffmpeg", "ffmpeg-python")
    import ffmpeg

    os.makedirs(outdir, exist_ok=True)
    processed_path = os.path.join(outdir, f"{base}__processed.m4a")

    filters: List[str] = []
    if speed and abs(speed - 1.0) > 1e-6:
        # Speed up / slow down via atempo
        filters.append(f"atempo={speed}")

    stream = ffmpeg.input(input_path)
    kwargs = {"c:a": "aac"}
    if filters:
        kwargs["af"] = ",".join(filters)

    out = (
        ffmpeg
        .output(stream, processed_path, **kwargs)
        .overwrite_output()
    )

    try:
        out.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:  # type: ignore
        raise RuntimeError(
            f"ffmpeg processing failed: {e.stderr.decode('utf-8', errors='ignore')}") from e

    return {
        "processed_path": processed_path,
        "filters": kwargs.get("af", ""),
        "speed": speed,
    }


# Old transcribe_file (text-only) removed; use the JSON/usage-enabled version above.


def main():
    parser = argparse.ArgumentParser(description="YouTube audio → chunks → OpenAI transcription + summary")
    parser.add_argument("url", help="Video URL (single)")
    parser.add_argument("--chunk-seconds", type=int, default=180,
                        help="Chunk length in seconds (default: 180)")
    parser.add_argument("--outdir", default="outputs",
                        help="Output directory for files (default: outputs)")
    parser.add_argument("--model", default="gpt-4o-mini-transcribe",
                        help="OpenAI transcription model (default: gpt-4o-mini-transcribe)")
    parser.add_argument("--language", default="auto",
                        help="Language code or 'auto' (default: auto)")
    parser.add_argument("--price-per-mtoken", type=float, default=3.0,
                        help="USD price per 1M tokens for cost estimate (default: 3.0)")
    parser.add_argument("--speed", type=float, default=1.5,
                        help="Audio playback speed factor via ffmpeg atempo (default: 1.5)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip the final summarization step")
    parser.add_argument("--summary-model", default="gpt-4o-mini",
                        help="Model to use for summary (default: gpt-4o-mini)")
    parser.add_argument("--summary-prompt-file", default="summary_prompt.txt",
                        help="Path to a file with the summary prompt (default: summary_prompt.txt)")
    parser.add_argument("--summary-price-in-per-mtoken", type=float, default=0.15,
                        help="USD price per 1M input tokens for summary model (default: 0.15 for gpt-4o-mini)")
    parser.add_argument("--summary-price-out-per-mtoken", type=float, default=0.6,
                        help="USD price per 1M output tokens for summary model (default: 0.6 for gpt-4o-mini)")
    parser.add_argument("--s3-bucket", default="youtube-transcribe-summary",
                        help="Optional S3 bucket name to upload the entire run directory")
    parser.add_argument("--s3-prefix", default="runs",
                        help="Optional S3 key prefix for uploads when --s3-bucket is provided")
    args = parser.parse_args()

    # Dependency checks
    _check_binaries()
    _require("yt_dlp", "yt-dlp")
    _require("ffmpeg", "ffmpeg-python")
    _require("openai", "openai")
    if args.s3_bucket:
        _require("boto3", "boto3")

    from openai import OpenAI  # type: ignore

    url = args.url
    outdir = args.outdir
    chunk_seconds = int(args.chunk_seconds)
    model = args.model
    language = args.language
    price_per_mtoken = float(args.price_per_mtoken)

    # Create per-run output directory (timestamped)
    os.makedirs(outdir, exist_ok=True)
    # Timezone-aware UTC now (compat across Python versions)
    try:
        UTC = dt.UTC  # Python 3.11+
    except AttributeError:
        UTC = dt.timezone.utc  # Older versions
    now_utc = dt.datetime.now(UTC)
    run_id = now_utc.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(outdir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Downloading audio from: {url}")
    info = download_audio(url, run_dir)
    audio_path = info["audio_path"]
    video_id = info["id"]
    title = info.get("title") or video_id

    base = safe_slug(f"{video_id}_{title}")
    print(f"Downloaded to: {audio_path}")

    # Optional preprocessing: trim silence and change speed
    processing_info: Dict[str, Any] = {}
    processed_path = audio_path
    try:
        processing_info = process_audio_ffmpeg(
            audio_path, run_dir, base, speed=float(args.speed))
        processed_path = processing_info.get("processed_path", audio_path)
        print(f"Processed audio: {processed_path}")
    except Exception as e:
        print(f"Processing skipped due to error: {e}")

    duration = probe_duration_seconds(processed_path)
    print(f"Audio duration (processed): {duration:.2f}s")

    print(f"Splitting into ~{chunk_seconds}s chunks using ffmpeg...")
    chunk_paths = split_audio_ffmpeg(processed_path, run_dir, base, chunk_seconds)
    print(f"Produced {len(chunk_paths)} chunks")

    # OpenAI client (uses OPENAI_API_KEY)
    client = OpenAI()

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

    print("Transcribing chunks...")
    for idx, cpath in enumerate(chunk_paths):
        print(f" - Transcribing chunk {idx+1}/{len(chunk_paths)}: {os.path.basename(cpath)}")
        try:
            resp = transcribe_file(cpath, client=client, model=model, language=language)
            text = resp["text"]
            usage_raw = resp.get("usage") if isinstance(resp, dict) else None
            usage_json = to_jsonable_usage(usage_raw)
            combined_text_parts.append(text)

            # Write per-chunk API output (JSON: text + usage)
            chunk_txt = re.sub(r"\.m4a$", ".txt", cpath)
            with open(chunk_txt, "w", encoding="utf-8") as f:
                json.dump({"text": text, "usage": usage_json}, f, ensure_ascii=False, indent=2)

            # Tokens and cost
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

            print(f"   usage: input={in_tokens} output={out_tokens} total={tot_tokens} | cost=${chunk_cost:.6f}")
        except Exception as e:
            err = str(e)
            print(f"   error: {err}")
            metadata["chunks"].append({
                "index": idx,
                "file": cpath,
                "start_seconds": idx * chunk_seconds,
                "error": err,
            })
            # Continue with next chunk

    # Combined transcript
    combined_text = "\n\n".join(combined_text_parts)
    combined_path = os.path.join(run_dir, "combined.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    # Optional summarization of combined transcript
    summary_info: Dict[str, Any] = {}
    if not args.no_summary and combined_text.strip():
        prompt_file = os.path.join(os.getcwd(), args.summary_prompt_file)
        default_prompt = (
            "Summarize the transcript in clear prose, capturing key points, "
            "structure, and conclusions. Limit to about 200 words. "
            "Write the summary in the same language as the transcript; do not translate."
        )
        prompt_text = read_text_file(prompt_file).strip() or default_prompt
        # Ensure language consistency regardless of external prompt contents
        prompt_text = prompt_text.strip() + "\n\nPlease write the summary in the same language as the transcript (do not translate)."
        # Estimate tokens for input (prompt + transcript)
        est_tokens = estimate_tokens(prompt_text + "\n\n" + combined_text, model=args.summary_model)
        est_input_cost = (est_tokens / 1_000_000.0) * float(args.summary_price_in_per_mtoken)
        print(f"Generating summary with {args.summary_model} (estimated input tokens: {est_tokens}, est input cost: ${est_input_cost:.6f})...")
        sum_resp = summarize_transcript(client, model=args.summary_model,
                                        transcript_text=combined_text,
                                        prompt_text=prompt_text)
        if "summary" in sum_resp:
            summary_text = sum_resp["summary"]
            summary_path = os.path.join(run_dir, "summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            print(f"Summary written: {summary_path}")
        else:
            summary_text = ""
            summary_path = None
            print(f"Summary generation error: {sum_resp.get('error')}")

        # Pricing based on usage (actual)
        usage_clean = to_jsonable_usage(sum_resp.get("usage")) if isinstance(sum_resp, dict) else {}
        actual_in = int(usage_clean.get("input_tokens", 0) or 0)
        actual_out = int(usage_clean.get("output_tokens", 0) or 0)
        actual_total = int(usage_clean.get("total_tokens", actual_in + actual_out) or 0)
        in_rate = float(args.summary_price_in_per_mtoken)
        out_rate = float(args.summary_price_out_per_mtoken)
        actual_cost_in = (actual_in / 1_000_000.0) * in_rate
        actual_cost_out = (actual_out / 1_000_000.0) * out_rate
        actual_cost_total = actual_cost_in + actual_cost_out

        print(f"Summary usage: input={actual_in} output={actual_out} total={actual_total}")
        print(f"Summary cost: input=${actual_cost_in:.6f} output=${actual_cost_out:.6f} total=${actual_cost_total:.6f}")

        summary_info = {
            "model": args.summary_model,
            "prompt_file": args.summary_prompt_file,
            "estimated_input_tokens": est_tokens,
            "estimated_input_cost_usd": round(est_input_cost, 6),
            "price_per_mtoken_usd": {
                "input": in_rate,
                "output": out_rate,
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

    # Metadata
    metadata["combined_transcript_file"] = combined_path
    if summary_info:
        metadata["summary"] = summary_info
    # Aggregate stats and cost
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

    s3_uploads: List[Dict[str, str]] = []
    if args.s3_bucket:
        run_subdir = os.path.basename(run_dir)
        base_prefix = args.s3_prefix.strip("/")
        upload_prefix = f"{base_prefix}/{run_subdir}" if base_prefix else run_subdir
        display_prefix = f"/{upload_prefix}" if upload_prefix else ""
        print(f"Uploading run artifacts to s3://{args.s3_bucket}{display_prefix}...")
        try:
            s3_uploads = upload_directory_to_s3(run_dir, args.s3_bucket, upload_prefix)
            metadata["s3_uploads"] = s3_uploads
            metadata["s3_destination"] = {
                "bucket": args.s3_bucket,
                "prefix": upload_prefix,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"Uploaded {len(s3_uploads)} files to S3.")
        except Exception as s3_exc:
            print(f"S3 upload error: {s3_exc}")

    print("Done.")
    print(f"Combined transcript: {combined_path}")
    print(f"Metadata: {meta_path}")
    print("Summary:")
    print(f" - Chunks: {len(chunk_paths)}")
    print(f" - Input tokens: {total_input_tokens}")
    print(f" - Output tokens: {total_output_tokens}")
    print(f" - Total tokens: {total_usage_tokens}")
    print(f" - Price per 1M input tokens: ${price_per_mtoken}")
    print(f" - Estimated total cost: ${total_cost_usd:.6f}")
    if s3_uploads:
        print(f" - S3 uploads: {len(s3_uploads)} files")
    # Summarizing stats (if applicable)
    if 'summary' in metadata and isinstance(metadata['summary'], dict):
        s = metadata['summary']
        model_name = s.get('model', args.summary_model)
        est_in = int(s.get('estimated_input_tokens', 0) or 0)
        est_in_cost = float(s.get('estimated_input_cost_usd', 0.0) or 0.0)
        price_in = s.get('price_per_mtoken_usd', {}).get('input', None)
        price_out = s.get('price_per_mtoken_usd', {}).get('output', None)
        act_in = int(s.get('actual_input_tokens', 0) or 0)
        act_out = int(s.get('actual_output_tokens', 0) or 0)
        act_tot = int(s.get('actual_total_tokens', act_in + act_out) or 0)
        act_costs = s.get('actual_cost_usd', {}) if isinstance(s.get('actual_cost_usd'), dict) else {}
        act_cost_in = act_costs.get('input', 0.0) or 0.0
        act_cost_out = act_costs.get('output', 0.0) or 0.0
        act_cost_total = act_costs.get('total', 0.0) or 0.0
        sfile = s.get('summary_file')
        serr = s.get('error')
        print("Summarizing:")
        print(f" - Model: {model_name}")
        if price_in is not None and price_out is not None:
            print(f" - Price per 1M tokens: input=${price_in} output=${price_out}")
        print(f" - Estimated input tokens: {est_in} (est input cost=${est_in_cost:.6f})")
        print(f" - Actual tokens: input={act_in} output={act_out} total={act_tot}")
        print(f" - Actual cost: input=${act_cost_in:.6f} output=${act_cost_out:.6f} total=${act_cost_total:.6f}")
        if sfile:
            print(f" - Summary file: {sfile}")
        if serr:
            print(f" - Error: {serr}")
    else:
        print("Summarizing: skipped")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
