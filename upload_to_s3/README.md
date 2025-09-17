# Uploading Transcription Runs to Amazon S3

This directory contains a variant of `youtube_audio_transcribe.py` that uploads every file produced during a run (downloaded audio, chunk transcripts, metadata, summary, etc.) to Amazon S3 once processing finishes.

## 1. Install Requirements

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt boto3
```

Make sure `ffmpeg` and `ffprobe` are available on your `PATH` and `OPENAI_API_KEY` is set.

## 2. Configure AWS Credentials

The uploader uses the default AWS credential chain that `boto3` supports. Pick whichever option fits your workflow:

- **Environment variables (no AWS CLI required):**
  1. In the AWS console create or reuse an IAM access key pair (Access key ID + Secret access key).
  2. Export the values before running the script:
     ```bash
     export AWS_ACCESS_KEY_ID=AKIA...
     export AWS_SECRET_ACCESS_KEY=yourSecret
     export AWS_DEFAULT_REGION=us-east-1      # choose your region
     # export AWS_SESSION_TOKEN=...           # only when using temporary credentials
     ```
     On Windows use `setx` or PowerShell’s `$Env:` syntax.
- **Shared credentials files:** manually create `~/.aws/credentials` (and optionally `~/.aws/config`) with a `[default]` profile containing the same keys.
- **AWS CLI (`aws configure`):** still works if you prefer the CLI to write the files for you.

Whichever method you choose, make sure the credentials have permission to put objects into your target bucket (see policy example below). Then create or choose the S3 bucket and optional key prefix that will receive the run artifacts.

## 3. Run the Script

```bash
python youtube_audio_transcribe.py "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --chunk-seconds 180 \
  --outdir outputs \
  --model gpt-4o-mini-transcribe \
  --language auto \
  --speed 1.5 \
  --summary-model gpt-4o-mini \
  --s3-bucket my-transcripts-bucket \
  --s3-prefix demo-runs
```

- `--s3-bucket` is required to trigger uploads.
- `--s3-prefix` is optional; when provided the run uploads land under `s3://<bucket>/<prefix>/<run_id>/`. Without it the script still creates a folder named after the run ID at the root.
- All other CLI flags match those in the root script and remain optional.

After completion, the script prints how many files were uploaded and records the list under `s3_uploads` inside the run’s `metadata.json`.
