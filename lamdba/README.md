# AWS Lambda Deployment Guide

This directory contains a Lambda entrypoint (`lambda_handler.py`) that wraps the existing YouTube → transcript → summary workflow so it can be invoked by AWS Lambda. Prepared in collaboration with OpenAI’s Codex assistant.

**Important:** This is a demo implementation. YouTube frequently blocks unauthenticated scraping and may require login cookies or additional headers. If yt-dlp reports "Sign in to confirm you’re not a bot", export cookies from an authenticated browser profile and wire them into the Lambda (see the yt-dlp wiki). Out of the box, the handler does not ship any cookie management, so downloads can fail for protected videos.

## 1. Prerequisites

- An AWS account with permission to create Lambda functions, IAM roles, and S3 buckets.
- An S3 bucket to hold the transcription outputs (optional but recommended).
- An OpenAI API key stored in AWS Secrets Manager (see **Step 1a** below). Set the Lambda environment variable `OPENAI_SECRET_NAME` to the secret’s identifier. (Optional: set `OPENAI_SECRET_JSON_KEY` when the secret holds a JSON document, default key is `OPENAI_API_KEY`.)
- FFmpeg/FFprobe binaries packaged for Lambda (e.g., via a Lambda layer or container image).
- Python dependencies: `yt-dlp`, `ffmpeg-python`, `openai`, `tiktoken` (plus their transitive deps).

### 1a. Store the OpenAI API key in Secrets Manager

1. Open Secrets Manager → **Store a new secret**.
2. Choose **Other type of secret**.
3. For a simple string secret, select **Plaintext** and paste the key (`sk-...`). Alternatively, store JSON such as `{ "OPENAI_API_KEY": "sk-..." }` if you prefer key/value format.
4. Give the secret a name (e.g., `openai/api-key`). Note the full ARN or name.
5. Leave rotation disabled unless you plan to rotate the key automatically.

CLI equivalent (plain string):
```bash
aws secretsmanager create-secret \
  --name openai/api-key \
  --secret-string "sk-YourOpenAIKey"
```

CLI for JSON payload:
```bash
aws secretsmanager create-secret \
  --name openai/api-key \
  --secret-string '{"OPENAI_API_KEY":"sk-YourOpenAIKey"}'
```

You will reference this secret name/ARN in the Lambda environment variable `OPENAI_SECRET_NAME`. If you stored JSON with a custom field, set `OPENAI_SECRET_JSON_KEY` accordingly.

## 2. Package Dependencies

Lambda needs all native binaries and Python modules to be available at runtime. There are two common approaches:

### Option A – Lambda Layer (zip)

1. Create a working folder (e.g., `layer-build`).
2. Install Python dependencies targeting the Lambda runtime:
   ```bash
   # For x86_64 Lambda functions (default)
   pip install \
     --platform manylinux2014_x86_64 \
     --implementation cp \
     --only-binary=:all: \
     --upgrade \
     --target layer-build/python \
     yt-dlp ffmpeg-python openai tiktoken

   # For arm64 Lambda functions
   pip install \
     --platform manylinux2014_aarch64 \
     --implementation cp \
     --only-binary=:all: \
     --upgrade \
     --target layer-build/python \
     yt-dlp ffmpeg-python openai tiktoken
   ```
3. Add FFmpeg and FFprobe binaries to the layer (place them under `layer-build/bin/`). Options:
   - **Static download (quickest):**
     Pick the archive that matches your Lambda architecture (`python3.11` runtime defaults to `x86_64`, but you may have switched to `arm64`).
     ```bash
     # For x86_64 functions (default)
     curl -L -o ffmpeg-release-amd64-static.tar.xz \
       https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
     tar -xf ffmpeg-release-amd64-static.tar.xz

     # For arm64 functions
     curl -L -o ffmpeg-release-arm64-static.tar.xz \
       https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz
     tar -xf ffmpeg-release-arm64-static.tar.xz

     mkdir -p layer-build/bin
     cp ffmpeg-*/ffmpeg ffmpeg-*/ffprobe layer-build/bin/
     ```
     These static builds run on Amazon Linux 2 with no additional libraries.
     *(macOS users can `brew install wget` if they prefer `wget` over `curl`.)*
   - **Container-based build (reproducible):** use an Amazon Linux 2 image matching your architecture (`amazonlinux:2` for x86_64, `amazonlinux:2023` + `--platform linux/arm64` for arm64 when using Docker Desktop).
     ```bash
     # x86_64 example
     docker run --rm -it -v "$(pwd)/layer-build:/opt" amazonlinux:2 bash
     yum update -y
     yum install -y tar xz wget
     wget -O /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
     cd /tmp && tar -xf ffmpeg.tar.xz
     mkdir -p /opt/bin
     cp ffmpeg-*static/ffmpeg ffmpeg-*static/ffprobe /opt/bin/
     exit
     ```
     For arm64, pass `--platform linux/arm64` to `docker run` and download `ffmpeg-release-arm64-static.tar.xz` instead.
   - **Reuse a published Lambda layer:** AWS SAR has community FFmpeg layers if you prefer to attach an existing artifact instead of shipping binaries yourself.
4. Zip the layer contents:
   ```bash
   cd layer-build
   zip -r ../youtube-transcribe-layer.zip .
   ```
5. Publish the zip as a Lambda layer:
   ```bash
   aws lambda publish-layer-version \
     --layer-name youtube-transcribe-deps \
     --zip-file fileb://youtube-transcribe-layer.zip \
     --compatible-runtimes python3.11 \
     --region <your-region>
   ```
   Note the returned `LayerVersionArn` and attach it to your Lambda function:
   ```bash
   aws lambda update-function-configuration \
     --function-name <your-function-name> \
     --layers <layer-version-arn> \
     --region <your-region>
   ```
   *(Console alternative: Lambda → Layers → Create layer → upload zip → pick runtime; then Lambda → Functions → Configuration → Layers → Add layer.)*

### Option B – Container Image

1. Start from the AWS provided Python base image (e.g., `public.ecr.aws/lambda/python:3.11`).
2. Copy the project files into the image.
3. Install the Python dependencies with `pip install -r requirements.txt`.
4. Install FFmpeg/FFprobe (e.g., with `yum`/`apt` depending on the base image).
5. Set the image’s `CMD` to `lamdba.lambda_handler.handler`.

## 3. Create the Lambda Function

1. Zip the Lambda sources (the handler is standalone, no other repo files required):
   ```bash
   zip -r youtube-transcribe-code.zip lamdba
   ```
2. Create a new Lambda function using the Python 3.11 runtime (or the runtime that matches your dependencies).
3. Set the handler to:
   ```
   lamdba.lambda_handler.handler
   ```
4. Increase memory (at least 2048 MB recommended) and timeout (5–15 minutes depending on video length).
5. Optional: increase ephemeral storage above the default 512 MB if you expect long videos (`Configuration → General configuration → Edit` → set “Ephemeral storage” up to 10 GB).
6. Attach the dependency layer or select the container image built in the previous step.
7. Configure an execution role with permissions to write to the S3 bucket (if used) and to read the secret:
   ```
   s3:PutObject, s3:GetObject, s3:AbortMultipartUpload (as needed)
   secretsmanager:GetSecretValue on your OpenAI secret
   ```
   Example inline policy (replace placeholders with your values):
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:PutObject",
           "s3:GetObject",
           "s3:AbortMultipartUpload"
         ],
         "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/YOUR_PREFIX/*"
       },
       {
         "Effect": "Allow",
         "Action": "secretsmanager:GetSecretValue",
         "Resource": "arn:aws:secretsmanager:YOUR_REGION:YOUR_ACCOUNT_ID:secret:openai/api-key-*"
       }
     ]
   }
   ```
8. Add environment variables for Secrets Manager access, for example:
   ```
   OPENAI_SECRET_NAME=openai/api-key
   OPENAI_SECRET_JSON_KEY=OPENAI_API_KEY  # optional when the secret is JSON
   ```

## 4. Invoke the Function

Sample test event:
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "chunk_seconds": 180,
  "model": "gpt-4o-mini-transcribe",
  "language": "auto",
  "price_per_mtoken": 3.0,
  "summary_model": "gpt-4o-mini",
  "summary_price_in_per_mtoken": 0.15,
  "summary_price_out_per_mtoken": 0.6,
  "s3_bucket": "my-transcripts-bucket",
  "s3_prefix": "lambda-runs/demo"
}
```

Response payload structure:
- `status`: `ok` or `error`.
- `run_id`: timestamp-based identifier.
- `output_dir`: local path inside Lambda (`/tmp/...`).
- `combined_transcript`: path to `combined.txt`.
- `summary_file`: path to `summary.txt` (if generated).
- `metadata_file`: path to `metadata.json`.
- `summary_text`: summary content (if generated).
- `metadata`: JSON metadata dictionary mirroring `metadata.json`.
- `s3_uploads`: list of uploaded `{bucket, key}` objects when S3 is configured.

If the status is `error`, `message` explains the failure.

## 5. Operational Notes

- Lambda has a hard cap on execution time (15 minutes). For longer videos, reduce `chunk_seconds`, skip summary, or consider AWS Batch/ECS.
- Ensure your OpenAI account has sufficient quota; costs depend on transcript length and summary output.
- `tiktoken` is optional (the handler falls back to a rough heuristic if it cannot load the tokenizer), but including it gives better cost estimates.
- For automated pipelines, trigger the Lambda via API Gateway, EventBridge Scheduler, or another service. Ensure the invoker provides the YouTube URL and, if needed, an S3 prefix to keep outputs organized.
