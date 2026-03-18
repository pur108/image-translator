# Image Translator

A manga/comic image translation API that detects text in images using OCR, translates it, and renders the translated text back onto the image.

## Features

- OCR-based text detection (PaddleOCR)
- Translation via DeepL or OpenAI
- Automatic text inpainting and re-rendering
- Async job processing with Celery + Redis
- Single image and batch translation endpoints
- Thai and English language support

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Redis

## Setup

```bash
# Install dependencies
uv sync

# Create .env from example and fill in your API keys
cp .env.example .env
```

Edit `.env` with your configuration:

| Variable | Description |
|---|---|
| `TRANSLATION_PROVIDER` | `deepl` or `openai` |
| `DEEPL_API_KEY` | DeepL API key (if using DeepL) |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_STORAGE_BUCKET` | Supabase storage bucket name |
| `REDIS_URL` | Redis connection URL (default: `redis://localhost:6379/0`) |

## Running Locally

### 1. Start Redis

```bash
# macOS (Homebrew)
brew services start redis

# Or run directly
redis-server
```

### 2. Start the API server

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Start the Celery worker (separate terminal)

```bash
uv run celery -A app.tasks.celery_app worker --loglevel=info
```

The API will be available at `http://localhost:8001`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/translate` | Submit a single image translation job |
| `POST` | `/api/v1/translate/batch` | Submit a batch translation job |
| `GET` | `/api/v1/status/{job_id}` | Check job status |
| `GET` | `/api/v1/result/{job_id}` | Get translated image |
| `GET` | `/api/v1/batch/{batch_id}` | Get batch status |

### Example: Translate an image

```bash
curl -X POST http://localhost:8001/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://your-project.supabase.co/storage/v1/object/public/your-bucket/image.png",
    "source_lang": "th",
    "target_lang": "en"
  }'
```

API docs are available at `http://localhost:8001/docs` in non-production environments.
