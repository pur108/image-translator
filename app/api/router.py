import json
import uuid

import redis
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.config import settings
from app.schemas import BatchTranslateRequest, TranslateRequest
from app.tasks.translate_task import translate_batch, translate_image

router = APIRouter(prefix="/api/v1")


def _get_redis() -> redis.Redis:
    return redis.from_url(settings.REDIS_URL)


def _validate_supabase_url(url: str) -> None:
    expected_prefix = (
        f"{settings.SUPABASE_URL}/storage/v1/object/public/"
        f"{settings.SUPABASE_STORAGE_BUCKET}/"
    )
    if not url.startswith(expected_prefix):
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": {
                    "code": "INVALID_URL",
                    "message": "URL must be a Supabase storage public URL",
                },
            },
        )


@router.post("/translate", status_code=202)
async def create_translation_job(req: TranslateRequest):
    url_str = str(req.image_url)
    _validate_supabase_url(url_str)

    job_id = str(uuid.uuid4())
    r = _get_redis()
    r.hset(f"job:{job_id}", mapping={"status": "pending"})
    r.expire(f"job:{job_id}", settings.RESULT_TTL_SECONDS)

    translate_image.delay(
        job_id, url_str, req.source_lang.value, req.target_lang.value
    )

    return {"success": True, "data": {"job_id": job_id}}


@router.post("/translate/batch", status_code=202)
async def create_batch_translation(req: BatchTranslateRequest):
    for url in req.image_urls:
        _validate_supabase_url(str(url))

    batch_id = str(uuid.uuid4())
    urls = [str(u) for u in req.image_urls]
    callback = str(req.callback_url) if req.callback_url else None

    translate_batch.delay(
        batch_id, urls, req.source_lang.value, req.target_lang.value, callback
    )

    return {"success": True, "data": {"batch_id": batch_id}}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    r = _get_redis()
    job_data = r.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": {"code": "NOT_FOUND", "message": "Job not found"},
            },
        )

    status = job_data.get(b"status", b"unknown").decode()
    result = {"job_id": job_id, "status": status}
    if status == "failed":
        result["error"] = job_data.get(b"error", b"").decode()

    return {"success": True, "data": result}


@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    r = _get_redis()
    job_data = r.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": {"code": "NOT_FOUND", "message": "Job not found"},
            },
        )

    status = job_data.get(b"status", b"").decode()
    if status != "done":
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "error": {
                    "code": "JOB_NOT_READY",
                    "message": f"Job status is '{status}', not 'done'",
                },
            },
        )

    image_bytes = job_data.get(b"result", b"")
    content_type = job_data.get(b"content_type", b"image/png").decode()
    return Response(content=image_bytes, media_type=content_type)


@router.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    r = _get_redis()
    batch_data = r.hgetall(f"batch:{batch_id}")
    if not batch_data:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": {"code": "NOT_FOUND", "message": "Batch not found"},
            },
        )

    job_ids = json.loads(batch_data.get(b"job_ids", b"[]").decode())
    jobs = []
    for jid in job_ids:
        jdata = r.hgetall(f"job:{jid}")
        jobs.append(
            {
                "job_id": jid,
                "status": jdata.get(b"status", b"unknown").decode()
                if jdata
                else "unknown",
            }
        )

    all_done = all(j["status"] in ("done", "failed") for j in jobs)

    return {
        "success": True,
        "data": {
            "batch_id": batch_id,
            "status": "done" if all_done else "processing",
            "total": len(jobs),
            "completed": int(batch_data.get(b"completed", 0)),
            "jobs": jobs,
        },
    }
