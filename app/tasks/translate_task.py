import json
import logging
import uuid
from typing import Any, cast

import httpx
import redis
from celery import Task

from app.config import settings
from app.pipeline.processor import TranslationPipeline
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

_pool = redis.ConnectionPool.from_url(settings.REDIS_URL)


def _get_redis() -> redis.Redis:  # type: ignore[type-arg]
    return redis.Redis(connection_pool=_pool)


class TranslateTask(Task):
    _pipeline: TranslationPipeline | None = None

    @property
    def pipeline(self) -> TranslationPipeline:
        if self._pipeline is None:
            self._pipeline = TranslationPipeline(settings)
        return self._pipeline


_BATCH_COMPLETE_LUA = """
local completed = redis.call('HINCRBY', KEYS[1], 'completed', 1)
local total = tonumber(redis.call('HGET', KEYS[1], 'total') or 0)
if completed >= total and total > 0 then
    redis.call('HSET', KEYS[1], 'status', 'done')
    return 1
end
return 0
"""


@celery_app.task(
    base=TranslateTask,
    bind=True,
    name="translate_image",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    max_retries=3,
)
def translate_image(
    self: TranslateTask, job_id: str, image_url: str, source_lang: str, target_lang: str
) -> None:
    r = _get_redis()
    job_key = f"job:{job_id}"

    try:
        r.hset(job_key, mapping={"status": "processing"})

        result_bytes = self.pipeline.process(image_url, source_lang, target_lang)

        pipe = r.pipeline()
        pipe.hset(job_key, mapping={
            "status": "done",
            "content_type": "image/png",
            "result": result_bytes,
        })
        pipe.expire(job_key, settings.RESULT_TTL_SECONDS)
        pipe.execute()

        batch_id_raw: bytes | None = cast(bytes | None, r.hget(job_key, "batch_id"))
        if batch_id_raw:
            _complete_batch_job(r, batch_id_raw.decode())

    except Exception as exc:
        r.hset(job_key, mapping={"status": "failed", "error": str(exc)})
        r.expire(job_key, settings.RESULT_TTL_SECONDS)
        raise


def _complete_batch_job(r: redis.Redis, batch_id: str) -> None:  # type: ignore[type-arg]
    batch_key = f"batch:{batch_id}"
    is_done = r.eval(_BATCH_COMPLETE_LUA, 1, batch_key)

    if is_done:
        batch_data: dict[bytes, bytes] = cast(dict[bytes, bytes], r.hgetall(batch_key))
        callback_url = batch_data.get(b"callback_url")
        if callback_url:
            _send_callback(
                callback_url.decode(),
                batch_id,
                json.loads(batch_data.get(b"job_ids", b"[]").decode()),
            )


def _send_callback(callback_url: str, batch_id: str, job_ids: list[str]) -> None:
    try:
        httpx.post(
            callback_url,
            json={"batch_id": batch_id, "status": "done", "job_ids": job_ids},
            timeout=10.0,
        )
    except Exception:
        logger.exception("Batch callback failed for %s", batch_id)


@celery_app.task(name="translate_batch")
def translate_batch(
    batch_id: str,
    image_urls: list[str],
    source_lang: str,
    target_lang: str,
    callback_url: str | None,
) -> None:
    r = _get_redis()
    batch_key = f"batch:{batch_id}"
    job_ids = [str(uuid.uuid4()) for _ in image_urls]

    mapping: dict[str, Any] = {
        "status": "processing",
        "job_ids": json.dumps(job_ids),
        "total": len(job_ids),
        "completed": 0,
    }
    if callback_url:
        mapping["callback_url"] = callback_url
    r.hset(batch_key, mapping=mapping)
    r.expire(batch_key, settings.RESULT_TTL_SECONDS)

    for job_id, url in zip(job_ids, image_urls):
        r.hset(f"job:{job_id}", mapping={"status": "pending", "batch_id": batch_id})
        translate_image.delay(job_id, url, source_lang, target_lang)  # type: ignore[attr-defined]
