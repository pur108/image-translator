from celery import Celery

from app.config import settings

celery_app = Celery(
    "translate_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=settings.RESULT_TTL_SECONDS,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

celery_app.conf.include = ["app.tasks.translate_task"]
