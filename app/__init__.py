from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import router
from app.config import settings
from app.errors import TranslationError


def create_app() -> FastAPI:
    application = FastAPI(
        title="Manga Translation API",
        version="1.0.0",
        docs_url="/docs" if settings.APP_ENV != "production" else None,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    application.include_router(router)

    @application.exception_handler(TranslationError)
    async def translation_error_handler(request: Request, exc: TranslationError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {"code": exc.code, "message": exc.message},
            },
        )

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    return application


app = create_app()
