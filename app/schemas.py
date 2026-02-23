from enum import Enum
from typing import Optional

from pydantic import BaseModel, HttpUrl, field_validator


class LangCode(str, Enum):
    TH = "th"
    EN = "en"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class TranslateRequest(BaseModel):
    image_url: HttpUrl
    source_lang: LangCode
    target_lang: LangCode

    @field_validator("target_lang")
    @classmethod
    def langs_must_differ(cls, v, info):
        if info.data.get("source_lang") and v == info.data["source_lang"]:
            raise ValueError("source_lang and target_lang must be different")
        return v


class BatchTranslateRequest(BaseModel):
    image_urls: list[HttpUrl]
    source_lang: LangCode
    target_lang: LangCode
    callback_url: Optional[HttpUrl] = None

    @field_validator("target_lang")
    @classmethod
    def langs_must_differ(cls, v, info):
        if info.data.get("source_lang") and v == info.data["source_lang"]:
            raise ValueError("source_lang and target_lang must be different")
        return v

    @field_validator("image_urls")
    @classmethod
    def at_least_one_url(cls, v):
        if not v:
            raise ValueError("image_urls must contain at least one URL")
        return v


class SuccessResponse(BaseModel):
    success: bool = True
    data: dict


class ErrorResponseBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorResponseBody
