import cv2
import httpx
import numpy as np

from app.config import Settings
from app.errors import (
    FileTooLargeError,
    ImageDownloadError,
    InvalidFileTypeError,
    InvalidURLError,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


class ImageDownloader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.expected_prefix = (
            f"{settings.SUPABASE_URL}/storage/v1/object/public/"
            f"{settings.SUPABASE_STORAGE_BUCKET}/"
        )
        self.client = httpx.Client(timeout=30.0)

    def validate_url(self, url: str) -> None:
        if not url.startswith(self.expected_prefix):
            raise InvalidURLError(f"URL must start with {self.expected_prefix}")

    def download(self, url: str) -> np.ndarray:
        self.validate_url(url)

        try:
            response = self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ImageDownloadError(
                f"Failed to download image: HTTP {e.response.status_code}"
            )
        except httpx.RequestError as e:
            raise ImageDownloadError(f"Failed to download image: {e}")

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise InvalidFileTypeError(
                f"Content-Type '{content_type}' is not supported. Only JPEG and PNG are allowed."
            )

        raw_bytes = response.content
        if len(raw_bytes) > self.settings.MAX_IMAGE_SIZE_BYTES:
            raise FileTooLargeError(
                f"Image size ({len(raw_bytes)} bytes) exceeds {self.settings.MAX_IMAGE_SIZE_BYTES} byte limit"
            )

        img_array = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            raise InvalidFileTypeError("Failed to decode image. File may be corrupted.")

        max_dim = self.settings.MAX_PROCESS_DIMENSION
        h, w = img_array.shape[:2]
        if max_dim and max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img_array = cv2.resize(
                img_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )

        return img_array
