class TranslationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class InvalidURLError(TranslationError):
    def __init__(
        self, message: str = "URL must be a valid Supabase storage public URL"
    ):
        super().__init__("INVALID_URL", message, 400)


class InvalidFileTypeError(TranslationError):
    def __init__(self, message: str = "Only JPEG and PNG images are supported"):
        super().__init__("INVALID_FILE_TYPE", message, 400)


class FileTooLargeError(TranslationError):
    def __init__(self, message: str = "Image exceeds 10MB limit"):
        super().__init__("FILE_TOO_LARGE", message, 400)


class QuotaExceededError(TranslationError):
    def __init__(self, message: str = "DeepL translation quota exceeded"):
        super().__init__("QUOTA_EXCEEDED", message, 429)


class OCRError(TranslationError):
    def __init__(self, message: str = "OCR processing failed"):
        super().__init__("OCR_FAILED", message, 500)


class InpaintError(TranslationError):
    def __init__(self, message: str = "Image inpainting failed"):
        super().__init__("INPAINT_FAILED", message, 500)


class ImageDownloadError(TranslationError):
    def __init__(self, message: str = "Failed to download image"):
        super().__init__("IMAGE_DOWNLOAD_FAILED", message, 502)
