import deepl

from app.config import Settings
from app.errors import QuotaExceededError, TranslationError

LANG_MAP_SOURCE = {
    "th": "TH",
    "en": "EN",
}

LANG_MAP_TARGET = {
    "th": "TH",
    "en": "EN-US",
}


class TranslationService:
    def __init__(self, settings: Settings):
        server_url = None
        if settings.DEEPL_FREE_API:
            server_url = "https://api-free.deepl.com"
        self.translator = deepl.Translator(
            settings.DEEPL_API_KEY, server_url=server_url
        )

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        if not texts:
            return []

        deepl_source = LANG_MAP_SOURCE.get(source_lang)
        deepl_target = LANG_MAP_TARGET.get(target_lang)

        if not deepl_source or not deepl_target:
            raise TranslationError(
                "UNSUPPORTED_LANGUAGE",
                f"Language pair {source_lang}->{target_lang} is not supported",
                400,
            )

        try:
            results = self.translator.translate_text(
                texts,
                source_lang=deepl_source,
                target_lang=deepl_target,
            )
        except deepl.QuotaExceededException:
            raise QuotaExceededError()
        except deepl.DeepLException as e:
            raise TranslationError(
                "TRANSLATION_FAILED", f"DeepL API error: {e}", 502
            )

        if isinstance(results, list):
            return [r.text for r in results]
        return [results.text]
