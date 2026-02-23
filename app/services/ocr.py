from dataclasses import dataclass

import numpy as np
from paddleocr import PaddleOCR

from app.errors import OCRError


@dataclass
class TextRegion:
    bbox: list[list[int]]  # 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text: str
    confidence: float


class OCRService:
    def __init__(self):
        self._models: dict[str, PaddleOCR] = {}

    def _get_model(self, lang: str) -> PaddleOCR:
        paddle_lang = "en" if lang == "en" else "th"
        if paddle_lang not in self._models:
            self._models[paddle_lang] = PaddleOCR(
                lang=paddle_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._models[paddle_lang]

    # def _get_model(self, lang: str) -> PaddleOCR:
    #     paddle_lang = "en" if lang == "en" else "th"
    #     if paddle_lang not in self._models:
    #         self._models[paddle_lang] = PaddleOCR(
    #             use_doc_orientation_classify=False,
    #             use_doc_unwarping=False,
    #             use_textline_orientation=False,
    #             text_detection_model_name="PP-OCRv5_server_det",
    #             text_recognition_model_name="PP-OCRv5_server_rec",
    #         )
    #     return self._models[paddle_lang]

    # def _get_model(self, lang: str) -> PaddleOCR:
    #     paddle_lang = "en" if lang == "en" else "th"
    #     if paddle_lang not in self._models:
    #         self._models[paddle_lang] = PaddleOCR(
    #             use_doc_orientation_classify=True,
    #             use_doc_unwarping=True,
    #             use_textline_orientation=True,
    #             text_detection_model_name="PP-OCRv5_server_det",
    #             text_recognition_model_name=self.REC_MODELS[paddle_lang],
    #         )
    #     return self._models[paddle_lang]

    def detect_and_recognize(
        self, image: np.ndarray, lang: str, confidence_threshold: float = 0.5
    ) -> list[TextRegion]:
        model = self._get_model(lang)

        try:
            results = list(model.predict(image))
        except Exception as e:
            raise OCRError(f"PaddleOCR processing failed: {e}")

        if not results:
            return []

        res = results[0]
        rec_texts = res.get("rec_texts", [])
        if not rec_texts:
            return []

        rec_scores = res.get("rec_scores", [])
        rec_polys = res.get("rec_polys", [])

        regions: list[TextRegion] = []
        for i, text in enumerate(rec_texts):
            confidence = rec_scores[i]
            if confidence < confidence_threshold:
                continue
            poly = rec_polys[i]  # numpy array of shape (4, 2)
            bbox_int = [[int(p[0]), int(p[1])] for p in poly]
            regions.append(TextRegion(bbox=bbox_int, text=text, confidence=confidence))

        return regions
