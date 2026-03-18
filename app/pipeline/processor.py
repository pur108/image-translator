import logging
import time

import cv2
import numpy as np

from app.config import Settings
from app.services.image_downloader import ImageDownloader
from app.services.ocr import OCRService
from app.services.inpaint import InpaintService
from app.services.translation import TranslationService
from app.services.text_grouping import TextGrouper
from app.services.text_renderer import TextRenderer


logger = logging.getLogger(__name__)


class TranslationPipeline:
    def __init__(self, settings: Settings):
        self.downloader = ImageDownloader(settings)
        self.ocr = OCRService()
        self.grouper = TextGrouper()
        self.inpainter = InpaintService(fast_inpaint=settings.FAST_INPAINT)
        self.translator = TranslationService(settings)
        self.renderer = TextRenderer(settings)
        logger.info("Warming up OCR models...")
        self.ocr.warmup()
        logger.info("Pipeline ready")

    def process(self, image_url: str, source_lang: str, target_lang: str) -> bytes:
        t0 = time.time()

        # Step 1: Download and validate image
        img_array = self.downloader.download(image_url)
        logger.info("Download: %.1fs", time.time() - t0)

        # Step 2: OCR - detect text regions and read text
        t1 = time.time()
        detections = self.ocr.detect_and_recognize(img_array, lang=source_lang)
        logger.info("OCR: %.1fs (%d regions)", time.time() - t1, len(detections))

        if not detections:
            return self._encode_image(img_array)

        # Step 3: Group nearby text regions into speech bubbles
        grouped = self.grouper.group(detections)
        if not grouped:
            return self._encode_image(img_array)

        # Step 4: Create mask from bounding boxes and inpaint
        t2 = time.time()
        mask = self.inpainter.create_mask(img_array.shape, grouped)
        inpainted = self.inpainter.inpaint(img_array, mask)
        logger.info("Inpaint: %.1fs", time.time() - t2)

        # Step 5: Translate all detected text
        t3 = time.time()
        texts = [d.text for d in grouped]
        translated_texts = self.translator.translate_batch(
            texts, source_lang=source_lang, target_lang=target_lang
        )
        logger.info("Translate: %.1fs (%d texts)", time.time() - t3, len(texts))

        # Step 6: Render translated text onto inpainted image
        t4 = time.time()
        result = self.renderer.render(
            inpainted, grouped, translated_texts, target_lang
        )
        logger.info("Render: %.1fs", time.time() - t4)

        logger.info("Total pipeline: %.1fs", time.time() - t0)
        return self._encode_image(result)

    def _encode_image(self, image: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode result image")
        return buffer.tobytes()
