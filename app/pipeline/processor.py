import cv2
import numpy as np

from app.config import Settings
from app.services.image_downloader import ImageDownloader
from app.services.ocr import OCRService
from app.services.inpaint import InpaintService
from app.services.translation import TranslationService
from app.services.text_grouping import TextGrouper
from app.services.text_renderer import TextRenderer


class TranslationPipeline:
    def __init__(self, settings: Settings):
        self.downloader = ImageDownloader(settings)
        self.ocr = OCRService()
        self.grouper = TextGrouper()
        self.inpainter = InpaintService()
        self.translator = TranslationService(settings)
        self.renderer = TextRenderer(settings)

    def process(self, image_url: str, source_lang: str, target_lang: str) -> bytes:
        # Step 1: Download and validate image
        img_array = self.downloader.download(image_url)

        # Step 2: OCR - detect text regions and read text
        detections = self.ocr.detect_and_recognize(img_array, lang=source_lang)

        if not detections:
            return self._encode_image(img_array)

        # Step 3: Group nearby text regions into speech bubbles
        grouped = self.grouper.group(detections)
        if not grouped:
            return self._encode_image(img_array)

        # Step 4: Create mask from bounding boxes and inpaint
        mask = self.inpainter.create_mask(img_array.shape, grouped)
        inpainted = self.inpainter.inpaint(img_array, mask)

        # Step 5: Translate all detected text
        texts = [d.text for d in grouped]
        translated_texts = self.translator.translate_batch(
            texts, source_lang=source_lang, target_lang=target_lang
        )

        # Step 6: Render translated text onto inpainted image
        result = self.renderer.render(
            inpainted, grouped, translated_texts, target_lang
        )

        return self._encode_image(result)

    def _encode_image(self, image: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode result image")
        return buffer.tobytes()
