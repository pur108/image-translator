import os

import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
from simple_lama_inpainting.models.model import download_model, LAMA_MODEL_URL

from app.errors import InpaintError
from app.services.ocr import TextRegion


def _load_lama_cpu() -> SimpleLama:
    """Load LaMa model forcing CPU, even if checkpoint has CUDA tensors."""
    model_path = os.environ.get("LAMA_MODEL")
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"LaMa model not found: {model_path}")
    if not model_path:
        model_path = download_model(LAMA_MODEL_URL)

    lama = object.__new__(SimpleLama)
    lama.device = torch.device("cpu")
    lama.model = torch.jit.load(model_path, map_location="cpu")
    lama.model.eval()
    lama.model.to(lama.device)
    return lama


class InpaintService:
    def __init__(self, fast_inpaint: bool = True):
        self.fast_inpaint = fast_inpaint
        self._lama: SimpleLama | None = None
        if not fast_inpaint:
            self._lama = _load_lama_cpu()

    def create_mask(
        self,
        image_shape: tuple,
        regions: list[TextRegion],
        dilation_pixels: int = 5,
    ) -> np.ndarray:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for region in regions:
            pts = np.array(region.bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        if dilation_pixels > 0:
            kernel = np.ones(
                (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1), np.uint8
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            if self.fast_inpaint:
                return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

            # Convert BGR (OpenCV) to RGB (PIL)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_mask = Image.fromarray(mask)

            # Run LaMa inpainting
            result_pil = self._lama(pil_image, pil_mask)

            # Convert back to BGR numpy
            result_rgb = np.array(result_pil)
            return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise InpaintError(f"Inpainting failed: {e}")
