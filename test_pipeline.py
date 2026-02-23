"""
Layer-by-layer pipeline tester.

Usage:
    python test_pipeline.py --image-url "https://..." --source-lang th --target-lang en
    python test_pipeline.py --image-url "https://..." --source-lang th --target-lang en --job-id 42
"""

import argparse
import json
import os
import uuid

import cv2
import numpy as np

from app.config import Settings

RESULT_DIR = "./result"


def get_output_dir(job_id: str) -> str:
    path = os.path.join(RESULT_DIR, f"job{job_id}")
    os.makedirs(path, exist_ok=True)
    return path


# ── Layer 1: Download ──
def test_download(settings: Settings, image_url: str, out_dir: str) -> np.ndarray:
    from app.services.image_downloader import ImageDownloader

    print("[1/6] Downloading image...")
    downloader = ImageDownloader(settings)
    img = downloader.download(image_url)
    cv2.imwrite(os.path.join(out_dir, "1_downloaded.png"), img)
    print(f"      Saved 1_downloaded.png  (shape: {img.shape})")
    return img


# ── Layer 2: OCR ──
def test_ocr(image: np.ndarray, source_lang: str, out_dir: str):
    from app.services.ocr import OCRService

    print("[2/6] Running OCR...")
    ocr = OCRService()
    regions = ocr.detect_and_recognize(image, lang=source_lang)
    print(f"      Detected {len(regions)} text region(s)")

    # Save JSON
    data = [
        {"bbox": r.bbox, "text": r.text, "confidence": round(r.confidence, 4)}
        for r in regions
    ]
    with open(os.path.join(out_dir, "2_ocr.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Save visual with bounding boxes
    vis = image.copy()
    for r in regions:
        pts = np.array(r.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            vis,
            f"{r.confidence:.2f}",
            (pts[0][0], pts[0][1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(os.path.join(out_dir, "2_ocr_visual.png"), vis)
    print("      Saved 2_ocr.json + 2_ocr_visual.png")
    return regions


# ── Layer 3: Text Grouping ──
def test_grouping(image: np.ndarray, regions, out_dir: str):
    from app.services.text_grouping import TextGrouper

    print("[3/6] Grouping text regions...")
    grouper = TextGrouper()
    grouped = grouper.group(regions)
    print(f"      {len(regions)} regions → {len(grouped)} group(s)")

    # Save JSON
    data = [
        {"bbox": r.bbox, "text": r.text, "confidence": round(r.confidence, 4)}
        for r in grouped
    ]
    with open(os.path.join(out_dir, "3_grouped.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Save visual with grouped bounding boxes (blue) over OCR boxes (green)
    vis = image.copy()
    for r in regions:
        pts = np.array(r.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    for r in grouped:
        pts = np.array(r.bbox, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imwrite(os.path.join(out_dir, "3_grouped_visual.png"), vis)
    print("      Saved 3_grouped.json + 3_grouped_visual.png")
    return grouped


# ── Layer 4: Inpainting ──
def test_inpaint(image: np.ndarray, regions, out_dir: str) -> np.ndarray:
    from app.services.inpaint import InpaintService

    print("[4/6] Inpainting (removing text)...")
    inpainter = InpaintService()
    mask = inpainter.create_mask(image.shape, regions)
    cv2.imwrite(os.path.join(out_dir, "4_mask.png"), mask)
    inpainted = inpainter.inpaint(image, mask)
    cv2.imwrite(os.path.join(out_dir, "4_inpainted.png"), inpainted)
    print("      Saved 4_mask.png + 4_inpainted.png")
    return inpainted


# ── Layer 5: Translation ───
def test_translate(
    settings: Settings, regions, source_lang: str, target_lang: str, out_dir: str
) -> list[str]:
    from app.services.translation import TranslationService

    print("[5/6] Translating text...")
    translator = TranslationService(settings)
    texts = [r.text for r in regions]
    translated = translator.translate_batch(texts, source_lang, target_lang)

    data = [
        {"original": orig, "translated": trans}
        for orig, trans in zip(texts, translated)
    ]
    with open(os.path.join(out_dir, "5_translated.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"      Translated {len(translated)} text(s)")
    print("      Saved 5_translated.json")
    return translated


# ── Layer 6: Text Rendering ───
def test_render(
    settings: Settings,
    inpainted: np.ndarray,
    regions,
    translated: list[str],
    target_lang: str,
    out_dir: str,
) -> np.ndarray:
    from app.services.text_renderer import TextRenderer

    print("[6/6] Rendering translated text...")
    renderer = TextRenderer(settings)
    result = renderer.render(inpainted, regions, translated, target_lang)
    cv2.imwrite(os.path.join(out_dir, "6_rendered.png"), result)
    print("      Saved 6_rendered.png")
    return result


# ── Main ───
def main():
    parser = argparse.ArgumentParser(description="Test pipeline layer by layer")
    parser.add_argument("--image-url", required=True, help="Supabase image URL")
    parser.add_argument("--source-lang", required=True, choices=["th", "en"])
    parser.add_argument("--target-lang", required=True, choices=["th", "en"])
    parser.add_argument("--job-id", default=None, help="Job ID (default: auto UUID)")
    args = parser.parse_args()

    job_id = args.job_id or uuid.uuid4().hex[:8]
    out_dir = get_output_dir(job_id)
    settings = Settings()

    print(f"=== Pipeline Test  job={job_id} ===")
    print(f"    Output: {out_dir}/")
    print()

    # Layer 1
    image = test_download(settings, args.image_url, out_dir)

    # Layer 2
    regions = test_ocr(image, args.source_lang, out_dir)
    if not regions:
        print("\nNo text detected — stopping here.")
        return

    # Layer 3
    grouped = test_grouping(image, regions, out_dir)
    if not grouped:
        print("\nNo text groups after filtering — stopping here.")
        return

    # Layer 4
    inpainted = test_inpaint(image, grouped, out_dir)

    # Layer 5
    translated = test_translate(
        settings, grouped, args.source_lang, args.target_lang, out_dir
    )

    # Layer 6
    test_render(settings, inpainted, grouped, translated, args.target_lang, out_dir)

    print(f"\n=== Done! Check {out_dir}/ ===")


if __name__ == "__main__":
    main()
