"""Local PaddleOCR image validation and text extraction."""

from __future__ import annotations

import io
import warnings
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}


class OcrError(ValueError):
    """Raised when an uploaded image cannot be validated or read."""


class PaddleOcrEngine(Protocol):
    def predict(self, image: np.ndarray) -> object: ...


def decode_image(image_bytes: bytes, filename: str) -> np.ndarray:
    """Validate upload size, extension, and image content, then return RGB pixels."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise OcrError("The uploaded image is empty or unreadable.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise OcrError("The image exceeds the 10 MB upload limit.")
    if not isinstance(filename, str):
        raise OcrError("The uploaded image must have a PNG or JPEG filename.")

    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise OcrError("Upload a PNG, JPG, or JPEG image.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as probe:
                if probe.format not in ALLOWED_IMAGE_FORMATS:
                    raise OcrError("The file content is not a PNG or JPEG image.")
                if probe.width * probe.height > MAX_IMAGE_PIXELS:
                    raise OcrError("The image exceeds the 20-megapixel limit.")
                probe.verify()

            with Image.open(io.BytesIO(image_bytes)) as source:
                return np.asarray(source.convert("RGB")).copy()
    except OcrError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise OcrError("The uploaded file is not a valid, readable PNG or JPEG image.") from exc


def create_ocr_engine() -> PaddleOcrEngine:
    """Create the CPU-only PaddleOCR v3 engine on first use."""
    try:
        from paddleocr import PaddleOCR
    except (ImportError, OSError) as exc:
        raise OcrError(
            "Image analysis is unavailable because PaddleOCR dependencies are "
            "not installed on the server."
        ) from exc

    try:
        return PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
    except Exception as exc:
        raise OcrError(
            "PaddleOCR could not initialize. Check the server's dependencies, "
            "network access, and available memory."
        ) from exc


def _result_mapping(result: object) -> Mapping[str, object]:
    if isinstance(result, Mapping):
        return result

    result_json = getattr(result, "json", None)
    if isinstance(result_json, Mapping):
        return result_json.get("res", result_json)
    return {}


def _box_coordinates(box: object) -> tuple[float, float, float]:
    points = np.asarray(box, dtype=float).reshape(-1, 2)
    top = float(points[:, 1].min())
    bottom = float(points[:, 1].max())
    left = float(points[:, 0].min())
    return (top + bottom) / 2, top, left


def _ordered_texts(results: Sequence[object]) -> list[str]:
    lines: list[tuple[float, float, float, str]] = []
    for result in results:
        data = _result_mapping(result)
        texts = data.get("rec_texts", [])
        boxes = data.get("rec_boxes", data.get("rec_polys", []))
        if not isinstance(boxes, (list, tuple, np.ndarray)):
            boxes = ()
        if not isinstance(texts, (list, tuple, np.ndarray)):
            continue
        for index, text in enumerate(texts):
            value = str(text).strip()
            if not value:
                continue
            if index < len(boxes):
                try:
                    center_y, top, left = _box_coordinates(boxes[index])
                except (TypeError, ValueError, IndexError):
                    center_y, top, left = float(index), float(index), 0.0
            else:
                center_y, top, left = float(index), float(index), 0.0
            lines.append((center_y, top, left, value))
    return [line[3] for line in sorted(lines)]


def extract_text_from_image(
    image_bytes: bytes,
    filename: str,
    engine: PaddleOcrEngine | None = None,
) -> str:
    """Run PaddleOCR locally and return recognized text in reading order."""
    image = decode_image(image_bytes, filename)
    if engine is None:
        engine = create_ocr_engine()

    try:
        results = engine.predict(image)
    except Exception as exc:
        raise OcrError("PaddleOCR could not read this image. Try another image.") from exc

    if results is None:
        return ""
    if not isinstance(results, (list, tuple)):
        results = [results]
    return "\n".join(_ordered_texts(results))
