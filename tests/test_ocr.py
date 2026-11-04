from __future__ import annotations

import builtins
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr import OcrError, decode_image, extract_text_from_image


def png_bytes() -> bytes:
    image = Image.new("RGB", (80, 40), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class PaddleOcrTests(unittest.TestCase):
    def test_extracts_detected_text_in_reading_order(self) -> None:
        engine = Mock()
        engine.predict.return_value = [
            {
                "rec_texts": ["Right", "Second line", "Left"],
                "rec_boxes": [
                    [40, 5, 70, 20],
                    [5, 30, 60, 45],
                    [5, 5, 30, 20],
                ],
            }
        ]

        text = extract_text_from_image(png_bytes(), "posting.png", engine=engine)

        self.assertEqual(text, "Left\nRight\nSecond line")
        engine.predict.assert_called_once()

    def test_empty_detection_returns_empty_text(self) -> None:
        engine = Mock()
        engine.predict.return_value = [{"rec_texts": [], "rec_boxes": []}]

        self.assertEqual(
            extract_text_from_image(png_bytes(), "posting.png", engine=engine),
            "",
        )

    def test_validates_filename_and_image_content(self) -> None:
        with self.assertRaisesRegex(OcrError, "PNG, JPG, or JPEG"):
            decode_image(png_bytes(), "posting.gif")
        with self.assertRaisesRegex(OcrError, "valid, readable"):
            decode_image(b"not an image", "posting.png")
        invalid_gif = io.BytesIO()
        Image.new("RGB", (10, 10)).save(invalid_gif, format="GIF")
        with self.assertRaisesRegex(OcrError, "content is not"):
            decode_image(
                invalid_gif.getvalue(),
                "posting.png",
            )

    def test_rejects_uploads_over_size_limit_before_decoding(self) -> None:
        with self.assertRaisesRegex(OcrError, "10 MB"):
            decode_image(b"x" * (10 * 1024 * 1024 + 1), "posting.png")

    def test_reports_engine_failures_as_friendly_ocr_errors(self) -> None:
        engine = Mock()
        engine.predict.side_effect = RuntimeError("internal engine details")

        with self.assertRaisesRegex(OcrError, "could not read this image"):
            extract_text_from_image(png_bytes(), "posting.png", engine=engine)

    def test_reports_missing_paddleocr_dependency(self) -> None:
        original_import = builtins.__import__

        def import_without_paddleocr(name, *args, **kwargs):
            if name == "paddleocr":
                raise ModuleNotFoundError("missing dependency")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_paddleocr):
            with self.assertRaisesRegex(OcrError, "dependencies are not installed"):
                from ocr import create_ocr_engine

                create_ocr_engine()
