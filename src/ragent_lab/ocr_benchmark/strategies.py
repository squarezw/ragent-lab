"""
OCR model implementations for various OCR engines.
"""

from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import time

from .base import OCRModel, OCRResult


class PaddleOCRModel(OCRModel):
    """PaddleOCR model wrapper."""

    def __init__(self, lang: str = 'ch', use_angle_cls: bool = True,
                 use_gpu: bool = False):
        """
        Initialize PaddleOCR model.

        Args:
            lang: Language code ('ch', 'en', etc.)
            use_angle_cls: Whether to use angle classification
            use_gpu: Whether to use GPU
        """
        super().__init__(
            name=f"PaddleOCR ({lang})",
            description=f"PaddleOCR model with {lang} language support",
            supports_batch=False,
            supports_languages=['ch', 'en', 'korean', 'japan', 'french', 'german']
        )
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._ocr = None

    def _get_ocr(self):
        """Get or initialize PaddleOCR instance."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    use_angle_cls=self.use_angle_cls,
                    use_gpu=self.use_gpu,
                    show_log=False
                )
            except ImportError:
                raise ImportError(
                    "PaddleOCR is not installed. "
                    "Please install it with: pip install paddleocr"
                )
        return self._ocr

    def recognize(self, image_path: Union[str, Path], **kwargs) -> OCRResult:
        """
        Recognize text from an image using PaddleOCR.

        Args:
            image_path: Path to the image file
            **kwargs: Additional parameters for PaddleOCR

        Returns:
            OCRResult containing recognized text and metadata
        """
        ocr = self._get_ocr()
        image_path = str(image_path)

        start_time = time.time()
        try:
            result = ocr.ocr(image_path, cls=self.use_angle_cls)
            processing_time = time.time() - start_time

            if not result or not result[0]:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    model_name=self.name,
                    processing_time=processing_time,
                    metadata={
                        'lang': self.lang,
                        'use_angle_cls': self.use_angle_cls,
                        'use_gpu': self.use_gpu,
                        'status': 'no_text_found'
                    }
                )

            # Extract text and confidence from result
            texts = []
            confidences = []
            bounding_boxes = []

            for line in result[0]:
                box, (text, confidence) = line
                texts.append(text)
                confidences.append(confidence)
                bounding_boxes.append({
                    'box': box,
                    'text': text,
                    'confidence': confidence
                })

            full_text = '\n'.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                model_name=self.name,
                processing_time=processing_time,
                metadata={
                    'lang': self.lang,
                    'use_angle_cls': self.use_angle_cls,
                    'use_gpu': self.use_gpu,
                    'num_lines': len(texts)
                },
                bounding_boxes=bounding_boxes
            )

        except Exception as e:
            processing_time = time.time() - start_time
            raise RuntimeError(f"PaddleOCR recognition failed: {e}")


class TesseractOCRModel(OCRModel):
    """Tesseract OCR model wrapper."""

    def __init__(self, lang: str = 'eng+chi_sim'):
        """
        Initialize Tesseract OCR model.

        Args:
            lang: Language code(s) (e.g., 'eng', 'chi_sim', 'eng+chi_sim')
        """
        super().__init__(
            name=f"Tesseract ({lang})",
            description=f"Tesseract OCR with {lang} language support",
            supports_batch=False,
            supports_languages=['eng', 'chi_sim', 'chi_tra', 'jpn', 'kor']
        )
        self.lang = lang

    def recognize(self, image_path: Union[str, Path], **kwargs) -> OCRResult:
        """
        Recognize text from an image using Tesseract.

        Args:
            image_path: Path to the image file
            **kwargs: Additional parameters for pytesseract

        Returns:
            OCRResult containing recognized text and metadata
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError(
                "pytesseract or Pillow is not installed. "
                "Please install with: pip install pytesseract Pillow"
            )

        image_path = str(image_path)
        start_time = time.time()

        try:
            # Open image
            image = Image.open(image_path)

            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.lang, **kwargs)

            # Get detailed data for confidence
            data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)

            # Calculate average confidence (filter out -1 values)
            confidences = [conf for conf in data['conf'] if conf != -1]
            avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

            processing_time = time.time() - start_time

            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence,
                model_name=self.name,
                processing_time=processing_time,
                metadata={
                    'lang': self.lang,
                    'num_words': len([w for w in data['text'] if w.strip()]),
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            raise RuntimeError(f"Tesseract OCR recognition failed: {e}")


class DeepSeekOCRModel(OCRModel):
    """DeepSeek OCR model wrapper (placeholder for future implementation)."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize DeepSeek OCR model.

        Args:
            model_path: Path to the DeepSeek OCR model
        """
        super().__init__(
            name="DeepSeek OCR",
            description="DeepSeek OCR model",
            supports_batch=True,
            supports_languages=['en', 'zh']
        )
        self.model_path = model_path
        self._model = None

    def _get_model(self):
        """Get or load DeepSeek OCR model."""
        if self._model is None:
            # TODO: Implement DeepSeek OCR model loading
            # This is a placeholder for future implementation
            raise NotImplementedError(
                "DeepSeek OCR is not yet implemented. "
                "Please check the DeepSeek documentation for installation instructions."
            )
        return self._model

    def recognize(self, image_path: Union[str, Path], **kwargs) -> OCRResult:
        """
        Recognize text from an image using DeepSeek OCR.

        Args:
            image_path: Path to the image file
            **kwargs: Additional parameters

        Returns:
            OCRResult containing recognized text and metadata
        """
        # TODO: Implement DeepSeek OCR recognition
        raise NotImplementedError(
            "DeepSeek OCR recognition is not yet implemented. "
            "This is a placeholder for future development."
        )


# Factory functions for creating OCR models
def paddleocr_model(lang: str = 'ch', use_angle_cls: bool = True,
                   use_gpu: bool = False) -> PaddleOCRModel:
    """Create PaddleOCR model."""
    return PaddleOCRModel(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu)


def tesseract_ocr_model(lang: str = 'eng+chi_sim') -> TesseractOCRModel:
    """Create Tesseract OCR model."""
    return TesseractOCRModel(lang=lang)


def deepseek_ocr_model(model_path: Optional[str] = None) -> DeepSeekOCRModel:
    """Create DeepSeek OCR model."""
    return DeepSeekOCRModel(model_path=model_path)
