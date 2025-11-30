"""
Registry for OCR models.
"""

from typing import Dict, List, Optional, Callable
from .base import OCRModel
from .strategies import (
    PaddleOCRModel,
    TesseractOCRModel,
    DeepSeekOCRModel,
    paddleocr_model,
    tesseract_ocr_model,
    deepseek_ocr_model
)


# Global registry for OCR models
_OCR_REGISTRY: Dict[str, Dict] = {}


def register_ocr_model(name: str, factory: Callable, description: str = "",
                      requires: Optional[List[str]] = None):
    """
    Register an OCR model in the registry.

    Args:
        name: Model name/identifier
        factory: Factory function that creates the model
        description: Model description
        requires: List of required packages
    """
    _OCR_REGISTRY[name] = {
        'factory': factory,
        'description': description,
        'requires': requires or []
    }


def get_ocr_model(name: str, **kwargs) -> OCRModel:
    """
    Get an OCR model by name.

    Args:
        name: Model name/identifier
        **kwargs: Additional parameters for the model factory

    Returns:
        OCRModel instance

    Raises:
        ValueError: If model not found in registry
    """
    if name not in _OCR_REGISTRY:
        available = ', '.join(_OCR_REGISTRY.keys())
        raise ValueError(f"OCR model '{name}' not found. Available models: {available}")

    model_info = _OCR_REGISTRY[name]
    return model_info['factory'](**kwargs)


def get_all_ocr_models() -> Dict[str, Dict]:
    """
    Get all registered OCR models.

    Returns:
        Dictionary of all registered models with their metadata
    """
    return _OCR_REGISTRY.copy()


def list_ocr_models() -> List[str]:
    """
    List all registered OCR model names.

    Returns:
        List of model names
    """
    return list(_OCR_REGISTRY.keys())


# Register default OCR models
register_ocr_model(
    name='paddleocr',
    factory=paddleocr_model,
    description='PaddleOCR - Fast and accurate OCR with Chinese support',
    requires=['paddleocr', 'paddlepaddle']
)

register_ocr_model(
    name='paddleocr_en',
    factory=lambda **kwargs: paddleocr_model(lang='en', **kwargs),
    description='PaddleOCR - English language model',
    requires=['paddleocr', 'paddlepaddle']
)

register_ocr_model(
    name='tesseract',
    factory=tesseract_ocr_model,
    description='Tesseract OCR - Open source OCR engine',
    requires=['pytesseract', 'Pillow']
)

register_ocr_model(
    name='tesseract_en',
    factory=lambda **kwargs: tesseract_ocr_model(lang='eng', **kwargs),
    description='Tesseract OCR - English only',
    requires=['pytesseract', 'Pillow']
)

register_ocr_model(
    name='deepseek_ocr',
    factory=deepseek_ocr_model,
    description='DeepSeek OCR - Advanced OCR model (placeholder)',
    requires=['deepseek']  # Placeholder
)
