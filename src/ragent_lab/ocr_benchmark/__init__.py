"""
OCR Benchmark module for testing different OCR SDKs.

This module provides a comprehensive benchmarking framework for various OCR
(Optical Character Recognition) engines including PaddleOCR, DeepSeek OCR,
Tesseract, and others.
"""

from .base import OCRModel, OCRResult, BenchmarkResult
from .strategies import (
    PaddleOCRModel,
    TesseractOCRModel,
)
from .registry import (
    register_ocr_model,
    get_ocr_model,
    get_all_ocr_models,
    list_ocr_models
)
from .benchmark import OCRBenchmark

__all__ = [
    # Base classes
    'OCRModel',
    'OCRResult',
    'BenchmarkResult',

    # OCR models
    'PaddleOCRModel',
    'TesseractOCRModel',

    # Registry functions
    'register_ocr_model',
    'get_ocr_model',
    'get_all_ocr_models',
    'list_ocr_models',

    # Benchmark
    'OCRBenchmark',
]
