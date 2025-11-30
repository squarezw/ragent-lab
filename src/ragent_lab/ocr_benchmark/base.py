"""
Base classes and utilities for OCR benchmark testing.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import time


@dataclass
class OCRResult:
    """Result of OCR operation on a single image."""
    text: str
    confidence: float
    model_name: str
    processing_time: float  # in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    bounding_boxes: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        """Validate OCR result after initialization."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")


@dataclass
class BenchmarkResult:
    """Result of OCR benchmark test."""
    model_name: str
    total_images: int
    successful: int
    failed: int
    average_confidence: float
    average_processing_time: float
    total_time: float
    results: List[OCRResult]
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_images == 0:
            return 0.0
        return self.successful / self.total_images

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return 1.0 - self.success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            'model_name': self.model_name,
            'total_images': self.total_images,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'average_confidence': self.average_confidence,
            'average_processing_time': self.average_processing_time,
            'total_time': self.total_time,
            'errors': self.errors
        }


class OCRModel(ABC):
    """Abstract base class for OCR models."""

    def __init__(self, name: str, description: str,
                 supports_batch: bool = False,
                 supports_languages: Optional[List[str]] = None):
        """
        Initialize OCR model.

        Args:
            name: Model name
            description: Model description
            supports_batch: Whether model supports batch processing
            supports_languages: List of supported language codes
        """
        self.name = name
        self.description = description
        self.supports_batch = supports_batch
        self.supports_languages = supports_languages or ['en', 'zh']

    @abstractmethod
    def recognize(self, image_path: Union[str, Path], **kwargs) -> OCRResult:
        """
        Recognize text from an image.

        Args:
            image_path: Path to the image file
            **kwargs: Additional model-specific parameters

        Returns:
            OCRResult containing recognized text and metadata
        """
        pass

    def recognize_batch(self, image_paths: List[Union[str, Path]],
                       **kwargs) -> List[OCRResult]:
        """
        Recognize text from multiple images.

        Args:
            image_paths: List of paths to image files
            **kwargs: Additional model-specific parameters

        Returns:
            List of OCRResult for each image
        """
        if self.supports_batch:
            # Override in subclass for optimized batch processing
            return self._recognize_batch_optimized(image_paths, **kwargs)
        else:
            # Fallback to sequential processing
            return [self.recognize(img, **kwargs) for img in image_paths]

    def _recognize_batch_optimized(self, image_paths: List[Union[str, Path]],
                                   **kwargs) -> List[OCRResult]:
        """
        Optimized batch processing (to be implemented in subclass).

        Args:
            image_paths: List of paths to image files
            **kwargs: Additional model-specific parameters

        Returns:
            List of OCRResult for each image
        """
        raise NotImplementedError("Batch processing not implemented for this model")

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return f"OCRModel(name='{self.name}', supports_batch={self.supports_batch})"
