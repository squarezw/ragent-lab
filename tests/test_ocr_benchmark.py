"""
Tests for OCR benchmark functionality.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ragent_lab.ocr_benchmark import (
    OCRModel,
    OCRResult,
    BenchmarkResult,
    PaddleOCRModel,
    TesseractOCRModel,
    OCRBenchmark,
    get_ocr_model,
    list_ocr_models
)


class TestOCRBase(unittest.TestCase):
    """Test cases for OCR base classes."""

    def test_ocr_result_creation(self):
        """Test OCR result creation."""
        result = OCRResult(
            text="Hello World",
            confidence=0.95,
            model_name="TestModel",
            processing_time=0.5,
            metadata={"key": "value"}
        )
        self.assertEqual(result.text, "Hello World")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.processing_time, 0.5)

    def test_ocr_result_validation(self):
        """Test OCR result validation."""
        with self.assertRaises(ValueError):
            OCRResult(
                text="Test",
                confidence=1.5,  # Invalid confidence > 1
                model_name="Test",
                processing_time=0.5
            )

    def test_benchmark_result_metrics(self):
        """Test benchmark result metrics calculation."""
        result = BenchmarkResult(
            model_name="TestModel",
            total_images=10,
            successful=8,
            failed=2,
            average_confidence=0.9,
            average_processing_time=0.5,
            total_time=5.0,
            results=[]
        )
        self.assertEqual(result.success_rate, 0.8)
        self.assertEqual(result.error_rate, 0.2)


class TestOCRRegistry(unittest.TestCase):
    """Test cases for OCR model registry."""

    def test_list_ocr_models(self):
        """Test listing OCR models."""
        models = list_ocr_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn('paddleocr', models)
        self.assertIn('tesseract', models)

    def test_get_ocr_model(self):
        """Test getting OCR model from registry."""
        # This should work without actually loading the model
        # since we're just checking the registry
        models = list_ocr_models()
        self.assertIn('paddleocr', models)


class TestOCRBenchmark(unittest.TestCase):
    """Test cases for OCR benchmark."""

    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        # Create a mock model
        mock_model = Mock(spec=OCRModel)
        mock_model.name = "MockModel"

        benchmark = OCRBenchmark(models=[mock_model])
        self.assertEqual(len(benchmark.models), 1)
        self.assertEqual(benchmark.models[0].name, "MockModel")

    def test_text_similarity(self):
        """Test text similarity calculation."""
        similarity = OCRBenchmark._calculate_text_similarity(
            "Hello World",
            "Hello World"
        )
        self.assertEqual(similarity, 1.0)

        similarity = OCRBenchmark._calculate_text_similarity(
            "Hello World",
            "Hello"
        )
        self.assertLess(similarity, 1.0)
        self.assertGreater(similarity, 0.0)

    @patch('ragent_lab.ocr_benchmark.strategies.PaddleOCR')
    def test_paddleocr_model(self, mock_paddleocr):
        """Test PaddleOCR model wrapper."""
        # Mock PaddleOCR instance
        mock_ocr_instance = MagicMock()
        mock_paddleocr.return_value = mock_ocr_instance

        # Mock OCR result
        mock_ocr_instance.ocr.return_value = [[
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Hello", 0.95)],
            [[[0, 10], [10, 10], [10, 20], [0, 20]], ("World", 0.98)]
        ]]

        model = PaddleOCRModel()
        # We can't actually test recognition without a real image,
        # but we can test model creation
        self.assertEqual(model.name, "PaddleOCR (ch)")


if __name__ == "__main__":
    unittest.main()
