"""
Example usage of OCR benchmark feature.

This script demonstrates how to use the OCR benchmark module to test
different OCR engines on a set of images.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ragent_lab.ocr_benchmark import (
    OCRBenchmark,
    PaddleOCRModel,
    TesseractOCRModel,
    get_ocr_model,
    list_ocr_models
)


def example_basic_usage():
    """Example: Basic OCR benchmark usage."""
    print("="*80)
    print("Example 1: Basic OCR Benchmark")
    print("="*80)

    # List available OCR models
    print("\nAvailable OCR models:")
    for model_name in list_ocr_models():
        print(f"  - {model_name}")

    # Create benchmark with specific models
    # Note: These models require installation of their dependencies
    try:
        benchmark = OCRBenchmark(models=['paddleocr', 'tesseract'])
        print(f"\nCreated benchmark with {len(benchmark.models)} models")
    except Exception as e:
        print(f"\nNote: Some models may not be available: {e}")
        print("Install required packages:")
        print("  - PaddleOCR: pip install paddleocr paddlepaddle")
        print("  - Tesseract: pip install pytesseract Pillow")


def example_single_model():
    """Example: Test a single OCR model."""
    print("\n" + "="*80)
    print("Example 2: Testing a Single OCR Model")
    print("="*80)

    try:
        # Create a single model instance
        model = PaddleOCRModel(lang='ch', use_gpu=False)
        print(f"\nCreated model: {model.name}")
        print(f"Supports languages: {model.supports_languages}")

        # Example: Recognize text from an image
        # image_path = "path/to/your/image.jpg"
        # result = model.recognize(image_path)
        # print(f"Recognized text: {result.text}")
        # print(f"Confidence: {result.confidence:.4f}")
        # print(f"Processing time: {result.processing_time:.4f}s")

        print("\nTo use this model, provide an image path:")
        print("  result = model.recognize('path/to/image.jpg')")

    except ImportError as e:
        print(f"\nModel dependencies not installed: {e}")


def example_benchmark_comparison():
    """Example: Compare multiple OCR models."""
    print("\n" + "="*80)
    print("Example 3: Comparing Multiple OCR Models")
    print("="*80)

    # Sample image paths (replace with your actual images)
    sample_images = [
        # "path/to/image1.jpg",
        # "path/to/image2.png",
        # "path/to/image3.jpg",
    ]

    if not sample_images:
        print("\nTo run a benchmark comparison:")
        print("1. Prepare a set of test images")
        print("2. Add image paths to the sample_images list")
        print("3. Run the benchmark:")
        print("\n   benchmark = OCRBenchmark(models=['paddleocr', 'tesseract'])")
        print("   results = benchmark.benchmark_all(sample_images)")
        print("   benchmark.print_results(results)")
        return

    try:
        # Create benchmark with multiple models
        benchmark = OCRBenchmark(models=['paddleocr', 'tesseract'])

        # Run benchmark on all images
        results = benchmark.benchmark_all(sample_images)

        # Print formatted results
        benchmark.print_results(results)

        # Compare models
        comparison = benchmark.compare_models(results)
        print("\nModel Comparison:")
        for model_info in comparison['models']:
            print(f"  {model_info['name']}:")
            print(f"    Success Rate: {model_info['success_rate']:.2%}")
            print(f"    Avg Confidence: {model_info['average_confidence']:.4f}")
            print(f"    Avg Time: {model_info['average_processing_time']:.4f}s")

    except Exception as e:
        print(f"\nError running benchmark: {e}")


def example_with_ground_truth():
    """Example: Benchmark with ground truth for accuracy measurement."""
    print("\n" + "="*80)
    print("Example 4: Benchmark with Ground Truth")
    print("="*80)

    # Sample images with expected text
    ground_truth = {
        # "path/to/image1.jpg": "Expected text for image 1",
        # "path/to/image2.jpg": "Expected text for image 2",
    }

    if not ground_truth:
        print("\nTo measure OCR accuracy against ground truth:")
        print("1. Prepare test images with known text")
        print("2. Create a ground_truth dictionary:")
        print("   ground_truth = {")
        print("       'image1.jpg': 'Expected text',")
        print("       'image2.jpg': 'Another text',")
        print("   }")
        print("3. Run benchmark with ground truth:")
        print("   results = benchmark.benchmark_all(image_paths, ground_truth=ground_truth)")
        return

    try:
        benchmark = OCRBenchmark(models=['paddleocr'])
        image_paths = list(ground_truth.keys())

        # Run benchmark with ground truth
        results = benchmark.benchmark_all(image_paths, ground_truth=ground_truth)

        # Check accuracy in results
        for model_name, result in results.items():
            print(f"\n{model_name} Results:")
            for ocr_result in result.results:
                accuracy = ocr_result.metadata.get('accuracy', 0.0)
                print(f"  Image: {ocr_result.metadata.get('image_path', 'unknown')}")
                print(f"  Accuracy: {accuracy:.2%}")

    except Exception as e:
        print(f"\nError: {e}")


def example_custom_model():
    """Example: Create and use a custom OCR model."""
    print("\n" + "="*80)
    print("Example 5: Creating a Custom OCR Model")
    print("="*80)

    print("\nTo create a custom OCR model, inherit from OCRModel:")
    print("""
    from ragent_lab.ocr_benchmark import OCRModel, OCRResult
    import time

    class CustomOCRModel(OCRModel):
        def __init__(self):
            super().__init__(
                name="Custom OCR",
                description="My custom OCR implementation",
                supports_batch=False
            )

        def recognize(self, image_path, **kwargs):
            start_time = time.time()
            # Your OCR implementation here
            text = "recognized text"
            confidence = 0.95
            processing_time = time.time() - start_time

            return OCRResult(
                text=text,
                confidence=confidence,
                model_name=self.name,
                processing_time=processing_time,
                metadata={}
            )

    # Use the custom model
    model = CustomOCRModel()
    result = model.recognize("image.jpg")
    """)


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("OCR BENCHMARK EXAMPLES")
    print("="*80)

    example_basic_usage()
    example_single_model()
    example_benchmark_comparison()
    example_with_ground_truth()
    example_custom_model()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nFor more information, see the documentation:")
    print("  - src/ragent_lab/ocr_benchmark/")
    print("  - tests/test_ocr_benchmark.py")
    print("\nRequired dependencies:")
    print("  - PaddleOCR: pip install paddleocr paddlepaddle")
    print("  - Tesseract: pip install pytesseract Pillow")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
