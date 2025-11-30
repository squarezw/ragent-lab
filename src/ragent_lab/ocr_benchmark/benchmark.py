"""
OCR Benchmark testing framework.
"""

from typing import List, Dict, Union, Optional
from pathlib import Path
import time
from dataclasses import dataclass

from .base import OCRModel, OCRResult, BenchmarkResult
from .registry import get_ocr_model, list_ocr_models


class OCRBenchmark:
    """OCR Benchmark testing framework."""

    def __init__(self, models: Optional[List[Union[str, OCRModel]]] = None):
        """
        Initialize OCR benchmark.

        Args:
            models: List of model names or OCRModel instances to benchmark.
                   If None, all registered models will be used.
        """
        self.models = []

        if models is None:
            # Use all registered models
            model_names = list_ocr_models()
            for name in model_names:
                try:
                    self.models.append(get_ocr_model(name))
                except Exception as e:
                    print(f"Warning: Could not load model '{name}': {e}")
        else:
            # Use specified models
            for model in models:
                if isinstance(model, str):
                    self.models.append(get_ocr_model(model))
                elif isinstance(model, OCRModel):
                    self.models.append(model)
                else:
                    raise ValueError(f"Invalid model type: {type(model)}")

    def benchmark_single_model(self, model: OCRModel,
                              image_paths: List[Union[str, Path]],
                              ground_truth: Optional[Dict[str, str]] = None,
                              **kwargs) -> BenchmarkResult:
        """
        Benchmark a single OCR model on a set of images.

        Args:
            model: OCR model to benchmark
            image_paths: List of image file paths
            ground_truth: Optional dictionary mapping image paths to expected text
            **kwargs: Additional parameters for OCR recognition

        Returns:
            BenchmarkResult containing benchmark statistics
        """
        results = []
        errors = []
        total_confidence = 0.0
        total_processing_time = 0.0
        successful = 0
        failed = 0

        start_time = time.time()

        for image_path in image_paths:
            try:
                result = model.recognize(image_path, **kwargs)
                results.append(result)
                total_confidence += result.confidence
                total_processing_time += result.processing_time
                successful += 1

                # Check accuracy against ground truth if provided
                if ground_truth:
                    expected_text = ground_truth.get(str(image_path))
                    if expected_text:
                        accuracy = self._calculate_text_similarity(
                            result.text, expected_text
                        )
                        result.metadata['accuracy'] = accuracy

            except Exception as e:
                failed += 1
                errors.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })

        total_time = time.time() - start_time
        total_images = len(image_paths)

        avg_confidence = total_confidence / successful if successful > 0 else 0.0
        avg_processing_time = total_processing_time / successful if successful > 0 else 0.0

        return BenchmarkResult(
            model_name=model.name,
            total_images=total_images,
            successful=successful,
            failed=failed,
            average_confidence=avg_confidence,
            average_processing_time=avg_processing_time,
            total_time=total_time,
            results=results,
            errors=errors
        )

    def benchmark_all(self, image_paths: List[Union[str, Path]],
                     ground_truth: Optional[Dict[str, str]] = None,
                     **kwargs) -> Dict[str, BenchmarkResult]:
        """
        Benchmark all models on a set of images.

        Args:
            image_paths: List of image file paths
            ground_truth: Optional dictionary mapping image paths to expected text
            **kwargs: Additional parameters for OCR recognition

        Returns:
            Dictionary mapping model names to BenchmarkResult
        """
        results = {}

        for model in self.models:
            print(f"Benchmarking {model.name}...")
            try:
                result = self.benchmark_single_model(
                    model, image_paths, ground_truth, **kwargs
                )
                results[model.name] = result
            except Exception as e:
                print(f"Error benchmarking {model.name}: {e}")

        return results

    def compare_models(self, benchmark_results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """
        Compare benchmark results across models.

        Args:
            benchmark_results: Dictionary of benchmark results from benchmark_all()

        Returns:
            Dictionary containing comparison statistics
        """
        comparison = {
            'models': [],
            'metrics': {}
        }

        for model_name, result in benchmark_results.items():
            comparison['models'].append({
                'name': model_name,
                'success_rate': result.success_rate,
                'average_confidence': result.average_confidence,
                'average_processing_time': result.average_processing_time,
                'total_time': result.total_time
            })

        # Find best performing models
        if comparison['models']:
            comparison['metrics']['fastest'] = min(
                comparison['models'],
                key=lambda x: x['average_processing_time']
            )
            comparison['metrics']['most_confident'] = max(
                comparison['models'],
                key=lambda x: x['average_confidence']
            )
            comparison['metrics']['highest_success_rate'] = max(
                comparison['models'],
                key=lambda x: x['success_rate']
            )

        return comparison

    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0 and 1
        """
        # Simple character-level similarity
        # For more advanced comparison, consider using libraries like difflib or nltk
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()

        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # Calculate Levenshtein distance ratio
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()

    def print_results(self, benchmark_results: Dict[str, BenchmarkResult]):
        """
        Print benchmark results in a formatted way.

        Args:
            benchmark_results: Dictionary of benchmark results
        """
        print("\n" + "="*80)
        print("OCR BENCHMARK RESULTS")
        print("="*80)

        for model_name, result in benchmark_results.items():
            print(f"\n{model_name}:")
            print(f"  Total Images: {result.total_images}")
            print(f"  Successful: {result.successful}")
            print(f"  Failed: {result.failed}")
            print(f"  Success Rate: {result.success_rate:.2%}")
            print(f"  Average Confidence: {result.average_confidence:.4f}")
            print(f"  Average Processing Time: {result.average_processing_time:.4f}s")
            print(f"  Total Time: {result.total_time:.4f}s")

            if result.errors:
                print(f"  Errors:")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"    - {error['image_path']}: {error['error']}")

        # Print comparison
        print("\n" + "-"*80)
        comparison = self.compare_models(benchmark_results)
        if 'metrics' in comparison and comparison['metrics']:
            print("\nPerformance Summary:")
            if 'fastest' in comparison['metrics']:
                fastest = comparison['metrics']['fastest']
                print(f"  Fastest: {fastest['name']} ({fastest['average_processing_time']:.4f}s)")
            if 'most_confident' in comparison['metrics']:
                confident = comparison['metrics']['most_confident']
                print(f"  Most Confident: {confident['name']} ({confident['average_confidence']:.4f})")
            if 'highest_success_rate' in comparison['metrics']:
                success = comparison['metrics']['highest_success_rate']
                print(f"  Highest Success Rate: {success['name']} ({success['success_rate']:.2%})")

        print("="*80 + "\n")
