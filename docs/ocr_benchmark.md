# OCR Benchmark Feature

OCR (Optical Character Recognition) Benchmark is a comprehensive testing framework for evaluating and comparing different OCR engines.

## Overview

The OCR Benchmark module provides a unified interface for testing various OCR SDKs including:

- **PaddleOCR**: Fast and accurate OCR with excellent Chinese language support
- **Tesseract OCR**: Open-source OCR engine with broad language support
- **DeepSeek OCR**: Advanced OCR model (placeholder for future implementation)
- **Custom OCR models**: Extensible framework for adding your own OCR implementations

## Features

- **Multi-model support**: Test and compare multiple OCR engines simultaneously
- **Performance metrics**: Track processing time, confidence scores, and success rates
- **Ground truth comparison**: Measure accuracy against known text
- **Batch processing**: Process multiple images efficiently
- **Extensible architecture**: Easy to add new OCR models
- **Detailed reporting**: Comprehensive benchmark results with visualizations

## Installation

### Basic Installation

```bash
# Install the main package
pip install -r requirements.txt
```

### OCR Dependencies

Install OCR-specific dependencies:

```bash
# Install all OCR dependencies
pip install -r requirements-ocr.txt
```

Or install specific OCR engines:

```bash
# PaddleOCR
pip install paddleocr paddlepaddle

# Tesseract OCR
pip install pytesseract Pillow

# Additional image processing
pip install opencv-python
```

### System Requirements

**For Tesseract OCR**, you also need to install the Tesseract engine:

- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Quick Start

### Basic Usage

```python
from ragent_lab.ocr_benchmark import OCRBenchmark, list_ocr_models

# List available models
print(list_ocr_models())

# Create benchmark with specific models
benchmark = OCRBenchmark(models=['paddleocr', 'tesseract'])

# Benchmark on images
image_paths = ['image1.jpg', 'image2.png']
results = benchmark.benchmark_all(image_paths)

# Print results
benchmark.print_results(results)
```

### Single Model Usage

```python
from ragent_lab.ocr_benchmark import PaddleOCRModel

# Create a model instance
model = PaddleOCRModel(lang='ch', use_gpu=False)

# Recognize text from an image
result = model.recognize('document.jpg')

print(f"Text: {result.text}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Processing time: {result.processing_time:.3f}s")
```

### With Ground Truth

```python
from ragent_lab.ocr_benchmark import OCRBenchmark

# Define ground truth
ground_truth = {
    'invoice1.jpg': 'Invoice #12345',
    'receipt.jpg': 'Total: $99.99'
}

benchmark = OCRBenchmark(models=['paddleocr'])
results = benchmark.benchmark_all(
    image_paths=list(ground_truth.keys()),
    ground_truth=ground_truth
)

# Check accuracy
for model_name, result in results.items():
    for ocr_result in result.results:
        accuracy = ocr_result.metadata.get('accuracy', 0)
        print(f"Accuracy: {accuracy:.2%}")
```

## Available Models

### PaddleOCR

```python
from ragent_lab.ocr_benchmark import PaddleOCRModel

# Chinese + English (default)
model = PaddleOCRModel(lang='ch')

# English only
model = PaddleOCRModel(lang='en')

# With GPU support
model = PaddleOCRModel(lang='ch', use_gpu=True)

# Disable angle classification (faster but less accurate)
model = PaddleOCRModel(lang='ch', use_angle_cls=False)
```

**Supported languages**: Chinese, English, Korean, Japanese, French, German

### Tesseract OCR

```python
from ragent_lab.ocr_benchmark import TesseractOCRModel

# English + Simplified Chinese
model = TesseractOCRModel(lang='eng+chi_sim')

# English only
model = TesseractOCRModel(lang='eng')

# Traditional Chinese
model = TesseractOCRModel(lang='chi_tra')
```

**Supported languages**: English, Chinese (Simplified/Traditional), Japanese, Korean, and [many more](https://github.com/tesseract-ocr/tessdata)

### Using Registry

```python
from ragent_lab.ocr_benchmark import get_ocr_model, list_ocr_models

# List all registered models
models = list_ocr_models()
# ['paddleocr', 'paddleocr_en', 'tesseract', 'tesseract_en', 'deepseek_ocr']

# Get a model by name
model = get_ocr_model('paddleocr')
result = model.recognize('image.jpg')
```

## Benchmark Metrics

The benchmark provides comprehensive metrics:

- **Total Images**: Number of images processed
- **Successful**: Number of successfully processed images
- **Failed**: Number of failed images
- **Success Rate**: Percentage of successful recognitions
- **Average Confidence**: Mean confidence score across all results
- **Average Processing Time**: Mean time per image
- **Total Time**: Total benchmark duration

### Example Output

```
================================================================================
OCR BENCHMARK RESULTS
================================================================================

PaddleOCR (ch):
  Total Images: 10
  Successful: 10
  Failed: 0
  Success Rate: 100.00%
  Average Confidence: 0.9234
  Average Processing Time: 0.3421s
  Total Time: 3.4567s

Tesseract (eng+chi_sim):
  Total Images: 10
  Successful: 9
  Failed: 1
  Success Rate: 90.00%
  Average Confidence: 0.8567
  Average Processing Time: 0.5123s
  Total Time: 5.2345s

--------------------------------------------------------------------------------

Performance Summary:
  Fastest: PaddleOCR (ch) (0.3421s)
  Most Confident: PaddleOCR (ch) (0.9234)
  Highest Success Rate: PaddleOCR (ch) (100.00%)
================================================================================
```

## Creating Custom OCR Models

Extend the framework with your own OCR implementation:

```python
from ragent_lab.ocr_benchmark import OCRModel, OCRResult
import time

class CustomOCRModel(OCRModel):
    def __init__(self):
        super().__init__(
            name="Custom OCR",
            description="My custom OCR implementation",
            supports_batch=False,
            supports_languages=['en', 'zh']
        )
        # Initialize your model here

    def recognize(self, image_path, **kwargs):
        start_time = time.time()

        # Your OCR logic here
        text = self._your_ocr_function(image_path)
        confidence = self._calculate_confidence()

        processing_time = time.time() - start_time

        return OCRResult(
            text=text,
            confidence=confidence,
            model_name=self.name,
            processing_time=processing_time,
            metadata={'custom_field': 'value'}
        )

# Register your model
from ragent_lab.ocr_benchmark import register_ocr_model

register_ocr_model(
    name='custom_ocr',
    factory=lambda: CustomOCRModel(),
    description='My custom OCR model',
    requires=['your_dependencies']
)
```

## Advanced Usage

### Batch Processing

```python
model = PaddleOCRModel()

# Process multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model.recognize_batch(image_paths)

for result in results:
    print(f"Text: {result.text[:50]}...")
```

### Comparing Models

```python
benchmark = OCRBenchmark(models=['paddleocr', 'tesseract'])
results = benchmark.benchmark_all(image_paths)

# Get comparison statistics
comparison = benchmark.compare_models(results)

# Access specific metrics
fastest = comparison['metrics']['fastest']
most_confident = comparison['metrics']['most_confident']

print(f"Fastest model: {fastest['name']}")
print(f"Most confident: {most_confident['name']}")
```

### Accessing Bounding Boxes

```python
model = PaddleOCRModel()
result = model.recognize('document.jpg')

# Access bounding boxes (if available)
if result.bounding_boxes:
    for bbox in result.bounding_boxes:
        print(f"Text: {bbox['text']}")
        print(f"Confidence: {bbox['confidence']:.2%}")
        print(f"Box: {bbox['box']}")
```

## Performance Tips

1. **Use GPU acceleration** for PaddleOCR when processing many images:
   ```python
   model = PaddleOCRModel(use_gpu=True)
   ```

2. **Choose appropriate language models** to improve accuracy:
   ```python
   # For English documents only
   model = PaddleOCRModel(lang='en')
   ```

3. **Disable angle classification** if images are already properly oriented:
   ```python
   model = PaddleOCRModel(use_angle_cls=False)
   ```

4. **Batch processing** when supported:
   ```python
   results = model.recognize_batch(image_paths)
   ```

## Troubleshooting

### PaddleOCR Issues

**Import Error**: Install PaddleOCR and PaddlePaddle:
```bash
pip install paddleocr paddlepaddle
```

**GPU Error**: Ensure CUDA is properly installed or use CPU:
```python
model = PaddleOCRModel(use_gpu=False)
```

### Tesseract Issues

**Tesseract not found**: Install the Tesseract engine system-wide:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

**Language data missing**: Install language packs:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr-chi-sim tesseract-ocr-chi-tra
```

## Examples

See the `examples/ocr_benchmark_example.py` file for comprehensive examples:

```bash
python examples/ocr_benchmark_example.py
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_ocr_benchmark.py -v
```

Or using unittest:

```bash
python tests/test_ocr_benchmark.py
```

## API Reference

### Classes

- `OCRModel`: Abstract base class for OCR models
- `OCRResult`: Result of OCR operation
- `BenchmarkResult`: Result of benchmark test
- `PaddleOCRModel`: PaddleOCR implementation
- `TesseractOCRModel`: Tesseract OCR implementation
- `OCRBenchmark`: Benchmark testing framework

### Functions

- `list_ocr_models()`: List all registered models
- `get_ocr_model(name)`: Get a model by name
- `register_ocr_model(name, factory, ...)`: Register a custom model

## Contributing

To add a new OCR engine:

1. Create a new class inheriting from `OCRModel`
2. Implement the `recognize()` method
3. Add factory function in `strategies.py`
4. Register in `registry.py`
5. Add tests in `tests/test_ocr_benchmark.py`
6. Update documentation

## License

MIT License

## Support

For issues and questions:
- GitHub Issues: [Project Issues](https://github.com/squarezw/ragent-lab/issues)
- Documentation: `docs/ocr_benchmark.md`
