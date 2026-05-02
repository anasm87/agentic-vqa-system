# Agentic VQA System

This project implements a Production-Grade Visual Question Answering (VQA) Pipeline. The pipeline is designed to take an image (e.g., a dashcam frame) and a text prompt (e.g., "Describe the state of the road, vehicles, and traffic lights"), process them using an open-source Vision-Language Model (VLM), and return a descriptive answer.

## Setup and Installation

1. **Activate your Virtual Environment** (Recommended):
   ```bash
   # On Windows
   .\venv\Scripts\activate
   
   # On Linux/macOS
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   Install all required packages via `pip`.
   ```bash
   pip install -r requirements.txt
   ```
   *Core dependencies include: `torch`, `transformers`, `Pillow`, and `accelerate`.*

## Architecture

The system leverages **PyTorch** for computation and **Hugging Face Transformers** for accessing state-of-the-art Vision-Language Models (VLM).

### PyTorch and Hugging Face VLM Integration
- **Framework**: Built with **PyTorch**, ensuring efficient tensor operations and GPU acceleration where available.
- **Model Library**: Integrated with the **Hugging Face `transformers`** library to utilize the `BlipForQuestionAnswering` architecture.
- **Pre-trained Model**: Defaults to `Salesforce/blip-vqa-base`, which is optimized for visual question answering tasks.
- **Processing**: Uses `BlipProcessor` for multimodal input handling (image + text).

### `VisionAnalyzer` Class
- **Hardware Acceleration**: Automatically detects and runs on a GPU (`cuda`) if available, and gracefully falls back to the `cpu` if not.
- **`analyze_frame(image_path: str, prompt: str) -> str`**:
  - **`image_path`**: The file path to the image you want to analyze.
  - **`prompt`**: The question or command about the image.
  - **Returns**: A string containing the model's generated answer.

## Usage

Here is a simple example of how to use the `VisionAnalyzer` in your code:

```python
from src.vqa_engine import VisionAnalyzer

# 1. Initialize the analyzer
analyzer = VisionAnalyzer()

# 2. Define your image and prompt
image_path = "path/to/your/image.jpg"
prompt = "Describe the state of the road, vehicles, and traffic lights."

# 3. Run inference
result = analyzer.analyze_frame(image_path, prompt)

print("VQA Output:", result)
```

## Testing

A test script is included to verify the VQA pipeline is working correctly. It creates a placeholder dashcam image (drawing a road, sky, traffic lights, and a car) and runs it through the model.

To execute the test:
```bash
python tests/test_inference.py
```

**Expected behavior**: The script will download/generate a sample image, load the `blip-vqa-base` model, and print the model's text output to the console based on the sample image and the prompt.