# VLM Extract

A simple Python library to extract text from documents and images using Vision Language Models, with a unified async API that works across multiple VLM providers.

## Features

- **Async-first API**: Simple `await extract_text(file_path, provider="ollama")` function
- **Multiple VLM Providers**: Ollama (local), OpenAI (cloud), LocalAI, and extensible architecture
- **File Format Support**: Images (PNG, JPEG, JPG, GIF, BMP, WebP, TIFF, HEIC) and documents (PDF, DOCX, PPTX, XLSX, EPUB, HTML)
- **Smart PDF Processing**: Automatic detection and fast extraction of text-based PDFs using PyMuPDF, with VLM fallback for image-based PDFs
- **Environment-based Configuration**: Provider settings configurable via `.env`
- **Batch Processing**: Handle multiple files concurrently
- **Error Resilience**: Retry logic and graceful failure handling
- **Comprehensive Testing**: 85% test coverage with integration tests
- **Terminal Command**: Simple command-line interface for file processing
- **PDF Support**: Extract text from PDF documents (multi-page support)

## Quick Start

1. **Install the library**:
   ```bash
   pip install vlm-extract
   ```

2. **Configure your provider** (copy `env.example` to `.env`):
   ```bash
   cp env.example .env
   # Edit .env with your preferred provider settings
   ```

3. **Terminal Usage**

Use the built-in terminal command for quick text extraction:

```bash
# Extract text from a single file
python examples/extract_text.py image.png
python examples/extract_text.py document.jpg
python examples/extract_text.py document.pdf
python examples/extract_text.py /path/to/your/file.png
```

The command shows:
- Current configuration (provider, model, URL)
- Progress indication
- Extracted text in clean format
- Error messages with troubleshooting tips

## Configuration

Copy `env.example` to `.env` and configure your preferred VLM provider:

```bash
# VLM Provider Configuration
VLM_PROVIDER=ollama
VLM_BASE_URL=http://localhost:11434
VLM_API_KEY=your_api_key_here
VLM_MODEL=llava
VLM_TIMEOUT=30
VLM_MAX_RETRIES=3

# PDF Processing Configuration
PDF_TEXT_EXTRACTION_ENABLED=true
PDF_MIN_TEXT_RATIO=0.1
PDF_FALLBACK_TO_VLM=true
```

## Smart PDF Processing

VLM Extract automatically optimizes PDF processing:

- **Text-based PDFs**: Uses PyMuPDF for fast, accurate text extraction (~10-100x faster)
- **Image-based PDFs**: Uses VLM for OCR processing
- **Automatic Detection**: Intelligently chooses the best method for each PDF
- **Transparent Fallback**: Seamlessly switches between methods as needed

### PDF Processing Configuration

```bash
# Enable/disable PyMuPDF text extraction
PDF_TEXT_EXTRACTION_ENABLED=true

# Minimum text ratio to consider PDF as text-based (0.0-1.0)
PDF_MIN_TEXT_RATIO=0.1

# Fallback to VLM if PyMuPDF extraction fails
PDF_FALLBACK_TO_VLM=true
```

## Supported Providers

- **Ollama**: Local models (llava, bakllava, qwen2.5vl, etc.)
- **OpenAI**: Cloud models (gpt-4-vision-preview) - Coming soon
- **LocalAI**: Local models with LocalAI server - Coming soon
- **Extensible**: Easy to add more providers

## Examples

### Basic Usage
```python
from vlm_extract import extract_text, Provider

# Extract text from an image
text = await extract_text("image.png", provider=Provider.OLLAMA)

# Extract text from a PDF (automatically optimized)
text = await extract_text("document.pdf", provider=Provider.OLLAMA)
```

### Batch Processing
```python
from vlm_extract import extract_text_batch

# Process multiple files
results = await extract_text_batch(["image1.png", "document.pdf"], provider=Provider.OLLAMA)
```

### Configuration Access
```python
from vlm_extract import config

print(f"Current provider: {config.vlm.provider.value}")
print(f"Model: {config.vlm.model}")
```

## Development

```bash
# Setup development environment
uv venv .venv
source .venv/bin/activate
uv sync

# Run tests
pytest

# Run with coverage
pytest --cov=vlm_extract --cov-report=term-missing

# Install in development mode
uv pip install -e .
```

## License

MIT License 