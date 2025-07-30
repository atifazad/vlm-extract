# VLM Extract

A simple Python library to extract text from documents and images using Vision Language Models, with a unified async API that works across multiple VLM providers.

## Features

- **Async-first API**: Simple `await extract_text(file_path, provider="ollama")` function
- **Multiple VLM Providers**: Ollama (local), OpenAI (cloud), LocalAI, and extensible architecture
- **File Format Support**: Images (PNG, JPEG, JPG, GIF, BMP, WebP, TIFF, HEIC) and documents (PDF, DOCX, PPTX, XLSX, EPUB, HTML)
- **Environment-based Configuration**: Provider settings configurable via `.env`
- **Batch Processing**: Handle multiple files concurrently
- **Error Resilience**: Retry logic and graceful failure handling
- **Comprehensive Testing**: 85% test coverage with integration tests
- **Terminal Command**: Simple command-line interface for file processing

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
```

## Supported Providers

- **Ollama**: Local models (llava, bakllava, qwen2.5vl, etc.)
- **OpenAI**: Cloud models (gpt-4-vision-preview) - Coming soon
- **LocalAI**: Local models with LocalAI server - Coming soon
- **Extensible**: Easy to add more providers



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
```


## License

MIT License 