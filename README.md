# VLM Extract

A simple Python library to extract text from documents and images using Vision Language Models, with a unified async API that works across multiple VLM providers.

## Features

- **Async-first API**: Simple `await extract_text(file_path, provider="ollama")` function
- **Multiple VLM Providers**: Ollama (local), OpenAI (cloud), LocalAI, and extensible architecture
- **File Format Support**: Images (PNG, JPEG, GIF, BMP, WebP, TIFF, HEIC) and documents (PDF, DOCX, PPTX, XLSX, EPUB, HTML)
- **Environment-based Configuration**: No hardcoded values, everything configurable via `.env`
- **Batch Processing**: Handle multiple files concurrently
- **Error Resilience**: Retry logic and graceful failure handling

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

3. **Extract text**:
   ```python
   import asyncio
   from vlm_extract import extract_text, Provider

   async def main():
       text = await extract_text("path/to/document.png", provider=Provider.OLLAMA)
       print(text)

   asyncio.run(main())
   ```

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

- **Ollama**: Local models (llava, bakllava, etc.)
- **OpenAI**: Cloud models (gpt-4-vision-preview)
- **LocalAI**: Local models with LocalAI server
- **Extensible**: Easy to add more providers

## Development

```bash
# Setup development environment
uv venv .venv
source .venv/bin/activate
uv sync

# Run tests
pytest

# Install in development mode
uv pip install -e .
```

## License

MIT License 