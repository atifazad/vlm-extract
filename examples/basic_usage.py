#!/usr/bin/env python3
"""Basic usage example for VLM Extract library."""

import asyncio
from pathlib import Path
from vlm_extract import extract_text, extract_text_batch, config, Provider


async def basic_extraction():
    """Basic text extraction from a single file."""
    print("=== Basic Text Extraction ===")
    
    # Example file path (replace with your actual image file)
    file_path = Path("example.png")
    
    if file_path.exists():
        try:
            print(f"Extracting text from: {file_path}")
            text = await extract_text(file_path, provider=Provider.OLLAMA)
            print(f"Extracted text:\n{text}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"File not found: {file_path}")
        print("Create an image file with text to test extraction.")


async def batch_extraction():
    """Batch text extraction from multiple files."""
    print("\n=== Batch Text Extraction ===")
    
    # Example file paths (replace with your actual files)
    file_paths = [
        Path("image1.png"),
        Path("image2.jpg"),
        Path("document.pdf")
    ]
    
    # Filter existing files
    existing_files = [f for f in file_paths if f.exists()]
    
    if existing_files:
        try:
            print(f"Extracting text from {len(existing_files)} files...")
            results = await extract_text_batch(existing_files, provider=Provider.OLLAMA)
            
            for file_path, result in zip(existing_files, results):
                if isinstance(result, Exception):
                    print(f"{file_path}: Error - {result}")
                else:
                    print(f"{file_path}: {result[:100]}...")
        except Exception as e:
            print(f"Batch extraction error: {e}")
    else:
        print("No example files found. Create some image files to test batch extraction.")


async def configuration_demo():
    """Demonstrate configuration options."""
    print("\n=== Configuration Demo ===")
    
    print(f"Default provider: {config.vlm.provider.value}")
    print(f"VLM base URL: {config.vlm.base_url}")
    print(f"VLM model: {config.vlm.model}")
    print(f"VLM timeout: {config.vlm.timeout}s")
    print(f"VLM max retries: {config.vlm.max_retries}")
    print(f"Max file size: {config.file.max_file_size_mb}MB")
    print(f"Supported image formats: {', '.join(config.file.supported_image_formats)}")
    print(f"Supported document formats: {', '.join(config.file.supported_document_formats)}")


async def main():
    """Run all examples."""
    print("VLM Extract - Basic Usage Examples")
    print("=" * 50)
    
    # Show configuration
    await configuration_demo()
    
    # Basic extraction
    await basic_extraction()
    
    # Batch extraction
    await batch_extraction()
    
    print("\n" + "=" * 50)
    print("To test with real files:")
    print("1. Create image files with text in them")
    print("2. Make sure Ollama is running with a vision model (e.g., 'llava')")
    print("3. Update the file paths in this script")
    print("4. Run: python examples/basic_usage.py")


if __name__ == "__main__":
    asyncio.run(main()) 