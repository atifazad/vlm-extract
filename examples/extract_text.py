#!/usr/bin/env python3
"""Terminal command for extracting text from files using VLM Extract."""

import asyncio
import sys
from pathlib import Path
from vlm_extract import extract_text, Provider, config


async def extract_text_from_file(file_path: str):
    """Extract text from a single file."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    if not path.is_file():
        print(f"❌ Error: Path is not a file: {file_path}")
        sys.exit(1)
    
    print("VLM Extract - Text Extraction")
    print("=" * 40)
    print(f"File: {file_path}")
    print(f"Provider: {config.vlm.provider.value}")
    print(f"Model: {config.vlm.model}")
    print(f"Base URL: {config.vlm.base_url}")
    print()
    
    try:
        print("🔄 Extracting text...")
        text = await extract_text(path, provider=Provider.OLLAMA)
        
        if text.strip():
            print("✅ Text extracted successfully:")
            print("-" * 40)
            print(text)
            print("-" * 40)
        else:
            print("⚠️  No text extracted (empty result)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure the model is pulled: ollama pull llava")
        print("3. Check if the image contains readable text")
        print("4. Verify the file format is supported")
        sys.exit(1)


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python extract_text.py <file_path>")
        print("Example: python extract_text.py image.png")
        print("Example: python extract_text.py document.jpg")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        asyncio.run(extract_text_from_file(file_path))
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 