"""Pytest configuration and fixtures."""

import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test environment variables
    os.environ["VLM_PROVIDER"] = "ollama"
    
    # Global configuration
    os.environ["VLM_TIMEOUT"] = "10"
    os.environ["VLM_MAX_RETRIES"] = "2"
    
    # Provider-specific configuration for Ollama
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["OLLAMA_MODEL"] = "llava"
    os.environ["OLLAMA_API_KEY"] = ""
    
    # Provider-specific configuration for OpenAI
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
    os.environ["OPENAI_MODEL"] = "gpt-4o"
    os.environ["OPENAI_API_KEY"] = "test_openai_key"
    
    # File and batch processing configuration
    os.environ["MAX_FILE_SIZE_MB"] = "10"
    os.environ["BATCH_SIZE"] = "3"
    os.environ["BATCH_TIMEOUT"] = "30"


@pytest.fixture
def test_image_path():
    """Get test image path from test_resources folder."""
    test_images_dir = Path("test_resources")
    test_file = test_images_dir / "test_image.png"
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test image not found: {test_file}. Please add test_image.png to the test_images folder.")
    
    yield test_file


@pytest.fixture
def test_text_image_path():
    """Get test image with text from test_images folder."""
    test_images_dir = Path("test_resources")
    test_file = test_images_dir / "test_image.png"
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test image not found: {test_file}. Please add test_image.png to the test_images folder.")
    
    yield test_file 