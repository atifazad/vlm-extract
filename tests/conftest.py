"""Pytest configuration and fixtures."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables from .env file."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set test-specific overrides only for non-critical values
    # Preserve actual API keys and URLs from .env
    os.environ["VLM_TIMEOUT"] = "30"  # Use same timeout as .env for tests
    os.environ["VLM_MAX_RETRIES"] = "2"  # Fewer retries for tests
    os.environ["MAX_FILE_SIZE_MB"] = "10"  # Smaller file size limit for tests
    os.environ["BATCH_SIZE"] = "3"  # Smaller batch size for tests
    os.environ["BATCH_TIMEOUT"] = "30"  # Shorter batch timeout for tests
    
    # Only override provider if not set in .env
    if not os.getenv("VLM_PROVIDER"):
        os.environ["VLM_PROVIDER"] = "ollama"
    
    # Only override Ollama settings if not set in .env
    if not os.getenv("OLLAMA_BASE_URL"):
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    if not os.getenv("OLLAMA_MODEL"):
        os.environ["OLLAMA_MODEL"] = "llava"
    
    # Only override OpenAI settings if not set in .env
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
    if not os.getenv("OPENAI_MODEL"):
        os.environ["OPENAI_MODEL"] = "gpt-4o"
    
    # Never override API keys - use the real ones from .env
    # This allows tests to work with real credentials when available


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