"""Pytest configuration and fixtures."""

import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set test environment variables
    os.environ["VLM_PROVIDER"] = "ollama"
    os.environ["VLM_BASE_URL"] = "http://localhost:11434"
    os.environ["VLM_MODEL"] = "llava"
    os.environ["VLM_TIMEOUT"] = "10"
    os.environ["VLM_MAX_RETRIES"] = "2"
    os.environ["MAX_FILE_SIZE_MB"] = "10"
    os.environ["BATCH_SIZE"] = "3"
    os.environ["BATCH_TIMEOUT"] = "30"


@pytest.fixture
def test_image_path():
    """Create a test image file."""
    test_file = Path("test_image.png")
    # Create a simple test image (1x1 pixel PNG) with valid checksum
    test_file.write_bytes(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    yield test_file
    # Cleanup
    test_file.unlink(missing_ok=True)


@pytest.fixture
def test_text_image_path():
    """Create a test image with text (placeholder)."""
    # For now, return the same test image
    # In real testing, this would be an image with actual text
    test_file = Path("test_text_image.png")
    # Create a simple test image (1x1 pixel PNG) with valid checksum
    test_file.write_bytes(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    yield test_file
    # Cleanup
    test_file.unlink(missing_ok=True) 