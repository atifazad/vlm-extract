"""Tests for core functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch
from vlm_extract import extract_text, extract_text_batch


class TestCore:
    """Test core functionality."""

    @pytest.mark.asyncio
    async def test_extract_text_file_not_found(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            await extract_text("nonexistent.png")

    @pytest.mark.asyncio
    async def test_extract_text_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        # Create a temporary file for testing
        test_file = Path("test.png")
        test_file.write_bytes(b"fake image data")
        
        try:
            with pytest.raises(ValueError, match="Unsupported provider"):
                await extract_text(test_file, provider="unsupported")
        finally:
            test_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_extract_text_batch_empty_list(self):
        """Test batch extraction with empty list."""
        results = await extract_text_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_extract_text_batch_with_exceptions(self):
        """Test batch extraction handles exceptions gracefully."""
        # Create a temporary file for testing
        test_file = Path("test.png")
        test_file.write_bytes(b"fake image data")
        
        try:
            # Mock the provider to raise an exception
            with patch('vlm_extract.providers.ollama.OllamaProvider.extract_text') as mock_extract:
                mock_extract.side_effect = Exception("Test error")
                
                results = await extract_text_batch([test_file])
                assert len(results) == 1
                assert isinstance(results[0], Exception)
        finally:
            test_file.unlink(missing_ok=True) 