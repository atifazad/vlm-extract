"""Integration tests for core functionality."""

import pytest
from pathlib import Path
from vlm_extract import extract_text, extract_text_batch, Provider


class TestCoreIntegration:
    """Integration tests for core functionality."""

    @pytest.mark.asyncio
    async def test_extract_text_with_ollama(self, test_image_path):
        """Test extract_text with Ollama provider."""
        try:
            result = await extract_text(test_image_path, provider=Provider.OLLAMA)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error", "invalid format"]):
                pytest.skip(f"Ollama server not accessible or test image invalid: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_extract_text_batch_with_ollama(self, test_image_path):
        """Test batch extraction with Ollama provider."""
        try:
            results = await extract_text_batch([test_image_path], provider=Provider.OLLAMA)
            assert len(results) == 1
            assert isinstance(results[0], str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error", "invalid format"]):
                pytest.skip(f"Ollama server not accessible or test image invalid: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_extract_text_with_default_provider(self, test_image_path):
        """Test extract_text with default provider from environment."""
        try:
            result = await extract_text(test_image_path)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error", "invalid format"]):
                pytest.skip(f"Ollama server not accessible or test image invalid: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_extract_text_unsupported_provider(self, test_image_path):
        """Test error handling for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            await extract_text(test_image_path, provider="unsupported")

    @pytest.mark.asyncio
    async def test_extract_text_openai_invalid_api_key(self, test_image_path):
        """Test that OpenAI provider handles invalid API key."""
        try:
            await extract_text(test_image_path, provider=Provider.OPENAI)
        except ValueError as e:
            # Should fail with invalid API key error
            assert "Invalid OpenAI API key" in str(e)
        except Exception as e:
            # Other errors (like connection issues) are acceptable
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error"]):
                pytest.skip(f"OpenAI API not accessible: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_extract_text_localai_not_implemented(self, test_image_path):
        """Test that LocalAI provider is not yet implemented."""
        with pytest.raises(NotImplementedError, match="LocalAI provider not yet implemented"):
            await extract_text(test_image_path, provider=Provider.LOCALAI) 