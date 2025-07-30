"""Tests for cross-provider compatibility."""

import pytest
from pathlib import Path
from vlm_extract import extract_text, extract_text_batch, Provider, config


class TestCrossProviderCompatibility:
    """Test cross-provider compatibility."""

    @pytest.mark.asyncio
    async def test_provider_switching_ollama(self, test_image_path):
        """Test switching to Ollama provider."""
        try:
            result = await extract_text(test_image_path, provider=Provider.OLLAMA)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error"]):
                pytest.skip(f"Ollama server not accessible: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_provider_switching_openai(self, test_image_path):
        """Test switching to OpenAI provider."""
        # Skip if no OpenAI API key
        if not config.get_provider_config("openai").get("api_key"):
            pytest.skip("OpenAI API key not configured")
        
        try:
            result = await extract_text(test_image_path, provider=Provider.OPENAI)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "rate limit"]):
                pytest.skip(f"OpenAI API not accessible: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_provider_switching_string(self, test_image_path):
        """Test provider switching using string names."""
        try:
            result = await extract_text(test_image_path, provider="ollama")
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error"]):
                pytest.skip(f"Ollama server not accessible: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_batch_processing_cross_provider(self, test_image_path):
        """Test batch processing with different providers."""
        file_paths = [test_image_path, test_image_path]  # Same file twice for testing
        
        # Test with Ollama
        try:
            results = await extract_text_batch(file_paths, provider=Provider.OLLAMA)
            assert isinstance(results, list)
            assert len(results) == 2
            # Check that results are either strings or exceptions
            for result in results:
                assert isinstance(result, (str, Exception))
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error"]):
                pytest.skip(f"Ollama server not accessible: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_unsupported_provider(self, test_image_path):
        """Test error handling for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            await extract_text(test_image_path, provider="unsupported")

    @pytest.mark.asyncio
    async def test_configuration_driven_provider_selection(self, test_image_path):
        """Test that provider selection works based on configuration."""
        # Test with default provider from config
        try:
            result = await extract_text(test_image_path)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error"]):
                pytest.skip(f"Default provider not accessible: {e}")
            else:
                raise

    def test_provider_enum_values(self):
        """Test that provider enum has correct values."""
        assert Provider.OLLAMA.value == "ollama"
        assert Provider.OPENAI.value == "openai"
        assert Provider.LOCALAI.value == "localai"

    def test_config_provider_mapping(self):
        """Test that config correctly maps provider names."""
        # Test string to enum conversion
        assert Provider("ollama") == Provider.OLLAMA
        assert Provider("openai") == Provider.OPENAI
        
        # Test invalid provider
        with pytest.raises(ValueError):
            Provider("invalid") 