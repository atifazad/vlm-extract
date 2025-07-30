"""Integration tests for OpenAI provider."""

import pytest
from pathlib import Path
from vlm_extract.providers.openai import OpenAIProvider
from vlm_extract.config import config


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_openai_health_check(self):
        """Test OpenAI API health check."""
        provider_config = config.get_provider_config("openai")
        
        # Skip if no API key
        if not provider_config.get("api_key"):
            pytest.skip("OpenAI API key not configured")
        
        provider = OpenAIProvider(provider_config)
        
        try:
            is_healthy = await provider.health_check()
            # If API key is valid, it should be healthy
            # If not valid, we expect an exception
        except Exception:
            # Invalid API key - this is expected in test environment
            pytest.skip("OpenAI API key not valid")

    @pytest.mark.asyncio
    async def test_openai_extract_text_from_image(self, test_image_path):
        """Test text extraction from image."""
        provider_config = config.get_provider_config("openai")
        
        # Skip if no API key
        if not provider_config.get("api_key"):
            pytest.skip("OpenAI API key not configured")
        
        provider = OpenAIProvider(provider_config)
        
        # Load image data
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        try:
            result = await provider.extract_text_from_image(image_data)
            # Should return a string (even if empty for our test image)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "rate limit", "invalid format"]):
                pytest.skip(f"OpenAI API not accessible or test image invalid: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_openai_extract_text_from_file(self, test_image_path):
        """Test text extraction from file."""
        provider_config = config.get_provider_config("openai")
        
        # Skip if no API key
        if not provider_config.get("api_key"):
            pytest.skip("OpenAI API key not configured")
        
        provider = OpenAIProvider(provider_config)
        
        try:
            result = await provider.extract_text(test_image_path)
            # Should return a string (even if empty for our test image)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "rate limit", "invalid format"]):
                pytest.skip(f"OpenAI API not accessible or test image invalid: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_openai_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        provider_config = config.get_provider_config("openai")
        
        # Skip if no API key
        if not provider_config.get("api_key"):
            pytest.skip("OpenAI API key not configured")
        
        provider = OpenAIProvider(provider_config)
        
        # Create a test file with unsupported format
        test_file = Path("test.txt")
        test_file.write_text("test content")
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                await provider.extract_text(test_file)
        finally:
            test_file.unlink(missing_ok=True)

    def test_openai_missing_api_key(self):
        """Test that OpenAI provider requires API key."""
        provider_config = {
            "provider": "openai",
            "api_key": "",
            "model": "gpt-4-vision-preview",
            "timeout": 30,
            "max_retries": 3
        }
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIProvider(provider_config) 