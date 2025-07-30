"""Integration tests for OpenAI provider."""

import pytest
from pathlib import Path
from vlm_extract.providers.openai import OpenAIProvider
from vlm_extract.config import config


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_openai_health_check(self):
        """Test OpenAI health check."""
        provider_config = config.get_provider_config("openai")
        
        # Skip if no API key
        if not provider_config.get("api_key"):
            pytest.skip("OpenAI API key not configured")
        
        provider = OpenAIProvider(provider_config)
        
        # Health check should return boolean
        result = await provider.health_check()
        assert isinstance(result, bool)

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
            assert isinstance(result, str)
        except ValueError as e:
            # Handle invalid API key error
            if "Invalid OpenAI API key" in str(e):
                pytest.skip(f"OpenAI API key is invalid: {e}")
            else:
                raise
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "rate limit"]):
                pytest.skip(f"OpenAI API not accessible: {e}")
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
            assert isinstance(result, str)
        except ValueError as e:
            # Handle invalid API key error
            if "Invalid OpenAI API key" in str(e):
                pytest.skip(f"OpenAI API key is invalid: {e}")
            else:
                raise
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "rate limit"]):
                pytest.skip(f"OpenAI API not accessible: {e}")
            else:
                raise

    def test_openai_missing_api_key(self):
        """Test that OpenAI provider requires API key."""
        provider_config = {
            "provider": "openai",
            "api_key": "",
            "model": "gpt-4o",
            "timeout": 30,
            "max_retries": 3
        }
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIProvider(provider_config) 