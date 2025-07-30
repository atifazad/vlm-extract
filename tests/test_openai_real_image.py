"""Test OpenAI with real image."""

import pytest
from pathlib import Path
from vlm_extract.providers.openai import OpenAIProvider
from vlm_extract.config import config


@pytest.mark.asyncio
async def test_openai_with_real_image(test_image_path):
    """Test OpenAI with real image file from test_images folder."""
    provider_config = config.get_provider_config("openai")
    
    # Skip if no API key
    if not provider_config.get("api_key"):
        pytest.skip("OpenAI API key not configured")
    
    provider = OpenAIProvider(provider_config)
    
    try:
        result = await provider.extract_text(test_image_path)
        print(f"Extracted text: {result}")
        assert isinstance(result, str)
        print("✅ OpenAI test with real image PASSED")
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        raise 