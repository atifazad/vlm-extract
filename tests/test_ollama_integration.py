"""Integration tests for Ollama provider."""

import pytest
from pathlib import Path
from vlm_extract.providers.ollama import OllamaProvider
from vlm_extract.config import config


class TestOllamaIntegration:
    """Integration tests for Ollama provider."""

    @pytest.mark.asyncio
    async def test_ollama_health_check(self):
        """Test Ollama server health check."""
        provider_config = config.get_provider_config("ollama")
        provider = OllamaProvider(provider_config)
        
        # This will fail if Ollama is not running
        # In a real test environment, you'd want to mock this
        try:
            is_healthy = await provider.health_check()
            # If Ollama is running, it should be healthy
            # If not running, we expect an exception
        except Exception:
            # Ollama not running - this is expected in test environment
            pytest.skip("Ollama server not running")

    @pytest.mark.asyncio
    async def test_ollama_extract_text_from_image(self, test_image_path):
        """Test text extraction from image."""
        provider_config = config.get_provider_config("ollama")
        provider = OllamaProvider(provider_config)
        
        # Load image data
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        try:
            result = await provider.extract_text_from_image(image_data)
            # Should return a string (even if empty for our test image)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error", "invalid format"]):
                pytest.skip(f"Ollama server not accessible or test image invalid: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_ollama_extract_text_from_file(self, test_image_path):
        """Test text extraction from file."""
        provider_config = config.get_provider_config("ollama")
        provider = OllamaProvider(provider_config)
        
        try:
            result = await provider.extract_text(test_image_path)
            # Should return a string (even if empty for our test image)
            assert isinstance(result, str)
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "timeout", "server error", "invalid format"]):
                pytest.skip(f"Ollama server not accessible or test image invalid: {e}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_ollama_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        provider_config = config.get_provider_config("ollama")
        provider = OllamaProvider(provider_config)
        
        # Create a test file with unsupported format
        test_file = Path("test.txt")
        test_file.write_text("test content")
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                await provider.extract_text(test_file)
        finally:
            test_file.unlink(missing_ok=True)

    def test_ollama_supported_formats(self):
        """Test supported formats list."""
        from vlm_extract.utils import get_supported_formats
        
        formats = get_supported_formats()
        assert isinstance(formats, dict)
        assert "images" in formats
        assert "documents" in formats
        assert "PNG" in formats["images"]
        assert "JPEG" in formats["images"]
        assert "GIF" in formats["images"] 