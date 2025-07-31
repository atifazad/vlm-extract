"""Tests for configuration management."""

import pytest
from vlm_extract.config import Config, VLMConfig, Provider


class TestConfig:
    """Test configuration classes."""

    def test_provider_enum(self):
        """Test Provider enum values."""
        assert Provider.OLLAMA.value == "ollama"
        assert Provider.OPENAI.value == "openai"
        assert Provider.LOCALAI.value == "localai"

    def test_config_provider_selection(self):
        """Test provider configuration selection from environment."""
        config = Config()
        
        # Test that provider is correctly read from environment
        # We can't assert a specific provider since it depends on VLM_PROVIDER env var
        assert config.vlm.provider in [Provider.OLLAMA, Provider.OPENAI, Provider.LOCALAI]
        
        # Test provider config retrieval for the current provider
        current_provider = config.vlm.provider
        provider_config = config.get_provider_config(current_provider)
        assert provider_config["provider"] == current_provider.value
        assert "base_url" in provider_config
        assert "model" in provider_config
        assert "api_key" in provider_config
        assert "timeout" in provider_config
        assert "max_retries" in provider_config
        
        # Test that provider-specific configuration is loaded
        provider_name = current_provider.value.upper()
        assert provider_config["base_url"] is not None
        assert provider_config["model"] is not None

    def test_provider_string_conversion(self):
        """Test provider string to enum conversion."""
        config = Config()
        
        # Test with string provider
        provider_config = config.get_provider_config("openai")
        assert provider_config["provider"] == "openai"
        
        # Test with enum provider
        provider_config = config.get_provider_config(Provider.OPENAI)
        assert provider_config["provider"] == "openai" 