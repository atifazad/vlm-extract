"""Tests for configuration management."""

import pytest
from vlm_extract.config import Config, VLMConfig, Provider


class TestConfig:
    """Test configuration classes."""

    def test_vlm_config_defaults(self):
        """Test VLM configuration defaults."""
        config = VLMConfig()
        assert config.provider == Provider.OLLAMA
        assert config.base_url == "http://localhost:11434"
        assert config.model == "llava"
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_provider_enum(self):
        """Test Provider enum values."""
        assert Provider.OLLAMA.value == "ollama"
        assert Provider.OPENAI.value == "openai"
        assert Provider.LOCALAI.value == "localai"

    def test_config_provider_selection(self):
        """Test provider configuration selection."""
        config = Config()
        
        # Test default provider
        assert config.vlm.provider == Provider.OLLAMA
        
        # Test provider config retrieval
        provider_config = config.get_provider_config(Provider.OLLAMA)
        assert provider_config["provider"] == "ollama"
        assert "base_url" in provider_config
        assert "model" in provider_config
        assert "api_key" in provider_config
        assert "timeout" in provider_config
        assert "max_retries" in provider_config

    def test_provider_string_conversion(self):
        """Test provider string to enum conversion."""
        config = Config()
        
        # Test with string provider
        provider_config = config.get_provider_config("openai")
        assert provider_config["provider"] == "openai"
        
        # Test with enum provider
        provider_config = config.get_provider_config(Provider.OPENAI)
        assert provider_config["provider"] == "openai" 