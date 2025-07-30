"""Configuration management for VLM Extract."""

import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Provider(Enum):
    """VLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    LOCALAI = "localai"


class VLMConfig(BaseModel):
    """Unified VLM configuration with provider-specific resolution."""
    provider: Provider = Field(default_factory=lambda: Provider(os.getenv("VLM_PROVIDER", "ollama")))
    
    @property
    def base_url(self) -> str:
        """Get provider-specific base URL."""
        provider = self.provider.value.upper()
        return os.getenv(f"{provider}_BASE_URL", "")
    
    @property
    def api_key(self) -> str:
        """Get provider-specific API key."""
        provider = self.provider.value.upper()
        return os.getenv(f"{provider}_API_KEY", "")
    
    @property
    def model(self) -> str:
        """Get provider-specific model."""
        provider = self.provider.value.upper()
        return os.getenv(f"{provider}_MODEL", "")
    
    @property
    def timeout(self) -> int:
        """Get timeout configuration."""
        return int(os.getenv("VLM_TIMEOUT", "30"))
    
    @property
    def max_retries(self) -> int:
        """Get max retries configuration."""
        return int(os.getenv("VLM_MAX_RETRIES", "3"))
    
    @field_validator('provider')
    @classmethod
    def validate_provider_config(cls, v):
        """Validate that provider-specific configuration is provided."""
        provider_name = v.value.upper()
        
        # Check if provider-specific config exists
        base_url = os.getenv(f"{provider_name}_BASE_URL")
        model = os.getenv(f"{provider_name}_MODEL")
        
        if not base_url:
            raise ValueError(f"Missing {provider_name}_BASE_URL configuration for provider {v.value}")
        if not model:
            raise ValueError(f"Missing {provider_name}_MODEL configuration for provider {v.value}")
        
        return v


class FileConfig(BaseModel):
    """File processing configuration."""
    max_file_size_mb: int = Field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "50")))
    supported_image_formats: List[str] = Field(default=["PNG", "JPEG", "JPG", "GIF", "BMP", "WEBP", "TIFF", "HEIC"])
    supported_document_formats: List[str] = Field(default=["PDF", "DOCX", "PPTX", "XLSX", "EPUB", "HTML"])


class BatchConfig(BaseModel):
    """Batch processing configuration."""
    batch_size: int = Field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "5")))
    batch_timeout: int = Field(default_factory=lambda: int(os.getenv("BATCH_TIMEOUT", "60")))


class Config(BaseModel):
    """Main configuration class."""
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    file: FileConfig = Field(default_factory=FileConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)

    def get_provider_config(self, provider: Optional[str | Provider] = None) -> dict:
        """Get configuration for a specific provider."""
        if isinstance(provider, str):
            try:
                provider = Provider(provider)
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        
        provider = provider or self.vlm.provider
        
        # Get provider-specific configuration
        provider_name = provider.value.upper()
        
        config = {
            "provider": provider.value,
            "base_url": os.getenv(f"{provider_name}_BASE_URL", ""),
            "api_key": os.getenv(f"{provider_name}_API_KEY", ""),
            "model": os.getenv(f"{provider_name}_MODEL", ""),
            "timeout": self.vlm.timeout,
            "max_retries": self.vlm.max_retries,
        }
        
        return config


# Global configuration instance
config = Config() 