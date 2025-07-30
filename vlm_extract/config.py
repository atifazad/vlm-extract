"""Configuration management for VLM Extract."""

import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Provider(Enum):
    """VLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    LOCALAI = "localai"


class VLMConfig(BaseModel):
    """Unified VLM configuration."""
    provider: Provider = Field(default_factory=lambda: Provider(os.getenv("VLM_PROVIDER")))
    base_url: str = Field(default_factory=lambda: os.getenv("VLM_BASE_URL"))
    api_key: str = Field(default_factory=lambda: os.getenv("VLM_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("VLM_MODEL"))
    timeout: int = Field(default_factory=lambda: int(os.getenv("VLM_TIMEOUT", "30")))
    max_retries: int = Field(default_factory=lambda: int(os.getenv("VLM_MAX_RETRIES", "3")))


class FileConfig(BaseModel):
    """File processing configuration."""
    max_file_size_mb: int = Field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "50")))
    supported_image_formats: List[str] = Field(
        default_factory=lambda: os.getenv("SUPPORTED_IMAGE_FORMATS", "PNG,JPEG,GIF,BMP,WEBP,TIFF,HEIC").split(",")
    )
    supported_document_formats: List[str] = Field(
        default_factory=lambda: os.getenv("SUPPORTED_DOCUMENT_FORMATS", "PDF,DOCX,PPTX,XLSX,EPUB,HTML").split(",")
    )


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
        return {
            "provider": provider.value,
            "base_url": self.vlm.base_url,
            "api_key": self.vlm.api_key,
            "model": self.vlm.model,
            "timeout": self.vlm.timeout,
            "max_retries": self.vlm.max_retries,
        }


# Global configuration instance
config = Config() 