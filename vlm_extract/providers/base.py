"""Abstract base class for VLM provider adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path


class BaseProvider(ABC):
    """Abstract base class for VLM provider adapters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        self.config = config

    @abstractmethod
    async def extract_text(self, file_path: Path) -> str:
        """Extract text from a file using the VLM provider."""
        pass

    @abstractmethod
    async def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image data using the VLM provider."""
        pass

    async def validate_file(self, file_path: Path) -> bool:
        """Validate if the file can be processed by this provider."""
        return file_path.exists() and file_path.is_file()



    async def health_check(self) -> bool:
        """Check if the provider is healthy and available."""
        return True 