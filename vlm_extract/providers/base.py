"""Abstract base class for VLM provider adapters."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseProvider(ABC):
    """Abstract base class for VLM provider adapters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        self.config = config

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

    async def extract_text_with_fallback(self, file_path: Path, pre_extracted_text: Optional[str] = None) -> str:
        """
        Extract text with optional pre-extracted text fallback.
        
        Args:
            file_path: Path to the file to process
            pre_extracted_text: Text already extracted (e.g., by PyMuPDF)
            
        Returns:
            Extracted text as string
        """
        # If text was already extracted (e.g., by PyMuPDF), return it
        if pre_extracted_text:
            return pre_extracted_text
        
        # Otherwise, use the normal VLM extraction process
        return await self.extract_text(file_path) 