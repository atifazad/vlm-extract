"""Document processing utilities for VLM Extract."""

import io
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import subprocess
import tempfile
import os
from .config import config
from .utils import get_file_extension


class DocumentProcessor:
    """Document processor for converting various formats to images."""

    def __init__(self):
        """Initialize document processor."""
        # Get supported formats from configuration
        self.supported_formats = config.file.supported_document_formats

    def convert_document_to_images(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """Convert document to list of (page_number, image_data) tuples."""
        file_ext = get_file_extension(file_path).upper()
        
        if file_ext == "PDF":
            return self._convert_pdf_to_images(file_path)
        else:
            raise ValueError(f"Unsupported document format: {file_ext}")

    def _convert_pdf_to_images(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """Convert PDF to images."""
        try:
            from .pdf_processor import PDFProcessor
            pdf_processor = PDFProcessor()
            return pdf_processor.extract_pages_as_images(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF {file_path}: {e}")

    def is_supported_format(self, file_path: Path) -> bool:
        """Check if document format is supported."""
        extension = file_path.suffix.upper()
        return extension in [f".{fmt}" for fmt in self.supported_formats]

    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return self.supported_formats 