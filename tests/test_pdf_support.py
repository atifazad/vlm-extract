"""Tests for PDF support functionality."""

import pytest
from pathlib import Path
from vlm_extract.pdf_processor import PDFProcessor


class TestPDFProcessor:
    """Test PDF processor functionality."""

    def test_pdf_processor_initialization(self):
        """Test PDF processor can be initialized."""
        processor = PDFProcessor()
        assert processor is not None

    def test_is_valid_pdf_with_nonexistent_file(self):
        """Test PDF validation with non-existent file."""
        processor = PDFProcessor()
        pdf_path = Path("nonexistent.pdf")
        
        assert not processor.is_valid_pdf(pdf_path)

    def test_get_page_count_with_nonexistent_file(self):
        """Test page count with non-existent file."""
        processor = PDFProcessor()
        pdf_path = Path("nonexistent.pdf")
        
        with pytest.raises(RuntimeError, match="Failed to read PDF"):
            processor.get_page_count(pdf_path)

    def test_extract_pages_as_images_with_nonexistent_file(self):
        """Test page extraction with non-existent file."""
        processor = PDFProcessor()
        pdf_path = Path("nonexistent.pdf")
        
        with pytest.raises(RuntimeError, match="Failed to process PDF"):
            processor.extract_pages_as_images(pdf_path)


class TestPDFIntegration:
    """Test PDF integration with Ollama provider."""

    @pytest.mark.asyncio
    async def test_pdf_extraction_not_implemented_yet(self):
        """Test that PDF extraction is not yet fully implemented."""
        # This test documents that PDF extraction uses placeholder images
        # In a real implementation, you'd use proper PDF-to-image conversion
        from vlm_extract.utils import get_supported_formats
        
        # Test that PDF is in supported formats
        formats = get_supported_formats()
        assert "PDF" in formats["documents"] 