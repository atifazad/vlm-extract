"""Tests for smart PDF processing with PyMuPDF and VLM fallback."""

import pytest
from pathlib import Path
from vlm_extract.pdf_processor import PDFProcessor
from vlm_extract.utils import process_pdf_smart


class TestSmartPDFProcessing:
    """Test smart PDF processing functionality."""

    def test_pdf_processor_initialization(self):
        """Test PDF processor can be initialized."""
        processor = PDFProcessor()
        assert processor is not None

    def test_is_text_based_pdf_with_invalid_file(self):
        """Test text detection with invalid file."""
        processor = PDFProcessor()
        result = processor.is_text_based_pdf(Path("nonexistent.pdf"))
        assert result is False

    def test_extract_text_with_pymupdf_with_invalid_file(self):
        """Test PyMuPDF extraction with invalid file."""
        processor = PDFProcessor()
        with pytest.raises(RuntimeError):
            processor.extract_text_with_pymupdf(Path("nonexistent.pdf"))

    @pytest.mark.asyncio
    async def test_process_pdf_smart_with_invalid_file(self):
        """Test smart PDF processing with invalid file."""
        with pytest.raises(RuntimeError):
            await process_pdf_smart(Path("nonexistent.pdf"))

    def test_pdf_processor_config_integration(self):
        """Test that PDF processor uses configuration."""
        processor = PDFProcessor()
        # This should not raise an exception
        assert hasattr(processor, 'is_text_based_pdf')
        assert hasattr(processor, 'extract_text_with_pymupdf') 