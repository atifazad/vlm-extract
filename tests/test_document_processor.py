"""Tests for document processing functionality."""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from vlm_extract.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test document processor functionality."""

    def test_document_processor_initialization(self):
        """Test document processor can be initialized."""
        processor = DocumentProcessor()
        assert processor is not None
        assert "PDF" in processor.supported_formats

    def test_is_supported_format(self):
        """Test format support detection."""
        processor = DocumentProcessor()
        
        # Test supported formats
        assert processor.is_supported_format(Path("test.pdf"))  # PDF is supported
        
        # Test unsupported formats
        assert not processor.is_supported_format(Path("test.txt"))
        assert not processor.is_supported_format(Path("test.docx"))
        assert not processor.is_supported_format(Path("test.pptx"))
        assert not processor.is_supported_format(Path("test.xlsx"))
        assert not processor.is_supported_format(Path("test.epub"))
        assert not processor.is_supported_format(Path("test.html"))
        assert not processor.is_supported_format(Path("test.png"))

    def test_convert_document_to_images_unsupported_format(self):
        """Test conversion with unsupported format."""
        processor = DocumentProcessor()
        file_path = Path("test.txt")
        
        with pytest.raises(ValueError, match="Unsupported document format"):
            processor.convert_document_to_images(file_path)

    def test_convert_pdf_to_images(self):
        """Test PDF to images conversion."""
        processor = DocumentProcessor()
        file_path = Path("test.pdf")
        
        with patch('vlm_extract.pdf_processor.PDFProcessor') as mock_pdf_processor:
            mock_processor_instance = MagicMock()
            mock_processor_instance.extract_pages_as_images.return_value = [(1, MagicMock())]
            mock_pdf_processor.return_value = mock_processor_instance
            
            result = processor.convert_document_to_images(file_path)
            assert len(result) == 1
            assert result[0][0] == 1

    def test_convert_pdf_to_images_error(self):
        """Test PDF conversion error."""
        processor = DocumentProcessor()
        file_path = Path("test.pdf")
        
        with patch('vlm_extract.pdf_processor.PDFProcessor') as mock_pdf_processor:
            mock_processor_instance = MagicMock()
            mock_processor_instance.extract_pages_as_images.side_effect = Exception("PDF error")
            mock_pdf_processor.return_value = mock_processor_instance
            
            with pytest.raises(RuntimeError, match="Failed to convert PDF"):
                processor.convert_document_to_images(file_path) 