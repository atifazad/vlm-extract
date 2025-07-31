"""Tests for document processing functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from vlm_extract.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test document processor functionality."""

    def test_document_processor_initialization(self):
        """Test document processor can be initialized."""
        processor = DocumentProcessor()
        assert processor is not None
        assert "DOCX" in processor.supported_formats
        assert "PPTX" in processor.supported_formats
        assert "XLSX" in processor.supported_formats
        assert "EPUB" in processor.supported_formats
        assert "HTML" in processor.supported_formats

    def test_is_supported_format(self):
        """Test format support detection."""
        processor = DocumentProcessor()
        
        # Test supported formats
        assert processor.is_supported_format(Path("test.docx"))
        assert processor.is_supported_format(Path("test.pptx"))
        assert processor.is_supported_format(Path("test.xlsx"))
        assert processor.is_supported_format(Path("test.epub"))
        assert processor.is_supported_format(Path("test.html"))
        assert processor.is_supported_format(Path("test.pdf"))  # PDF is now supported
        
        # Test unsupported formats
        assert not processor.is_supported_format(Path("test.txt"))
        assert not processor.is_supported_format(Path("test.png"))

    def test_convert_document_to_images_unsupported_format(self):
        """Test conversion with unsupported format."""
        processor = DocumentProcessor()
        file_path = Path("test.txt")
        
        with pytest.raises(ValueError, match="Unsupported document format"):
            processor.convert_document_to_images(file_path)

    @patch('subprocess.check_output')
    @patch('subprocess.run')
    def test_convert_docx_to_images(self, mock_run, mock_check_output):
        """Test DOCX to images conversion."""
        processor = DocumentProcessor()
        file_path = Path("test.docx")
        
        # Mock pandoc output
        mock_check_output.return_value = "<html>test content</html>"
        
        # Mock wkhtmltopdf
        mock_run.return_value = MagicMock()
        
        with patch.object(processor, '_html_to_images') as mock_html_to_images:
            mock_html_to_images.return_value = [(1, MagicMock())]
            
            result = processor.convert_document_to_images(file_path)
            assert len(result) == 1
            assert result[0][0] == 1

    @patch('subprocess.check_output')
    def test_convert_docx_to_images_pandoc_error(self, mock_check_output):
        """Test DOCX conversion with pandoc error."""
        processor = DocumentProcessor()
        file_path = Path("test.docx")
        
        # Mock pandoc error
        mock_check_output.side_effect = FileNotFoundError("pandoc not found")
        
        with pytest.raises(RuntimeError, match="pandoc not found"):
            processor.convert_document_to_images(file_path)

    @patch('subprocess.run')
    def test_convert_pptx_to_images(self, mock_run):
        """Test PPTX to images conversion."""
        processor = DocumentProcessor()
        file_path = Path("test.pptx")
        
        # Mock libreoffice
        mock_run.return_value = MagicMock()
        
        with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = True
                
                with patch('vlm_extract.pdf_processor.PDFProcessor') as mock_pdf_processor:
                    mock_processor_instance = MagicMock()
                    mock_processor_instance.extract_pages_as_images.return_value = [(1, MagicMock())]
                    mock_pdf_processor.return_value = mock_processor_instance
                    
                    result = processor.convert_document_to_images(file_path)
                    assert len(result) == 1

    @patch('subprocess.run')
    def test_convert_pptx_to_images_libreoffice_error(self, mock_run):
        """Test PPTX conversion with libreoffice error."""
        processor = DocumentProcessor()
        file_path = Path("test.pptx")
        
        # Mock libreoffice error
        mock_run.side_effect = FileNotFoundError("libreoffice not found")
        
        with pytest.raises(RuntimeError, match="libreoffice not found"):
            processor.convert_document_to_images(file_path)

    @patch('subprocess.run')
    def test_convert_xlsx_to_images(self, mock_run):
        """Test XLSX to images conversion."""
        processor = DocumentProcessor()
        file_path = Path("test.xlsx")
        
        # Mock libreoffice
        mock_run.return_value = MagicMock()
        
        with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = True
                
                with patch('vlm_extract.pdf_processor.PDFProcessor') as mock_pdf_processor:
                    mock_processor_instance = MagicMock()
                    mock_processor_instance.extract_pages_as_images.return_value = [(1, MagicMock())]
                    mock_pdf_processor.return_value = mock_processor_instance
                    
                    result = processor.convert_document_to_images(file_path)
                    assert len(result) == 1

    @patch('subprocess.run')
    def test_convert_epub_to_images(self, mock_run):
        """Test EPUB to images conversion."""
        processor = DocumentProcessor()
        file_path = Path("test.epub")
        
        # Mock calibre
        mock_run.return_value = MagicMock()
        
        with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = True
                
                with patch('vlm_extract.pdf_processor.PDFProcessor') as mock_pdf_processor:
                    mock_processor_instance = MagicMock()
                    mock_processor_instance.extract_pages_as_images.return_value = [(1, MagicMock())]
                    mock_pdf_processor.return_value = mock_processor_instance
                    
                    result = processor.convert_document_to_images(file_path)
                    assert len(result) == 1

    @patch('subprocess.run')
    def test_convert_epub_to_images_calibre_error(self, mock_run):
        """Test EPUB conversion with calibre error."""
        processor = DocumentProcessor()
        file_path = Path("test.epub")
        
        # Mock calibre error
        mock_run.side_effect = FileNotFoundError("calibre not found")
        
        with pytest.raises(RuntimeError, match="calibre not found"):
            processor.convert_document_to_images(file_path)

    def test_convert_html_to_images(self):
        """Test HTML to images conversion."""
        processor = DocumentProcessor()
        file_path = Path("test.html")
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "<html>test</html>"
            
            with patch.object(processor, '_html_to_images') as mock_html_to_images:
                mock_html_to_images.return_value = [(1, MagicMock())]
                
                result = processor.convert_document_to_images(file_path)
                assert len(result) == 1

    @patch('subprocess.run')
    def test_html_to_images(self, mock_run):
        """Test HTML to images conversion."""
        processor = DocumentProcessor()
        html_content = "<html><body>test</body></html>"
        
        # Mock wkhtmltopdf
        mock_run.return_value = MagicMock()
        
        with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = MagicMock()
                
                with patch('pathlib.Path.exists') as mock_exists:
                    mock_exists.return_value = True
                    
                    with patch('vlm_extract.pdf_processor.PDFProcessor') as mock_pdf_processor:
                        mock_processor_instance = MagicMock()
                        mock_processor_instance.extract_pages_as_images.return_value = [(1, MagicMock())]
                        mock_pdf_processor.return_value = mock_processor_instance
                        
                        result = processor._html_to_images(html_content)
                        assert len(result) == 1

    @patch('subprocess.run')
    def test_html_to_images_wkhtmltopdf_error(self, mock_run):
        """Test HTML conversion with wkhtmltopdf error."""
        processor = DocumentProcessor()
        html_content = "<html><body>test</body></html>"
        
        # Mock wkhtmltopdf error
        mock_run.side_effect = FileNotFoundError("wkhtmltopdf not found")
        
        with pytest.raises(RuntimeError, match="wkhtmltopdf not found"):
            processor._html_to_images(html_content) 