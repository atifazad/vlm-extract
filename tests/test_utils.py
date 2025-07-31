"""Tests for utility functions."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from vlm_extract.utils import (
    get_file_extension,
    is_image_file,
    is_document_file,
    validate_file_size,
    validate_file,
    load_image_data,
    get_supported_formats,
    process_file_for_vlm,
)


class TestUtils:
    """Test utility functions."""

    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension(Path("test.png")) == "PNG"
        assert get_file_extension(Path("test.PNG")) == "PNG"
        assert get_file_extension(Path("test.pdf")) == "PDF"
        assert get_file_extension(Path("test.docx")) == "DOCX"
        assert get_file_extension(Path("test")) == ""

    def test_is_image_file(self):
        """Test image file detection."""
        assert is_image_file(Path("test.png"))
        assert is_image_file(Path("test.jpg"))
        assert is_image_file(Path("test.jpeg"))
        assert is_image_file(Path("test.gif"))
        assert is_image_file(Path("test.bmp"))
        assert is_image_file(Path("test.webp"))
        assert is_image_file(Path("test.tiff"))
        assert is_image_file(Path("test.heic"))
        
        assert not is_image_file(Path("test.pdf"))
        assert not is_image_file(Path("test.docx"))
        assert not is_image_file(Path("test.txt"))

    def test_is_document_file(self):
        """Test document file detection."""
        assert is_document_file(Path("test.pdf"))
        # Only PDF is supported for documents
        assert not is_document_file(Path("test.docx"))
        assert not is_document_file(Path("test.pptx"))
        assert not is_document_file(Path("test.xlsx"))
        assert not is_document_file(Path("test.epub"))
        assert not is_document_file(Path("test.html"))
        assert not is_document_file(Path("test.png"))
        assert not is_document_file(Path("test.txt"))

    def test_validate_file_size(self):
        """Test file size validation."""
        with patch('pathlib.Path.stat') as mock_stat:
            # Test file within size limit
            mock_stat.return_value.st_size = 5 * 1024 * 1024  # 5MB
            assert validate_file_size(Path("test.png"))
            
            # Test file exceeding size limit
            mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB
            assert not validate_file_size(Path("test.png"))

    def test_validate_file_nonexistent(self):
        """Test file validation with non-existent file."""
        is_valid, error_msg = validate_file(Path("nonexistent.png"))
        assert not is_valid
        assert "File not found" in error_msg

    def test_validate_file_not_file(self):
        """Test file validation with directory."""
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('pathlib.Path.is_file') as mock_is_file:
                mock_exists.return_value = True
                mock_is_file.return_value = False
                
                is_valid, error_msg = validate_file(Path("test.png"))
                assert not is_valid
                assert "Path is not a file" in error_msg

    def test_validate_file_too_large(self):
        """Test file validation with oversized file."""
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('pathlib.Path.is_file') as mock_is_file:
                with patch('vlm_extract.utils.validate_file_size') as mock_validate_size:
                    mock_exists.return_value = True
                    mock_is_file.return_value = True
                    mock_validate_size.return_value = False
                    
                    is_valid, error_msg = validate_file(Path("test.png"))
                    assert not is_valid
                    assert "File too large" in error_msg

    def test_validate_file_unsupported_format(self):
        """Test file validation with unsupported format."""
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('pathlib.Path.is_file') as mock_is_file:
                with patch('vlm_extract.utils.validate_file_size') as mock_validate_size:
                    with patch('vlm_extract.utils.is_image_file') as mock_is_image:
                        with patch('vlm_extract.utils.is_document_file') as mock_is_document:
                            mock_exists.return_value = True
                            mock_is_file.return_value = True
                            mock_validate_size.return_value = True
                            mock_is_image.return_value = False
                            mock_is_document.return_value = False
                            
                            is_valid, error_msg = validate_file(Path("test.txt"))
                            assert not is_valid
                            assert "Unsupported file format" in error_msg

    def test_validate_file_valid(self):
        """Test file validation with valid file."""
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('pathlib.Path.is_file') as mock_is_file:
                with patch('vlm_extract.utils.validate_file_size') as mock_validate_size:
                    with patch('vlm_extract.utils.is_image_file') as mock_is_image:
                        mock_exists.return_value = True
                        mock_is_file.return_value = True
                        mock_validate_size.return_value = True
                        mock_is_image.return_value = True
                        
                        is_valid, error_msg = validate_file(Path("test.png"))
                        assert is_valid
                        assert error_msg == ""

    @pytest.mark.asyncio
    async def test_load_image_data(self):
        """Test image data loading."""
        test_data = b"fake image data"
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = test_data
            
            result = await load_image_data(Path("test.png"))
            assert result == test_data

    def test_get_supported_formats(self):
        """Test supported formats retrieval."""
        formats = get_supported_formats()
        
        assert "images" in formats
        assert "documents" in formats
        assert "PNG" in formats["images"]
        assert "PDF" in formats["documents"]

    @pytest.mark.asyncio
    async def test_process_file_for_vlm_image(self):
        """Test processing image file for VLM."""
        test_data = b"fake image data"
        
        with patch('vlm_extract.utils.validate_file') as mock_validate:
            with patch('vlm_extract.utils.is_image_file') as mock_is_image:
                with patch('vlm_extract.utils.load_image_data') as mock_load_data:
                    mock_validate.return_value = (True, "")
                    mock_is_image.return_value = True
                    mock_load_data.return_value = test_data
                    
                    result = await process_file_for_vlm(Path("test.png"))
                    assert len(result) == 1
                    assert result[0] == test_data

    @pytest.mark.asyncio
    async def test_process_file_for_vlm_pdf(self):
        """Test processing PDF file for VLM."""
        test_data = b"fake image data"
        
        with patch('vlm_extract.utils.validate_file') as mock_validate:
            with patch('vlm_extract.utils.is_image_file') as mock_is_image:
                with patch('vlm_extract.utils._process_pdf_for_vlm') as mock_process_pdf:
                    mock_validate.return_value = (True, "")
                    mock_is_image.return_value = False
                    mock_process_pdf.return_value = [test_data]
                    
                    result = await process_file_for_vlm(Path("test.pdf"))
                    assert len(result) == 1
                    assert result[0] == test_data

    @pytest.mark.asyncio
    async def test_process_file_for_vlm_document(self):
        """Test processing document file for VLM."""
        test_data = b"fake image data"
        
        with patch('vlm_extract.utils.validate_file') as mock_validate:
            with patch('vlm_extract.utils.is_image_file') as mock_is_image:
                with patch('vlm_extract.utils._process_pdf_for_vlm') as mock_process_pdf:
                    mock_validate.return_value = (True, "")
                    mock_is_image.return_value = False
                    mock_process_pdf.return_value = [test_data]
                    
                    result = await process_file_for_vlm(Path("test.pdf"))
                    assert len(result) == 1
                    assert result[0] == test_data

    @pytest.mark.asyncio
    async def test_process_file_for_vlm_unsupported(self):
        """Test processing unsupported file for VLM."""
        with patch('vlm_extract.utils.validate_file') as mock_validate:
            with patch('vlm_extract.utils.is_image_file') as mock_is_image:
                with patch('vlm_extract.utils.is_document_file') as mock_is_document:
                    mock_validate.return_value = (True, "")
                    mock_is_image.return_value = False
                    mock_is_document.return_value = False
                    
                    with pytest.raises(ValueError, match="Unsupported file format"):
                        await process_file_for_vlm(Path("test.txt"))

    @pytest.mark.asyncio
    async def test_process_file_for_vlm_validation_error(self):
        """Test processing file with validation error."""
        with patch('vlm_extract.utils.validate_file') as mock_validate:
            mock_validate.return_value = (False, "File not found")
            
            with pytest.raises(ValueError, match="File not found"):
                await process_file_for_vlm(Path("test.png"))

    @pytest.mark.asyncio
    async def test_process_pdf_for_vlm(self):
        """Test PDF processing for VLM."""
        test_data = b"fake image data"
        
        with patch('vlm_extract.utils.validate_file') as mock_validate:
            with patch('vlm_extract.utils.is_image_file') as mock_is_image:
                with patch('vlm_extract.utils._process_pdf_for_vlm') as mock_process_pdf:
                    mock_validate.return_value = (True, "")
                    mock_is_image.return_value = False
                    mock_process_pdf.return_value = [test_data]
                    
                    result = await process_file_for_vlm(Path("test.pdf"))
                    assert len(result) == 1
                    assert result[0] == test_data

    @pytest.mark.asyncio
    async def test_process_document_for_vlm(self):
        """Test document processing for VLM."""
        test_data = b"fake image data"
        
        with patch('vlm_extract.utils.validate_file') as mock_validate:
            with patch('vlm_extract.utils.is_image_file') as mock_is_image:
                with patch('vlm_extract.utils._process_pdf_for_vlm') as mock_process_pdf:
                    mock_validate.return_value = (True, "")
                    mock_is_image.return_value = False
                    mock_process_pdf.return_value = [test_data]
                    
                    result = await process_file_for_vlm(Path("test.pdf"))
                    assert len(result) == 1
                    assert result[0] == test_data 