"""Utility functions for file handling and format detection."""

import io
from pathlib import Path
from typing import List, Union
from .config import config
from .pdf_processor import PDFProcessor


def get_file_extension(file_path: Path) -> str:
    """Get file extension in uppercase."""
    return file_path.suffix.upper().lstrip(".")


def is_image_file(file_path: Path) -> bool:
    """Check if file is a supported image format."""
    extension = get_file_extension(file_path)
    return extension in config.file.supported_image_formats


def is_document_file(file_path: Path) -> bool:
    """Check if file is a supported document format."""
    extension = get_file_extension(file_path)
    return extension in config.file.supported_document_formats


def validate_file_size(file_path: Path) -> bool:
    """Validate file size against configured limits."""
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    return file_size_mb <= config.file.max_file_size_mb


def validate_file(file_path: Path) -> tuple[bool, str]:
    """
    Validate file for processing.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    if not validate_file_size(file_path):
        return False, f"File too large: {file_path}"
    
    extension = get_file_extension(file_path)
    if not (is_image_file(file_path) or is_document_file(file_path)):
        return False, f"Unsupported file format: {extension}"
    
    return True, ""


async def load_image_data(file_path: Path) -> bytes:
    """Load image data as bytes."""
    with open(file_path, "rb") as f:
        return f.read()


def get_supported_formats() -> dict[str, List[str]]:
    """Get all supported file formats."""
    return {
        "images": config.file.supported_image_formats,
        "documents": config.file.supported_document_formats,
    }


async def process_file_for_vlm(file_path: Path) -> List[bytes]:
    """
    Process file and return list of image data ready for VLM processing.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        List of image data bytes (one for images, multiple for PDFs)
        
    Raises:
        ValueError: If file format is not supported
        RuntimeError: If processing fails
    """
    # Validate file first
    is_valid, error_msg = validate_file(file_path)
    if not is_valid:
        raise ValueError(error_msg)
    
    if is_image_file(file_path):
        # Single image file
        image_data = await load_image_data(file_path)
        return [image_data]
    
    elif is_document_file(file_path):
        # Document file (PDF only)
        if file_path.suffix.upper() == ".PDF":
            # Use smart PDF processing
            result, method = await process_pdf_smart(file_path)
            
            if method == "pymupdf":
                # PyMuPDF already extracted text, return empty list to skip VLM
                # The text will be returned directly by the provider
                return []
            else:
                # VLM processing needed, return image data
                return result
        else:
            raise ValueError(f"Unsupported document format: {file_path.suffix}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


async def _process_pdf_for_vlm(pdf_path: Path) -> List[bytes]:
    """Process PDF file and return list of page images as bytes."""
    pdf_processor = PDFProcessor()
    
    try:
        # Get PDF pages as images
        page_images = pdf_processor.extract_pages_as_images(pdf_path)
        
        if not page_images:
            raise RuntimeError("No pages could be extracted from PDF")
        
        # Convert PIL images to bytes
        image_data_list = []
        for page_num, image in page_images:
            try:
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                image_data_list.append(img_byte_arr.getvalue())
            except Exception as e:
                raise RuntimeError(f"Failed to convert page {page_num} to image: {e}")
        
        return image_data_list
        
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF {pdf_path}: {e}")


async def process_pdf_smart(pdf_path: Path) -> tuple[str, str]:
    """
    Smart PDF processing: use PyMuPDF for text-based PDFs, VLM for image-based PDFs.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted_text, processing_method)
    """
    pdf_processor = PDFProcessor()
    
    # Check if PDF is text-based
    if pdf_processor.is_text_based_pdf(pdf_path):
        # Fast path: extract text directly with PyMuPDF
        text = pdf_processor.extract_text_with_pymupdf(pdf_path)
        return text, "pymupdf"
    else:
        # Fallback: convert to images for VLM processing
        image_data_list = await _process_pdf_for_vlm(pdf_path)
        return image_data_list, "vlm"


async def _process_document_for_vlm(file_path: Path) -> List[bytes]:
    """Process document file and return list of page images as bytes."""
    from .document_processor import DocumentProcessor
    
    document_processor = DocumentProcessor()
    
    try:
        # Get document pages as images
        page_images = document_processor.convert_document_to_images(file_path)
        
        if not page_images:
            raise RuntimeError("No pages could be extracted from document")
        
        # Convert PIL images to bytes
        image_data_list = []
        for page_num, image in page_images:
            try:
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                image_data_list.append(img_byte_arr.getvalue())
            except Exception as e:
                raise RuntimeError(f"Failed to convert page {page_num} to image: {e}")
        
        return image_data_list
        
    except Exception as e:
        raise RuntimeError(f"Failed to process document {file_path}: {e}") 