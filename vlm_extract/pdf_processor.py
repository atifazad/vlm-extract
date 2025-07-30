"""PDF processing utilities for VLM Extract."""

import io
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from pypdf import PdfReader
from pdf2image import convert_from_path


class PDFProcessor:
    """PDF processor for converting pages to images."""

    def __init__(self):
        """Initialize PDF processor."""
        pass

    def extract_pages_as_images(self, pdf_path: Path) -> List[Tuple[int, Image.Image]]:
        """
        Extract PDF pages as PIL Images using pdf2image.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of tuples (page_number, image)
        """
        images = []
        
        try:
            # Convert PDF pages to images using pdf2image
            pdf_images = convert_from_path(pdf_path, dpi=200)
            
            for page_num, image in enumerate(pdf_images):
                images.append((page_num + 1, image))
                    
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF {pdf_path}: {e}")
            
        return images



    def get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF."""
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF {pdf_path}: {e}")

    def is_valid_pdf(self, pdf_path: Path) -> bool:
        """Check if a file is a valid PDF."""
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages) > 0
        except:
            return False 