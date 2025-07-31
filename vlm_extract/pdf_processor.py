"""PDF processing utilities for VLM Extract."""

import io
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from pypdf import PdfReader
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from .config import config


class PDFProcessor:
    """PDF processor with smart text/image detection and extraction."""

    def __init__(self):
        """Initialize PDF processor."""
        pass

    def is_text_based_pdf(self, pdf_path: Path) -> bool:
        """
        Check if PDF contains extractable text using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF contains significant text content
        """
        # Check if text extraction is enabled
        if not config.file.pdf_text_extraction_enabled:
            return False
            
        try:
            doc = fitz.open(pdf_path)
            total_text = ""
            total_pages = len(doc)
            
            for page in doc:
                text = page.get_text()
                if text.strip():
                    total_text += text
            
            doc.close()
            
            # Calculate text ratio (text characters vs total pages)
            text_length = len(total_text.strip())
            if total_pages == 0:
                return False
                
            # Consider it text-based if we have substantial text content
            # and the text ratio meets the minimum threshold
            text_ratio = text_length / (total_pages * 100)  # Normalize by pages
            return text_ratio >= config.file.pdf_min_text_ratio and text_length > 50
            
        except Exception:
            # If PyMuPDF fails, assume it's image-based
            return False

    def extract_text_with_pymupdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using PyMuPDF (fast path for text-based PDFs).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    if len(doc) > 1:
                        all_text.append(f"Page {page_num + 1}:\n{text.strip()}")
                    else:
                        all_text.append(text.strip())
            
            doc.close()
            
            return "\n\n".join(all_text) if all_text else "No text could be extracted"
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF {pdf_path}: {e}")

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