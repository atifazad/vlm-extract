"""Document processing utilities for VLM Extract."""

import io
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import subprocess
import tempfile
import os


class DocumentProcessor:
    """Document processor for converting various formats to images."""

    def __init__(self):
        """Initialize document processor."""
        self.supported_formats = ["DOCX", "PPTX", "XLSX", "EPUB", "HTML"]

    def convert_document_to_images(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """
        Convert document to list of images.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of tuples (page_number, image)
        """
        extension = file_path.suffix.upper()
        
        if extension == ".DOCX":
            return self._convert_docx_to_images(file_path)
        elif extension == ".PPTX":
            return self._convert_pptx_to_images(file_path)
        elif extension == ".XLSX":
            return self._convert_xlsx_to_images(file_path)
        elif extension == ".EPUB":
            return self._convert_epub_to_images(file_path)
        elif extension == ".HTML":
            return self._convert_html_to_images(file_path)
        else:
            raise ValueError(f"Unsupported document format: {extension}")

    def _convert_docx_to_images(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """Convert DOCX to images using pandoc and wkhtmltopdf."""
        try:
            # Use pandoc to convert DOCX to HTML
            html_content = subprocess.check_output(
                ["pandoc", str(file_path), "-t", "html"],
                text=True,
                stderr=subprocess.PIPE
            )
            
            # Convert HTML to image using wkhtmltopdf
            return self._html_to_images(html_content)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert DOCX {file_path}: {e}")
        except FileNotFoundError:
            raise RuntimeError("pandoc not found. Please install pandoc for DOCX support.")

    def _convert_pptx_to_images(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """Convert PPTX to images using libreoffice."""
        try:
            # Use libreoffice to convert PPTX to PDF
            with tempfile.TemporaryDirectory() as temp_dir:
                subprocess.run([
                    "libreoffice", "--headless", "--convert-to", "pdf",
                    "--outdir", temp_dir, str(file_path)
                ], check=True, capture_output=True)
                
                # Find the generated PDF
                pdf_path = Path(temp_dir) / f"{file_path.stem}.pdf"
                if pdf_path.exists():
                    from .pdf_processor import PDFProcessor
                    pdf_processor = PDFProcessor()
                    return pdf_processor.extract_pages_as_images(pdf_path)
                else:
                    raise RuntimeError("Failed to convert PPTX to PDF")
                    
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert PPTX {file_path}: {e}")
        except FileNotFoundError:
            raise RuntimeError("libreoffice not found. Please install libreoffice for PPTX support.")

    def _convert_xlsx_to_images(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """Convert XLSX to images using libreoffice."""
        try:
            # Use libreoffice to convert XLSX to PDF
            with tempfile.TemporaryDirectory() as temp_dir:
                subprocess.run([
                    "libreoffice", "--headless", "--convert-to", "pdf",
                    "--outdir", temp_dir, str(file_path)
                ], check=True, capture_output=True)
                
                # Find the generated PDF
                pdf_path = Path(temp_dir) / f"{file_path.stem}.pdf"
                if pdf_path.exists():
                    from .pdf_processor import PDFProcessor
                    pdf_processor = PDFProcessor()
                    return pdf_processor.extract_pages_as_images(pdf_path)
                else:
                    raise RuntimeError("Failed to convert XLSX to PDF")
                    
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert XLSX {file_path}: {e}")
        except FileNotFoundError:
            raise RuntimeError("libreoffice not found. Please install libreoffice for XLSX support.")

    def _convert_epub_to_images(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """Convert EPUB to images using calibre."""
        try:
            # Use calibre to convert EPUB to PDF
            with tempfile.TemporaryDirectory() as temp_dir:
                subprocess.run([
                    "ebook-convert", str(file_path), str(Path(temp_dir) / "output.pdf")
                ], check=True, capture_output=True)
                
                pdf_path = Path(temp_dir) / "output.pdf"
                if pdf_path.exists():
                    from .pdf_processor import PDFProcessor
                    pdf_processor = PDFProcessor()
                    return pdf_processor.extract_pages_as_images(pdf_path)
                else:
                    raise RuntimeError("Failed to convert EPUB to PDF")
                    
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert EPUB {file_path}: {e}")
        except FileNotFoundError:
            raise RuntimeError("calibre not found. Please install calibre for EPUB support.")

    def _convert_html_to_images(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """Convert HTML to images using wkhtmltopdf."""
        try:
            # Read HTML content
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return self._html_to_images(html_content)
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert HTML {file_path}: {e}")

    def _html_to_images(self, html_content: str) -> List[Tuple[int, Image.Image]]:
        """Convert HTML content to images using wkhtmltopdf."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write HTML to temporary file
                html_file = Path(temp_dir) / "input.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Convert HTML to PDF using wkhtmltopdf
                pdf_file = Path(temp_dir) / "output.pdf"
                subprocess.run([
                    "wkhtmltopdf", "--quiet", str(html_file), str(pdf_file)
                ], check=True, capture_output=True)
                
                if pdf_file.exists():
                    from .pdf_processor import PDFProcessor
                    pdf_processor = PDFProcessor()
                    return pdf_processor.extract_pages_as_images(pdf_file)
                else:
                    raise RuntimeError("Failed to convert HTML to PDF")
                    
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert HTML to PDF: {e}")
        except FileNotFoundError:
            raise RuntimeError("wkhtmltopdf not found. Please install wkhtmltopdf for HTML support.")

    def is_supported_format(self, file_path: Path) -> bool:
        """Check if document format is supported."""
        extension = file_path.suffix.upper()
        return extension in [f".{fmt}" for fmt in self.supported_formats]

    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return self.supported_formats 