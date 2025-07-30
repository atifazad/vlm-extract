"""Utility functions for file handling and format detection."""

from pathlib import Path
from typing import List
from .config import config


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