"""VLM Extract - A simple library to extract text from documents and images using Vision Language Models."""

from .core import extract_text, extract_text_batch
from .config import Config, config, Provider

__version__ = "0.1.0"
__all__ = ["extract_text", "extract_text_batch", "Config", "config", "Provider"] 