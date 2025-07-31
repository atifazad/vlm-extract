"""Core interface for VLM Extract."""

import asyncio
from pathlib import Path
from typing import Optional, List
from .config import config, Provider
from .providers import OllamaProvider


async def extract_text(
    file_path: str | Path,
    provider: Optional[str | Provider] = None,
    **kwargs
) -> str:
    """
    Extract text from a document or image using Vision Language Models.
    
    Args:
        file_path: Path to the file to extract text from
        provider: VLM provider to use (defaults to VLM_PROVIDER env var)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Extracted text as string
    
    Raises:
        ValueError: If provider is not supported
        FileNotFoundError: If file doesn't exist
        Exception: Provider-specific errors
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get provider configuration
    if isinstance(provider, str):
        try:
            provider = Provider(provider)
        except ValueError:
            raise ValueError(f"Unsupported provider: {provider}")
    
    provider = provider or config.vlm.provider
    provider_config = config.get_provider_config(provider)
    
    # Create provider instance
    if provider == Provider.OLLAMA:
        provider_instance = OllamaProvider(provider_config)
    elif provider == Provider.OPENAI:
        from .providers.openai import OpenAIProvider
        provider_instance = OpenAIProvider(provider_config)
    elif provider == Provider.LOCALAI:
        # TODO: Implement LocalAI provider
        raise NotImplementedError("LocalAI provider not yet implemented")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Check if this is a PDF that might have been processed by PyMuPDF
    if file_path.suffix.upper() == ".PDF":
        from .utils import process_pdf_smart
        result, method = await process_pdf_smart(file_path)
        
        if method == "pymupdf":
            # PyMuPDF already extracted text, return it directly
            return result
        else:
            # VLM processing needed, use the image data
            image_data_list = result
            return await _extract_text_from_images(provider_instance, image_data_list)
    else:
        # Non-PDF file, use normal processing
        from .utils import process_file_for_vlm
        image_data_list = await process_file_for_vlm(file_path)
        return await _extract_text_from_images(provider_instance, image_data_list)


async def _extract_text_from_images(provider_instance, image_data_list: List[bytes]) -> str:
    """Extract text from list of image data using the provider."""
    all_text = []
    for i, image_data in enumerate(image_data_list):
        try:
            page_text = await provider_instance.extract_text_from_image(image_data)
            if page_text.strip():
                if len(image_data_list) > 1:
                    all_text.append(f"Page {i + 1}:\n{page_text}")
                else:
                    all_text.append(page_text)
        except Exception as e:
            if len(image_data_list) > 1:
                all_text.append(f"Page {i + 1}: Error extracting text - {e}")
            else:
                raise e
    
    return "\n\n".join(all_text) if all_text else "No text could be extracted"


async def extract_text_batch(
    file_paths: List[str | Path],
    provider: Optional[str | Provider] = None,
    **kwargs
) -> List[str]:
    """
    Extract text from multiple files concurrently.
    
    Args:
        file_paths: List of file paths to extract text from
        provider: VLM provider to use (defaults to VLM_PROVIDER env var)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        List of extracted text strings
    """
    tasks = [
        extract_text(file_path, provider, **kwargs)
        for file_path in file_paths
    ]
    return await asyncio.gather(*tasks, return_exceptions=True) 