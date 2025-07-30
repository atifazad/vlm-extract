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
        # TODO: Implement OpenAI provider
        raise NotImplementedError("OpenAI provider not yet implemented")
    elif provider == Provider.LOCALAI:
        # TODO: Implement LocalAI provider
        raise NotImplementedError("LocalAI provider not yet implemented")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Extract text
    return await provider_instance.extract_text(file_path)


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