"""Ollama provider implementation."""

import base64
from pathlib import Path
from typing import Dict, Any, List
import httpx
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """Ollama provider for text extraction from images."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider with configuration."""
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "llava")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)

    async def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image data using Ollama."""
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": "Extract and return all the text visible in this image. Return only the text content, no explanations.",
            "images": [image_base64],
            "stream": False
        }

        # Make request with retry logic
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    return result.get("response", "").strip()
                    
            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise ValueError(f"Model '{self.model}' not found. Please pull it first: ollama pull {self.model}")
                elif e.response.status_code == 500:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"Ollama server error: {e.response.text}")
                    continue
                else:
                    raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to extract text from image: {str(e)}")
                continue

    async def health_check(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False 