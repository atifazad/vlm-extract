"""OpenAI provider implementation."""

import base64
from pathlib import Path
from typing import Dict, Any
import httpx
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI provider for text extraction from images."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider with configuration."""
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-4o")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

    async def extract_text(self, file_path: Path) -> str:
        """Extract text from a file using OpenAI."""
        from ..utils import process_file_for_vlm
        
        # Process file to get image data
        image_data_list = await process_file_for_vlm(file_path)
        
        # Extract text from each image
        all_text = []
        for i, image_data in enumerate(image_data_list):
            try:
                page_text = await self.extract_text_from_image(image_data)
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

    async def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from image data using OpenAI."""
        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Prepare request payload for OpenAI Vision API
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract and return all the text visible in this image. Return only the text content, no explanations."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

        # Make request with retry logic
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content.strip()
                    
            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    raise TimeoutError(f"OpenAI request timed out after {self.timeout}s")
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise ValueError("Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable.")
                elif e.response.status_code == 429:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError("OpenAI rate limit exceeded. Please try again later.")
                    continue
                elif e.response.status_code == 400:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", "Bad request")
                    raise ValueError(f"OpenAI API error: {error_msg}")
                else:
                    raise RuntimeError(f"OpenAI API error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to extract text from image: {str(e)}")
                continue

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception:
            return False 