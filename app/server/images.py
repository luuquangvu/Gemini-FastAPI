"""OpenAI-compatible Image Generation router (/v1/images/generations)."""

import base64
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.server.chat import _resolve_model_name
from app.server.middleware import (
    get_media_store_dir,
    get_media_token,
    get_temp_dir,
    verify_api_key,
)
from app.services import GeminiClientPool

router = APIRouter()

# Official Imagen aspect ratios supported by Google AI Studio & Imagen 3
VALID_ASPECT_RATIOS = (
    "1:1",
    "9:16",
    "16:9",
    "3:4",
    "4:3",
    "3:2",
    "2:3",
    "5:4",
    "4:5",
    "21:9",
    "4:1",
    "1:4",
    "8:1",
    "1:8",
)


class CreateImageRequest(BaseModel):
    """Request payload for image generation."""

    prompt: str = Field(..., max_length=32000, description="Prompt text for image generation")
    model: str | None = Field(default="gemini-flash", description="Model identifier")
    n: int | None = Field(default=1, ge=1, le=10, description="Number of images to generate")
    size: str | None = Field(
        default="1:1",
        description=(
            "Aspect ratio or resolution: 1:1, 16:9, 9:16, 21:9, 4:3, 3:4, 3:2, 2:3, "
            "5:4, 4:5, 4:1, 1:4, 8:1, 1:8, or standard OpenAI sizes"
        ),
    )
    response_format: Literal["url", "b64_json"] | None = Field(
        default="b64_json",
        description="The format in which the generated images are returned",
    )
    quality: Literal["standard", "hd", "1k", "2k", "4k", "auto"] | None = Field(
        default="auto",
        description="Image resolution/quality preference",
    )
    style: Literal["vivid", "natural"] | None = Field(default=None)
    user: str | None = Field(default=None)


class ImageData(BaseModel):
    """Generated image object in OpenAI-compatible response format."""

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImagesResponse(BaseModel):
    """OpenAI-compatible Images response."""

    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageData] = Field(default_factory=list)


def _resolve_aspect_ratio(size: str | None) -> str | None:
    """Map user-provided size or OpenAI resolution string to Imagen aspect ratio."""
    if not size:
        return None
    s = size.strip().lower()

    # Direct match with official aspect ratios
    for ar in VALID_ASPECT_RATIOS:
        if s == ar.lower():
            return ar

    # Standard OpenAI pixel dimension mappings
    mapping = {
        "1024x1024": "1:1",
        "512x512": "1:1",
        "2048x2048": "1:1",
        "1792x1024": "16:9",
        "1920x1080": "16:9",
        "1280x720": "16:9",
        "1024x1792": "9:16",
        "1080x1920": "9:16",
        "720x1280": "9:16",
        "1024x768": "4:3",
        "800x600": "4:3",
        "768x1024": "3:4",
        "600x800": "3:4",
        "2560x1080": "21:9",
        "3440x1440": "21:9",
    }
    return mapping.get(s)


@router.post(
    "/v1/images/generations",
    response_model=ImagesResponse,
    response_model_exclude_none=True,
    tags=["Images"],
)
async def create_image(
    request: CreateImageRequest,
    raw_request: Request,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    """Handle OpenAI-compatible Image Generation API requests."""
    pool = GeminiClientPool()
    target_model = request.model or "gemini-flash"
    try:
        resolved_model = _resolve_model_name(pool, target_model)
    except ValueError as exc:
        if request.model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        resolved_model = "gemini-3.8-flash"
    except Exception:
        resolved_model = "gemini-3.8-flash"

    try:
        wrapper = await pool.acquire(require_account=True)
        session = wrapper.start_chat(model=wrapper.usable_model(resolved_model))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to acquire Gemini client: {e}",
        ) from e

    prompt_clean = request.prompt.strip()
    prompt_lower = prompt_clean.lower()

    # Base generation command (keep user prompt natural without injecting rigid wrappers)
    trigger_words = (
        "generate",
        "create",
        "draw",
        "make an image",
        "picture of",
        "photo of",
        "disegna",
        "crea",
        "genera",
    )
    if any(prompt_lower.startswith(w) for w in trigger_words):
        generation_prompt = prompt_clean
    else:
        generation_prompt = f"Generate an image of {prompt_clean}"

    # Aspect Ratio handling: append clean aspect ratio directive if requested
    ar = _resolve_aspect_ratio(request.size)
    if ar and f"{ar} aspect ratio" not in prompt_lower and f"aspect ratio {ar}" not in prompt_lower:
        generation_prompt = f"{generation_prompt} in {ar} aspect ratio"

    # Optional resolution preference (1k, 2k, 4k) without polluting artistic style
    if request.quality in ("1k", "2k", "4k") and request.quality not in prompt_lower:
        generation_prompt = f"{generation_prompt} in {request.quality} resolution"

    # OpenAI standard style param (only if explicitly requested by client)
    if request.style == "vivid" and "vivid" not in prompt_lower:
        generation_prompt = f"{generation_prompt}, vivid style"
    elif request.style == "natural" and "natural" not in prompt_lower:
        generation_prompt = f"{generation_prompt}, natural style"

    try:
        resp = await session.send_message(generation_prompt)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error during generation: {e}",
        ) from e

    images = resp.images or []
    if not images:
        fallback_text = (resp.text.strip()[:200]) if resp.text else ""
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The model did not return a generated image. Response text: {fallback_text}",
        )

    base_url = str(raw_request.base_url)
    media_store = get_media_store_dir()
    image_data_list: list[ImageData] = []

    # Internal curl_cffi session for downloading image bytes
    http_session = getattr(wrapper, "client", None)

    for img in images[: (request.n or 1)]:
        try:
            saved_path = await img.save(path=str(media_store), client=http_session)
            if not saved_path:
                continue

            saved_file = Path(saved_path)
            fname = saved_file.name
            raw_bytes = saved_file.read_bytes()

            if request.response_format == "url":
                token = get_media_token(fname)
                img_url = f"{base_url}media/{fname}?token={token}"
                image_data_list.append(ImageData(url=img_url, revised_prompt=generation_prompt))
            else:
                b64_str = base64.b64encode(raw_bytes).decode("utf-8")
                image_data_list.append(
                    ImageData(b64_json=b64_str, revised_prompt=generation_prompt)
                )
        except Exception:
            if getattr(img, "url", None):
                image_data_list.append(ImageData(url=img.url, revised_prompt=generation_prompt))

    if not image_data_list:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to persist or retrieve generated image data",
        )

    return ImagesResponse(created=int(time.time()), data=image_data_list)
