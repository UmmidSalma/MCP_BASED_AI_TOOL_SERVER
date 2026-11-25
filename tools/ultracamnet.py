"""Local image captioning behind the existing ultracamnet_predict tool."""
import asyncio
import base64
import binascii
import io
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from security.validation import validate_image_file

load_dotenv()

_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_BASE64_LENGTH = 4 * ((_MAX_IMAGE_BYTES + 2) // 3)
_DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"
_model: Any = None
_processor: Any = None
_loaded_model_name: str | None = None
_model_lock = threading.Lock()


def _load_model() -> tuple[Any, Any]:
    """Load and cache the configured captioning model and processor."""
    global _model, _processor, _loaded_model_name
    model_name = os.environ.get("IMAGE_CAPTION_MODEL", _DEFAULT_MODEL).strip()
    if not model_name:
        raise ValueError("IMAGE_CAPTION_MODEL must name a model or local model directory.")

    with _model_lock:
        if _model is not None and _loaded_model_name == model_name:
            return _processor, _model

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            missing = exc.name or "torch or transformers"
            raise RuntimeError(
                "BLIP image captioning dependencies are unavailable in this MCP "
                f"server's Python environment (missing {missing!r}). Install them "
                "there with `python -m pip install -r requirements-ml.txt`."
            ) from exc

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to("cpu")
        model.eval()
        _processor = processor
        _model = model
        _loaded_model_name = model_name
        return processor, model


def _decode_image(image_base64: str) -> bytes:
    if len(image_base64) > _MAX_BASE64_LENGTH:
        raise ValueError("Image exceeds the 5 MB size limit.")
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("image_base64 must contain valid standard base64 data.") from exc
    if not image_bytes:
        raise ValueError("image_base64 decoded to an empty file.")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 5 MB size limit.")
    return image_bytes


def _generate_caption(image: Any) -> dict[str, str]:
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(generated[0], skip_special_tokens=True).strip()
    if not caption:
        raise RuntimeError("The image captioning model returned an empty caption.")
    return {"caption": caption}


async def predict_image(
    image_path: str | None = None,
    image_base64: str | None = None,
) -> dict[str, str]:
    """Generate a caption for exactly one local image path or base64 image."""
    if (image_path is None) == (image_base64 is None):
        return {"error": "Provide exactly one of image_path or image_base64."}
    try:
        from PIL import Image, UnidentifiedImageError

        image_bytes = (
            validate_image_file(image_path).read_bytes()
            if image_path is not None
            else _decode_image(image_base64)
        )
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb_image = image.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("The supplied data is not a supported image.") from exc

        return await asyncio.to_thread(_generate_caption, rgb_image)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        return {"error": str(exc)}
    except ImportError as exc:
        return {"error": f"Image captioning dependency is unavailable: {exc.name}."}
    except Exception as exc:
        return {"error": f"Image captioning failed: {exc}"}
