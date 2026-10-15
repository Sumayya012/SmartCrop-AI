"""Safe image loading and model-input preparation."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS  # decompression-bomb guard (raises above 2x this)

_RESAMPLE = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR}


class ImageError(ValueError):
    """User-facing image problem."""


@dataclass
class LoadedImage:
    image: Image.Image
    warnings: list[str] = field(default_factory=list)


def load_image(data: bytes) -> LoadedImage:
    """Decode upload bytes into an RGB image with basic validation and quality warnings."""
    if not data:
        raise ImageError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ImageError(f"Image is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if img.format not in {"JPEG", "PNG", "WEBP"}:
                raise ImageError("Unsupported format. Please upload a JPG, PNG, or WEBP image.")
            if img.width * img.height > config.MAX_IMAGE_PIXELS:
                raise ImageError("Image has too many pixels. Please upload a smaller image.")
            rgb = ImageOps.exif_transpose(img).convert("RGB")
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise ImageError("We couldn't read that image. Please upload a valid JPG, PNG, or WEBP file.") from exc

    warnings: list[str] = []
    if min(rgb.size) < config.MIN_IMAGE_SIDE:
        warnings.append("The image is very small; predictions may be unreliable.")
    brightness = float(np.asarray(rgb.convert("L"), dtype=np.float32).mean())
    if brightness < 40:
        warnings.append("The image looks very dark; use better lighting if possible.")
    elif brightness > 225:
        warnings.append("The image looks overexposed; avoid direct glare if possible.")
    return LoadedImage(rgb, warnings)


def to_model_batch(image: Image.Image) -> np.ndarray:
    """RGB image -> float32 array (1, H, W, 3) with raw 0-255 values.

    The saved model scales pixels to [-1, 1] internally (MobileNetV2 preprocess_input), so no
    scaling is applied here. Dividing by 255 as well would silently degrade predictions.
    """
    resized = image.convert("RGB").resize(config.IMAGE_SIZE, _RESAMPLE.get(config.RESIZE_METHOD, Image.Resampling.NEAREST))
    return np.asarray(resized, dtype=np.float32)[None, ...]
