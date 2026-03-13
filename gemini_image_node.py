import os
import warnings
import logging
import random
import json
import base64
import time
import requests

import torch
import numpy as np
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore", message="IMAGE_SAFETY is not a valid FinishReason")

from .image_utils import create_placeholder_image, prepare_batch_images

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# ── Default API configuration ──────────────────────────────────────────────
APIYI_BASE_URL = "https://api.apiyi.com"
DEFAULT_IMAGE_API_KEY = ""

IMAGE_MODELS = [
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
]

# Aspect ratio → imageSize mapping
ASPECT_RATIOS = {
    "1:1": "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "4:3": "4:3",
    "3:4": "3:4",
    "3:2": "3:2",
    "2:3": "2:3",
    "21:9": "21:9",
    "5:4": "5:4",
    "4:5": "4:5",
}

RESOLUTIONS = ["1K", "2K", "4K"]


def _resolve_image_api_key(external_key: str = "") -> str:
    """Return the best available image API key."""
    return DEFAULT_IMAGE_API_KEY


def verify_image_api_key(api_key: str) -> tuple:
    """Verify an image API key against apiyi.com.
    Returns (is_valid: bool, message: str).
    """
    if not api_key:
        return False, "No API key provided."
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.get(
            f"{APIYI_BASE_URL}/v1/models",
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            return True, "Image API key is valid ✅"
        elif resp.status_code in (401, 403):
            return False, f"Invalid API key (HTTP {resp.status_code})"
        else:
            return True, f"API key accepted (HTTP {resp.status_code})"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


class IFGeminiImageGen:
    """
    Image generation node using apiyi.com Google-native generateContent API.
    Receives a text prompt from an upstream node (e.g. IFGeminiPrompt).
    No text input widget — the prompt comes from the left-side 'text' connector.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "model_name": (IMAGE_MODELS, {"default": "gemini-2.5-flash-image"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "use_random_seed": ("BOOLEAN", {"default": False}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 10}),
                "aspect_ratio": (
                    list(ASPECT_RATIOS.keys()),
                    {"default": "1:1"},
                ),
                "resolution": (RESOLUTIONS, {"default": "2K"}),
                "max_images": ("INT", {"default": 4, "min": 1, "max": 16}),
                "external_api_key": ("STRING", {"default": ""}),
                "api_call_delay": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "generate_images"
    CATEGORY = "ImpactFrames💥🎞️/LLM"

    # ── Main execution ─────────────────────────────────────────────────────
    def generate_images(
        self,
        text,
        model_name="gemini-2.5-flash-image",
        images=None,
        seed=0,
        use_random_seed=False,
        batch_count=1,
        aspect_ratio="1:1",
        resolution="2K",
        max_images=4,
        external_api_key="",
        api_call_delay=1.0,
    ):
        """Generate images via apiyi.com Google-native generateContent endpoint."""

        api_key = _resolve_image_api_key(external_api_key)
        if not api_key:
            return (
                "ERROR: No image API key found.",
                create_placeholder_image(),
            )

        prompt = text

        # Seed
        if use_random_seed:
            operation_seed = random.randint(0, 0xFFFFFFFF)
        else:
            operation_seed = seed

        # Timeout mapping based on resolution
        timeout_map = {"1K": 60, "2K": 90, "4K": 180}
        timeout = timeout_map.get(resolution, 90)

        all_image_bytes = []
        all_text = []
        status_lines = []

        for batch_idx in range(batch_count):
            if batch_idx > 0 and api_call_delay > 0:
                time.sleep(api_call_delay)

            current_seed = (operation_seed + batch_idx) % (2**31 - 1)

            # Build the Google-native API URL
            api_url = (
                f"{APIYI_BASE_URL}/v1beta/models/"
                f"{model_name}:generateContent"
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # Build contents with optional reference images
            parts = [{"text": prompt}]

            if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                ref_pil_images = prepare_batch_images(images, max_images, max_size=2048)
                for ref_img in ref_pil_images:
                    buf = BytesIO()
                    ref_img.save(buf, format="PNG")
                    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": b64_data,
                        }
                    })

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {
                        "aspectRatio": ASPECT_RATIOS.get(aspect_ratio, "1:1"),
                        "imageSize": resolution,
                    },
                },
            }

            try:
                logger.info(
                    f"IFGeminiImageGen: Batch {batch_idx+1}/{batch_count}, "
                    f"model={model_name}, seed={current_seed}, "
                    f"aspect={aspect_ratio}, res={resolution}"
                )
                logger.info(f"IFGeminiImageGen: Prompt = {prompt}")
                logger.info(f"IFGeminiImageGen: API URL = {api_url}")

                resp = requests.post(
                    api_url, headers=headers, json=payload, timeout=timeout
                )
                logger.info(f"IFGeminiImageGen: HTTP status = {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()

                # Log response structure for debugging
                logger.info(f"IFGeminiImageGen: Response keys = {list(data.keys())}")
                candidates = data.get("candidates", [])
                logger.info(f"IFGeminiImageGen: candidates count = {len(candidates)}")
                if candidates:
                    candidate_keys = list(candidates[0].keys())
                    finish_reason_log = candidates[0].get("finishReason", "N/A")
                    logger.info(f"IFGeminiImageGen: candidate[0] keys = {candidate_keys}, finishReason = {finish_reason_log}")

                # Parse Google-native response
                batch_images = []
                batch_text = ""

                if not candidates:
                    logger.warning("IFGeminiImageGen: API returned empty candidates")
                else:
                    content = candidates[0].get("content", {}) or {}
                    parts = content.get("parts", []) or []
                    logger.info(f"IFGeminiImageGen: parts count = {len(parts)}")
                    if parts:
                        for i, p in enumerate(parts):
                            p_keys = list(p.keys())
                            logger.info(f"IFGeminiImageGen: part[{i}] keys = {p_keys}")

                    if not parts:
                        logger.warning("IFGeminiImageGen: No parts in response content")
                        # Log raw response for debugging
                        raw_text = json.dumps(data, ensure_ascii=False)[:1000]
                        logger.warning(f"IFGeminiImageGen: Raw response (truncated) = {raw_text}")
                    else:
                        for part in parts:
                            # Image data
                            inline_data = part.get("inlineData", {}) or {}
                            image_base64 = inline_data.get("data", "")

                            if image_base64:
                                try:
                                    img_bytes = base64.b64decode(image_base64)
                                    batch_images.append(img_bytes)
                                    mime_type = inline_data.get("mimeType", "image/png")
                                    logger.info(f"IFGeminiImageGen: Extracted image ({mime_type})")
                                except Exception as dec_err:
                                    logger.error(f"Base64 decode error: {dec_err}")

                            # Text data
                            if "text" in part and part["text"]:
                                batch_text += part["text"] + "\n"

                if batch_images:
                    all_image_bytes.extend(batch_images)
                    status_lines.append(
                        f"Batch {batch_idx+1} (seed {current_seed}): "
                        f"Generated {len(batch_images)} image(s)"
                    )
                else:
                    # Check for safety block
                    finish_reason = ""
                    if candidates:
                        finish_reason = candidates[0].get("finishReason", "")
                    status_lines.append(
                        f"Batch {batch_idx+1} (seed {current_seed}): "
                        f"No images (finishReason={finish_reason})"
                    )
                    if "SAFETY" in finish_reason.upper():
                        status_lines.append(
                            "⚠️ Blocked for safety. Try modifying your prompt."
                        )

                if batch_text.strip():
                    all_text.append(f"Batch {batch_idx+1}:\n{batch_text.strip()}")

            except requests.exceptions.Timeout:
                status_lines.append(
                    f"Batch {batch_idx+1}: Timeout after {timeout}s "
                    f"(try lower resolution)"
                )
            except requests.exceptions.HTTPError as e:
                body = ""
                try:
                    body = resp.text[:300]
                except Exception:
                    pass
                status_lines.append(
                    f"Batch {batch_idx+1}: HTTP error {e}\n{body}"
                )
                logger.error(f"IFGeminiImageGen HTTP error: {e}\n{body}")
            except Exception as e:
                status_lines.append(f"Batch {batch_idx+1}: Error – {str(e)[:200]}")
                logger.error(f"IFGeminiImageGen error: {e}", exc_info=True)

        # ── Convert collected images to tensor ─────────────────────────────
        status_text = "\n".join(status_lines)

        if all_image_bytes:
            try:
                pil_images = []
                for ib in all_image_bytes:
                    try:
                        pil_img = Image.open(BytesIO(ib)).convert("RGB")
                        pil_images.append(pil_img)
                    except Exception as pil_err:
                        logger.error(f"PIL decode error: {pil_err}")

                if not pil_images:
                    raise ValueError("All image bytes failed to decode")

                # Normalise to first image size
                w0, h0 = pil_images[0].size
                for i in range(1, len(pil_images)):
                    if pil_images[i].size != (w0, h0):
                        pil_images[i] = pil_images[i].resize((w0, h0), Image.LANCZOS)

                tensors = []
                for pil_img in pil_images:
                    arr = np.array(pil_img).astype(np.float32) / 255.0
                    tensors.append(torch.from_numpy(arr)[None,])

                image_tensor = torch.cat(tensors, dim=0)

                result_text = (
                    f"Generated {len(pil_images)} image(s) with {model_name}\n"
                    f"Prompt: {prompt}\n"
                    f"Seed: {operation_seed} | Aspect: {aspect_ratio} | Res: {resolution}\n"
                    f"Size: {w0}×{h0}\n"
                )
                if all_text:
                    result_text += "\n--- Model text ---\n" + "\n\n".join(all_text)
                result_text += f"\n\n--- Status ---\n{status_text}"

                return result_text, image_tensor

            except Exception as proc_err:
                return (
                    f"Error processing images: {proc_err}\n\n{status_text}",
                    create_placeholder_image(),
                )
        else:
            return (
                f"No images generated.\n{status_text}",
                create_placeholder_image(),
            )