import os
import warnings
import logging
import random
import json
import base64
import requests
from io import BytesIO

import torch

warnings.filterwarnings("ignore", message="IMAGE_SAFETY is not a valid FinishReason")

from .image_utils import prepare_batch_images

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# ── Default API configuration ──────────────────────────────────────────────
APIYI_BASE_URL = "https://api.apiyi.com"
DEFAULT_TEXT_API_KEY = ""

PROMPT_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-pro-preview-thinking",
    "gemini-3.1-pro-preview-thinking",
]

# ── Mode-specific system prompts ──────────────────────────────────────────
# Automatically selected based on whether images are provided
MODE_SYSTEM_PROMPTS = {
    "generate": (
        "You are an expert image prompt engineer. Your task is to take the user's description "
        "and transform it into a highly detailed, vivid image generation prompt. "
        "Focus on: subject details (appearance, clothing, expression, pose), "
        "environment/background (setting, atmosphere, lighting conditions), "
        "composition (camera angle, framing, depth of field), "
        "style (photorealistic, illustration, painting, etc.), "
        "and mood/color palette. "
        "Output ONLY the final image prompt text, no explanations or commentary."
    ),
    "edit": (
        "You are an expert multimodal image prompt engineer. "
        "The user has provided one or more reference images along with a text description. "
        "Your workflow:\n"
        "1. First, carefully analyze each provided image — identify the subject (face, body, identity features), "
        "clothing, accessories, pose, expression, background, lighting, style, and any notable details.\n"
        "2. Then, combine your image analysis with the user's text instructions to produce a single, "
        "comprehensive image generation/editing prompt.\n"
        "3. The output prompt must clearly specify:\n"
        "   - What to preserve from the reference images (identity, specific features, pose, etc.)\n"
        "   - What to change or create based on the user's instructions (background, outfit, style, etc.)\n"
        "   - Detailed visual descriptions for the final desired result\n"
        "   - Lighting, composition, and style directives\n"
        "Output ONLY the final image prompt text, no explanations or commentary."
    ),
}


def _resolve_text_api_key(external_key: str = "") -> str:
    """Return the best available text API key."""
    return DEFAULT_TEXT_API_KEY


def verify_text_api_key(api_key: str) -> tuple:
    """Verify an API key against apiyi.com /v1/models endpoint.
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
            return True, "Text API key is valid ✅"
        elif resp.status_code in (401, 403):
            return False, f"Invalid API key (HTTP {resp.status_code})"
        else:
            return True, f"API key accepted (HTTP {resp.status_code})"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


class IFGeminiPrompt:
    """
    A prompt node that can optionally accept images for multimodal analysis.
    Calls the apiyi.com OpenAI-compatible chat completions endpoint.
    Outputs text and optionally passes through input images unchanged.
    """

    def __init__(self):
        pass

    # ── ComfyUI Node Definition ────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Create a vivid word-picture representation of this scene, "
                            "include elements that characterize the subject, costume, "
                            "prop elements, the action, the background, layout and "
                            "composition elements present on the scene, be sure to "
                            "mention the style and mood of the scene."
                        ),
                    },
                ),
                "model_name": (PROMPT_MODELS, {"default": "gemini-2.5-flash"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "max_images": ("INT", {"default": 4, "min": 1, "max": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "use_random_seed": ("BOOLEAN", {"default": False}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
                "external_api_key": ("STRING", {"default": ""}),
                "structured_output": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "images")
    FUNCTION = "generate_text"
    CATEGORY = "ImpactFrames💥🎞️/LLM"

    # ── Main execution ─────────────────────────────────────────────────────
    def generate_text(
        self,
        prompt,
        model_name="gemini-2.5-flash",
        temperature=0.8,
        images=None,
        max_images=4,
        seed=0,
        use_random_seed=False,
        max_output_tokens=8192,
        external_api_key="",
        structured_output=False,
    ):
        """Call apiyi.com /v1/chat/completions (OpenAI-compatible) with optional images."""

        api_key = _resolve_text_api_key(external_api_key)
        if not api_key:
            return self._make_result("ERROR: No text API key found.", images)

        # Seed
        if use_random_seed:
            operation_seed = random.randint(0, 0xFFFFFFFF)
        else:
            operation_seed = seed

        # Build request
        api_url = f"{APIYI_BASE_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Build user content: text + optional images (OpenAI multimodal format)
        has_images = (
            images is not None
            and isinstance(images, torch.Tensor)
            and images.nelement() > 0
        )

        if has_images:
            # Convert images to base64 for multimodal API call
            pil_images = prepare_batch_images(images, max_images, max_size=1024)
            logger.info(f"IFGeminiPrompt: Including {len(pil_images)} image(s) in API request")

            # OpenAI-compatible multimodal content format
            user_content = [{"type": "text", "text": prompt}]
            for idx, pil_img in enumerate(pil_images):
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_data}",
                    },
                })
                logger.info(f"IFGeminiPrompt: Image {idx+1} size: {pil_img.size}")
        else:
            user_content = prompt

        # Auto-select mode based on whether images are provided
        mode = "edit" if has_images else "generate"
        system_prompt = MODE_SYSTEM_PROMPTS[mode]
        logger.info(f"IFGeminiPrompt: Auto-selected mode = {mode}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if operation_seed != 0:
            payload["seed"] = operation_seed
        if structured_output:
            payload["response_format"] = {"type": "json_object"}

        try:
            logger.info(
                f"IFGeminiPrompt: Calling {api_url} with model={model_name}, "
                f"mode={mode}, temp={temperature}, seed={operation_seed}, "
                f"images={'yes ('+str(len(pil_images))+')' if has_images else 'no'}"
            )
            resp = requests.post(api_url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            # Extract text from OpenAI-compatible response
            choices = data.get("choices", [])
            if not choices:
                return self._make_result("ERROR: No choices returned from API.", images)

            content = choices[0].get("message", {}).get("content", "")

            if structured_output:
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    content = (
                        f"Warning: Requested JSON output but received non-JSON:\n\n{content}"
                    )

            usage = data.get("usage", {})
            logger.info(
                f"IFGeminiPrompt: Success. "
                f"Tokens – prompt: {usage.get('prompt_tokens', '?')}, "
                f"completion: {usage.get('completion_tokens', '?')}"
            )
            return self._make_result(content, images)

        except requests.exceptions.HTTPError as e:
            body = ""
            try:
                body = resp.text[:500]
            except Exception:
                pass
            error_msg = f"HTTP error: {e}\n{body}"
            logger.error(f"IFGeminiPrompt: {error_msg}")
            return self._make_result(f"ERROR: {error_msg}", images)

        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            logger.error(f"IFGeminiPrompt: {error_msg}", exc_info=True)
            return self._make_result(f"ERROR: {error_msg}", images)

    @staticmethod
    def _make_result(text, images):
        """Build the return tuple, passing through images if available."""
        if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
            return (text, images)
        else:
            # Return an empty 1x1 placeholder so the IMAGE output is always valid
            import numpy as np
            from PIL import Image as PILImage
            placeholder = PILImage.new("RGB", (1, 1), color=(0, 0, 0))
            arr = np.array(placeholder).astype(np.float32) / 255.0
            empty_tensor = torch.from_numpy(arr)[None,]
            return (text, empty_tensor)