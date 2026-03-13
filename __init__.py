from .gemini_prompt_node import IFGeminiPrompt
from .gemini_image_node import IFGeminiImageGen

NODE_CLASS_MAPPINGS = {
    "IFGeminiPromptNode": IFGeminiPrompt,
    "IFGeminiImageGenNode": IFGeminiImageGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IFGeminiPromptNode": "IF Gemini Prompt",
    "IFGeminiImageGenNode": "IF Gemini Image Gen",
}

# Path to web directory relative to this file
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]