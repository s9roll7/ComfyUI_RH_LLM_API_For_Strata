from openai import OpenAI
import time
from PIL import Image
import numpy as np
import base64
import os
import io

def encode_image_b64(ref_image):
    """
    Encode ComfyUI IMAGE tensor to base64 JPEG without resizing.

    Notes:
    - Keep original resolution (no resize).
    - Avoid temporary files (in-memory encoding).
    """
    i = 255.0 * ref_image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    # Use JPEG to match the existing OpenAI-compatible payload mime label.
    img.save(buf, format="JPEG", quality=95, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _get_video_file_path(video):
    """
    Try to extract a filesystem path from a ComfyUI VIDEO object.
    Returns None if it cannot be resolved.
    """
    # VideoFromFile type (private attribute)
    if hasattr(video, "_VideoFromFile__file"):
        path = getattr(video, "_VideoFromFile__file", None)
        if isinstance(path, str) and os.path.exists(path):
            return path

    # Stream-like sources
    if hasattr(video, "get_stream_source"):
        try:
            stream_source = video.get_stream_source()
            if isinstance(stream_source, str) and os.path.exists(stream_source):
                return stream_source
        except Exception:
            pass

    # Common attributes
    for attr in ("path", "file"):
        if hasattr(video, attr):
            path = getattr(video, attr, None)
            if isinstance(path, str) and os.path.exists(path):
                return path

    return None


def encode_video_b64(video):
    """
    Encode ComfyUI VIDEO object to base64 MP4 bytes.

    Notes:
    - No ffmpeg processing, no compression, no resizing.
    - If a file path is available, read it directly.
    - Otherwise, try saving via save_to() to a temp mp4 and read back.
    """
    video_path = _get_video_file_path(video)
    if video_path:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    if hasattr(video, "save_to"):
        temp_path = f"temp_video_{time.time()}.mp4"
        try:
            video.save_to(temp_path)
            with open(temp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    raise ValueError(f"Unable to read video data from object type: {type(video)}")

class RH_LLMAPI_Node():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hello"}),
                "temperature": ("FLOAT", {"default": 0.6}),
                "seed": ("INT", {"default": 100}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "video": ("VIDEO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("describe",)
    FUNCTION = "rh_run_llmapi"
    CATEGORY = "Runninghub"

    def rh_run_llmapi(self, api_baseurl, api_key, model, role, prompt, temperature, seed, ref_image=None, video=None):

        client = OpenAI(api_key=api_key, base_url=api_baseurl)

        # Priority: video > image > text (align with reference node behavior)
        if video is not None:
            base64_video = encode_video_b64(video)
            messages = [
                {'role': 'system', 'content': f'{role}'},
                {'role': 'user',
                 'content': [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{base64_video}"
                            }
                        },
                    ]},
            ]
        elif ref_image is None:
            messages = [
                {'role': 'system', 'content': f'{role}'},
                {'role': 'user', 'content': f'{prompt}'},
            ]
        else:
            base64_image = encode_image_b64(ref_image)
            messages = [
                {'role': 'system', 'content': f'{role}'},
                {'role': 'user', 
                 'content': [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ]},
            ]
        completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        if completion is not None and hasattr(completion, 'choices'):
            prompt = completion.choices[0].message.content
        else:
            prompt = 'Error'
        return (prompt,)