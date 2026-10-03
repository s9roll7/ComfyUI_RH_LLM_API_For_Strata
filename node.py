from openai import OpenAI

import time
from PIL import Image
import numpy as np
import base64
import os
import io
import requests


def encode_image_b64(image):
    """
    Encode one ComfyUI IMAGE tensor/image array to base64 JPEG.

    The caller is expected to pass a single image with shape HxWxC.
    Original resolution is preserved.
    """
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()

    i = 255.0 * image
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_image_batch_b64(ref_image):
    """
    Encode a ComfyUI IMAGE batch to a list of base64 JPEG strings.

    ComfyUI IMAGE normally has shape:
        [B, H, W, C]

    Every image in the batch is encoded and returned.
    """
    images = ref_image.cpu().numpy()

    encoded = []
    for image in images:
        encoded.append(encode_image_b64(image))

    return encoded


def _get_video_file_path(video):
    """
    Try to extract a filesystem path from a ComfyUI VIDEO object.
    Returns None if it cannot be resolved.
    """

    if hasattr(video, "_VideoFromFile__file"):
        path = getattr(video, "_VideoFromFile__file", None)
        if isinstance(path, str) and os.path.exists(path):
            return path

    if hasattr(video, "get_stream_source"):
        try:
            stream_source = video.get_stream_source()
            if isinstance(stream_source, str) and os.path.exists(stream_source):
                return stream_source
        except Exception:
            pass

    for attr in ("path", "file"):
        if hasattr(video, attr):
            path = getattr(video, attr, None)
            if isinstance(path, str) and os.path.exists(path):
                return path

    return None


def encode_video_b64(video):
    """
    Encode ComfyUI VIDEO object to base64 MP4 bytes.

    No ffmpeg processing, no compression, no resizing.
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

    raise ValueError(
        f"Unable to read video data from object type: {type(video)}"
    )


def unload_model(api_baseurl, api_key):
    """
    Unload the model from Strata.

    Strata v0.1.30:
        POST /unload

    api_baseurl may be:
        http://127.0.0.1:8080/v1
        http://127.0.0.1:8080
    """

    base_url = api_baseurl.strip().rstrip("/")

    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    unload_url = base_url + "/unload"

    headers = {
        "Content-Type": "application/json"
    }

    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    try:
        response = requests.post(
            unload_url,
            headers=headers,
            json={},
            timeout=30
        )

        print(
            f"[RH_LLMAPI] POST {unload_url} "
            f"-> {response.status_code} {response.text}",
            flush=True
        )

        response.raise_for_status()

    except Exception as e:
        print(
            f"[RH_LLMAPI] Failed to unload model: {e}",
            flush=True
        )


class RH_LLMAPI_Node():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8080/v1"
                    },
                    {
                        "multiline": True
                    }
                ),

                "api_key": (
                    "STRING",
                    {
                        "default": "dummy"
                    }
                ),

                "model": (
                    "STRING",
                    {
                        "default": ""
                    }
                ),

                "role": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are a helpful assistant"
                    }
                ),

                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello"
                    }
                ),

                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.6
                    }
                ),

                "thinking_level": (
                    [
                        "none",
                        "low",
                        "medium",
                        "high"
                    ],
                    {
                        "default": "medium"
                    }
                ),

                "seed": (
                    "INT",
                    {
                        "default": 100
                    }
                ),

                "unload_after": (
                    "BOOLEAN",
                    {
                        "default": False
                    }
                ),
            },

            "optional": {
                # IMAGE is a ComfyUI batch input.
                # All images in the batch are sent to the LLM.
                "ref_image_batch": (
                    "IMAGE",
                ),

                "video": (
                    "VIDEO",
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
    )

    RETURN_NAMES = (
        "describe",
    )

    FUNCTION = "rh_run_llmapi"

    CATEGORY = "Runninghub"

    def rh_run_llmapi(
        self,
        api_baseurl,
        api_key,
        model,
        role,
        prompt,
        temperature,
        thinking_level,
        seed,
        unload_after,
        ref_image_batch=None,
        video=None
    ):
        client = OpenAI(
            api_key=api_key,
            base_url=api_baseurl
        )

        # Priority:
        # video > image batch > text

        if video is not None:

            base64_video = encode_video_b64(video)

            messages = [
                {
                    "role": "system",
                    "content": f"{role}"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": (
                                    "data:video/mp4;base64,"
                                    f"{base64_video}"
                                )
                            }
                        },
                    ],
                },
            ]

        elif ref_image_batch is None:

            messages = [
                {
                    "role": "system",
                    "content": f"{role}"
                },
                {
                    "role": "user",
                    "content": f"{prompt}"
                },
            ]

        else:

            # ComfyUI IMAGE is a batch:
            # [B, H, W, C]

            base64_images = encode_image_batch_b64(
                ref_image_batch
            )

            user_content = [
                {
                    "type": "text",
                    "text": f"{prompt}"
                }
            ]

            for base64_image in base64_images:

                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                f"{base64_image}"
                            )
                        },
                    }
                )

            messages = [
                {
                    "role": "system",
                    "content": f"{role}"
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ]

        # ---------------------------------------------------------
        # OpenAI-compatible request
        #
        # Strata-specific:
        #     reasoning_effort
        #
        # none / low / medium / high
        # ---------------------------------------------------------

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=thinking_level
        )

        if completion is not None and hasattr(
            completion,
            "choices"
        ):
            result = completion.choices[0].message.content
        else:
            result = "Error"

        # ---------------------------------------------------------
        # Unload Strata model after inference
        # ---------------------------------------------------------

        if unload_after:
            unload_model(api_baseurl, api_key)

        return (result,)

