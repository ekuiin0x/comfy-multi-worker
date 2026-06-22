"""ComfyUI node: download a LoRA from a URL at request time, then apply it.

Lets the serverless worker accept an arbitrary CivitAI (or any HTTP) LoRA URL in
the workflow instead of needing the file baked into the image. The file is cached
in models/loras by filename, so a warm worker only downloads each LoRA once.

CivitAI auth: pass the API token via the `token` input; it's appended as a
`?token=` query param (CivitAI's documented method). No token header is sent, so
the signed-storage redirect isn't rejected.
"""
from __future__ import annotations

import hashlib
import os
import urllib.parse
import urllib.request

import comfy.sd
import comfy.utils
import folder_paths

_CHUNK = 1 << 20  # 1 MiB
_EXTS = (".safetensors", ".pt", ".ckpt")


def _with_token(url: str, token: str) -> str:
    if not token or "token=" in url:
        return url
    sep = "&" if urllib.parse.urlparse(url).query else "?"
    return f"{url}{sep}token={urllib.parse.quote(token)}"


def _download(url: str, filename: str, token: str) -> str:
    lora_dir = folder_paths.get_folder_paths("loras")[0]
    os.makedirs(lora_dir, exist_ok=True)
    if not filename:
        filename = hashlib.sha1(url.encode()).hexdigest()[:16]
    if not filename.endswith(_EXTS):
        filename += ".safetensors"
    path = os.path.join(lora_dir, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    tmp = path + ".part"
    req = urllib.request.Request(_with_token(url, token), headers={"User-Agent": "comfyui-lora-from-url"})
    with urllib.request.urlopen(req, timeout=900) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            f.write(chunk)
    if os.path.getsize(tmp) == 0:
        os.remove(tmp)
        raise RuntimeError(f"downloaded 0 bytes from {url}")
    os.replace(tmp, path)
    return path


class LoraLoaderFromURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "url": ("STRING", {"default": "", "multiline": False}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
                "token": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, model, clip, url, strength_model, strength_clip, filename="", token=""):
        url = (url or "").strip()
        if not url or (strength_model == 0 and strength_clip == 0):
            return (model, clip)
        path = _download(url, filename.strip(), token.strip())
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)


NODE_CLASS_MAPPINGS = {"LoraLoaderFromURL": LoraLoaderFromURL}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraLoaderFromURL": "Load LoRA from URL"}
