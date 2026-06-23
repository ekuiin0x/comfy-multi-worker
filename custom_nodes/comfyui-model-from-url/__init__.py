"""ComfyUI node: download a diffusion model (FLUX/SD UNet) from a URL at request
time, then load it as a MODEL.

Lets the serverless worker run an arbitrary CivitAI (or any HTTP) checkpoint URL
in the workflow instead of baking every checkpoint into the image. The file is
cached in models/diffusion_models by filename, so a warm worker only downloads
each model once. Mirrors comfyui-lora-from-url so the token never enters the
image: pass it via the `token` input and it is appended as `?token=` at request
time (CivitAI's documented method; no header, so the signed-storage redirect is
not rejected).

Use for UNet-only checkpoints (FLUX transformer, split SDXL UNet) -> pair with a
DualCLIPLoader + VAELoader pointing at baked text-encoders/VAE.
"""
from __future__ import annotations

import hashlib
import os
import urllib.parse
import urllib.request

import torch

import comfy.sd
import folder_paths

_CHUNK = 1 << 20  # 1 MiB
_EXTS = (".safetensors", ".pt", ".ckpt", ".sft")


def _with_token(url: str, token: str) -> str:
    if not token or "token=" in url:
        return url
    sep = "&" if urllib.parse.urlparse(url).query else "?"
    return f"{url}{sep}token={urllib.parse.quote(token)}"


def _download(url: str, filename: str) -> str:
    dst_dir = folder_paths.get_folder_paths("diffusion_models")[0]
    os.makedirs(dst_dir, exist_ok=True)
    if not filename:
        filename = hashlib.sha1(url.encode()).hexdigest()[:16]
    if not filename.endswith(_EXTS):
        filename += ".safetensors"
    path = os.path.join(dst_dir, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "comfyui-model-from-url"})
    with urllib.request.urlopen(req, timeout=1800) as resp, open(tmp, "wb") as f:
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


class UNetLoaderFromURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
                "token": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, url, weight_dtype, filename="", token=""):
        url = _with_token((url or "").strip(), token.strip())
        if not url:
            raise RuntimeError("UNetLoaderFromURL: empty url")
        path = _download(url, filename.strip())
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        return (model,)


NODE_CLASS_MAPPINGS = {"UNetLoaderFromURL": UNetLoaderFromURL}
NODE_DISPLAY_NAME_MAPPINGS = {"UNetLoaderFromURL": "Load UNet/Checkpoint from URL"}
