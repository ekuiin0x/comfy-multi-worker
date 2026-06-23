# Custom worker-comfyui image: image base models baked in + LoRAs on demand.
#
# THE PATTERN: every base model is one `comfy model download` into the ComfyUI
# folder for its type. LoRAs are the exception -- they are NEVER baked; the
# custom node pulls them by URL at request time into models/loras.
#
#   models/checkpoints      all-in-one image checkpoints (SD1.5, SDXL, Pony, FLUX fp8)
#   models/loras            LoRAs -> filled on demand by the node below
#
# SIZE: ~30 GB (FLUX 17 + Pony 6.6 + Counterfeit 4 + RealisticVision 2). Image-only
# worker. (Wan2.2 I2V video models were dropped to keep builds/rollouts fast --
# re-add a diffusion_models block if video is needed again.)
FROM runpod/worker-comfyui:5.8.6-base

# ===== FLUX.1-dev fp8 all-in-one (~17 GB) -> CheckpointLoaderSimple =====
RUN comfy model download \
    --url "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
    --relative-path models/checkpoints \
    --filename flux1-dev-fp8.safetensors

# ===== Pony Diffusion V6 XL (SDXL/Pony, ~6.6 GB) -> CheckpointLoaderSimple =====
RUN comfy model download \
    --url "https://civitai.com/api/download/models/290640" \
    --relative-path models/checkpoints \
    --filename ponyDiffusionV6XL.safetensors

# ===== Counterfeit-V3.0 (anime SD 1.5, ~4 GB) -> CheckpointLoaderSimple =====
# SD 1.5 base for anime-style SD-1.5 LoRAs.
RUN comfy model download \
    --url "https://civitai.com/api/download/models/57618" \
    --relative-path models/checkpoints \
    --filename counterfeitV30.safetensors

# ===== Realistic Vision V6.0 B1 (photoreal SD 1.5, ~2 GB) -> CheckpointLoaderSimple =====
# Realistic SD 1.5 base so SD-1.5 LoRAs render photoreal humans (Counterfeit is anime-only).
RUN comfy model download \
    --url "https://civitai.com/api/download/models/245598" \
    --relative-path models/checkpoints \
    --filename realisticVisionV60B1.safetensors

# ===== LoRAs ON DEMAND (any base) =====
# Not baked. This node downloads a LoRA from a URL at request time into
# models/loras (shared by FLUX and SDXL/Pony LoRAs alike) and caches it.
COPY custom_nodes/comfyui-lora-from-url /comfyui/custom_nodes/comfyui-lora-from-url

# =============================================================================
# IDENTITY-PRESERVING EDIT PIPELINE -- phase 1: InstantID core
# Appended at the END so the cached base-model layers above never rebuild.
# Locks a model's face as conditioning so she can be regenerated in new poses/
# scenes/outfits WITHOUT drifting identity -- the thing pure img2img can't do.
# (Phase 2 adds OpenPose ControlNet + FaceDetailer once this core is verified.)
# =============================================================================

# Build tools: insightface ships no cp312 wheel -> it compiles from source.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*

# cython+numpy MUST precede insightface (its setup.py imports them at build).
RUN uv pip install --no-cache cython numpy

# InstantID custom node + its deps (insightface, onnxruntime, onnxruntime-gpu).
RUN git clone --depth=1 https://github.com/cubiq/ComfyUI_InstantID /comfyui/custom_nodes/ComfyUI_InstantID && \
    uv pip install --no-cache -r /comfyui/custom_nodes/ComfyUI_InstantID/requirements.txt

# InstantID IP-Adapter (face conditioning, ~1.69 GB) -> models/instantid
RUN comfy model download \
    --url "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin" \
    --relative-path models/instantid \
    --filename ip-adapter.bin

# InstantID ControlNet (~2.5 GB) -> models/controlnet
RUN comfy model download \
    --url "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors" \
    --relative-path models/controlnet \
    --filename instantid_controlnet.safetensors

# antelopev2 face-analysis models -> EXACT path models/insightface/models/antelopev2/
# (this exact subpath is the #1 InstantID footgun; the old antelopev2.zip 404s.)
RUN for f in 1k3d68 2d106det genderage glintr100 scrfd_10g_bnkps; do \
      comfy model download \
        --url "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/$f.onnx" \
        --relative-path models/insightface/models/antelopev2 \
        --filename "$f.onnx"; \
    done

# RealVisXL V5.0 photoreal SDXL base (HF, NO CivitAI token, ~6.5 GB).
# Chosen over a token-gated CivitAI checkpoint to delete the build's #1 failure
# point; photoreal SDXL gives InstantID strong identity fidelity.
RUN comfy model download \
    --url "https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors" \
    --relative-path models/checkpoints \
    --filename realvisxlV50.safetensors

# 4x-UltraSharp upscaler (~67 MB) -> models/upscale_models
RUN comfy model download \
    --url "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" \
    --relative-path models/upscale_models \
    --filename 4x-UltraSharp.pth
