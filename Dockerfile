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
