# Custom worker-comfyui image: several base models baked in + LoRAs on demand.
#
# THE PATTERN: every base model is one `comfy model download` into the ComfyUI
# folder for its type. LoRAs are the exception -- they are NEVER baked; the
# custom node pulls them by URL at request time into models/loras.
#
#   models/checkpoints      all-in-one image checkpoints (SD1.5, SDXL, Pony, FLUX fp8)
#   models/diffusion_models standalone UNet/diffusion weights (Wan, split FLUX)
#   models/text_encoders    text encoders (umt5 for Wan, t5/clip_l for split FLUX)
#   models/vae              VAEs
#   models/clip_vision      CLIP-vision encoders (some I2V models)
#   models/loras            LoRAs -> filled on demand by the node below
#
# SIZE WARNING: this image is ~64 GB (FLUX 17 + Pony 6.6 + Counterfeit 4 + Wan2.2 I2V ~35.6).
# First cold pull per worker takes a few minutes; cached afterwards. Drop any
# model block you don't need to shrink it.
FROM runpod/worker-comfyui:5.8.6-base

# ===== IMAGE: FLUX.1-dev fp8 all-in-one (~17 GB) -> CheckpointLoaderSimple =====
RUN comfy model download \
    --url "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
    --relative-path models/checkpoints \
    --filename flux1-dev-fp8.safetensors

# ===== IMAGE: Pony Diffusion V6 XL (SDXL/Pony, ~6.6 GB) -> CheckpointLoaderSimple =====
RUN comfy model download \
    --url "https://civitai.com/api/download/models/290640" \
    --relative-path models/checkpoints \
    --filename ponyDiffusionV6XL.safetensors

# ===== IMAGE: Counterfeit-V3.0 (anime SD 1.5, ~4 GB) -> CheckpointLoaderSimple =====
# SD 1.5 base so SD-1.5-only LoRAs work (e.g. anime-background style LoRAs).
RUN comfy model download \
    --url "https://civitai.com/api/download/models/57618" \
    --relative-path models/checkpoints \
    --filename counterfeitV30.safetensors

# ===== VIDEO: Wan 2.2 I2V 14B (fp8 scaled). Two diffusion models + encoder + vae =====
RUN comfy model download \
    --url "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
    --relative-path models/diffusion_models \
    --filename wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
RUN comfy model download \
    --url "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
    --relative-path models/diffusion_models \
    --filename wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
RUN comfy model download \
    --url "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    --relative-path models/text_encoders \
    --filename umt5_xxl_fp8_e4m3fn_scaled.safetensors
RUN comfy model download \
    --url "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    --relative-path models/vae \
    --filename wan_2.1_vae.safetensors

# ===== LoRAs ON DEMAND (any base) =====
# Not baked. This node downloads a LoRA from a URL at request time into
# models/loras (shared by FLUX and SDXL/Pony LoRAs alike) and caches it.
COPY custom_nodes/comfyui-lora-from-url /comfyui/custom_nodes/comfyui-lora-from-url
