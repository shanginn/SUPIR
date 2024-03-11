import os
import subprocess
import time

# Constants
CACHE_DIR = os.path.abspath("weights")
MODEL_CACHE = f"{CACHE_DIR}/AIGC_pretrain"

os.environ["HF_HOME"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

# URLs for model weights
WEIGHTS_URLS = {
    "SUPIR_v0Q": "https://weights.replicate.delivery/default/SUPIR-v0Q.ckpt",
    "LLAVA": "https://weights.replicate.delivery/default/llava-v1.5-13b.tar",
    "SDXL": "https://weights.replicate.delivery/default/stable-diffusion-xl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors",
    "SDXL_CLIP1": "https://weights.replicate.delivery/default/clip-vit-large-patch14.tar",
    "SDXL_CLIP2": "https://weights.replicate.delivery/default/CLIP-ViT-bigG-14-laion2B-39B-b160k.tar",
}

# Paths for model weights
WEIGHTS_PATHS = {
    "SUPIR_v0Q": f"{MODEL_CACHE}/SUPIR_cache/SUPIR-v0Q.ckpt",
    "SDXL": f"{MODEL_CACHE}/SDXL_cache/sd_xl_base_1.0_0.9vae.safetensors",
    "SDXL_CLIP1": f"{MODEL_CACHE}/clip-vit-large-patch14",
    "SDXL_CLIP2": f"{MODEL_CACHE}/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin",
}


def download(url: str, dest: str):
    start = time.time()
    print(f"Downloading URL: {url}")
    print(f"Destination: {dest}")
    args = ["pget"]
    if url.endswith((".tar", ".zip")):
        args.append("-x")
    subprocess.check_call(args + [url, dest], close_fds=False)
    print(f"Download completed in {time.time() - start:.2f} seconds")


def download_weights():
    for model_dir in [
        MODEL_CACHE,
        f"{MODEL_CACHE}/SUPIR_cache",
        f"{MODEL_CACHE}/SDXL_cache",
        f"{MODEL_CACHE}/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ]:
        os.makedirs(model_dir, exist_ok=True)

    for model_name, url in WEIGHTS_URLS.items():
        path = WEIGHTS_PATHS[model_name]
        if not os.path.exists(path):
            download(url, path)

