# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import copy
import os
import subprocess
import time
from PIL import Image
from cog import BasePredictor, Input, Path
import torch
import logging

from SUPIR.util import (
    create_SUPIR_model,
    PIL2Tensor,
    Tensor2PIL,
    convert_dtype,
)
from llava.llava_agent import LLavaAgent
import CKPT_PTH

logging.basicConfig(
    format="(%(asctime)s) %(name)s:%(lineno)d [%(levelname)s] | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


SUPIR_v0Q_URL = "https://weights.replicate.delivery/default/SUPIR-v0Q.ckpt"
SUPIR_v0F_URL = "https://weights.replicate.delivery/default/SUPIR-v0F.ckpt"
LLAVA_URL = "https://weights.replicate.delivery/default/llava-v1.5-13b.tar"
LLAVA_CLIP_URL = (
    "https://weights.replicate.delivery/default/clip-vit-large-patch14-336.tar"
)
SDXL_URL = "https://weights.replicate.delivery/default/stable-diffusion-xl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors"
SDXL_CLIP1_URL = "https://weights.replicate.delivery/default/clip-vit-large-patch14.tar"
SDXL_CLIP2_URL = (
    "https://weights.replicate.delivery/default/CLIP-ViT-bigG-14-laion2B-39B-b160k.tar"
)

MODEL_CACHE = "/opt/data/private/AIGC_pretrain/"  # Follow the default in CKPT_PTH.py
LLAVA_CLIP_PATH = CKPT_PTH.LLAVA_CLIP_PATH
LLAVA_MODEL_PATH = CKPT_PTH.LLAVA_MODEL_PATH
SDXL_CLIP1_PATH = CKPT_PTH.SDXL_CLIP1_PATH
SDXL_CLIP2_CACHE = f"{MODEL_CACHE}/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k"
SDXL_CKPT = f"{MODEL_CACHE}/SDXL_cache/sd_xl_base_1.0_0.9vae.safetensors"
SUPIR_CKPT_F = f"{MODEL_CACHE}/SUPIR_cache/SUPIR-v0F.ckpt"
SUPIR_CKPT_Q = f"{MODEL_CACHE}/SUPIR_cache/SUPIR-v0Q.ckpt"


def download_weights(url, dest, extract=True):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    args = ["pget"]
    if extract:
        args.append("-x")
    subprocess.check_call(args + [url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    model: torch.nn.Module
    llava_agent: LLavaAgent
    supir_device = "cuda:0"
    llava_device = "cuda:0"

    def setup(self) -> None:
        for model_dir in [
            MODEL_CACHE,
            f"{MODEL_CACHE}/SUPIR_cache",
            f"{MODEL_CACHE}/SDXL_cache",
        ]:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

        if not os.path.exists(SUPIR_CKPT_Q):
            download_weights(SUPIR_v0Q_URL, SUPIR_CKPT_Q, extract=False)
        if not os.path.exists(SUPIR_CKPT_F):
            download_weights(SUPIR_v0F_URL, SUPIR_CKPT_F, extract=False)
        if not os.path.exists(LLAVA_MODEL_PATH):
            download_weights(LLAVA_URL, LLAVA_MODEL_PATH)
        if not os.path.exists(LLAVA_CLIP_PATH):
            download_weights(LLAVA_CLIP_URL, LLAVA_CLIP_PATH)
        if not os.path.exists(SDXL_CLIP1_PATH):
            download_weights(SDXL_CLIP1_URL, SDXL_CLIP1_PATH)
        if not os.path.exists(SDXL_CKPT):
            download_weights(SDXL_URL, SDXL_CKPT, extract=False)
        if not os.path.exists(SDXL_CKPT):
            download_weights(SDXL_CLIP2_URL, SDXL_CKPT)

        # # load LLaVA
        self.llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=self.llava_device, load_8bit=True, load_4bit=False)

        self.model = create_SUPIR_model("options/SUPIR_v0.yaml", SUPIR_sign="Q")
        self.model.half()
        self.model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
        self.model.ae_dtype = convert_dtype("bf16")
        self.model.model.dtype = convert_dtype("bf16")
        self.model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(self.model.first_stage_model.denoise_encoder)

        self.model.to(self.supir_device)

    def predict(
        self,
        image: Path = Input(description="Low quality input image."),
        upscale: int = Input(
            description="Upsampling ratio of given inputs.",
            default=2
        ),
        min_size: float = Input(
            description="Minimum resolution of output images.", default=1024
        ),
        edm_steps: int = Input(
            description="Number of steps for EDM Sampling Schedule.",
            ge=1,
            le=500,
            default=50,
        ),
        use_llava: bool = Input(
            description="Use LLaVA model to get captions.", default=True
        ),
        a_prompt: str = Input(
            description="Additive positive prompt for the inputs.",
            default="hyper detailed, maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect",
        ),
        n_prompt: str = Input(
            description="Negative prompt for the inputs.",
            default="blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth",
        ),
        color_fix_type: str = Input(
            description="Color Fixing Type..",
            choices=["None", "AdaIn", "Wavelet"],
            default="Wavelet",
        ),
        s_stage1: int = Input(
            description="Control Strength of Stage1 (negative means invalid).",
            default=-1,
        ),
        s_churn: float = Input(
            description="Original churn hy-param of EDM.", default=5
        ),
        s_noise: float = Input(
            description="Original noise hy-param of EDM.", default=1.003
        ),
        s_cfg: float = Input(
            description=" Classifier-free guidance scale for prompts.",
            ge=1,
            le=20,
            default=7.5,
        ),
        s_stage2: float = Input(description="Control Strength of Stage2.", default=1.0),
        linear_CFG: bool = Input(
            description="Linearly (with sigma) increase CFG from 'spt_linear_CFG' to s_cfg.",
            default=False,
        ),
        linear_s_stage2: bool = Input(
            description="Linearly (with sigma) increase s_stage2 from 'spt_linear_s_stage2' to s_stage2.",
            default=False,
        ),
        spt_linear_CFG: float = Input(
            description="Start point of linearly increasing CFG.", default=1.0
        ),
        spt_linear_s_stage2: float = Input(
            description="Start point of linearly increasing s_stage2.", default=0.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        logger.info(f"GPU memory usage at the start: {torch.cuda.memory_allocated(self.supir_device) / 1024**3:.2f} GB")

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        model = self.model

        lq_img = Image.open(str(image))
        lq_img, h0, w0 = PIL2Tensor(lq_img, upsacle=upscale, min_size=min_size)
        lq_img = lq_img.unsqueeze(0).to(self.supir_device)[:, :3, :, :]

        # step 2: LLaVA
        captions = [""]
        if use_llava:
            start = time.time()
            # step 1: Pre-denoise for LLaVA)
            clean_images = model.batchify_denoise(lq_img)
            clean_pil_img = Tensor2PIL(clean_images[0], h0, w0)

            captions = self.llava_agent.gen_image_caption([clean_pil_img])
            print(f"Captions from LLaVA: {captions}")

            del clean_images, clean_pil_img
            print(f"LLaVA took: {time.time() - start} seconds")
            logger.info(
                f"GPU memory usage after LLaVA: {torch.cuda.memory_allocated(self.llava_device) / 1024 ** 3:.2f} GB")

        start = time.time()

        # step 3: Diffusion Process
        samples = model.batchify_sample(
            lq_img,
            captions,
            num_steps=edm_steps,
            restoration_scale=s_stage1,
            s_churn=s_churn,
            s_noise=s_noise,
            cfg_scale=s_cfg,
            control_scale=s_stage2,
            seed=seed,
            num_samples=1,
            p_p=a_prompt,
            n_p=n_prompt,
            color_fix_type=color_fix_type,
            use_linear_CFG=linear_CFG,
            use_linear_control_scale=linear_s_stage2,
            cfg_scale_start=spt_linear_CFG,
            control_scale_start=spt_linear_s_stage2,
        )
        print(f"Diffusion Process took: {time.time() - start} seconds")
        logger.info(
            f"GPU memory usage after Diffusion Process: {torch.cuda.memory_allocated(self.supir_device) / 1024 ** 3:.2f} GB")

        out_path = f"/tmp/{seed}.png"
        Tensor2PIL(samples[0], h0, w0).save(out_path)

        return Path(out_path)
