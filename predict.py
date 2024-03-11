import os

cache_dir = os.path.abspath("weights")
os.environ["PYANNOTE_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

import copy
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

logging.basicConfig(
    format="(%(asctime)s) %(name)s:%(lineno)d [%(levelname)s] | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    model: torch.nn.Module
    supir_device = "cuda:0"

    def setup(self) -> None:
        self.model = create_SUPIR_model("options/SUPIR_v0_Juggernautv9_lightning.yaml", SUPIR_sign="Q")
        # self.model.half()
        # self.model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
        self.model.ae_dtype = convert_dtype("bf16")
        self.model.model.dtype = convert_dtype("bf16")
        self.model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(self.model.first_stage_model.denoise_encoder)

        self.model.to(self.supir_device)

    def predict(
        self,
        image: Path = Input(description="Low quality input image."),
        captions: str = Input(
            description="Captions for the image",
            default="a professional, detailed, high-quality photo"
        ),
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
        linear_cfg: bool = Input(
            description="Linearly (with sigma) increase CFG from 'spt_linear_CFG' to s_cfg.",
            default=False,
        ),
        linear_s_stage2: bool = Input(
            description="Linearly (with sigma) increase s_stage2 from 'spt_linear_s_stage2' to s_stage2.",
            default=False,
        ),
        spt_linear_cfg: float = Input(
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

        start = time.time()

        # step 3: Diffusion Process
        samples = model.batchify_sample(
            lq_img,
            [captions],
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
            use_linear_CFG=linear_cfg,
            use_linear_control_scale=linear_s_stage2,
            cfg_scale_start=spt_linear_cfg,
            control_scale_start=spt_linear_s_stage2,
        )
        print(f"Diffusion Process took: {time.time() - start} seconds")
        logger.info(
            f"GPU memory usage after Diffusion Process: {torch.cuda.memory_allocated(self.supir_device) / 1024 ** 3:.2f} GB")

        out_path = f"/tmp/{seed}.png"
        Tensor2PIL(samples[0], h0, w0).save(out_path)

        return Path(out_path)
