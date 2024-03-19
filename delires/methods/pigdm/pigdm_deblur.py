import os
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from tqdm import tqdm
from logging import Logger
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from delires.utils.utils_image import get_infos_img
import delires.methods.utils.utils_agem as utils_agem
import delires.methods.utils.utils_image as utils_image
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMDeblurConfig
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
from delires.methods.diffpir.guided_diffusion.respace import SpacedDiffusion
from delires.methods.diffpir.utils import utils_model
from delires.methods.diffpir.utils.delires_utils import plot_sequence
from delires.methods.pigdm.pigdm_sampling import pigdm_sampling

from delires.params import RESTORED_DATA_PATH, CLEAN_DATA_PATH, OPERATORS_PATH



def apply_PiGDM_for_deblurring(
        config: PiGDMDeblurConfig,
        clean_image_filename: str,
        degraded_image_filename: str,
        kernel_filename: str,
        clean_image: torch.Tensor,
        degraded_image: torch.Tensor,
        kernel: torch.Tensor,
        model: UNet2DModel,
        scheduler: DDPMScheduler,
        img_ext: str = "png",
        logger: Logger = None,
        device = "cpu"
    ) -> tuple[np.ndarray, dict]:
    """
    Apply PiGDM for deblurring to a given degraded image.

    ARGUMENTS:
        - clean_image: torch.Tensor float32 of shape (1,C,W,H) containing the clean image (with value clipping because loader from PNG).
        - degraded_image: torch.Tensor float32 of shape (1,C,W,H) containing the degraded image (with or without value clipping).
        - kernel: torch.Tensor float32 of shape (1,1,W,H) containing the blur kernel.

    TIPS:
        - sample["kernel"], sample["L"] and sample["H"] must be torch.Tensor with float values in [0,1]
        - x taken as input of the function <forward_model> must be a torch.Tensor with float values in [0,1]
    """

    # setup model and scheduler
    model = model.to(device)
    scheduler.set_timesteps(config.timesteps)

    if logger is not None: # debug logs
        logger.debug(get_infos_img(clean_image))
        logger.debug(get_infos_img(degraded_image))
        logger.debug(get_infos_img(kernel))

    # setup data and kernel
    sample = {
        "H": clean_image.to(device),
        "L": degraded_image.to(device),
        "kernel": kernel.to(device),
        "sigma": config.noise_level_img,
    }

    if logger is not None: # debug logs
        logger.debug(get_infos_img(sample["H"]))
        logger.debug(get_infos_img(sample["L"]))
        logger.debug(get_infos_img(sample["kernel"]))

    # log informations
    if logger is not None:
        logger.info(f"- device: {device}")
        logger.info(f"- timesteps: {config.timesteps}")
        logger.info(f"* clean image: {clean_image_filename}")
        logger.info(f"* degraded image: {degraded_image_filename}")
        logger.info(f"* kernel: {kernel_filename}")

    # Forward model (cuda GPU implementation)
    # forward_model = lambda x: agem.fft_blur(x, sample['kernel'].to(device))
    # If you are using an MPS gpu use the following forward_model instead
    # CPU fallback implementation (no MPS support for torch.roll, fft2, Complex Float, etc.)
    guidance = lambda y, x, sigma, r: utils_agem.deblurring_guidance(y, x, sample['kernel'], sigma=sigma, r=r).to(device)
    
    # Degraded image y = A x + noise
    y = sample['L'].to(device)

    # PiGDM sampling
    res = pigdm_sampling(
        config,
        model,
        scheduler, 
        y, 
        guidance, 
        scale=1,
        device=device,
        logger=logger,
    )

    # Ground truth image x
    x = sample['H'].to(device)

    clean_image = utils_image.tensor2uint(x)
    degraded_image = utils_image.tensor2uint(y)
    restored_image = utils_image.tensor2uint(res.to(device))

    # print(get_infos_img(clean_image))
    # print(get_infos_img(degraded_image))
    # print(get_infos_img(restored_image))

    psnr = utils_image.calculate_psnr(restored_image, clean_image)

    metrics = {"psnr": psnr}

    return restored_image, metrics