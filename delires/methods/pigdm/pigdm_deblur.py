import os
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from tqdm import tqdm
from logging import Logger
from diffusers import DDPMScheduler, UNet2DModel

from delires.utils.utils_image import get_infos_img
import delires.methods.utils.utils_agem as utils_agem
import delires.utils.utils_image as utils_image
from delires.methods.pigdm.pigdm_configs import PiGDMDeblurConfig
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
from delires.methods.pigdm.pigdm_sampling import pigdm_sampling


def apply_PiGDM_for_deblurring(
        config: PiGDMDeblurConfig,
        clean_image_filename: str,
        degraded_image_filename: str,
        kernel_filename: str,
        clean_image: torch.Tensor,
        degraded_image: torch.Tensor,
        kernel: np.ndarray,
        model: UNet2DModel | UNetModel,
        scheduler: DDPMScheduler,
        img_ext: str = "png",
        logger: Logger = None,
        device = "cpu"
    ) -> tuple[np.ndarray, dict]:
    """
    Apply PiGDM for deblurring to a given degraded image.

    ARGUMENTS:
        - config: PiGDMDeblurConfig object containing the configuration for the DPS deblurring.
        - clean_image_filename: str containing the filename of the clean image.
        - degraded_image_filename: str containing the filename of the degraded image.
        - kernel_filename: str containing the filename of the blur kernel.
        - clean_image: torch.Tensor float32 of shape (1,C,W,H) containing the clean image (with value clipping because loader from PNG).
        - degraded_image: torch.Tensor float32 of shape (1,C,W,H) containing the degraded image (with or without value clipping).
        - kernel: np.ndarray float32 of shape (1,1,W,H) containing the blur kernel.
        - model: neural network type function to use as prior in the DPS sampling.
        - scheduler: DDPMScheduler object containing the scheduler for the DPS sampling.
        - img_ext: str containing the extension of the images (default: "png").
        - logger: Logger object to log the process (default to None meaning no logs).
        - device: str containing the device to use (default: "cpu").

    TIPS:
        - sample["kernel"], sample["L"] and sample["H"] must be torch.Tensor with float values in [0,1]
        - x taken as input of the function <forward_model> must be a torch.Tensor with float values in [0,1]
    
    RETURNS:
        - restored_image: np.ndarray uint8 of shape (W,H,C) containing the restored image.
        - metrics: dict containing the metrics of the restoration.
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
        "kernel": torch.tensor(kernel, dtype=torch.float32).to(device),
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

    # PiGDM sampling
    res = pigdm_sampling(
        config=config,
        model=model,
        scheduler=scheduler, 
        y=sample['L'].to(device), 
        guidance=guidance, 
        scale=1,
        device=device,
        logger=logger,
    )

    clean_image = utils_image.tensor2uint(sample['H'].to(device))
    degraded_image = utils_image.tensor2uint(sample['L'].to(device))
    restored_image = utils_image.tensor2uint(res.to(device))

    if logger is not None: # debug logs
        logger.debug(get_infos_img(clean_image))
        logger.debug(get_infos_img(degraded_image))
        logger.debug(get_infos_img(restored_image))

    psnr = utils_image.calculate_psnr(restored_image, clean_image)

    metrics = {"psnr": psnr}

    return restored_image, metrics