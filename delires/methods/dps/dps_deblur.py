from typing import Callable
import numpy as np
import torch
from tqdm import tqdm
from logging import Logger
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DModel

from delires.utils.utils_image import get_infos_img
import delires.methods.utils.utils_agem as utils_agem
import delires.methods.utils.utils_image as utils_image
from delires.methods.utils.utils import adapt_kernel_dps_pigdm, alpha_beta
from delires.methods.dps.dps_configs import DPSConfig, DPSDeblurConfig
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
from delires.methods.diffpir.utils.delires_utils import plot_sequence
from delires.methods.dps.dps_sampling import dps_sampling

from delires.params import RESTORED_DATA_PATH


def apply_DPS_for_deblurring(
        config: DPSDeblurConfig,
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
        device = "cpu",
    ) -> tuple[np.ndarray, dict]:
    """
    Apply DPS for deblurring to a given degraded image.

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
    forward_model = lambda x: utils_agem.fft_blur(x, sample['kernel'])
    # forward_model = lambda x: forward_model_cpu(x.to('cpu')).to(device)
    
    # Degraded image y = A x + noise
    y = sample['L'].to(device)

    # DPS sampling
    res = dps_sampling(
        model, 
        scheduler, 
        y, 
        forward_model, 
        scale=1, 
        scale_guidance=0, 
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