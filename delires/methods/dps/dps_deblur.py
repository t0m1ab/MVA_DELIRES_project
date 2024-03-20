from logging import Logger
import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel

import delires.methods.utils.utils_agem as utils_agem
import delires.utils.utils_image as utils_image
from delires.methods.dps.dps_configs import DPSDeblurConfig
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
from delires.methods.dps.dps_sampling import dps_sampling


def apply_DPS_for_deblurring(
        config: DPSDeblurConfig,
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
        device = "cpu",
    ) -> tuple[np.ndarray, dict]:
    """
    Apply DPS for deblurring to a given degraded image.

    ARGUMENTS:
        - config: DPSDeblurConfig object containing the configuration for the DPS deblurring.
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
        logger.debug(utils_image.get_infos_img(clean_image))
        logger.debug(utils_image.get_infos_img(degraded_image))
        logger.debug(utils_image.get_infos_img(kernel))

    # setup data and kernel
    sample = {
        "H": clean_image.to(device),
        "L": degraded_image.to(device),
        "kernel": torch.tensor(kernel, dtype=torch.float32).to(device),
        "sigma": config.noise_level_img,
    }

    if logger is not None: # debug logs
        logger.debug(utils_image.get_infos_img(sample["H"]))
        logger.debug(utils_image.get_infos_img(sample["L"]))
        logger.debug(utils_image.get_infos_img(sample["kernel"]))

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

    # DPS sampling
    res = dps_sampling(
        model=model, 
        scheduler=scheduler, 
        y=sample['L'].to(device), 
        forward_model=forward_model, 
        scale=1, 
        scale_guidance=0, 
        device=device,
        logger=logger,
    )

    clean_image = utils_image.tensor2uint(sample['H'].to(device))
    degraded_image = utils_image.tensor2uint(sample['L'].to(device))
    restored_image = utils_image.tensor2uint(res.to(device))

    if logger is not None: # debug logs
        logger.debug(utils_image.get_infos_img(clean_image))
        logger.debug(utils_image.get_infos_img(degraded_image))
        logger.debug(utils_image.get_infos_img(restored_image))

    psnr = utils_image.calculate_psnr(restored_image, clean_image)

    metrics = {"psnr": psnr}

    return restored_image, metrics