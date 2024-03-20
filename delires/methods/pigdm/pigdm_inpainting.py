from typing import Callable
import numpy as np
import torch
from tqdm import tqdm
from logging import Logger
from diffusers import DDPMScheduler, UNet2DModel

import delires.methods.utils.utils_agem as utils_agem
import delires.utils.utils_image as utils_image
from delires.methods.utils.utils import adapt_mask_dps_pigdm
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMInpaintingConfig
from delires.methods.pigdm.pigdm_sampling import pigdm_sampling

from delires.params import RESTORED_DATA_PATH


def apply_PiGDM_for_inpainting(
        config: PiGDMInpaintingConfig,
        clean_image_filename: str,
        degraded_image_filename: str,
        masks_filename: str,
        clean_image: torch.Tensor,
        degraded_image: torch.Tensor,
        mask: np.ndarray,
        model: UNet2DModel,
        scheduler: DDPMScheduler,
        img_ext: str = "png",
        logger: Logger = None,
        device = "cpu"
    ) -> tuple[np.ndarray, dict]:
    """
    Apply PiGDM for inpainting to a given degraded image.

    ARGUMENTS:
        - clean_image: torch.Tensor float32 of shape (1,C,W,H) containing the clean image (with value clipping because loader from PNG).
        - degraded_image: torch.Tensor float32 of shape (1,C,W,H) containing the degraded image (with or without value clipping).
        - mask: np.ndarray mask.

    TIPS:
        - sample["mask"], sample["L"] and sample["H"] must be torch.Tensor with float values in [0,1]
        - x taken as input of the function <forward_model> must be a torch.Tensor with float values in [0,1]
    """

    # setup model and scheduler
    model = model.to(device)
    scheduler.set_timesteps(config.timesteps)

    if logger is not None: # debug logs
        logger.debug(utils_image.get_infos_img(clean_image))
        logger.debug(utils_image.get_infos_img(degraded_image))
        logger.debug(utils_image.get_infos_img(mask))

    # setup data and mask
    sample = {
        "H": clean_image.to(device),
        "L": degraded_image.to(device),
        "mask": adapt_mask_dps_pigdm(mask).to(device),
        "sigma": config.noise_level_img,
    }

    if logger is not None: # debug logs
        logger.debug(utils_image.get_infos_img(sample["H"]))
        logger.debug(utils_image.get_infos_img(sample["L"]))
        logger.debug(utils_image.get_infos_img(sample["mask"]))

    # log informations
    if logger is not None:
        logger.info(f"timesteps: {config.timesteps}")
        logger.info(f"device: {device}")
        logger.info(f"Clean image: {clean_image_filename}")
        logger.info(f"Degraded image: {degraded_image_filename}")
        logger.info(f"Masks: {masks_filename}")

    # Guidance (cuda GPU implementation)
    guidance = lambda y, x, sigma, r: utils_agem.inpainting_guidance(y, x, sample['mask'], sigma=sigma, r=r).to(device)
    y = sample['L'].to(device)
    mask = sample['mask'].to(device)

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
    restored_image = utils_image.tensor2uint(res)

    # plt.figure(figsize=(10, 10/3))
    # plt.subplot(131)
    # plt.imshow(util.tensor2uint(y))
    # plt.axis('off')
    # plt.subplot(132)
    # plt.imshow(util.tensor2uint(res))
    # plt.axis('off')
    # plt.subplot(133)
    # plt.imshow(util.tensor2uint(x))
    # plt.axis('off')
    # plt.show()

    psnr = utils_image.calculate_psnr(utils_image.tensor2uint(res), utils_image.tensor2uint(sample['H']))

    metrics = {"psnr": psnr}

    return restored_image, metrics