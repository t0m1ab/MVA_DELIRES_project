import numpy as np
from logging import Logger
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from delires.utils.utils_image import get_infos_img
import delires.methods.utils.utils_agem as utils_agem
import delires.utils.utils_image as utils_image
from delires.methods.utils.utils import adapt_mask_dps_pigdm
from delires.methods.dps.dps_configs import DPSConfig, DPSInpaintingConfig
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
from delires.methods.diffpir.guided_diffusion.respace import SpacedDiffusion
from delires.methods.diffpir.utils import utils_model
from delires.methods.dps.dps_sampling import dps_sampling

from delires.params import RESTORED_DATA_PATH


def apply_DPS_for_inpainting(
        config: DPSInpaintingConfig,
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
    Apply DPS for inpainting to a given degraded image.

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
        logger.debug(get_infos_img(clean_image))
        logger.debug(get_infos_img(degraded_image))
        logger.debug(get_infos_img(mask))

    # setup data and mask
    sample = {
        "H": clean_image.to(device),
        "L": degraded_image.to(device),
        "mask": adapt_mask_dps_pigdm(mask).to(device),
        "sigma": config.noise_level_img,
    }

    if logger is not None: # debug logs
        logger.debug(get_infos_img(sample["H"]))
        logger.debug(get_infos_img(sample["L"]))
        logger.debug(get_infos_img(sample["mask"]))

    # log informations
    if logger is not None:
        logger.info(f"timesteps: {config.timesteps}")
        logger.info(f"device: {device}")
        logger.info(f"Clean image: {clean_image_filename}")
        logger.info(f"Degraded image: {degraded_image_filename}")
        logger.info(f"Masks: {masks_filename}")

    # Forward model (cuda GPU implementation)
    forward_model = lambda x: sample['mask'] * x
    y = sample['L'].to(device)
    
    # DPS sampling
    res = dps_sampling(
        model,
        scheduler, 
        y, 
        forward_model,
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