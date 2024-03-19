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

from delires.params import RESTORED_DATA_PATH, CLEAN_DATA_PATH, OPERATORS_PATH


def pigdm_sampling(
        config: PiGDMDeblurConfig,
        model: UNet2DModel | UNetModel, 
        scheduler: DDPMScheduler, 
        y: torch.Tensor, 
        guidance: Callable, 
        scale: int = 1, 
        device: str = "cpu",
        logger: Logger = None,
    ):
    """
    PiGDM with DDPM and intrinsic scale
    """
    sample_size = y.shape[-1]
    step_size = scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    input = torch.randn((1, 3, sample_size, sample_size)).to(device)

    for t in tqdm(scheduler.timesteps):
        # Computation of some hyper-params
        prev_t = t - step_size
        variance = scheduler._get_variance(t, prev_t)
        r = torch.sqrt(variance/(variance + 1))
        current_alpha_t = 1 / (1 + variance)

        # Predict noise
        input.requires_grad_()

        if isinstance(model, UNetModel): # diffpir nn
            ### NOTE: the code below mimics the logic of utils_model.model_fn for epsilon prediction using
            #### a UNetModel instance <model> and a SpacedDiffusion instance <diffusion> with the following settings:
            # from delires.methods.diffpir.guided_diffusion.respace import ModelMeanType, ModelVarType
            # assert diffusion.rescale_timesteps == False
            # assert diffusion.model_mean_type == ModelMeanType.EPSILON
            # assert diffusion.model_var_type == ModelVarType.LEARNED_RANGE

            batch_dim, channel_dim = input.shape[:2]
            vec_t = torch.tensor([t] * batch_dim, device=input.device)
            
            # output with 6 channels for each image
            model_output = model(input, vec_t)

            # according to the config, the first 3 channels are the epsilon_t we want
            noisy_residual, _ = torch.split(model_output, channel_dim, dim=1)
            
            # DEBUG: save intermediate epsilon_t (saved images must look like noise)
            # img_to_save = torch.clone(epsilon_t)
            # img_to_save = img_to_save[0].detach().cpu().numpy().copy().transpose(1, 2, 0)
            # img_to_save -= np.min(img_to_save)
            # img_to_save /= np.max(img_to_save)
            # plt.imsave(f"{RESTORED_DATA_PATH}/x_{t.item()}.png", img_to_save)
            if logger is not None: # debug logs
                logger.debug(f"t={t.item()}", get_infos_img(noisy_residual))
        
        elif isinstance(model, UNet2DModel): # huggingface nn
            ### NOTE: simply run an inference with the model which is supposed to return the noise epsilon_t
            noisy_residual = model(input, t).sample

        else:
            raise ValueError(f"Unknown model instance: {type(model)}")

        # Get x_prec and x0_hat
        pred = scheduler.step(noisy_residual, t, input)
        x0_hat = utils_agem.clean_output(pred.pred_original_sample)
        x_prec = pred.prev_sample

        # Guidance
        g = (guidance(y, x0_hat, sigma=config.noise_level_img, r=r).detach() * x0_hat).sum()
        grad = torch.autograd.grad(outputs=g, inputs=input)[0]
        input = input.detach_()

        # Update of x_t
        input = x_prec + scale * grad * (r ** 2) * torch.sqrt(current_alpha_t)

    return utils_agem.clean_output(input)


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
        - clean_image: np.ndarray float32 of shape (1,C,W,H) containing the clean image (with value clipping because loader from PNG).
        - degraded_image: np.ndarray float32 of shape (1,C,W,H) containing the degraded image (with or without value clipping).
        - kernel: np.ndarray float32 of shape (1,1,W,H) containing the blur kernel.

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

    ### ===== DEBUG: load specific debug data ===== ###

    # print(kernel_filename)
    # print(degraded_image_filename)
    # print(clean_image_filename)
    # for k, v in sample.items():
    #     print(k, get_infos_img(v) if isinstance(v, torch.Tensor) else None)
    #     print(v)
    #     print(100*"-")
    
    ### ==================== DEBUG ==================== ###

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