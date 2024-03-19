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

from delires.params import RESTORED_DATA_PATH


def dps_sampling(
        model: UNet2DModel | UNetModel, 
        scheduler: DDPMScheduler, 
        y: torch.Tensor, 
        forward_model: Callable, 
        scale: int = 1, 
        scale_guidance: int = 1,
        device: str = "cpu",
        logger: Logger = None,
    ):
    """
    DPS with DDPM and intrinsic scale
    """
    sample_size = y.shape[-1]

    # Init random noise
    x_T = torch.randn((1, 3, sample_size, sample_size)).to(device)
    x_t = x_T

    # plot_sequence(np.array(sigmas.cpu()), path=RESTORED_DATA_PATH, title="sigmas.png")
    # plot_sequence(scheduler.timesteps, path=RESTORED_DATA_PATH, title="timesteps.png")

    for t in tqdm(scheduler.timesteps, desc="DPS sampling"):

        # Predict noisy residual eps_theta(x_t)
        x_t.requires_grad_()

        if isinstance(model, UNetModel): # diffpir nn
            ### NOTE: the code below mimics the logic of utils_model.model_fn for epsilon prediction using
            #### a UNetModel instance <model> and a SpacedDiffusion instance <diffusion> with the following settings:
            # from delires.methods.diffpir.guided_diffusion.respace import ModelMeanType, ModelVarType
            # assert diffusion.rescale_timesteps == False
            # assert diffusion.model_mean_type == ModelMeanType.EPSILON
            # assert diffusion.model_var_type == ModelVarType.LEARNED_RANGE

            batch_dim, channel_dim = x_t.shape[:2]
            vec_t = torch.tensor([t] * batch_dim, device=x_t.device)
            
            # output with 6 channels for each image
            model_output = model(x_t, vec_t)

            # according to the config, the first 3 channels are the epsilon_t we want
            epsilon_t, _ = torch.split(model_output, channel_dim, dim=1)
            
            # DEBUG: save intermediate epsilon_t (saved images must look like noise)
            # img_to_save = torch.clone(epsilon_t)
            # img_to_save = img_to_save[0].detach().cpu().numpy().copy().transpose(1, 2, 0)
            # img_to_save -= np.min(img_to_save)
            # img_to_save /= np.max(img_to_save)
            # plt.imsave(f"{RESTORED_DATA_PATH}/x_{t.item()}.png", img_to_save)
            if logger is not None: # debug logs
                logger.debug(f"t={t.item()}", get_infos_img(epsilon_t))
        
        elif isinstance(model, UNet2DModel): # huggingface nn
            ### NOTE: simply run an inference with the model which is supposed to return the noise epsilon_t
            epsilon_t = model(x_t, t).sample

        else:
            raise ValueError(f"Unknown model instance: {type(model)}")
        
        # Get x0_hat and unconditional
        # x_{t-1} = a_t * x_t + b_t * epsilon(x_t) + sigma_t z_t
        # with b_t = eta_t
        predict = scheduler.step(epsilon_t, t, x_t)
        x0_hat  = utils_agem.clean_output(predict.pred_original_sample)
        x_prev  = predict.prev_sample # unconditional DDPM sample x_{t-1}'

        # Guidance
        f = torch.norm(forward_model(x0_hat) - y)
        g = torch.autograd.grad(f, x_t)[0]

        # compute variance schedule
        alpha_t, beta_t= alpha_beta(scheduler, t)

        # Guidance weight
        # eta_t = ...
        if (scale_guidance==1):
            eta_t =  beta_t / (alpha_t ** 0.5)
        else:
            eta_t = 1.0

        # DPS update rule = DDPM update rule + guidance
        x_t = x_prev - scale * eta_t * g
        x_t = x_t.detach_()

    return utils_agem.clean_output(x_t)


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