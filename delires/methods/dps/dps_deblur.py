import numpy as np
import torch
from tqdm import tqdm
from logging import Logger
from diffusers import DDPMPipeline

from delires.utils.utils_image import get_infos_img
import delires.methods.dps.utils.utils_agem as utils_agem
import delires.methods.dps.utils.utils_image as utils_image
from delires.methods.dps.dps_configs import DPSConfig, DPSDeblurConfig


def adapt_kernel_dps(kernel: np.ndarray) -> torch.Tensor:
    """ Convert kernel to float32 tensor with batch and channel dim and values in range [0,1] """
    kernel_with_batch_dims = np.expand_dims(kernel, axis=(0,1)) # add batch dim and channel dim
    return torch.tensor(kernel_with_batch_dims, dtype=torch.float32)


def adapt_image_dps(img: np.ndarray) -> torch.Tensor:
    """ Convert uint8 image to float32 tensor with batch channel, transpose dims and set values in range [0,1] """
    img_with_batch_dim = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0) / 255.
    return torch.tensor(img_with_batch_dim, dtype=torch.float32)


def alpha_beta(scheduler, t):
    prev_t = scheduler.previous_timestep(t)
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    return current_alpha_t, current_beta_t


def dps_sampling(
        model, 
        scheduler, 
        y, 
        forward_model, 
        nsamples=1, 
        scale=1, 
        scale_guidance=1,
        device: str = "cpu",
    ):
    """
    DPS with DDPM and intrinsic scale
    """
    sample_size = model.config.sample_size

    # Init random noise
    x_T = torch.randn((nsamples, 3, sample_size, sample_size)).to(device)
    x_t = x_T

    for t in tqdm(scheduler.timesteps, desc="DPS sampling"):

        # Predict noisy residual eps_theta(x_t)
        x_t.requires_grad_()
        epsilon_t = model(x_t, t).sample

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
        clean_image: np.ndarray,
        degraded_image: np.ndarray,
        kernel: np.ndarray,
        ddpm_model: DDPMPipeline,
        img_ext: str = "png",
        logger: Logger = None,
    ) -> tuple[np.ndarray, dict]:
    """
    Apply DPS for deblurring to a given degraded image.

    ARGUMENTS:
        [See delires.diffusers.diffpir.diffpir_deblur.apply_DiffPIR_for_deblurring]

    TIPS:
        - sample["kernel"], sample["L"] and sample["H"] must be torch.Tensor with float values in [0,1]
        - x taken as input of the function <forward_model> must be a torch.Tensor with float values in [0,1]
    """

    # scheduler = DDPMScheduler.from_pretrained(model_name)
    # model = UNet2DModel.from_pretrained(model_name).to(device)

    # setup device
    device = torch.device("cpu")
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()

    # setup model and scheduler
    ddpm = ddpm_model.to(device)
    scheduler = ddpm.scheduler
    model = ddpm.unet
    scheduler.set_timesteps(config.timesteps)

    if logger is not None:
        logger.debug(get_infos_img(clean_image))
        logger.debug(get_infos_img(degraded_image))
        logger.debug(get_infos_img(kernel))

    # setup data and kernel
    sample = {
        "H": adapt_image_dps(clean_image).to(device),
        "L": adapt_image_dps(degraded_image).to(device),
        "kernel": adapt_kernel_dps(kernel).to(device),
    }

    if logger is not None:
        logger.debug(get_infos_img(sample["H"]))
        logger.debug(get_infos_img(sample["L"]))
        logger.debug(get_infos_img(sample["kernel"]))

    # log informations
    if logger is not None:
        logger.info(f"model_name: {config.model_name}")
        logger.info(f"timesteps: {config.timesteps}")
        logger.info(f"device: {config.device}")
        logger.info(f"Clean image: {clean_image_filename}")
        logger.info(f"Degraded image: {degraded_image_filename}")
        logger.info(f"Kernel: {kernel_filename}")

    # Forward model (cuda GPU implementation)
    # forward_model = lambda x: agem.fft_blur(x, sample['kernel'].to(device))
    # If you are using an MPS gpu use the following forward_model instead
    # CPU fallback implementation (no MPS support for torch.roll, fft2, Complex Float, etc.)
    forward_model_cpu = lambda x: utils_agem.fft_blur(x, sample['kernel'])
    forward_model = lambda x: forward_model_cpu(x.to('cpu')).to(device)
    # Degraded image y = A x + noise
    y = sample['L'].to(device)
    # DPS sampling
    res = dps_sampling(model, scheduler, y, forward_model, 1, scale=1, scale_guidance=0)
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