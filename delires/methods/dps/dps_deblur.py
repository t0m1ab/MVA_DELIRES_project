import numpy as np
import torch
from tqdm import tqdm
from logging import Logger
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from delires.utils.utils_image import get_infos_img
import delires.methods.dps_pigdm_utils.utils_agem as utils_agem
import delires.methods.dps_pigdm_utils.utils_image as utils_image
from delires.methods.dps.dps_configs import DPSConfig, DPSDeblurConfig
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
from delires.methods.diffpir.guided_diffusion.respace import SpacedDiffusion
from delires.methods.diffpir.utils import utils_model
from delires.methods.diffpir.utils.delires_utils import plot_sequence

from delires.params import RESTORED_DATA_PATH


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
        config: DPSDeblurConfig,
        model, 
        scheduler, 
        y, 
        forward_model, 
        nsamples=1, 
        scale=1, 
        scale_guidance=1,
        device: str = "cpu",
        diffusion: SpacedDiffusion = None,
    ):
    """
    DPS with DDPM and intrinsic scale
    """
    sample_size = y.shape[-1]

    # Init random noise
    x_T = torch.randn((nsamples, 3, sample_size, sample_size)).to(device)
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

            print(f"t={t.item()}", get_infos_img(epsilon_t))

            # exit()

        elif isinstance(model, UNet2DModel): # hf nn
            ### NOTE: simply run an inference with the model which is supposed to return the noise epsilon_t
            epsilon_t = model(x_t, t).sample

        else:
            raise ValueError(f"Unknown model instance: {type(model)}")
        
        # print(get_infos_img(epsilon_t))

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
        model: UNet2DModel,
        scheduler: DDPMScheduler,
        diffusion: SpacedDiffusion,
        img_ext: str = "png",
        logger: Logger = None,
        device = "cpu",
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

    # setup model and scheduler
    model = model.to(device)
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
        logger.info(f"device: {device}")
        logger.info(f"Clean image: {clean_image_filename}")
        logger.info(f"Degraded image: {degraded_image_filename}")
        logger.info(f"Kernel: {kernel_filename}")

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
        config,
        model, 
        scheduler, 
        y, 
        forward_model, 
        1, 
        scale=1, 
        scale_guidance=0, 
        device=device,
        diffusion=diffusion,
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