import numpy as np
import torch

from diffusers import DDPMScheduler


def adapt_kernel_dps_pigdm(kernel: np.ndarray) -> torch.Tensor:
    """ Convert kernel to float32 tensor with batch and channel dim and values in range [0,1] """
    kernel_with_batch_dims = np.expand_dims(kernel, axis=(0,1)) # add batch dim and channel dim
    return torch.tensor(kernel_with_batch_dims, dtype=torch.float32)


def adapt_mask_dps_pigdm(mask: np.ndarray) -> torch.Tensor:
    """ Convert mask to float32 tensor with batch and channel dim """
    mask_with_batch_dims = np.transpose(mask, (2, 0, 1))
    mask_with_batch_dims = np.expand_dims(mask_with_batch_dims, axis=(0)) # add batch dim and channel dim
    return torch.tensor(mask_with_batch_dims, dtype=torch.float32)


def adapt_image_dps_pigdm(img: np.ndarray) -> torch.Tensor:
    """ Convert uint8 image to float32 tensor with batch channel, transpose dims and set values in range [0,1] """
    img_with_batch_dim = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0) / 255.
    return torch.tensor(img_with_batch_dim, dtype=torch.float32)


def alpha_beta(scheduler: DDPMScheduler, t: int) -> tuple[float, float]:
    """ Compute alpha_t and beta_t for a given timestep t. """
    prev_t = scheduler.previous_timestep(t)
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    return current_alpha_t, current_beta_t