import numpy as np
import torch

from diffusers import DDPMScheduler


def adapt_kernel_dps_pigdm(kernel: np.ndarray) -> torch.Tensor:
    """ Convert kernel to float32 tensor with batch and channel dim and values in range [0,1] """
    kernel_with_batch_dims = np.expand_dims(kernel, axis=(0,1)) # add batch dim and channel dim
    return torch.tensor(kernel_with_batch_dims, dtype=torch.float32)


def alpha_beta(scheduler: DDPMScheduler, t: int) -> tuple[float, float]:
    """ Compute alpha_t and beta_t for a given timestep t. """
    prev_t = scheduler.previous_timestep(t)
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    return current_alpha_t, current_beta_t