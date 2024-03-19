from typing import Callable
from tqdm import tqdm
from logging import Logger
import torch

from diffusers import DDPMScheduler, UNet2DModel
from delires.methods.pigdm.pigdm_configs import PiGDMDeblurConfig, PiGDMInpaintingConfig
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
import delires.methods.utils.utils_agem as utils_agem
from delires.utils.utils_image import get_infos_img


def pigdm_sampling(
        config: PiGDMDeblurConfig | PiGDMInpaintingConfig,
        model: UNet2DModel | UNetModel, 
        scheduler: DDPMScheduler, 
        y: torch.Tensor, 
        guidance: Callable,
        scale: int = 1, 
        device: str = "cpu",
        logger: Logger = None,
    ) -> torch.Tensor:
    """
    PiGDM with DDPM and intrinsic scale
    """
    sample_size = y.shape[-1]
    step_size = scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    print("STEP SIZE", step_size, config.timesteps)
    print(scheduler.config.num_train_timesteps)
    print(scheduler.num_inference_steps)
    print(scheduler.timesteps)

    input = torch.randn((1, 3, sample_size, sample_size)).to(device)

    for t in tqdm(scheduler.timesteps, desc="PiGDM sampling"):
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