from dataclasses import dataclass


@dataclass
class PiGDMConfig:
    model_name: str     = "google/ddpm-ema-celebahq-256"    # diffusion_ffhq_10m | 256x256_diffusion_uncond | google/ddpm-ema-celebahq-256


@dataclass
class PiGDMDeblurConfig(PiGDMConfig):
    
    timesteps: int      = 50                                # number of timesteps for scheduler
    noise_level_img     = 12.75/255.0                       # set AWGN noise level for LR image, default: 0
    num_train_timesteps = 1000
    iter_num            = 10                                # set number of iterations
    ddim_sample         = False                             # sampling method

    lambda_             = 1.0                               # key parameter lambda


SCHEDULER_CONFIG = {
    "_class_name": "DDPMScheduler",
    "_diffusers_version": "0.1.1",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "clip_sample": True,
    "num_train_timesteps": 1000,
    "trained_betas": None,
    "variance_type": "fixed_small"
}


def main():
    print(PiGDMConfig())
    print(PiGDMDeblurConfig())


if __name__ == "__main__":
    main()