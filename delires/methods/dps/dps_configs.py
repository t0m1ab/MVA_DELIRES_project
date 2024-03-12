from dataclasses import dataclass


@dataclass
class DPSConfig:
    model_name: str     = "diffusion_ffhq_10m"    # diffusion_ffhq_10m | 256x256_diffusion_uncond | google/ddpm-ema-celebahq-256


@dataclass
class DPSDeblurConfig(DPSConfig):
    model_name: str     = "diffusion_ffhq_10m"    # diffusion_ffhq_10m | 256x256_diffusion_uncond | google/ddpm-ema-celebahq-256

    timesteps: int      = 1000                              # number of timesteps for scheduler
    noise_level_img     = 12.75/255.0                       # set AWGN noise level for LR image, default: 0
    num_train_timesteps = 1000
    iter_num            = 10                                # set number of iterations
    ddim_sample         = False                             # sampling method

    lambda_             = 1.0                               # key parameter lambda


@dataclass
class DPSSchedulerConfig():
    _class_name: str = "DDPMScheduler"
    _diffusers_version: str = "0.1.1"
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    clip_sample: bool = True
    num_train_timesteps: int = 1000
    trained_betas: list | None = None
    variance_type: str = "fixed_small"


def main():
    print(DPSConfig())
    print(DPSDeblurConfig())


if __name__ == "__main__":
    main()