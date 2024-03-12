from dataclasses import dataclass


@dataclass
class PiGDMConfig:
    model_name: str     = "diffusion_ffhq_10m"    # diffusion_ffhq_10m | 256x256_diffusion_uncond | google/ddpm-ema-celebahq-256
    num_train_timesteps = 1000


@dataclass
class PiGDMDeblurConfig(PiGDMConfig):
    timesteps: int      = 100                               # number of timesteps for scheduler
    noise_level_img     = 12.75/255.0                       # set AWGN noise level for LR image, default: 0
    iter_num            = 10                                # set number of iterations
    ddim_sample         = False                             # sampling method
    lambda_             = 1.0                               # key parameter lambda


@dataclass
class PiGDMSchedulerConfig():
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
    print(PiGDMConfig())
    print(PiGDMDeblurConfig())


if __name__ == "__main__":
    main()