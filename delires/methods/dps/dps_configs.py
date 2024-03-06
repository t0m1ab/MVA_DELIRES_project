from dataclasses import dataclass


@dataclass
class DPSConfig:
    model_name: str     = "google/ddpm-ema-celebahq-256"    # diffusion_ffhq_10m | 256x256_diffusion_uncond
    device: str         = "cuda"                            # cpu | cuda


@dataclass
class DPSDeblurConfig(DPSConfig):
    model_name: str     = "google/ddpm-ema-celebahq-256"    # diffusion_ffhq_10m | 256x256_diffusion_uncond
    device: str         = "mps"                             # cpu | cuda

    timesteps: int      = 10                                # number of timesteps for scheduler



def main():
    print(DPSConfig())
    print(DPSDeblurConfig())


if __name__ == "__main__":
    main()