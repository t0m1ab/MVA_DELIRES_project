from dataclasses import dataclass

# AVIALABLE NETWORKS: diffusion_ffhq_10m | 256x256_diffusion_uncond | google/ddpm-ema-celebahq-256

@dataclass
class DiffPIRConfig:
    model_name: str          = "diffusion_ffhq_10m"
    num_train_timesteps: int = 1000


@dataclass
class DiffPIRDeblurConfig(DiffPIRConfig):
    iter_num: int              = 100                 # set number of iterations
    noise_level_img: float     = 12.75/255.0         # set AWGN noise level for LR image, default: 0
    
    testset_name: str = 'demo_test'                  # set testing set,  'imagenet_val' | 'ffhq_val'
    iter_num_U: int = 1             # set number of inner iterations, default: 1

    show_img: bool = False         # default: False
    # save_L: bool = True          # save LR image
    save_restoration: bool = False          # save restored image
    save_LEH: bool = False         # save zoomed LR, E and H images
    save_progressive: bool = False         # save generation process
    
    lambda_: float = 1.0           # key parameter lambda
    sub_1_analytic: bool = True          # use analytical solution
    
    log_process: bool = False
    ddim_sample: bool = False         # sampling method
    model_output_type: str = 'pred_xstart' # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode: str = 'DiffPIR'     # DiffPIR; DPS; vanilla
    skip_type: str = 'quad'        # uniform, quad
    eta: float = 0.0           # eta for ddim sampling
    zeta: float = 0.1  
    guidance_scale: float = 1.0   

    calc_LPIPS: bool = False
    use_DIY_kernel: bool = True
    blur_mode: str = 'Gaussian'    # Gaussian; motion      
    kernel_size: int = 61

    sf: int = 1
    task_current: str = 'deblur'          
    n_channels: int = 3             # fixed
    cwd: str = ''  
    seed: int = 0             # fixed


@dataclass
class DiffPIRInpaintingConfig(DiffPIRConfig):

    iter_num: int               = 100               # set number of iterations
    noise_level_img: float      = 12.75/255.0       # set AWGN noise level for LR image, default: 0
    
    iter_num_U: int = 1             # set number of inner iterations, default: 1

    show_img: bool = False         # default: False
    # save_L: bool = True          # save LR image
    save_restoration: bool = False          # save restored image
    save_LEH: bool = False         # save zoomed LR, E and H images
    save_progressive: bool = False         # save generation process
    save_progressive_mask: bool = False         # save generation process
    
    lambda_: float = 1.0           # key parameter lambda
    sub_1_analytic: bool = True          # use analytical solution
    
    log_process: bool = False
    ddim_sample: bool = False         # sampling method
    model_output_type: str = 'pred_xstart' # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode: str = 'DiffPIR'     # DiffPIR; DPS; vanilla
    skip_type: str = 'quad'        # uniform, quad
    eta: float = 0.0           # eta for ddim sampling
    zeta: float = 1.0  
    guidance_scale: float = 1.0   

    calc_LPIPS: bool = False

    task_current: str = 'ip'              # 'ip' for inpainting  
    n_channels: int = 3             # fixed
    cwd: str = ''  
    seed: int = 0             # fixed



def main():
    print(DiffPIRConfig())
    print(DiffPIRDeblurConfig())
    print(DiffPIRInpaintingConfig())


if __name__ == "__main__":
    # main()
    
    import numpy as np
    import os
    import delires.params as params
    thingy = np.load(os.path.join(params.RESTORED_DATA_PATH, 'test_exp_diffpir_deblur', 'metrics.npz'), allow_pickle=True)
    for key in thingy.keys():
        print(key)
        print(thingy[key])
