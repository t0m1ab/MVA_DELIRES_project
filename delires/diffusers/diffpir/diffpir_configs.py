from dataclasses import dataclass


@dataclass
class DiffPIRConfig:
    model_name: str     = "diffusion_ffhq_10m"      # diffusion_ffhq_10m | 256x256_diffusion_uncond
    device: str         = "cuda"                    # cpu | cuda


@dataclass
class DiffPIRDeblurConfig(DiffPIRConfig):

    noise_level_img     = 12.75/255.0                 # set AWGN noise level for LR image, default: 0
    model_name          = 'diffusion_ffhq_10m'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusion model
    testset_name        = 'demo_test'                  # set testing set,  'imagenet_val' | 'ffhq_val'
    num_train_timesteps = 1000
    iter_num            = 3           # set number of iterations
    iter_num_U          = 1             # set number of inner iterations, default: 1

    show_img            = False         # default: False
    # save_L              = True          # save LR image
    save_restoration    = False          # save restored image
    save_LEH            = False         # save zoomed LR, E and H images
    save_progressive    = False         # save generation process
    border              = 0
	
    lambda_             = 1.0           # key parameter lambda
    sub_1_analytic      = True          # use analytical solution
    
    log_process         = False
    ddim_sample         = False         # sampling method
    model_output_type   = 'pred_xstart' # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode       = 'DiffPIR'     # DiffPIR; DPS; vanilla
    skip_type           = 'quad'        # uniform, quad
    eta                 = 0.0           # eta for ddim sampling
    zeta                = 0.1  
    guidance_scale      = 1.0   

    calc_LPIPS          = True
    use_DIY_kernel      = True
    blur_mode           = 'Gaussian'    # Gaussian; motion      
    kernel_size         = 61

    sf                  = 1
    task_current        = 'deblur'          
    n_channels          = 3             # fixed
    cwd                 = ''  
    seed                = 0             # fixed
    device              = 'cuda'        # fixed


def main():
    print(DiffPIRConfig())
    print(DiffPIRDeblurConfig())


if __name__ == "__main__":
    main()