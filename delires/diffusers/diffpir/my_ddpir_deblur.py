import os.path
from pathlib import Path
import cv2
import logging
from tqdm import tqdm
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict

from delires.diffusers.diffpir.utils import utils_model
from delires.diffusers.diffpir.utils import utils_logger
from delires.diffusers.diffpir.utils import utils_sisr as sr
from delires.diffusers.diffpir.utils import utils_image as util
from delires.diffusers.diffpir.utils.delires_utils import (
    plot_sequence, 
    create_blur_kernel, 
    create_blurred_and_noised_image, 
    manually_build_image_path,
)
from delires.diffusers.diffpir.guided_diffusion.unet import UNetModel
from delires.diffusers.diffpir.guided_diffusion.respace import SpacedDiffusion

# from guided_diffusion import dist_util
from delires.diffusers.diffpir.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

@dataclass
class DiffPIRDeblurConfig:

    noise_level_img     = 12.75/255.0                 # set AWGN noise level for LR image, default: 0
    model_name          = 'diffusion_ffhq_10m'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusion model
    testset_name        = 'demo_test'                  # set testing set,  'imagenet_val' | 'ffhq_val'
    num_train_timesteps = 1000
    iter_num            = 100           # set number of iterations
    iter_num_U          = 1             # set number of inner iterations, default: 1

    show_img            = False         # default: False
    # save_L              = True          # save LR image
    save_restoration    = True          # save restored image
    save_LEH            = False         # save zoomed LR, E and H images
    save_progressive    = False         # save generation process
    save_kernel         = True          # save blur kernel
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


def method_apply_DiffPIR_for_deblurring(
        config: DiffPIRDeblurConfig,
        clean_image_filename: str,
        degraded_image_filename: str,
        kernel_filename: str,
        ext: str,
        clean_image: np.ndarray,
        degraded_image: np.ndarray,
        kernel: np.ndarray,
        k_4d: torch.Tensor,
        model: UNetModel,
        diffusion: SpacedDiffusion,
    ) -> None:
    """
    Apply the Diffusion PIR method to deblur an image.

    ARGUMENTS:
        - config: a DiffPIRDeblurConfig object
        - clean_image_filename: the name of the clean image (without extension, ex: "my_clean_image")
        - degraded_image_filename: the name of the degraded image (without extension, ex: "my_degraded_image")
        - kernel_filename: the name of the kernel (without extension, ex: "my_kernel")
        - ext: the extension of the image (ex: ".png")
        - clean_image: the clean image as a numpy array
        - degraded_image: the degraded image as a numpy array
        - kernel: the blur kernel (see delires.utils.delires_utils.create_blur_kernel() for more details)
        - k_4d: the blur kernel as a 4D tensor (see delires.utils.delires_utils.create_blur_kernel() for more details)
        - model: the UNetModel object
        - diffusion: the SpacedDiffusion object

    RETURNS:
        - None for the moment but save the results according to config and specified paths.
    """

    ### 1 - CONFIGURATION

    config = DiffPIRDeblurConfig()

    noise_level_model = config.noise_level_img # set noise level of model, default: 0
    skip = config.num_train_timesteps//config.iter_num # skip interval
    sigma = max(0.001,config.noise_level_img) # noise level associated with condition y

    # model_zoo = os.path.join(config.cwd, 'model_zoo') # fixed
    # testsets = os.path.join(config.cwd, 'testsets') # fixed
    # results = os.path.join(config.cwd, 'results') # fixed
    # model_path = os.path.join(model_zoo, config.model_name+'.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    torch.cuda.empty_cache()
    print(f"Using device {device}")

    result_name = f'{degraded_image_filename}_{config.task_current}_{config.generate_mode}_{config.model_name}_sigma{config.noise_level_img}_NFE{config.iter_num}_eta{config.eta}_zeta{config.zeta}_lambda{config.lambda_}_blurmode{config.blur_mode}'
    result_path = RESTORED_DATA_PATH
    Path(result_path).mkdir(parents=True, exist_ok=True)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(result_path, f"{logger_name}.log"))
    logger = logging.getLogger(logger_name)

    # load the kernel
    # kernel, k_4d = create_blur_kernel(
    #     use_DIY_kernel=config.use_DIY_kernel,
    #     blur_mode=config.blur_mode,
    #     kernel_size=config.kernel_size,
    #     seed=config.seed,
    #     cwd=config.cwd,
    # )

    # noise schedule 
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, config.num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod) # equivalent noise sigma on image

    noise_model_t           = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t           = 0
    
    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img) # start timestep of the diffusion process
    t_start                 = config.num_train_timesteps - 1              


    ### 2 - L_path, E_path <=> input_path, output_path

    # L_path = os.path.join(testsets, config.testset_name) # L_path, for Low-quality images
    # E_path = os.path.join(results, result_name) # E_path, for Estimated images
    # util.mkdir(E_path)


    ### 3 - MODEL

    # model_path = os.path.join(MODELS_PATH, f"{config.model_name}.pt")
    # if config.model_name == "diffusion_ffhq_10m":
    #     model_config = dict(
    #         model_path=model_path,
    #         num_channels=128,
    #         num_res_blocks=1,
    #         attention_resolutions="16",
    #     )
    # else:
    #     model_config = dict(
    #         model_path=model_path,
    #         num_channels=256,
    #         num_res_blocks=2,
    #         attention_resolutions="8,16,32",
    #     )
    # args = utils_model.create_argparser(model_config).parse_args([])
    # model, diffusion = create_model_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys()))
    # # model.load_state_dict(
    # #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # # )
    # model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if config.generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(config.model_name, config.noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(config.eta, config.zeta, config.lambda_, config.guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, config.skip_type, skip, noise_model_t))
    logger.info('use_DIY_kernel:{}, blur mode:{}'.format(config.use_DIY_kernel, config.blur_mode))
    # logger.info('Model path: {:s}'.format(model_path))
    logger.info(f"Clean image: {os.path.join(CLEAN_DATA_PATH, clean_image_filename)}")
    logger.info(f"Degraded image: {os.path.join(DEGRAGDED_DATA_PATH, degraded_image_filename)}")
    logger.info(f"Kernel: {os.path.join(DEGRAGDED_DATA_PATH, kernel_filename)}")
    
    if config.calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    

    ### 2 - LOAD IMAGES
        
    img_H = clean_image
    img_L = degraded_image
    

    ### 4 - RESTORATION LOGIC

    def __restoration_logic(
            img_L: np.ndarray,
            img_H: np.ndarray,
            img_name: str,
            ext: str,
            lambda_: float, 
            zeta: float, 
        ) -> None:
        """
        Perform restoration of a single image using the <config> parameters and <model> previously defined.

        ARGUMENTS:
            - img_L: the blurred and noised image (low quality images = original images)
            - img_H: the original image (estimated images = high quality images)
            - img_name: the name of the image (without extension, ex: "my_image")
            - ext: the extension of the image (ex: ".png")
            - lambda_: the lambda parameter
            - zeta: the zeta parameter
        """
        logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(config.eta, zeta, lambda_, config.guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if config.calc_LPIPS:
            test_results['lpips'] = []

        # if config.save_kernel:
        #     util.imsave(kernel*255.*200, os.path.join(E_path, "blur_kernel.jpeg"))

        # if config.save_L:
        #     util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_LR'+ext))
        
        model_out_type = config.model_output_type

        # --------------------------------
        # (2) get rhos and sigmas
        # --------------------------------

        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(config.num_train_timesteps):
            sigmas.append(reduced_alpha_cumprod[config.num_train_timesteps-1-i])
            if model_out_type == 'pred_xstart' and config.generate_mode == 'DiffPIR':
                sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
            #elif model_out_type == 'pred_x_prev':
            else:
                sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
            rhos.append(lambda_*(sigma**2)/(sigma_ks[i]**2))    
        rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)
        
        # --------------------------------
        # (3) initialize x, and pre-calculation
        # --------------------------------

        # x = util.single2tensor4(img_L).to(device)

        y = util.single2tensor4(img_L).to(device) # (1,3,256,256)

        # for y with given noise level, add noise from t_y
        t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * config.noise_level_img)
        sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
        x = sqrt_alpha_effective * (2*y-1) + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - \
                sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)
        # x = torch.randn_like(y)

        k_tensor = util.single2tensor4(np.expand_dims(kernel, 2)).to(device)

        FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, config.sf)

        # --------------------------------
        # (4) main iterations
        # --------------------------------

        progress_img = []
        # create sequence of timestep for sampling
        if config.skip_type == 'uniform':
            seq = [i*skip for i in range(config.iter_num)]
            if skip > 1:
                seq.append(config.num_train_timesteps-1)
        elif config.skip_type == "quad":
            seq = np.sqrt(np.linspace(0, config.num_train_timesteps**2, config.iter_num))
            seq = [int(s) for s in list(seq)]
            seq[-1] = seq[-1] - 1
        progress_seq = seq[::max(len(seq)//10,1)]
        if progress_seq[-1] != seq[-1]:
            progress_seq.append(seq[-1])

        # plot the values in <seq>
        plot_sequence(seq, path=RESTORED_DATA_PATH, title=f'seq_{img_name}')
        
        # reverse diffusion for one image from random noise
        for i in tqdm(range(len(seq))):
            curr_sigma = sigmas[seq[i]].cpu().numpy()
            # time step associated with the noise level sigmas[i]
            t_i = utils_model.find_nearest(reduced_alpha_cumprod,curr_sigma)
            # skip iters
            if t_i > t_start:
                continue
            for u in range(config.iter_num_U):
                # --------------------------------
                # step 1, reverse diffsuion step
                # --------------------------------

                # solve equation 6b with one reverse diffusion step
                if 'DPS' in config.generate_mode:
                    x = x.requires_grad_()
                    xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type='pred_x_prev_and_start', \
                                model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                else:
                    x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                            model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                #   model_out_type=model_out_type, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                # --------------------------------
                # step 2, FFT
                # --------------------------------

                if seq[i] != seq[-1]:
                    if config.generate_mode == 'DiffPIR':
                        if config.sub_1_analytic:
                            if model_out_type == 'pred_xstart':
                                tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                # when noise level less than given image noise, skip
                                if i < config.num_train_timesteps-noise_model_t: 
                                    x0_p = x0 / 2 + 0.5
                                    x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, config.sf)
                                    x0_p = x0_p * 2 - 1
                                    # effective x0
                                    x0 = x0 + config.guidance_scale * (x0_p-x0)
                                else:
                                    model_out_type = 'pred_x_prev'
                                    x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                            model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                                    # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                                    #   model_out_type=model_out_type, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                                    pass
                        else:
                            # zeta=0.28; lambda_=7
                            x0 = x0.requires_grad_()
                            # first order solver
                            def Tx(x):
                                x = x / 2 + 0.5
                                pad_2d = torch.nn.ReflectionPad2d(kernel.shape[0]//2)
                                x_deblur = F.conv2d(pad_2d(x), k_4d)
                                return x_deblur
                            norm_grad, norm = utils_model.grad_and_value(operator=Tx,x=x0, x_hat=x0, measurement=y)
                            x0 = x0 - norm_grad * norm / (rhos[t_i])
                            x0 = x0.detach_()
                            pass                               
                    elif 'DPS' in config.generate_mode:
                        def Tx(x):
                            x = x / 2 + 0.5
                            pad_2d = torch.nn.ReflectionPad2d(k.shape[0]//2)
                            x_deblur = F.conv2d(pad_2d(x), k_4d)
                            return x_deblur
                            #return kernel.forward(x)                         
                        if config.generate_mode == 'DPS_y0':
                            norm_grad, norm = utils_model.grad_and_value(operator=Tx,x=x, x_hat=x0, measurement=y)
                            #norm_grad, norm = utils_model.grad_and_value(operator=Tx,x=xt, x_hat=x0, measurement=y)    # does not work
                            x = xt - norm_grad * 1. #norm / (2*rhos[t_i]) 
                            x = x.detach_()
                            pass
                        elif config.generate_mode == 'DPS_yt':
                            y_t = sqrt_alphas_cumprod[t_i] * (2*y-1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(y) # add AWGN
                            y_t = y_t/2 + 0.5
                            ### it's equivalent to use x & xt (?), but with xt the computation is faster.
                            #norm_grad, norm = utils_model.grad_and_value(operator=Tx,x=x, x_hat=xt, measurement=y_t)
                            norm_grad, norm = utils_model.grad_and_value(operator=Tx,x=xt, x_hat=xt, measurement=y_t)
                            x = xt - norm_grad * lambda_ * norm / (rhos[t_i]) * 0.35
                            x = x.detach_()
                            pass

                if (config.generate_mode == 'DiffPIR' and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == config.iter_num_U-1):
                    #x = sqrt_alphas_cumprod[t_i] * (x0) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                    
                    t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                    # calculate \hat{\eposilon}
                    eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                    eta_sigma = config.eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                    x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                else:
                    # x = x0
                    pass
                    
                # set back to x_t from x_{t-1}
                if u < config.iter_num_U-1 and seq[i] != seq[-1]:
                    # x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)
                    sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                    x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                            sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)

            # save the process
            x_0 = (x/2+0.5)
            if config.save_progressive and (seq[i] in progress_seq):
                x_show = x_0.clone().detach().cpu().numpy()       #[0,1]
                x_show = np.squeeze(x_show)
                if x_show.ndim == 3:
                    x_show = np.transpose(x_show, (1, 2, 0))
                progress_img.append(x_show)
                if config.log_process:
                    logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                
                if config.show_img:
                    util.imshow(x_show)

        # --------------------------------
        # (3) img_E
        # --------------------------------

        img_E = util.tensor2uint(x_0)
            
        psnr = util.calculate_psnr(img_E, img_H, border=config.border)  # change with your own border
        test_results['psnr'].append(psnr)
        
        if config.calc_LPIPS:
            img_H_tensor = np.transpose(img_H, (2, 0, 1))
            img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
            img_H_tensor = img_H_tensor / 255 * 2 -1
            lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
            lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
            test_results['lpips'].append(lpips_score)
            logger.info('{:>10s} PSNR: {:.4f}dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(img_name+ext, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
        else:
            logger.info('{:>10s} PSNR: {:.4f}dB'.format(img_name+ext, psnr))

        if config.n_channels == 1:
            img_H = img_H.squeeze()

        if config.save_restoration:
            util.imsave(img_E, os.path.join(RESTORED_DATA_PATH, img_name+'_'+config.model_name+ext))

        if config.save_progressive:
            now = datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            img_total = cv2.hconcat(progress_img)
            if config.show_img:
                util.imshow(img_total,figsize=(80,4))
            util.imsave(img_total*255., os.path.join(RESTORED_DATA_PATH, img_name+'_sigma_{:.3f}_process_lambda_{:.3f}_{}_psnr_{:.4f}{}'.format(config.noise_level_img,lambda_,current_time,psnr,ext)))
                                                                        
        # --------------------------------
        # (4) img_LEH
        # --------------------------------

        if config.save_LEH:
            img_L = util.single2uint(img_L)
            k_v = kernel/np.max(kernel)*1.0
            k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
            k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_I = cv2.resize(img_L, (config.sf*img_L.shape[1], config.sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
            img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
            util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if config.show_img else None
            util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(RESTORED_DATA_PATH, img_name+'_LEH'+ext))

        # if config.save_L:
        #     util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_LR'+ext))
        
        # --------------------------------
        # Average PSNR and LPIPS
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), sigma: ({:.3f}): {:.4f} dB'.format(config.testset_name, noise_level_model, ave_psnr))

        if config.calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}) sigma: ({:.3f}): {:.4f}'.format(config.testset_name, noise_level_model, ave_lpips))


    ### 5 - APPLY RESTORATION
            
    lambdas = [config.lambda_*i for i in range(7,8)]
    for lambda_ in lambdas:
        for zeta_i in [config.zeta*i for i in range(3,4)]:
            __restoration_logic(
                img_L=img_L,
                img_H=img_H,
                img_name=degraded_image_filename,
                ext=ext,
                lambda_=lambda_,
                zeta=zeta_i, 
            )
        

def main():

    img = "69037" # image name without extension in the test location described in the configuration

    config = DiffPIRDeblurConfig()

    # Create the blur kernel
    k, k_4d = create_blur_kernel(
        use_DIY_kernel=config.use_DIY_kernel,
        blur_mode=config.blur_mode,
        kernel_size=config.kernel_size,
        seed=config.seed,
        cwd=config.cwd,
    )

    # Build path to the image <img>
    img_path = manually_build_image_path(img, config.testset_name, config.cwd)
    # print(f"Image path: {img_path}")

    # Create the degraded image
    img_L, img_H, img_name, ext = create_blurred_and_noised_image(
        kernel=k, 
        img=img_path,
        n_channels=config.n_channels,
        noise_level_img=config.noise_level_img,
    )

    # <method_apply_DiffPIR_for_deblurring> must be used in a Diffuser apply method


if __name__ == '__main__':

    main()
