import os.path
import cv2
import logging
from tqdm import tqdm
from dataclasses import dataclass


import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict
import hdf5storage
from functools import partial

from delires.methods.diffpir.utils import utils_model
from delires.methods.diffpir.utils import utils_logger
from delires.methods.diffpir.utils import utils_sisr as sr
from delires.methods.diffpir.utils import utils_image as util
from delires.methods.diffpir.utils.utils_resizer import Resizer
from delires.methods.diffpir.utils.delires_utils import (
    create_downsampled_image, 
    get_downsample_kernel, 
    manually_build_image_path, 
    plot_sequence,
)

# from guided_diffusion import dist_util
from delires.methods.diffpir.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

@dataclass
class DiffPIRSisrConfig:

    noise_level_img     = 12.75/255.0                 # set AWGN noise level for LR image, default: 0
    model_name          = 'diffusion_ffhq_10m'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusion model
    testset_name        = 'demo_test'                  # set testing set,  'imagenet_val' | 'ffhq_val'
    num_train_timesteps = 1000
    iter_num            = 100           # set number of iterations
    iter_num_U          = 1             # set number of inner iterations, default: 1
    sr_mode             = 'blur'        # 'blur', 'cubic' mode of sr up/down sampling

    show_img            = False         # default: False
    save_L              = True          # save LR image
    save_E              = True          # save estimated image
    save_LEH            = False         # save zoomed LR, E and H images
    save_progressive    = False         # save generation process
    save_kernel         = True          # save blur kernel
    # border              = 0
	
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
    
    sf                 = 4               # set scale factor, default: [2, 3, 4], [2], [3], [4]
    kernel_idx = 1                        # set kernel index for downsampling
    inIter                  = 1                 # iter num for sr solution (for iterative back-projection): 4-6
    gamma                   = 1/100             # coef for iterative sr solver 20steps: 0.05-0.10 for zeta=1, 0.09-0.13 for zeta=0 
    classical_degradation   = False             # set classical degradation or bicubic degradation

    calc_LPIPS          = True

    task_current        = 'sr'      # 'sr' for super resolution   
    n_channels          = 3             # fixed
    cwd                 = ''  


def apply_DiffPIR_for_sisr(
        config: DiffPIRSisrConfig,
        img_name: str,
        ext: str,
        img_L: np.ndarray,
        img_H: np.ndarray,
        kernel: np.ndarray,
        k_index: int
    ):

    """
    Apply the Diffusion PIR method to increase the resolution of an image.

    ARGUMENTS:
        - config: a DiffPIRDeblurConfig object
        - img_name: the name of the image (without extension, ex: "my_image")
        - ext: the extension of the image (ex: ".jpeg")
        - img_L: the blurred and noised image (low quality images = original images)
        - img_H: the original image (estimated images = high quality images)
        - kernels: the kernels used to blur and/or downsample the images
        - kernel_idx: the index of the kernel used to blur and/or downsample the images

    RETURNS:
        - None for the moment but save the results according to paths specified in the configuration.
    """

    ### 1 - CONFIGURATION

    config = DiffPIRSisrConfig()

    noise_level_model = config.noise_level_img # set noise level of model, default: 0
    skip = config.num_train_timesteps//config.iter_num # skip interval
    sigma = max(0.001,config.noise_level_img) # noise level associated with condition y

    model_zoo = os.path.join(config.cwd, 'model_zoo') # fixed
    testsets = os.path.join(config.cwd, 'testsets') # fixed
    results = os.path.join(config.cwd, 'results') # fixed
    result_name = f'{config.testset_name}_{config.task_current}_{config.generate_mode}_{config.model_name}_sigma{config.noise_level_img}_NFE{config.iter_num}_eta{config.eta}_zeta{config.zeta}_lambda{config.lambda_}_srmode{config.sr_mode}'
    model_path = os.path.join(model_zoo, config.model_name+'.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    torch.cuda.empty_cache()
    print(f"Using device {device}")

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

    L_path = os.path.join(testsets, config.testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    ### 3 - MODEL

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if config.model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    logger.info('Loading model %s', config.model_name)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if config.generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, sr_mode:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(config.model_name, config.sr_mode, config.noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(config.eta, config.zeta, config.lambda_, config.guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, config.skip_type, skip, noise_model_t))
    logger.info('analytic iter num:{}, gamma:{}'.format(config.inIter, config.gamma))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)

    ### 4 - RESTORATION LOGIC

    if config.calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    logger.info('--------- sf:{:>1d} --k:{:>2d} ---------'.format(config.sf, k_index))

    util.surf(kernel) if config.show_img else None

    def __restoration_logic(
            img_L: np.ndarray,
            img_H: np.ndarray,
            img_name: str,
            ext: str,
            lambda_: float, 
            zeta: float, 
        ): 
        border = config.sf
        
        logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, inIter:{:.3f}, gamma:{:.3f}, guidance_scale:{:.2f}'.format(config.eta, zeta, lambda_, config.inIter, config.gamma, config.guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['psnr_y'] = []
        if config.calc_LPIPS:
            test_results['lpips'] = []
        model_out_type = config.model_output_type
        
        if config.sr_mode == "cubic":
            # samplers used for IBP
            down_sample = Resizer(img_H_tensor.shape, 1/config.sf).to(device)
            up_sample = partial(F.interpolate, scale_factor=config.sf)

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

        x = cv2.resize(img_L, (img_L.shape[1]*config.sf, img_L.shape[0]*config.sf), interpolation=cv2.INTER_CUBIC)
        if np.ndim(x)==2:
            x = x[..., None]

        if config.classical_degradation:
            x = sr.shift_pixel(x, config.sf)
        x = util.single2tensor4(x).to(device)

        y = util.single2tensor4(img_L).to(device)   #(1,3,256,256) [0,1]

        # x = torch.randn_like(x)
        x = sqrt_alphas_cumprod[t_start] * (2*x-1) + sqrt_1m_alphas_cumprod[t_start] * torch.randn_like(x)

        k_tensor = util.single2tensor4(np.expand_dims(kernel, 2)).to(device) 

        FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, config.sf)

        # --------------------------------
        # (4) main iterations
        # --------------------------------

        progress_img = []
        # create sequence of timestep for sampling
        skip = config.num_train_timesteps//config.iter_num
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
        plot_sequence(seq, path=E_path, title=f'seq_{img_name}')
        
        # reverse diffusion for one image from random noise
        for i in tqdm(range(len(seq))):
            curr_sigma = sigmas[seq[i]].cpu().numpy()
            # time step associated with the noise level sigmas[i]
            t_i = utils_model.find_nearest(reduced_alpha_cumprod,curr_sigma)
            # skip iters
            if t_i > t_start:
                continue
            # repeat for semantic consistence: from repaint
            for u in range(config.iter_num_U):
                # --------------------------------
                # step 1, reverse diffsuion step
                # --------------------------------

                ### solve equation 6b with one reverse diffusion step
                if 'DPS' in config.generate_mode:
                    x = x.requires_grad_()
                    xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type='pred_x_prev_and_start', \
                                model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                else:
                    x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                            model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                #       model_out_type=model_out_type, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                # --------------------------------
                # step 2, FFT
                # --------------------------------
                if seq[i] != seq[-1]:
                    if config.generate_mode == 'DiffPIR':
                        if config.sub_1_analytic:
                            if model_out_type == 'pred_xstart':
                                # when noise level less than given image noise, skip
                                if i < config.num_train_timesteps-noise_model_t: 
                                    if config.sr_mode == 'blur':
                                        tau = rhos[t_i].float().repeat(1, 1, 1, 1)
                                        x0_p = x0 / 2 + 0.5
                                        x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, config.sf)
                                        x0_p = x0_p * 2 - 1
                                        # effective x0
                                        x0 = x0 + config.guidance_scale * (x0_p-x0)
                                    elif config.sr_mode == 'cubic': 
                                        # iterative back-projection (IBP) solution
                                        for _ in range(config.inIter):
                                            x0 = x0 / 2 + 0.5
                                            x0 = x0 + config.gamma * up_sample((y - down_sample(x0))) / (1+rhos[t_i])
                                            x0 = x0 * 2 - 1
                                else:
                                    model_out_type = 'pred_x_prev'
                                    x0 = utils_model.model_fn(x, noise_level=curr_sigma*255,model_out_type=model_out_type, \
                                            model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                                    # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                                    #       model_out_type=model_out_type, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                                    pass
                        else:
                            # zeta=0.25; lambda_=15: FFHQ
                            # zeta=0.35; lambda_=35: ImageNet
                            x0 = x0.requires_grad_()
                            # first order solver
                            down_sample = Resizer(x.shape, 1/config.sf).to(device)
                            #norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=x0/2+0.5, x_hat=x0, measurement=y)
                            norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=x0, x_hat=x0, measurement=2*y-1)
                                                
                            x0 = x0 - norm_grad * norm / (rhos[t_i]) 
                            x0 = x0.detach_()
                            pass                          
                    elif 'DPS' in config.generate_mode:
                        down_sample = Resizer(x.shape, 1/config.sf).to(device)                        
                        if config.generate_mode == 'DPS_y0':
                            norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=x, x_hat=x0, measurement=2*y-1)
                            #norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=xt, x_hat=x0, measurement=2*y-1)    # does not work
                            x = xt - norm_grad * 1. #norm / (2*rhos[t_i]) 
                            x = x.detach_()
                            pass
                        elif config.generate_mode == 'DPS_yt':
                            y_t = sqrt_alphas_cumprod[t_i] * (2*y-1) + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(y) # add AWGN
                            #y_t = y_t/2 + 0.5
                            #norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=x, x_hat=xt, measurement=y_t)    # no need to use
                            norm_grad, norm = utils_model.grad_and_value(operator=down_sample,x=xt, x_hat=xt, measurement=y_t)
                            x = xt - norm_grad * lambda_ * norm / (rhos[t_i]) * 0.35
                            x = x.detach_()
                            pass

                # add noise back to t=i-1
                if (config.generate_mode == 'DiffPIR' and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == config.iter_num_U-1):
                    #x = sqrt_alphas_cumprod[t_i] * (x0) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                    
                    t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                    eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                    # calculate \hat{\eposilon}
                    eta_sigma = config.eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                    x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                else:
                    #x = x0
                    pass
                    
                # set back to x_t from x_{t-1}
                if u < config.iter_num_U-1 and seq[i] != seq[-1]:
                    ### it's equivalent to use x & xt (?), but with xt the computation is faster.
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

        psnr = util.calculate_psnr(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        
        if config.calc_LPIPS:
            img_H_tensor = np.transpose(img_H, (2, 0, 1))
            img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
            img_H_tensor = img_H_tensor / 255 * 2 -1
            lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
            lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
            test_results['lpips'].append(lpips_score)
            logger.info('{:>10s} -- sf:{:>1d} --k:{:>2d} PSNR: {:.4f}dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(img_name+ext, config.sf, k_index, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
        else:
            logger.info('{:>10s} -- sf:{:>1d} --k:{:>2d} PSNR: {:.4f}dB'.format(img_name+ext, config.sf, k_index, psnr))

        if config.save_E:
            util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(config.sf)+'_k'+str(k_index)+'_'+config.model_name+ext))

        if config.n_channels == 1:
            img_H = img_H.squeeze()

        if config.save_progressive:
            now = datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            img_total = cv2.hconcat(progress_img)
            if config.show_img:
                util.imshow(img_total,figsize=(80,4))
            util.imsave(img_total*255., os.path.join(E_path, img_name+'_sigma_{:.3f}_process_lambda_{:.3f}_{}_psnr_{:.4f}{}'.format(config.noise_level_img,lambda_,current_time,psnr,ext)))
            
        # --------------------------------
        # (4) img_LEH
        # --------------------------------

        img_L = util.single2uint(img_L).squeeze()

        if config.save_LEH:
            k_v = k/np.max(k)*1.0
            if config.n_channels==1:
                k_v = util.single2uint(k_v)
            else:
                k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, config.n_channels]))
            k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_I = cv2.resize(img_L, (config.sf*img_L.shape[1], config.sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_I[:k_v.shape[0], -k_v.shape[1]:, ...] = k_v
            img_I[:img_L.shape[0], :img_L.shape[1], ...] = img_L
            util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if config.show_img else None
            util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_x'+str(config.sf)+'_k'+str(k_index)+'_LEH'+ext))

        if config.save_L:
            util.imsave(img_L, os.path.join(E_path, img_name+'_x'+str(config.sf)+'_k'+str(k_index)+'_LR'+ext))

        if config.n_channels == 3:
            img_E_y = util.rgb2ycbcr(img_E, only_y=True)
            img_H_y = util.rgb2ycbcr(img_H, only_y=True)
            psnr_y = util.calculate_psnr(img_E_y, img_H_y)
            test_results['psnr_y'].append(psnr_y)
            
        # --------------------------------
        # Average PSNR and LPIPS
        # --------------------------------

        ave_psnr_k = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f} dB'.format(config.testset_name, config.sf, k_index, noise_level_model, ave_psnr_k))

        if config.n_channels == 3:  # RGB image
            ave_psnr_y_k = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            logger.info('------> Average PSNR(Y) of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f} dB'.format(config.testset_name, config.sf, k_index, noise_level_model, ave_psnr_y_k))

        if config.calc_LPIPS:
            ave_lpips_k = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.3f}): {:.4f}'.format(config.testset_name, config.sf, k_index, noise_level_model, ave_lpips_k))

    ### 5 - APPLY RESTORATION
    
    lambdas = [config.lambda_*i for i in range(7,8)]
    for lambda_ in lambdas:
        #for zeta_i in [zeta*i for i in range(2,4)]:
        for zeta_i in [0.25]:
            __restoration_logic(
                img_L=img_L,
                img_H=img_H,
                img_name=img_name,
                ext=ext,
                lambda_=lambda_,
                zeta=zeta_i, 
                )


def main():

    img = "69037" # image name without extension in the test location described in the configuration

    config = DiffPIRSisrConfig()
    
    # Instantiate the blur kernel
    k_index = 0
    kernel = get_downsample_kernel(
        classical_degradation = config.classical_degradation,
        sf = config.sf,
        k_index = k_index,
        cwd = config.cwd,
    )

    # Build path to the image <img>  # TODO: remove this as it won't be necessary.
    img_path = manually_build_image_path(img, config.testset_name, config.cwd)
    print(f"Image path: {img_path}")

    # Create the degraded image
    img_L, img_H, img_name, ext = create_downsampled_image(
        kernel=kernel, 
        img=img_path,
        sr_mode=config.sr_mode,
        classical_degradation=config.classical_degradation,
        sf=config.sf,
        n_channels=config.n_channels,
        noise_level_img=config.noise_level_img,
    )

    # Apply the Diffusion PIR method to deblur the image
    apply_DiffPIR_for_sisr(
        config=config,
        img_name=img_name, 
        ext=ext,
        img_L=img_L, 
        img_H=img_H, 
        kernel=kernel,
        k_index=k_index,
    )


if __name__ == '__main__':

    main()
