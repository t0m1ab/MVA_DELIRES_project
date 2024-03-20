import os.path
from logging import Logger
import cv2
from tqdm import tqdm
from dataclasses import dataclass

import numpy as np
import torch
from datetime import datetime

from delires.methods.diffpir.diffpir_configs import DiffPIRInpaintingConfig
from delires.methods.diffpir.utils import utils_model
from delires.methods.diffpir.utils import utils_sisr as sr
from delires.methods.diffpir.utils import utils_image as util
from delires.methods.diffpir.utils.delires_utils import manually_build_image_path
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
from delires.methods.diffpir.guided_diffusion.respace import SpacedDiffusion

# from guided_diffusion import dist_util
from delires.methods.diffpir.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from delires.data import load_masks, create_masked_image

from delires.params import (
    MODELS_PATH,
    OPERATORS_PATH,
    RESTORED_DATA_PATH,
    CLEAN_DATA_PATH,
    DEGRADED_DATA_PATH,
)


def build_result_name(img_name: str, config: DiffPIRInpaintingConfig) -> str:
    result_name = f"{img_name}"
    result_name += f"_{config.task_current}"
    result_name += f"_{config.generate_mode}"
    result_name += f"_{config.model_name}"
    result_name += f"_sigma{config.noise_level_img}"
    result_name += f"_NFE{config.iter_num}"
    result_name += f"_eta{config.eta}"
    result_name += f"_zeta{config.zeta}"
    result_name += f"_lambda{config.lambda_}"
    return result_name


def apply_DiffPIR_for_inpainting(
        config: DiffPIRInpaintingConfig,
        clean_image_filename: str,
        degraded_image_filename: str,
        masks_filename: str,
        mask_index: int,
        clean_image: np.ndarray,
        degraded_image: np.ndarray,
        mask: np.ndarray,
        model: UNetModel,
        diffusion: SpacedDiffusion,
        img_ext: str = "png",
        logger: Logger = None,
        device = "cpu"
    ) -> tuple[np.ndarray, dict]:
    """
    Apply the Diffusion PIR method to deblur an image.

    ARGUMENTS:
        - config: a DiffPIRDeblurConfig object
        - clean_image_filename: the name of the clean image (without extension, ex: "my_clean_image")
        - degraded_image_filename: the name of the degraded image (without extension, ex: "my_degraded_image")
        - masks_filename: the name of the set of masks (without extension, ex: "my_masks")
        - mask_index: the index of the mask within the set of masks
        - clean_image: the clean image as a numpy array
        - degraded_image: the degraded image as a numpy array
        - mask: the mask as a numpy array
        - model: the UNetModel object
        - diffusion: the SpacedDiffusion object
        - img_ext: the extension of the image (ex: "png")
        - logger: a Logger object. If None, nothing will be logged.

    RETURNS:
        - img_E: the restored image as a numpy array
        - metrics: a dict containing the PSNR and LPIPS scores
    """

    ### 1 - CONFIGURATION ADJUSTMENTS + LOGGER + DEVICE
    """
    Define the configuration, the logger and the device to be used.
    """

    if config.calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # setup logger
    # result_name = build_result_name(degraded_image_filename, config)
    # logger_name = result_name
    # utils_logger.logger_info(logger_name, log_path=os.path.join(RESTORED_DATA_PATH, f"{logger_name}.log"))
    # logger = logging.getLogger(logger_name)
    if logger is not None:
        logger.info(f"Using device {device}")


    ### 2 - NOISE SCHEDULE
    """
    Define the noise schedule for the diffusion process.
    """

    noise_level_model = config.noise_level_img # set noise level of model, default: 0
    skip = config.num_train_timesteps // config.iter_num # skip interval
    sigma = max(0.001, config.noise_level_img) # noise level associated with condition y

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


    ### 3 - MODEL
    """
    Set the model in evaluation mode and send it to the device.
    """

    model.eval()
    if config.generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)


    ### 4 - LOGGING START
    """
    Logs useful information about the configuration and the image to be restored before the restoration starts.
    """

    if logger is not None:
        logger.info(f"model_name: {config.model_name} | image sigma: {config.noise_level_img} | model sigma: {noise_level_model}")
        logger.info(f"eta: {config.eta} | zeta: {config.zeta} | lambda: {config.lambda_} | guidance_scale: {config.guidance_scale}")
        logger.info(f"start step: {t_start} | skip_type: {config.skip_type} | skip interval: {skip} | skipstep analytic steps: {noise_model_t}")
        logger.info(f"Clean image: {clean_image_filename}")
        logger.info(f"Degraded image: {degraded_image_filename}")
        logger.info(f"Masks: {masks_filename}, index: {mask_index}")
    

    ### 5 - SETUP ADAPTED VAR NAMES FOR THE RESTORATION LOGIC
    """
    Original var names chosen by the authors were not very explicit but we didn't change them in the restoration logic to avoid mistakes.
    """
        
    img_H = clean_image
    img_L = degraded_image
    img_name = degraded_image_filename    
    lambda_ = config.lambda_ 
    zeta = config.zeta
    

    ### 6 - APPLY RESTORATION LOGIC
    """
    Perform restoration of a single image using the <config> parameters and <model> previously defined.
    """

    # --------------------------------
    # (1) init metrics dict
    # --------------------------------

    model_out_type = config.model_output_type
    metrics = {}

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

    # convert mask to tensor
    mask = util.single2tensor4(mask.astype(np.float32)).to(device) 
    
    y = util.single2tensor4(img_L).to(device) # (1,3,256,256)
    y = y * 2 -1        # [-1,1]

    # for y with given noise level, add noise from t_y
    t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * config.noise_level_img)
    sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
    x = sqrt_alpha_effective * (2*y-1) + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - \
            sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)

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
    # plot_sequence(seq, path=RESTORED_DATA_PATH, title=f'seq_{img_name}')
    
    # reverse diffusion for one image from random noise
    for i in tqdm(range(len(seq)), desc="DiffPIR sampling"):
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
            
            # add noise, make the image noise level consistent in pixel level
            if config.generate_mode == 'repaint':
                x = (sqrt_alphas_cumprod[t_i] * y + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)) * mask \
                        + (1-mask) * x

            # solve equation 6b with one reverse diffusion step
            if model_out_type == 'pred_xstart':
                x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                        model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
            else:
                x = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                        model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
            # x = utils_model.test_mode(model_fn, x, mode=0, refield=32, min_size=256, modulo=16, noise_level=sigmas[i].cpu().numpy()*255)
            # --------------------------------
            # step 2, closed-form solution
            # --------------------------------

            if (config.generate_mode == 'DiffPIR') and not (seq[i] == seq[-1]): 
                # solve sub-problem
                if config.sub_1_analytic:
                    if model_out_type == 'pred_xstart':
                        # when noise level less than given image noise, skip
                        if i < config.num_train_timesteps-noise_model_t:    
                            x0_p = (mask*y + rhos[t_i].float()*x0).div(mask+rhos[t_i])
                            x0 = x0 + config.guidance_scale * (x0_p-x0)
                        else:
                            model_out_type = 'pred_x_prev'
                            x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                                model_diffusion=model, diffusion=diffusion, ddim_sample=config.ddim_sample, alphas_cumprod=alphas_cumprod)
                            pass
                    elif model_out_type == 'pred_x_prev':
                        # when noise level less than given image noise, skip
                        if i < config.num_train_timesteps-noise_model_t:    
                            x = (mask*y + rhos[t_i].float()*x).div(mask+rhos[t_i]) # y-->yt ?
                        else:
                            pass
                else:
                    raise NotImplementedError("First order solver for data-fitting term is not implemented for inpainting yet")
                    # TODO: first order solver
                    # x = x - 1 / (2*rhos[t_i]) * (x - y_t) * mask 
                    pass

            if (model_out_type == 'pred_xstart') and not (seq[i] == seq[-1]):
                # x = sqrt_alphas_cumprod[t_i] * (x) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x) # x = sqrt_alphas_cumprod[t_i] * (x) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                
                t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                # calculate \hat{\eposilon}
                eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                eta_sigma = config.eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                            + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)

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
            if config.log_process and logger is not None:
                logger.info(f"{seq[i]} | steps {t_i} | np.max(x_show): {np.max(x_show):.4f} | np.min(x_show): {np.min(x_show):.4f}")            
            if config.show_img:
                util.imshow(x_show)
                
    # --------------------------------
    # (5) comppute scores with restored image
    # --------------------------------

    img_E = util.tensor2uint(x_0)
    
    # compute PSNR
    psnr = util.calculate_psnr(img_E, img_H)
    metrics['psnr'] = psnr
    
    if config.calc_LPIPS:
        img_H_tensor = np.transpose(img_H, (2, 0, 1))
        img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
        img_H_tensor = img_H_tensor / 255 * 2 -1
        lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
        lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
        metrics['lpips'] = lpips_score
        if logger is not None:
            logger.info(f"{img_name:>10s} PSNR: {psnr:.4f}dB LPIPS: {lpips_score:.4f}")
    else:
        if logger is not None:
            logger.info(f"{img_name:>10s} PSNR: {psnr:.4f}dB")

    if config.n_channels == 1:
        img_H = img_H.squeeze()

    if config.save_restoration:
        util.imsave(img_E, os.path.join(RESTORED_DATA_PATH, f"{img_name}_{config.model_name}.{img_ext}"))
        
    if config.save_LEH:
        util.imsave(np.concatenate([util.single2uint(img_L), img_E, img_H], axis=1), os.path.join(RESTORED_DATA_PATH, f"{img_name}_{config.model_name}_LEH.{img_ext}"))

    if config.save_progressive:
        now = datetime.now()
        current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        img_total = cv2.hconcat(progress_img)
        if config.show_img:
            util.imshow(img_total,figsize=(80,4))
        util.imsave(img_total*255., os.path.join(RESTORED_DATA_PATH, img_name+'_sigma_{:.3f}_process_lambda_{:.3f}_{}_psnr_{:.4f}.{}'.format(config.noise_level_img,lambda_,current_time,psnr,img_ext)))
        images = []
        y_t = np.squeeze((y/2+0.5).cpu().numpy())
        if y_t.ndim == 3:
            y_t = np.transpose(y_t, (1, 2, 0))
        if config.generate_mode in ['repaint','DiffPIR']:
            for x in progress_img:
                images.append((y_t)* mask+ (1-mask) * x)
            img_total = cv2.hconcat(images)
            if config.show_img:
                util.imshow(img_total,figsize=(80,4))
            if config.save_progressive_mask:
                util.imsave(img_total*255., os.path.join(RESTORED_DATA_PATH, img_name+'_process_mask_lambda_{:.3f}_{}{}'.format(lambda_,current_time,img_ext)))

    return img_E, metrics
    

def main():

    img = "69037" # image name without extension in the test location described in the configuration

    config = DiffPIRInpaintingConfig()

    # Build path to the image <img>
    img_path = manually_build_image_path(img, config.testset_name, config.cwd)
    # print(f"Image path: {img_path}")
    
    # Load mask
    mask = load_masks("random_masks")[0]

    # Create the degraded image
    img_L, img_H, img_name, img_ext = create_masked_image(
        mask=mask, 
        img=img_path,
        n_channels=config.n_channels,
        noise_level_img=config.noise_level_img,
    )

    # <method_apply_DiffPIR_for_deblurring> must be used in a Diffuser apply method


if __name__ == '__main__':

    main()
