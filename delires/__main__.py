import os
import numpy as np
import logging
import torch
import glob

from delires.params import (
    MODE,
    MODELS_PATH,
    KERNELS_PATH, 
    CLEAN_DATA_PATH, 
    DEGRADED_DATA_PATH, 
    RESTORED_DATA_PATH,
)
from delires.diffusers.diffuser import DIFFUSER_TYPE, DIFFUSERS
from delires.utils.utils_logger import logger_info
from delires.diffusers.diffpir.diffpir_diffuser import DiffPIRDiffuser
from delires.diffusers.diffpir.diffpir_configs import DiffPIRConfig, DiffPIRDeblurConfig
from delires.data import fetch_kernel_name_from_dataset
import delires.utils.utils_image as utils_image
from delires.metrics import compute_metrics


def run_experiment(
    exp_name: str,
    diffuser_type: DIFFUSER_TYPE,
    mode: MODE, degraded_dataset_name: str,
    diffuser_config,
    diffuser_mode_config,
    device: torch.DeviceObjType|str = "cpu",
    nb_gen: int = 1,
    kernel_name: str|None = None,
    calc_LPIPS: bool = False
    ):
    """ 
    Run an experiment for a given method.
    """
    experiment_path = os.path.join(RESTORED_DATA_PATH, exp_name)

    # Create experiment folder
    os.makedirs(experiment_path, exist_ok=False)
      
    # setup device
    device = torch.device("cpu")
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        
    # Instantiate logger
    logger_name = exp_name
    logger_info(logger_name, log_path=os.path.join(experiment_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    
    # Log information about the experiment
    logger.info(f"Using device {device}")
    # TODO: do this logging
    # logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(config.model_name, config.noise_level_img, noise_level_model))
    # logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(config.eta, config.zeta, config.lambda_, config.guidance_scale))
    # logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, config.skip_type, skip, noise_model_t))
    # logger.info('use_DIY_kernel:{}, blur mode:{}'.format(config.use_DIY_kernel, config.blur_mode))
    # logger.info('Model path: {:s}'.format(model_path))

    
    # Instantiate diffuser
    diffpir_diffuser = DIFFUSERS[diffuser_type](diffuser_config, logger=logger)
    kernel_name = fetch_kernel_name_from_dataset(degraded_dataset_name) if kernel_name is None else kernel_name
    if kernel_name is None:
        raise ValueError("No kernel found for the given dataset. Please manually select a kernel.")
    diffpir_diffuser.load_blur_kernel(kernel_name)
    
    
 
    # Get datasets
    clean_images_names = glob.glob(os.path.join(CLEAN_DATA_PATH, "*.png"))
    degraded_images_names = glob.glob(os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, "*.png"))
    if len(clean_images_names) != len(degraded_images_names):
        logger.warning("The number of clean and degraded images is different.")
    
    
    metrics = {"PSNR": []} # TODO: other metrics
    # Apply the method over the dataset
    for i in range(len(clean_images_names)):
        clean_image_name = clean_images_names[i]
        degraded_image_name = degraded_images_names[i]
        clean_image = utils_image.imread_uint(clean_image_name)
        degraded_image = np.load(degraded_image_name+".npy")
        
        # Logger
        logger.info(f"Clean image: {os.path.join(CLEAN_DATA_PATH, clean_image_name)}")
        logger.info(f"Degraded image: {os.path.join(DEGRADED_DATA_PATH, degraded_image_name)}")
        logger.info(f"Kernel: {os.path.join(DEGRADED_DATA_PATH, kernel_name)}")
        
        # Create image folder
        os.makedirs(os.path.join(experiment_path, degraded_image_name), exist_ok=False)  # TODO: check that correspondance between clean and generated images is correct
        
        psnrs_img = []
        for j in range(nb_gen):
            if mode == "deblur":
                result_image, intermediary_images = diffpir_diffuser.apply_debluring(diffuser_config, clean_image, degraded_image, kernel_name)  # TODO: what to do with intermediary images?
            else:
                raise NotImplementedError
            #### LIGNE MAGINOT !!!! BLITZKRIEG POUVANT TOMBER A TOUT MOMENT !!!!
            save_img(result_image, os.path.join(LOGDIR, exp_name, f"img_{i}", f"gen_{j}.png"))
            psnr = compute_metrics(result_image, clean_image)
            # logger.info(f"{img_name:>10s} PSNR: {psnr:.4f}dB LPIPS: {lpips_score:.4f}")
            # else:
            #     logger.info(f"{img_name:>10s} PSNR: {psnr:.4f}dB")
            psnrs_img.append(psnr)
            
        metrics["PSNR"].append(psnrs_img)
        
    # Save metrics
    np.savez(os.path.join(LOGDIR, exp_name, "metrics.npz"), **metrics)
    report_metrics(metrics, os.path.join(LOGDIR, exp_name, "metrics.txt"))        
    

def main():
    print("Hello world!")


if __name__ == "__main__":
    # Define experiment name
    exp_name = "test_exp_deblur"
    
    diffpir_config = DiffPIRConfig()
    diffpir_deblur_config = DiffPIRDeblurConfig()
    
    run_experiment(...)