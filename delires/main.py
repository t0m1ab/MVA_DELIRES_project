import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import torch
import glob

import delires.utils.utils as utils
from delires.utils.utils_logger import logger_info
import delires.utils.utils_image as utils_image
from delires.diffusers.register import DIFFUSER_TYPE, DIFFUSERS
from delires.diffusers.diffpir.diffpir_configs import DiffPIRConfig, DiffPIRDeblurConfig
from delires.data import all_files_exist
from delires.metrics import compute_metrics

from delires.params import (
    TASK,
    MODELS_PATH,
    KERNELS_PATH, 
    CLEAN_DATA_PATH, 
    DEGRADED_DATA_PATH, 
    RESTORED_DATA_PATH,
)


def run_experiment(
    exp_name: str,
    diffuser_type: DIFFUSER_TYPE,
    task: TASK, 
    degraded_dataset_name: str,
    diffuser_config: DiffPIRConfig,
    diffuser_task_config: DiffPIRDeblurConfig,
    nb_gen: int = 1,
    kernel_name: str|None = None,
    calc_LPIPS: bool = False
    ):
    """ 
    Run an experiment for a given method.
    """

    # create experiment folder
    experiment_path = os.path.join(RESTORED_DATA_PATH, exp_name)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
      
    # setup device
    device = torch.device("cpu")
    if diffuser_config.device == "cuda" or diffuser_task_config.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    # create logger for the experiment
    logger_name = exp_name
    logger_info(logger_name, log_path=os.path.join(experiment_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # instantiate diffuser
    dataset_infos = utils.load_json(os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, "dataset_info.json"))
    kernel_name = dataset_infos["kernel_name"] if "kernel_name" in dataset_infos else None
    if kernel_name is None:
        raise ValueError("No kernel found for the given dataset. Please manually select a kernel.")
    diffpir_diffuser = DIFFUSERS[diffuser_type](diffuser_config, logger=logger)
    diffpir_diffuser.load_blur_kernel(kernel_name)
    
    # check the datasets (same number of images, same names, same extensions, etc.)
    if not all_files_exist(filenames=dataset_infos["images"], path=CLEAN_DATA_PATH, ext="png"):
        raise ValueError(f"Some clean png images are missing in: {CLEAN_DATA_PATH}.")
    if not all_files_exist(filenames=dataset_infos["images"], path=os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name), ext="png"):
        raise ValueError(f"Some degraded png images are missing in {os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name)}.")
    if not all_files_exist(filenames=dataset_infos["images"], path=os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name), ext="npy"):
        raise ValueError(f"Some degraded npy images are missing in {os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name)}.")

    # apply the method over the dataset
    logger.info(f"### Starting experiment {exp_name} with {diffuser_type} for {task} task on device {device}.")
    exp_metrics = {"PSNR": {}} # TODO: other metrics
    for img_name in dataset_infos["images"]:
        
        psnr_values = []
        # lpips_values = []
        for gen_idx in range(nb_gen):

            # apply the method
            if task == "deblur":
                metrics = diffpir_diffuser.apply_debluring(
                    config=diffuser_task_config,
                    clean_image_filename=img_name,
                    degraded_image_filename=img_name,
                    degraded_dataset_name=degraded_dataset_name,
                    experiment_name=exp_name,
                    kernel_filename=kernel_name,
                )
            else:
                raise NotImplementedError
            
            # compute metrics
            psnr_values.append(metrics["psnr"])
            #### LIGNE MAGINOT !!!! BLITZKRIEG POUVANT TOMBER A TOUT MOMENT !!!!
            if calc_LPIPS:
                raise NotImplementedError
                # lpips_img = ...
                # lpips_values.append(lpips_img)
            
        exp_metrics["PSNR"][img_name] = list(psnr_values)
        # metrics["LPIPS"][img_name] = list(lpips_values)
        
    # save metrics
    np.savez(os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.npz"), **exp_metrics)
    
    ### TODO: fix this monkeyness
    utils.report_metrics(metrics, os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.txt"))        
    

def main():
    
    exp_name = "test_exp_diffpir_deblur"
    degraded_dataset_name = "blurred_dataset"
    
    run_experiment(
        exp_name=exp_name,
        diffuser_type="diffpir",
        task="deblur",
        degraded_dataset_name=degraded_dataset_name,
        diffuser_config=DiffPIRConfig(),
        diffuser_task_config=DiffPIRDeblurConfig(),
        nb_gen=1,
        kernel_name=None,
        calc_LPIPS=False,
    )


if __name__ == "__main__":
    main()