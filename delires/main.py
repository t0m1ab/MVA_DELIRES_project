import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import torch

import delires.utils.utils as utils
from delires.utils.utils_logger import logger_info
import delires.utils.utils_image as utils_image
from delires.methods.register import DIFFUSER_TYPE, DIFFUSERS
from delires.methods.diffpir.diffpir_configs import DiffPIRConfig, DiffPIRDeblurConfig
from delires.data import all_files_exist
from delires.fid import fid_score
from delires.metrics import data_consistency_mse, report_metrics, save_std_image

from delires.params import (
    TASK,
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
    ):
    """ 
    Run an experiment for a given method.
    """

    # create experiment folder
    experiment_path = os.path.join(RESTORED_DATA_PATH, exp_name)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    
    # Save configs
    utils.archive_kwargs(diffuser_config.__dict__, os.path.join(RESTORED_DATA_PATH, exp_name, "diffuser_config.json"))
    utils.archive_kwargs(diffuser_task_config.__dict__, os.path.join(RESTORED_DATA_PATH, exp_name, "diffuser_task_config.json"))

      
    # setup device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    
    # create logger for the experiment
    logger_name = exp_name
    logger_info(logger_name, log_path=os.path.join(experiment_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # instantiate diffuser
    dataset_infos = utils.load_json(os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, "dataset_info.json"))
    kernel_name = dataset_infos["kernel_name"] if "kernel_name" in dataset_infos else None
    if kernel_name is None:
        raise ValueError("No kernel found for the given dataset. Please manually select a kernel.")
    diffpir_diffuser = DIFFUSERS[diffuser_type](diffuser_config, logger=logger, device=device)
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
    exp_raw_metrics = {
        "MSE_to_clean": {},  # computed from generated images
        "MSE_to_degraded": {},  # computed from generated images
        "average_image_std": {},  # computed from generated images
        "coverage": {},  # TODO
        "LPIPS": {},  # computed on-the-run
        }
    for img_name in dataset_infos["images"]:
        img_mse_to_clean = []
        img_mse_to_degraded = []
        img_coverage = []
        img_lpips = []
        
        gen_images = []
        
        clean_image = utils_image.imread_uint(os.path.join(CLEAN_DATA_PATH, img_name+".png"))

        for gen_idx in range(nb_gen):

            # apply the method (don't save the image in apply_task by default because there are multiple generations to save)
            if task == "deblur":
                restored_image, metrics = diffpir_diffuser.apply_debluring(
                    config=diffuser_task_config,
                    clean_image_filename=img_name,
                    degraded_image_filename=img_name,
                    degraded_dataset_name=degraded_dataset_name,
                    experiment_name=exp_name,
                    kernel_filename=kernel_name,
                )
            else:
                raise NotImplementedError

            # save the restored image (with "_genX" suffix where X is the generation index)
            diffpir_diffuser.save_restored_image(
                restored_image=restored_image,
                restored_image_filename=f"gen{gen_idx}",
                path=os.path.join(RESTORED_DATA_PATH, exp_name, f"img_{img_name}"),
            )
            
            # compute and store metrics
            img_mse_to_clean.append(utils_image.mse(restored_image, clean_image))
            img_mse_to_degraded.append(data_consistency_mse(degraded_dataset_name, img_name, restored_image, task, kernel_name))
            img_coverage.append(0)
            if diffuser_task_config.calc_LPIPS:
                img_lpips.append(metrics["lpips"])
            
            # Append generated image to compute std image
            gen_images.append(restored_image)
        
        # Save std image of generated restorations
        std_image = np.mean(np.std(gen_images, axis=0), axis=-1)
        save_std_image(exp_name, img_name, std_image)
            
        exp_raw_metrics["MSE_to_clean"][img_name] = img_mse_to_clean
        exp_raw_metrics["MSE_to_degraded"][img_name] = img_mse_to_degraded
        exp_raw_metrics["average_image_std"][img_name] = [np.mean(std_image)]
        exp_raw_metrics["coverage"][img_name] = img_coverage
        exp_raw_metrics["LPIPS"][img_name] = list(img_lpips)
        
    # Save metrics once before computing FID
    np.savez(os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.npz"), **exp_raw_metrics)
    
    # Compute FID and save metrics
    fid = fid_score.calculate_fid_given_paths(paths=[CLEAN_DATA_PATH, os.path.join(RESTORED_DATA_PATH, exp_name)], batch_size=5, device=device, dims=2048)
    np.savez(os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.npz"), **exp_raw_metrics, fid=fid)
    
    report_metrics(exp_raw_metrics, fid, os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.csv"))        
    

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
        nb_gen=3,
        kernel_name=None,
    )


if __name__ == "__main__":
    main()
    