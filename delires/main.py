import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import torch

import delires.utils.utils as utils
from delires.utils.utils_logger import logger_info
import delires.utils.utils_image as utils_image
from delires.methods.register import DIFFUSER_TYPE, DIFFUSERS, DIFFUSER_CONFIG, TASK_CONFIG
from delires.methods.diffpir.diffpir_configs import DiffPIRConfig, DiffPIRDeblurConfig, DiffPIRInpaintingConfig
from delires.methods.dps.dps_configs import DPSConfig, DPSDeblurConfig, DPSInpaintingConfig
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMDeblurConfig, PiGDMInpaintingConfig
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
    diffuser_config: DIFFUSER_CONFIG,
    diffuser_task_config: TASK_CONFIG,
    dataset_subset: int = None,
    nb_gen: int = 1,
    fid_dims: int = 2048,
    fid_kept_eigenvectors: int|None = None,
    ):
    """ 
    Run an experiment for a given method.
    
    Arguments:
        - dataset_subset: Only the dataset_subset first images of the dataset are used for the experiment. If None, the whole dataset is used.
        - fid_dims: number of features of the layer of InceptionV3 chosen for the computation of FID. This impacts the interpretability of the measure as well as its scale.
        - fid_kept_eigenvectors: Number of eigenvectors to keep for the covariance matrix of the features of InceptionV3 retained for the computation of FID. Use this if the number of images in the dataset is not sufficient.
    """

    # create experiment folder
    experiment_path = os.path.join(RESTORED_DATA_PATH, exp_name)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
        
    # save configs
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
    diffuser = DIFFUSERS[diffuser_type](diffuser_config, logger=logger, device=device)
    if task == "deblur":
        operator_name = dataset_infos["kernel_name"] if "kernel_name" in dataset_infos else None
        if operator_name is None:
            raise ValueError("No kernel found for the given dataset. Please manually select a kernel.")
        diffuser.load_blur_kernel(operator_name)
    elif task == "inpaint":
        operator_name = dataset_infos["masks_name"] if "masks_name" in dataset_infos else None
        if operator_name is None:
            raise ValueError("No set of masks found for the given dataset. Please manually select a set of masks.")
    
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
    images = dataset_infos["images"][:dataset_subset] if dataset_subset is not None else dataset_infos["images"]
    for i, img_name in enumerate(images):
        mask_index = None
        img_mse_to_clean = []
        img_mse_to_degraded = []
        img_coverage = []
        # img_lpips = []
        
        gen_images = []
        
        # Load clean_image as uint. Used for computing metrics
        clean_image = utils_image.imread_uint(os.path.join(CLEAN_DATA_PATH, img_name+".png"))
        
        for gen_idx in range(nb_gen):
            # apply the method (don't save the image in apply_task by default because there are multiple generations to save)
            if task == "deblur":
                restored_image, metrics = diffuser.apply_debluring(
                    config=diffuser_task_config,
                    clean_image_filename=img_name,
                    degraded_image_filename=img_name,
                    degraded_dataset_name=degraded_dataset_name,
                    experiment_name=exp_name,
                    kernel_filename=operator_name,
                )
            elif task == "inpaint":
                mask_index = i
                diffuser.load_inpainting_mask(operator_name, mask_index)
                restored_image, metrics = diffuser.apply_inpainting(
                    config=diffuser_task_config,
                    clean_image_filename=img_name,
                    degraded_image_filename=img_name,
                    degraded_dataset_name=degraded_dataset_name,
                    masks_filename=operator_name,
                    mask_index=mask_index,
                )
            else:
                raise NotImplementedError("The given task is not implemented.")
            
            # save the restored image (with "_genX" suffix where X is the generation index)
            diffuser.save_restored_image(
                restored_image=restored_image,
                restored_image_filename=f"gen{gen_idx}",
                path=os.path.join(RESTORED_DATA_PATH, exp_name, f"img_{img_name}"),
            )
            
            # compute and store metrics
            img_mse_to_clean.append(utils_image.mse(restored_image, clean_image))
            img_mse_to_degraded.append(data_consistency_mse(degraded_dataset_name, img_name, restored_image, task, operator_name, mask_index))
            img_coverage.append(0)
            # if diffuser_task_config.calc_LPIPS:
                # img_lpips.append(metrics["lpips"])
            
            # Append generated image to compute std image
            gen_images.append(restored_image)
        
        # Save std image of generated restorations
        std_image = np.mean(np.std(gen_images, axis=0), axis=-1)
        save_std_image(exp_name, img_name, std_image)
            
        exp_raw_metrics["MSE_to_clean"][img_name] = img_mse_to_clean
        exp_raw_metrics["MSE_to_degraded"][img_name] = img_mse_to_degraded
        exp_raw_metrics["average_image_std"][img_name] = [np.mean(std_image)]
        exp_raw_metrics["coverage"][img_name] = img_coverage
        # if diffuser_task_config.calc_LPIPS:
            # exp_raw_metrics["LPIPS"][img_name] = list(img_lpips)
        
        # Save metrics as a checkpoint after processing one image
        np.savez(os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.npz"), **exp_raw_metrics)
    
    # Compute FID and save metrics
    fid = fid_score.calculate_fid_given_paths(paths=[CLEAN_DATA_PATH, os.path.join(RESTORED_DATA_PATH, exp_name)], batch_size=nb_gen, device=device, dims=fid_dims, keep_eigen=fid_kept_eigenvectors)
    np.savez(os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.npz"), **exp_raw_metrics, fid=fid)
    
    report_metrics(exp_raw_metrics, fid, os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.csv"), calc_LPIPS=False)        
    

def run_all_experiments():
    tasks = ["deblur", "inpaint"]
    diffusers = ["diffpir", "dps", "pigdm"]
    diffuser_configs = {"diffpir": DiffPIRConfig(), "dps": DPSConfig(), "pigdm": PiGDMConfig()}
    task_configs = {
        "deblur": {"diffpir": DiffPIRDeblurConfig(), "dps": DPSDeblurConfig(), "pigdm": PiGDMDeblurConfig()},
        "inpaint": {"diffpir": DiffPIRInpaintingConfig(), "dps": DPSInpaintingConfig(), "pigdm": PiGDMInpaintingConfig()}
    }
    for task in tasks:
        for diffuser in diffusers:
            exp_name = f"test_exp_{diffuser}_{task}"
            degraded_dataset_name = "blurred_ffhq" if task == "deblur" else "masked_ffhq"
            run_experiment(
                exp_name=exp_name,
                diffuser_type=diffuser,
                task=task,
                degraded_dataset_name=degraded_dataset_name,
                diffuser_config=diffuser_configs[diffuser],
                diffuser_task_config=task_configs[task][diffuser],
                dataset_subset = 2,
                nb_gen=1,
                fid_dims=192,
                fid_kept_eigenvectors=157,
            )
            
    # New experiments for assessment of generation variability
    nb_gen = 2
    for task in tasks:
        for diffuser in diffusers:
            exp_name = f"test_exp_{diffuser}_{task}_std"
            degraded_dataset_name = "blurred_ffhq" if task == "deblur" else "masked_ffhq"
            run_experiment(
                exp_name=exp_name,
                diffuser_type=diffuser,
                task=task,
                degraded_dataset_name=degraded_dataset_name,
                diffuser_config=diffuser_configs[diffuser],
                diffuser_task_config=task_configs[task][diffuser],
                dataset_subset = 1,
                nb_gen=nb_gen,
                fid_dims=192,
                fid_kept_eigenvectors=157,
            )
    
    


def main():
    
    # # Deblurring
    # exp_name = "test_exp_diffpir_deblur"
    # degraded_dataset_name = "blurred_ffhq_test20"
    
    # run_experiment(
    #     exp_name=exp_name,
    #     diffuser_type="diffpir",
    #     task="deblur",
    #     degraded_dataset_name=degraded_dataset_name,
    #     diffuser_config=DiffPIRConfig(),
    #     diffuser_task_config=DiffPIRDeblurConfig(),
    #     nb_gen=2,
    #     fid_dims=192,
    #     fid_kept_eigenvectors=157,
    # )
    
    # Inpainting
    exp_name = "test_exp_diffpir_inpaint"
    degraded_dataset_name = "masked_ffhq"
    
    run_experiment(
        exp_name=exp_name,
        diffuser_type="dps",
        task="inpaint",
        degraded_dataset_name=degraded_dataset_name,
        diffuser_config=DPSConfig(),
        diffuser_task_config=DPSInpaintingConfig(),
        nb_gen=2,
        fid_dims=192,
        fid_kept_eigenvectors=157,
    )
    
    run_all_experiments()


if __name__ == "__main__":
    main()
    