import os
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import logging
import torch

import delires.utils.utils as utils
import delires.utils.utils_image as utils_image
from delires.utils.utils_logger import logger_info
from delires.methods.dps.dps_configs import DPSDeblurConfig, DPSInpaintingConfig
from delires.methods.pigdm.pigdm_configs import PiGDMDeblurConfig, PiGDMInpaintingConfig
from delires.methods.diffpir.diffpir_configs import DiffPIRDeblurConfig, DiffPIRInpaintingConfig
from delires.metrics import data_consistency_mse, report_metrics, save_std_image
from delires.fid import fid_score

from delires.methods.register import DIFFUSERS, DIFFUSER_CONFIG, TASK_CONFIG, TASKS
from delires.params import CLEAN_DATA_PATH, DEGRADED_DATA_PATH, RESTORED_DATA_PATH


TASK_DIFFUSER_TO_CONFIG = {
    "deblur": {
        "diffpir": DiffPIRDeblurConfig(), 
        "dps": DPSDeblurConfig(), 
        "pigdm": PiGDMDeblurConfig()
    },
    "inpaint": {
        "diffpir": DiffPIRInpaintingConfig(), 
        "dps": DPSInpaintingConfig(), 
        "pigdm": PiGDMInpaintingConfig()
    }
}
TASK_TO_DEGRADED_DATASET = {
    "deblur": "blurred_ffhq_test20",
    "inpaint": "masked_ffhq_test20"
}


def run_experiment(
    exp_name: str,
    diffuser_type: str,
    task: str, 
    degraded_dataset_name: str,
    dataset_subset: int = None,
    nb_gen: int = 1,
    fid_dims: int = 2048,
    fid_kept_eigenvectors: int|None = None,
    ) -> None:
    """ 
    Run an experiment for a given method, task, config and degraded dataset.
    
    ARGUMENTS:
        - exp_name: name of the experiment.
        - diffuser_type: type of diffuser to use (should be on of DIFFUSERS.keys()).
        - task: task to perform (should be one of TASKS).
        - degraded_dataset_name: name of the degraded dataset to use (should be a subfolder in DEGRADED_DATA_PATH).
        - diffuser_config: configuration of the chosen diffuser_type.
        - dataset_subset: only the dataset_subset first images of the dataset are used for the experiment. If None, the whole dataset is used.
        - nb_gen: number of generations to perform for each image (to assess variability of the method).
        - fid_dims: number of features of the layer of InceptionV3 chosen for the computation of FID.
                    This impacts the interpretability of the measure as well as its scale.
        - fid_kept_eigenvectors: Number of eigenvectors to keep for the covariance matrix of the features of InceptionV3 retained for the computation of FID.
                                 Use this if the number of images in the dataset is not sufficient.
    """

    # sanity checks
    if not diffuser_type in DIFFUSERS.keys():
        raise ValueError(f"Unknown diffuser type: {diffuser_type}. Should be one of {list(DIFFUSERS.keys())}.")
    if not task in TASKS:
        raise ValueError(f"Unknown task: {task}. Should be one of {TASKS}.")

    # create experiment folder
    experiment_path = os.path.join(RESTORED_DATA_PATH, exp_name)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
        
    # save configs
    diffuser_config = DIFFUSER_CONFIG[diffuser_type]
    diffuser_task_config = TASK_DIFFUSER_TO_CONFIG[task][diffuser_type]
    utils.archive_kwargs(diffuser_config.__dict__, os.path.join(RESTORED_DATA_PATH, exp_name, f"{diffuser_type}_config.json"))
    utils.archive_kwargs(diffuser_task_config.__dict__, os.path.join(RESTORED_DATA_PATH, exp_name, f"{diffuser_type}_task_config.json"))
      
    # setup device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    
    # create logger for the experiment
    logger_name = exp_name
    logger_info(logger_name, log_path=os.path.join(experiment_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # get dataset_infos and instantiate diffuser
    dataset_infos = utils.load_json(os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, "dataset_info.json"))
    diffuser = DIFFUSERS[diffuser_type](diffuser_config, logger=logger, device=device)
    
    # check the datasets (same number of images, same names, same extensions, etc.)
    degraded_dataset_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name)
    if not utils.all_files_exist(filenames=dataset_infos["images"], path=CLEAN_DATA_PATH, ext="png"):
        raise ValueError(f"Some clean png images are missing in: {CLEAN_DATA_PATH}.")
    if not utils.all_files_exist(filenames=dataset_infos["images"], path=os.path.join(degraded_dataset_path, 'png/'), ext="png"):
        raise ValueError(f"Some degraded png images are missing in {os.path.join(degraded_dataset_path, 'png/')}.")
    if not utils.all_files_exist(filenames=dataset_infos["images"], path=os.path.join(degraded_dataset_path, 'npy/'), ext="npy"):
        raise ValueError(f"Some degraded npy images are missing in {os.path.join(degraded_dataset_path, 'npy/')}.")

    # apply the method over the dataset
    logger.info(f"### ===== Starting experiment {exp_name} with {diffuser_type} for {task} task on {device} ===== ###")
    exp_raw_metrics = {
        "MSE_to_clean": {}, # computed from generated images
        "MSE_to_degraded": {}, # computed from generated images
        "average_image_std": {}, # computed from generated images
        "coverage": {}, # TODO
        "LPIPS": {}, # computed on-the-run
        }
    images = dataset_infos["images"][:dataset_subset] if dataset_subset is not None else dataset_infos["images"]
    operator_family = dataset_infos["operator_family_name"]

    for img_name in images:
        
        # define the operator used for the image and store metrics for each generation
        operator_idx = dataset_infos["image_to_operator"][img_name]
        img_mse_to_clean = []
        img_mse_to_degraded = []
        img_coverage = []
        # img_lpips = []
        gen_images = []
        
        # load clean_image as uint. Used for computing metrics
        clean_image = utils_image.imread_uint(os.path.join(CLEAN_DATA_PATH, img_name+".png"))
        
        for gen_idx in range(nb_gen):
            
            # apply the method (don't save the image in apply_task by default because there are multiple generations to save)
            if task == "deblur":
                diffuser.load_blur_kernel(kernel_family=operator_family, kernel_idx=operator_idx)
                restored_image, _ = diffuser.apply_debluring(
                    config=diffuser_task_config,
                    clean_image_filename=img_name,
                    degraded_image_filename=img_name,
                    degraded_dataset_name=degraded_dataset_name,
                    experiment_name=exp_name,
                )
            elif task == "inpaint":
                diffuser.load_inpainting_mask(mask_family=operator_family, mask_idx=operator_idx)
                restored_image, _ = diffuser.apply_inpainting(
                    config=diffuser_task_config,
                    clean_image_filename=img_name,
                    degraded_image_filename=img_name,
                    degraded_dataset_name=degraded_dataset_name,
                    experiment_name=exp_name,
                )
            else:
                raise NotImplementedError("The given task is not implemented.")
            
            # save the restored image (with "_genX" suffix where X is the generation index)
            diffuser.save_restored_image(
                restored_image=restored_image,
                restored_image_filename=f"{img_name}_gen{gen_idx}",
                path=os.path.join(RESTORED_DATA_PATH, exp_name, f"{img_name}"),
            )
            
            # compute and store metrics
            img_mse_to_clean.append(utils_image.mse(restored_image, clean_image))
            img_mse_to_degraded.append(data_consistency_mse(
                degraded_dataset_name=degraded_dataset_name,
                degraded_image_filename=img_name,
                restored_image=restored_image, 
                task=task, 
                operator_family=operator_family, 
                operator_idx=operator_idx,
            ))
            img_coverage.append(0)
            # if diffuser_task_config.calc_LPIPS:
                # img_lpips.append(metrics["lpips"])
            
            # append generated image to compute std image
            gen_images.append(restored_image)
        
        # save std image of generated restorations
        std_image = np.mean(np.std(gen_images, axis=0), axis=-1)
        save_std_image(exp_name, img_name, std_image)
            
        exp_raw_metrics["MSE_to_clean"][img_name] = img_mse_to_clean
        exp_raw_metrics["MSE_to_degraded"][img_name] = img_mse_to_degraded
        exp_raw_metrics["average_image_std"][img_name] = [np.mean(std_image)]
        exp_raw_metrics["coverage"][img_name] = img_coverage
        # if diffuser_task_config.calc_LPIPS:
            # exp_raw_metrics["LPIPS"][img_name] = list(img_lpips)
        
        # save metrics as a checkpoint after processing one image
        np.savez(os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.npz"), **exp_raw_metrics)
    
    # Compute FID and save metrics
    fid = fid_score.calculate_fid_given_paths(
        paths=[CLEAN_DATA_PATH, os.path.join(RESTORED_DATA_PATH, exp_name)], 
        batch_size=nb_gen, 
        device=device, 
        dims=fid_dims, 
        keep_eigen=fid_kept_eigenvectors
    )
    np.savez(os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.npz"), **exp_raw_metrics, fid=fid)
    
    report_metrics(exp_raw_metrics, fid, os.path.join(RESTORED_DATA_PATH, exp_name, "metrics.csv"), calc_LPIPS=False)        


def run_all_experiments():
    """
    Run all experiments for all available diffusers and tasks with a our settings and FFHQ test data.
    """

    # loop 1: metrics computation
    for task in TASKS:
        for diffuser in DIFFUSERS:
            run_experiment(
                exp_name=f"test_exp_{diffuser}_{task}",
                diffuser_type=diffuser,
                task=task,
                degraded_dataset_name=TASK_TO_DEGRADED_DATASET[task],
                dataset_subset=2,
                nb_gen=2,
                fid_dims=192,
                fid_kept_eigenvectors=157,
            )
            
    # loop 2: assessment of generation variability
    for task in TASKS:
        for diffuser in DIFFUSERS:
            run_experiment(
                exp_name=f"test_exp_{diffuser}_{task}_std",
                diffuser_type=diffuser,
                task=task,
                degraded_dataset_name=TASK_TO_DEGRADED_DATASET[task],
                dataset_subset=1,
                nb_gen=4,
                fid_dims=192,
                fid_kept_eigenvectors=157,
            )
    

def main():

    parser = argparse.ArgumentParser(description="Run DPS, PiGDM and DiffPIR experiments on FFHQ test data.")
    
    parser.add_argument(
        "--deblur",
        "-d",
        dest="deblur",
        action="store_true",
        help="run demo deblurring experiments.",
    )
    parser.add_argument(
        "--inpaint",
        "-i",
        dest="inpaint",
        action="store_true",
        help="run demo inpainting experiments.",
    )
    parser.add_argument(
        "--dps",
        dest="dps",
        action="store_true",
        help="use DPS method.",
    )
    parser.add_argument(
        "--pigdm",
        dest="pigdm",
        action="store_true",
        help="use PiGDM method.",
    )
    parser.add_argument(
        "--diffpir",
        dest="diffpir",
        action="store_true",
        help="use DiffPIR method.",
    )

    args = parser.parse_args()

    if args.deblur or args.inpaint:

        diffuser_method = None
        if args.dps:
            diffuser_method = "dps"
        elif args.pigdm:
            diffuser_method = "pigdm"
        elif args.diffpir:
            diffuser_method = "diffpir"
        else:
            raise ValueError("Please specify a diffuser method using one of the following flags: --dps --pigdm --diffpir")

        task = "deblur" if args.deblur else "inpaint"

        run_experiment(
            exp_name=f"test_exp_{diffuser_method}_{task}",
            diffuser_type=diffuser_method,
            task=task,
            degraded_dataset_name=TASK_TO_DEGRADED_DATASET[task],
            dataset_subset=2,
            nb_gen=2,
            fid_dims=192,
            fid_kept_eigenvectors=157,
        )
    
    else:
        run_all_experiments()

    
if __name__ == "__main__":
    main()
    