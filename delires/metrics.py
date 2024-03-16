import os
import numpy as np
import csv
from pathlib import Path
import copy

from delires.params import (
    TASK,
    DEGRADED_DATA_PATH,
    RESTORED_DATA_PATH
)
import delires.utils.utils_image as utils_image
from delires.data import blur, load_blur_kernel, load_masks, apply_mask


def data_consistency_mse(degraded_dataset_name: str, degraded_image_filename: str, reconstructed_image: np.ndarray, task: TASK, operator_filename: str = None, mask_index:int = None):
    """
    Compute the data consistency norm between the clean and the reconstructed image.
    
    ARGUMENTS:
        - degraded_dataset_name (str): Name of the degraded dataset.
        - degraded_image_name (str): Name of the degraded image.
        - reconstructed_image (np.ndarray): the reconstructed image in uint8.
        - task (TASK): the task to perform.
        - operator_filename (str): Name of the operator (without extension).
    """
    # load degraded image
    degraded_image_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, f"{degraded_image_filename}.png")
    degraded_image = utils_image.imread_uint(degraded_image_path)

    if degraded_image.dtype != np.uint8 or reconstructed_image.dtype != np.uint8:
        raise ValueError("The degraded and reconstructed images must be in uint8 format.")
    
    if task == "deblur":
        kernel = load_blur_kernel(operator_filename)
        return utils_image.mse(degraded_image, blur(reconstructed_image, kernel))
    elif task == "inpaint":
        if mask_index is None:
            raise ValueError("mask_index must be provided to compute consistency norm for inpainting task.")
        mask = load_masks(operator_filename)[mask_index]
        masked_reconstruction = utils_image.single2uint(apply_mask(utils_image.uint2single(reconstructed_image), mask))
        return utils_image.mse(degraded_image, masked_reconstruction)
    else:
        raise NotImplementedError(f"Data consistency norm not implemented for task {task}.")
    

def process_raw_metrics(raw_metrics: dict, calc_LPIPS:bool = False):
    """
    Process the raw metrics to compute the final metrics.
    
    ARGUMENTS:
        - raw_metrics (dict): the raw metrics.
        - calc_LPIPS (bool): whether to compute LPIPS or not.
        
    RETURNS:
        - metrics (dict): the final metrics.
    """
    metrics = {
        "PSNR": {},
        "datafit_RMSE": {},
        "average_image_std": {},
        "coverage": {},
        "LPIPS": {},
    }
    
    # PSNR
    metrics["PSNR"]["overall"] = utils_image.calculate_psnr_from_mse(np.mean(list(raw_metrics["MSE_to_clean"].values())))
    for img_name, mse in raw_metrics["MSE_to_clean"].items():
        metrics["PSNR"][img_name] = utils_image.calculate_psnr_from_mse(np.mean(mse))
        
    # RMSE
    metrics["datafit_RMSE"]["overall"] = np.sqrt(np.mean(list(raw_metrics["MSE_to_degraded"].values())))
    for img_name, mse in raw_metrics["MSE_to_degraded"].items():
        metrics["datafit_RMSE"][img_name] = np.sqrt(np.mean(mse))
        
    # average image std
    metrics["average_image_std"]["overall"] = np.mean(list(raw_metrics["average_image_std"].values()))
    for img_name, average_std in raw_metrics["average_image_std"].items():
        metrics["average_image_std"][img_name] = np.mean(average_std)
    
    # coverage
    metrics["coverage"]["overall"] = np.mean(list(raw_metrics["coverage"].values()))
    for img_name, coverage in raw_metrics["coverage"].items():
        metrics["coverage"][img_name] = np.mean(coverage)
    
    # LPIPS
    if calc_LPIPS:
        metrics["LPIPS"]["overall"] = np.mean(list(raw_metrics["LPIPS"].values()))
        for img_name, lpips in raw_metrics["LPIPS"].items():
            metrics["LPIPS"][img_name] = np.mean(lpips)
    
    return metrics


def report_metrics(raw_metrics: dict, fid: float, exp_path: str, calc_LPIPS:bool = False):
    metrics = process_raw_metrics(raw_metrics, calc_LPIPS)
    img_names = list(metrics["PSNR"].keys())[1:]
    with open(exp_path, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if calc_LPIPS:
            fields = ["img", "PSNR", "datafit_RMSE", "average_image_std", "coverage", "LPIPS", "FID"]
        else:
            fields = ["img", "PSNR", "datafit_RMSE", "average_image_std", "coverage", "FID"]
        writer.writerow(fields)
        writer.writerow(["Overall"] + [metrics[field]["overall"] for field in fields[1:-1]] + [fid])
        for img in img_names:
            writer.writerow(
                [img]
                + [np.mean(metrics[field][img]) for field in fields[1:-1]]
                )
            
            
def save_std_image(exp_name, image_name, std_image, img_ext="png"):
    path = os.path.join(RESTORED_DATA_PATH, exp_name, f"std_images")
    Path(path).mkdir(parents=True, exist_ok=True)
    std_image_path = os.path.join(path, f"std_{image_name}.{img_ext}")
    utils_image.imsave(std_image, std_image_path)
