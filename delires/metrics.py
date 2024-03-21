import os
import numpy as np
import csv
from pathlib import Path
import copy

import delires.utils.utils_image as utils_image
from delires.data import apply_blur, apply_mask
import delires.utils.operators as operators
from delires.params import DEGRADED_DATA_PATH, RESTORED_DATA_PATH


def data_consistency_mse(
        degraded_dataset_name: str, 
        degraded_image_filename: str, 
        restored_image: np.ndarray,
        task: str, 
        operator_family: str, 
        operator_idx: int|str,
    ):
    """
    Compute the data consistency norm between the clean and the restored image.
    
    ARGUMENTS:
        - degraded_dataset_name (str): name of the degraded dataset.
        - degraded_image_filename (str): name of the degraded image.
        - restored_image (np.ndarray): the restored image.
        - task (str): the task to perform (should be one of TASKS).
        - operator_family (str): family of the operator (should be a subfolder of OPERATORS_PATH).
        - operator_idx (int|str): index of the operator.
    """
    # load degraded image
    degraded_image_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, "png/", f"{degraded_image_filename}.png")
    degraded_image = utils_image.imread_uint(degraded_image_path)

    # print("degraded_image", utils_image.get_infos_img(degraded_image))
    # print("restored_image", utils_image.get_infos_img(restored_image))

    if degraded_image.dtype != np.uint8 or restored_image.dtype != np.uint8:
        raise ValueError("The degraded and restored images must be in uint8 format.")
    
    if task == "deblur":

        kernel = operators.load_operator(operator_family=operator_family, operator_idx=operator_idx)
        blurred_reconstruction = utils_image.tensor2uint(apply_blur(utils_image.uint2tensor4(restored_image), kernel))

        # print("degraded_image", utils_image.get_infos_img(degraded_image))
        # print("blurred_reconstruction", utils_image.get_infos_img(blurred_reconstruction))

        return utils_image.mse(degraded_image, blurred_reconstruction)
    
    elif task == "inpaint":

        mask = operators.load_operator(operator_family=operator_family, operator_idx=operator_idx)
        masked_reconstruction = utils_image.single2uint(apply_mask(utils_image.uint2single(restored_image), mask[:,:,None]))

        # print("degraded_image", utils_image.get_infos_img(degraded_image))
        # print("masked_reconstruction", utils_image.get_infos_img(masked_reconstruction))

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