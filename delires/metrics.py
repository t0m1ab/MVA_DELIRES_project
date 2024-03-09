import os
import numpy as np
import csv


from delires.params import (
    TASK,
    DEGRADED_DATA_PATH
)
import delires.utils.utils_image as utils_image
from delires.data import blur, load_blur_kernel


def data_consistency_norm(degraded_dataset_name: str, degraded_image_filename: str, reconstructed_image: np.ndarray, task: TASK, kernel_filename: str = None):
    """
    Compute the data consistency norm between the clean and the reconstructed image.
    
    ARGUMENTS:
        - degraded_dataset_name (str): Name of the degraded dataset.
        - degraded_image_name (str): Name of the degraded image.
        - reconstructed_image (np.ndarray): the reconstructed image in uint8.
        - task (TASK): the task to perform.
        - kernel_filename (str): Name of the kernel (without extension).
    """
    # load degraded image
    degraded_image_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, f"{degraded_image_filename}.png")
    degraded_image = utils_image.imread_uint(degraded_image_path)

    if degraded_image.dtype != np.uint8 or reconstructed_image.dtype != np.uint8:
        raise ValueError("The degraded and reconstructed images must be in uint8 format.")
    
    if task == "deblur":
        if kernel_filename is None:
            raise ValueError("The kernel filename must be loaded to compute data consistency for deblur task.")
        kernel = load_blur_kernel(kernel_filename)
        return np.linalg.norm(degraded_image - blur(reconstructed_image, kernel))
    else:
        raise NotImplementedError(f"Data consistency norm not implemented for task {task}.")
    
    
def report_metrics(metrics, exp_path):
    img_names = list(metrics["PSNR"].keys())
    with open(exp_path, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fields = ["img", "PSNR", "l2_residual", "average_image_std", "FID", "coverage", "LPIPS"]
        writer.writerow(fields)
        writer.writerow(["Overall"] + [np.mean(list(metrics[field].values())) for field in fields[1:]])
        for img in img_names:
            writer.writerow(
                [img]
                + [np.mean(metrics[field][img]) for field in fields[1:]]
                )
            