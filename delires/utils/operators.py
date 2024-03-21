import os
from pathlib import Path
import argparse
from typing import List
import numpy as np
import torch

import delires.utils.utils as utils
# import delires.utils.utils_image as utils_image
import delires.utils.utils_plots as utils_plots
from delires.methods.diffpir.utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
# from delires.methods.utils.utils_agem import fft_blur
from delires.methods.utils import utils_inpaint as utils_inpaint
from delires.params import OPERATORS_PATH


def load_operator(
        filename: str = None, 
        operator_family: str = None, 
        operator_idx: str | int = None, 
        path: str = None
    )-> np.ndarray:
    """ 
    Load an operator stored as a .npy file in a subfolder family in path.
    Check filename first which can be either like f'{family_name}/{family_name}_{operator_idx}.npy' 
    or f'{familyname}_{operator_idx}.npy' (with or without .npy extension).
    If filename is None, then operator_family and operator_idx must be provided and combined to find the right operator.
    If path is None, then OPERATORS_PATH is used.
    """
    path = OPERATORS_PATH if path is None else path

    if filename is not None: # use filename
        filename = f"{filename}.npy" if not filename.endswith(".npy") else filename # add extension if not present
    else: # use operator_family and operator_idx
        if operator_family is None or operator_idx is None:
            raise ValueError("operator_family and operator_idx must be provided to load a operator if no valid filename is provided")
        filename = os.path.join(operator_family, f"{operator_idx}.npy")
    
    if os.path.isfile(os.path.join(path, filename)):
        operator = np.load(os.path.join(path, filename))
    else:
        raise ValueError(f"Operator file {filename} not found in: {path}")

    operator = np.squeeze(operator) # remove single dimensions

    if not operator.ndim == 2:
        raise ValueError(f"operator.ndim must be 2, but operator has shape {operator.shape}")

    return operator


def create_blur_kernel(
        blur_mode: str,
        kernel_size: int,
        seed: int = 0,
        kernel_save_name: str|None = "",
        device: str = "cpu",
    ):
    np.random.seed(seed=seed) # for reproducibility
    kernel_std = 3.0 if blur_mode == 'Gaussian' else 0.5
    if blur_mode == 'Gaussian':
        kernel_std_i = kernel_std * np.abs(np.random.rand()*2+1)
        kernel = GaussialBlurOperator(kernel_size=kernel_size, intensity=kernel_std_i, device=device)
    elif blur_mode == 'motion':
        kernel = MotionBlurOperator(kernel_size=kernel_size, intensity=kernel_std, device=device)
    else:
        raise ValueError(f"Unknown blur mode: {blur_mode}")

    # TODO: change this to avoid passing through device?
    k_tensor = kernel.get_kernel().to(device, dtype=torch.float)
    k = k_tensor.clone().detach().cpu().numpy() # [0,1]
    k = np.squeeze(k)
    
    if kernel_save_name is not None:
        np.save(os.path.join(OPERATORS_PATH, f"{kernel_save_name}.npy"), k)
    
    return k


def create_blur_kernels_family(
        number_kernels: int = 20,
        kernels_family_name: str = None,
    ):
    raise NotImplementedError("This function is not implemented yet. Please use matalab_kernels.py to generate blur kernels instead.")


def create_inpainting_masks_family(
        mask_type: str,
        mask_len_range: List[int] = None,
        mask_prob_range: List[float] = None,
        image_shape: List[int] = (256, 256),
        margin: List[int] = (16, 16),
        number_masks: int = 20,
        mask_family_name: str = None,
        path: str = None,
        seed: int = 0,
    ):
    """
    Build family of inpainting masks and save them in a subfolder in path.
    """

    path = path if path is not None else OPERATORS_PATH
    mask_family_path = os.path.join(path, mask_family_name)
    Path(mask_family_path).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed=seed) # for reproducibility

    # generate masks
    mask_gen = utils_inpaint.mask_generator(
        mask_type=mask_type, 
        mask_len_range=mask_len_range, 
        mask_prob_range=mask_prob_range, 
        img_shape=image_shape, 
        margin=margin
    )
    
    # save masks
    for i in range(number_masks):
        mask = mask_gen()
        np.save(os.path.join(mask_family_path, f"{i}.npy"), mask)

    # save masks infos
    masks_infos_dict = {
        "mask_family_name": mask_family_name,
        "number_masks": number_masks,
        "mask_type": mask_type,
        "mask_len_range": mask_len_range,
        "mask_prob_range": mask_prob_range,
        "image_shape": image_shape,
        "margin": margin,
    }
    utils.archive_kwargs(masks_infos_dict, os.path.join(mask_family_path, "mask_info.json"))


def check_integrity_operators_family(family_name: str, path: str = None) -> int:
    """ 
    Check the integrity of a family of operators. A family should be a set of .npy files in a subfolder of path.
    If there are n members in the family then filenems should be: 0.npy, 1.npy, ..., n-1.npy
    Returns the number of operators in the family.
    """
    path = OPERATORS_PATH if path is None else path
    family_path = os.path.join(path, family_name)
    if not os.path.isdir(family_path):
        raise FileNotFoundError(f"{family_name} not found in: {path}")
    family_files = utils.listdir(family_path, ext="npy")
    n_files = len(family_files)
    if n_files == 0:
        raise ValueError(f"{family_name} is empty")
    for idx in range(n_files):
        if f"{idx}.npy" not in family_files:
            raise ValueError(f"{family_name} is missing {idx}.npy")
    return n_files


def main():

    parser = argparse.ArgumentParser(description="Generate blur kernels/inpainting masks.")
    parser.add_argument(
        "--kernel",
        "-k",
        dest="kernel",
        action="store_true", 
        help="if set then create blur kernels family"
    )
    parser.add_argument(
        "--mask",
        "-m",
        dest="mask",
        action="store_true", 
        help="if set then create inpainting masks family"
    )
    parser.add_argument(
        "--samples",
        "-x",
        dest="n_samples",
        type=int,
        default=100, 
        help="specify the number of masks/kernels to generate"
    )
    parser.add_argument(
        "--plot",
        "-p",
        dest="plot",
        action="store_true", 
        help="if set then create a plot of the generated masks/kernels"
    )
    parser.add_argument(
        "--seed",
        "-s",
        dest="seed",
        type=int,
        default=None, 
        help="specify a random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.kernel: # create blur kernels
        print("create_blur_kernels_family is not implemented yet. Please use matalab_kernels.py to generate blur kernels instead.")
        # create_blur_kernels_family(
        #     number_kernels=20,
        #     kernels_family_name="gaussian_blur_kernels",
        # )
        # if args.plot:
        #     utils_plots.plot_operator_family(operator_family="gaussian_blur_kernels", n_samples=16)

    if args.mask: # create inpainting masks
        
        # create box masks
        create_inpainting_masks_family(
            mask_type="box",
            mask_len_range=[96, 128],
            mask_prob_range=None,
            image_shape=(256, 256),
            margin=(16, 16),
            number_masks=args.n_samples,
            mask_family_name="box_masks",
            seed=args.seed,
        )
        if args.plot:
            utils_plots.plot_operator_family(operator_family="box_masks", n_samples=16)

        # create random masks
        create_inpainting_masks_family(
            mask_type="random",
            mask_len_range=[96, 128],
            mask_prob_range=[0.5, 0.7],
            image_shape=(256, 256),
            margin=(16, 16),
            number_masks=args.n_samples,
            mask_family_name="random_masks",
            seed=args.seed,
        )
        if args.plot:
            utils_plots.plot_operator_family(operator_family="random_masks", n_samples=16)


if __name__ == "__main__":
    main()
    # command example 1: python operators.py --kernel -x 20 -s 42 -p (create 20 blur kernels with seed 42 and plot)
    # command example 2: python operators.py --mask -x 20 -s 42 -p (create 20 inpainting masks with seed 42 and plot)