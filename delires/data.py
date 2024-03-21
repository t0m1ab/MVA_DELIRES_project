import os
from pathlib import Path
import hdf5storage
import argparse
import numpy as np
from scipy import ndimage
import torch
from typing import List

import delires.utils.utils as utils
import delires.utils.utils_image as utils_image
import delires.utils.operators as operators
from delires.methods.utils.utils_agem import fft_blur
from delires.methods.utils import utils_inpaint as utils_inpaint
from delires.params import OPERATORS_PATH, CLEAN_DATA_PATH, DEGRADED_DATA_PATH


### BLURRING

def apply_blur(img: torch.Tensor, k: np.ndarray | torch.Tensor) -> torch.Tensor:
    """ Originally fft_blur from PiGDM utils_agem.py """
    k_tensor = torch.FloatTensor(k)[None, None]
    degraded_img = fft_blur(img, k_tensor)
    return degraded_img


def create_blurred_image(
        kernel: np.ndarray, 
        img: str,
        n_channels: int,
        noise_level_img: float,
        save_path: str|None = None,
        seed: int = 0,
        show_img: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """ 
    Create a blurred and noised image from a given clean image and a blur kernel.

    ARGUMENTS:
        - kernel: 2D float32 np.ndarray with values in [0,1], the blur kernel.    
    """

    k_shape = kernel.shape
    if len(k_shape) != 2 or k_shape[0] != k_shape[1]:
        raise ValueError(f"Kernel shape is not square: {k_shape}")

    img_name, ext = os.path.splitext(os.path.basename(img))
    clean_img = utils_image.imread_uint(img, n_channels=n_channels) # load PNG file as uint8
    clean_img = utils_image.modcrop(clean_img, 8) # crop to the nearest multiple of 8 for the size
    clean_img = utils_image.uint2tensor4(clean_img) # convert to tensor with float [0,1] values and shape (1, C, H, W)

    # apply blur kernel
    degraded_img = apply_blur(clean_img, kernel)
    
    # utils_image.imshow(degraded_img) if show_img else None

    # (1, C, H, W) with unclipped float32 values in [0-esp,1+eps]
    degraded_img_float32 = degraded_img + torch.randn(degraded_img.size()) * noise_level_img

    # (1, C, H, W) with uint8 values in [0,255]
    degraded_img_uint8 = utils_image.single2uint(degraded_img)
    
    # DEBUG logs
    # print(f"Image: {img_name}")
    # print("degraded_img_float32", utils_image.get_infos_img(degraded_img_float32))
    # print("degraded_img_uint8", utils_image.get_infos_img(degraded_img_uint8))

    if save_path is not None:
        # save as .npy to avoid uint8 clipping info loss
        Path(os.path.join(save_path, "npy")).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_path, "npy", f"{img_name}.npy"), degraded_img_float32)
        # save as .png for visualization (can also be loaded to get uint8 values)
        Path(os.path.join(save_path, ext.strip("."))).mkdir(parents=True, exist_ok=True)
        utils_image.imsave(degraded_img_uint8, os.path.join(save_path, ext.strip("."), f"{img_name}{ext}"))
    else:
        return degraded_img, clean_img, img_name, ext


### MASKING

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Apply a mask to an image in [0, 1] range. """
    return image * mask


def create_masked_image(
        mask: np.ndarray, 
        img: str,
        n_channels: int,
        noise_level_img: float,
        save_path: str|None = None,
        seed: int = 0,
        show_img: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """ Create a blurred and noised image from a given clean image and a blur kernel. """
    np.random.seed(seed=seed)  # for reproducibility
    
    img_name, ext = os.path.splitext(os.path.basename(img))
    clean_img = utils_image.imread_uint(img, n_channels=n_channels)
    clean_img = utils_image.modcrop(clean_img, 8)  # modcrop

    degraded_img = utils_image.uint2single(clean_img)
    degraded_img = degraded_img * 2 - 1
    degraded_img += np.random.normal(0, noise_level_img * 2, degraded_img.shape) # add AWGN
    degraded_img = degraded_img / 2 + 0.5
    degraded_img = apply_mask(degraded_img, mask[:,:,None])
    utils_image.imshow(degraded_img) if show_img else None

    if save_path is not None:
        Path(os.path.join(save_path, "npy")).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_path, "npy", f"{img_name}.npy"), degraded_img) # save as .npy to adapt to diffpir...
        Path(os.path.join(save_path, ext.strip("."))).mkdir(parents=True, exist_ok=True)
        utils_image.imsave(
            utils_image.single2uint(degraded_img), 
            os.path.join(save_path, ext.strip("."), f"{img_name}{ext}")
        ) # save as .png for visualization
    else:
        return degraded_img, clean_img, img_name, ext


### DATASET GENERATION

def generate_degraded_dataset(
    degraded_dataset_name: str,
    operator_family_name: str,
    mode: str,
    n_channels: int,
    noise_level_img: float,
    seed: int = 0,
    show_img: bool = False,
    ):
    """
    Generate a degraded dataset from a clean dataset using the first operators in the passed operator family.
    """

    # security checks
    if mode not in ["blur", "mask"]:
        raise ValueError(f"Unknown mode: {mode}. Should be 'blur' or 'mask'.")
    if 'kernel' in operator_family_name and mode == "mask":
        print(f"WARNING: You may be trying to generate a masked dataset from a kernel family...")
    if 'mask' in operator_family_name and mode == "blur":
        print(f"WARNING: You may be trying to generate a blurred dataset from a mask family...")
    n_operators = operators.check_integrity_operators_family(operator_family_name)

    print(f"Generating degraded dataset '{degraded_dataset_name}' from clean dataset using operators from {operator_family_name}.")
    clean_dataset_path = os.path.join(CLEAN_DATA_PATH)
    degraded_dataset_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name)
    
    # load clean dataset
    clean_dataset = utils.sorted_nicely(utils.listdir(clean_dataset_path))
    Path(degraded_dataset_path).mkdir(parents=True, exist_ok=True)
    image_names = [os.path.basename(f).split(".")[0] for f in clean_dataset] # remove ext

    # load operators indexes and create image-to-operator correspondance
    operators_indexes = list(range(n_operators))
    if len(operators_indexes) < len(image_names): # need to use the same operator for multiple images
        factor = int(np.ceil(len(image_names) / n_operators))
        operators_indexes = [int(x) for x in np.tile(operators_indexes, factor)]
    if len(operators_indexes) > len(image_names): # need to cut the list of operators indexes
        operators_indexes = operators_indexes[:len(image_names)]
    image_to_operator = {img_name: op_idx for img_name, op_idx in zip(image_names, operators_indexes)} 

    kwargs = {
        "degraded_dataset_name": degraded_dataset_name,
        "images": image_names,
        "operator_family_name": operator_family_name,
        "image_to_operator": image_to_operator,
        "n_channels": n_channels, 
        "noise_level_img": noise_level_img, 
        "seed": seed,
    }
    utils.archive_kwargs(kwargs, os.path.join(degraded_dataset_path, "dataset_info.json"))    
    
    for img in clean_dataset:

        op_idx = image_to_operator[os.path.basename(img).split(".")[0]]
        operator = operators.load_operator(operator_family=operator_family_name, operator_idx=op_idx)

        if mode == "blur":
            create_blurred_image(
                kernel=operator, 
                img=os.path.join(clean_dataset_path, img),
                n_channels=n_channels,
                noise_level_img=noise_level_img,
                save_path=degraded_dataset_path,
                seed=seed,
                show_img=show_img,
            )
        elif mode == "mask":
            create_masked_image(
                mask=operator, 
                img=os.path.join(clean_dataset_path, img),
                n_channels=n_channels,
                noise_level_img=noise_level_img,
                save_path=degraded_dataset_path,
                seed=seed,
                show_img=show_img,
            )
    
    if mode == "blur":
        print(f"Blurred dataset '{degraded_dataset_name}' generated.")
    elif mode == "mask":
        print(f"Masked dataset '{degraded_dataset_name}' generated.")
        
    return degraded_dataset_path


def main():

    parser = argparse.ArgumentParser(description="Generate kernels/masks/datasets.")
    parser.add_argument(
        "--kernel_family",
        "-k",
        dest="kernel_family",
        type=str,
        default=None,
        help="create a blurred dataset from the specified kernels family"
    )
    parser.add_argument(
        "--mask_family",
        "-m",
        dest="mask_family",
        type=str,
        default=None, 
        help="create a masked dataset from the specified masks family"
    )
    parser.add_argument(
        "--noise",
        "-n",
        dest="noise",
        type=int,
        default=10,
        help="specify the noise level for the degraded images in [0,255]",
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

    noise_level_img = float(args.noise)/255.0 # 0.05
    if noise_level_img < 0 or noise_level_img > 1:
        raise ValueError("Noise level must be in [0,255]")
    
    if args.kernel_family is not None: # generate blurred dataset
        
        generate_degraded_dataset(
            degraded_dataset_name=f"blurred_{os.path.basename(CLEAN_DATA_PATH)}", # "blurred_ffhq_test20"
            operator_family_name=args.kernel_family,
            mode="blur",
            n_channels=3, 
            noise_level_img=noise_level_img, 
            seed=args.seed, 
        )
        
    if args.mask_family is not None: # generate masked dataset
        
        generate_degraded_dataset(
            degraded_dataset_name=f"masked_{os.path.basename(CLEAN_DATA_PATH)}", # "masked_ffhq_test20"
            operator_family_name=args.mask_family, 
            mode="mask",
            n_channels=3, 
            noise_level_img=noise_level_img, 
            seed=args.seed, 
        )


if __name__ == "__main__":
    main()
    # command example 1: python data.py --kernel levin09 -n 15 -s 42 (create blurred dataset with Levin09 kernels, noise level 15 and seed 42)
    # command example 2: python data.py --mask box_masks -n 15 -s 42 (create masked dataset with box masks, noise level 15 and seed 42)