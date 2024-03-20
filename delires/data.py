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
import delires.utils.utils_plots as utils_plots
from delires.methods.diffpir.utils import utils_sisr as utils_sr
from delires.utils.utils_resizer import Resizer
from delires.methods.diffpir.utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from delires.methods.utils.utils_agem import fft_blur
from delires.methods.utils import utils_inpaint as utils_inpaint
from delires.params import OPERATORS_PATH, CLEAN_DATA_PATH, DEGRADED_DATA_PATH


def all_files_exist(filenames: list[str], ext: str = None, path: str = None) -> bool:
    """ Check if all files in a list exist. """
    path = "" if path is None else path
    if ext is not None:
        ext = ext[1:] if ext[0] == "." else ext
    else:
        ext = ""
    return all([os.path.isfile(os.path.join(path, f"{f}.{ext}")) for f in filenames])


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
        filename = os.path.join(operator_family, f"{operator_family}_{operator_idx}.npy")
    
    if os.path.isfile(os.path.join(path, filename)):
        operator = np.load(os.path.join(path, filename))
    else:
        raise ValueError(f"Operator file {filename} not found in: {path}")

    operator = np.squeeze(operator) # remove single dimensions

    if not operator.ndim == 2:
        raise ValueError(f"operator.ndim must be 2, but operator has shape {operator.shape}")

    return operator


### BLURRING

def blur(img: torch.Tensor, k: np.ndarray | torch.Tensor) -> torch.Tensor:
    """ Originally fft_blur from PiGDM utils_agem.py """
    k_tensor = torch.FloatTensor(k)[None, None]
    degraded_img = fft_blur(img, k_tensor)
    return degraded_img


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
    # TODO: change this to avoid passing  through device?
    k_tensor = kernel.get_kernel().to(device, dtype=torch.float)
    k = k_tensor.clone().detach().cpu().numpy()       #[0,1]
    k = np.squeeze(k)
    k = np.squeeze(k)
    
    if kernel_save_name is not None:
        np.save(os.path.join(OPERATORS_PATH, f"{kernel_save_name}.npy"), k)
    
    return k


def create_blurred_and_noised_image(
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
    degraded_img = blur(clean_img, kernel)
    
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
        np.save(os.path.join(save_path, f"{img_name}.npy"), degraded_img_float32)
        # save as .png for visualization (can also be loaded to get uint8 values)
        utils_image.imsave(degraded_img_uint8, os.path.join(save_path, f"{img_name}{ext}"))
    else:
        return degraded_img, clean_img, img_name, ext


def generate_degraded_dataset_blurred(
    degraded_dataset_name: str,
    kernel: np.ndarray,
    kernel_name: str,
    n_channels: int,
    noise_level_img: float,
    seed: int = 0,
    show_img: bool = False,
    ):
    """ Generate a degraded dataset from a clean dataset using a given blur kernel. """
    clean_base_dir = os.path.basename(CLEAN_DATA_PATH)
    print(f"Generating dataset '{degraded_dataset_name}' from '{clean_base_dir}' with blur kernel '{kernel_name}'...")
    clean_dataset_path = os.path.join(CLEAN_DATA_PATH)
    degraded_dataset_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name)
    Path(clean_dataset_path).mkdir(parents=True, exist_ok=True)
    Path(degraded_dataset_path).mkdir(parents=True, exist_ok=True)
    
    # Load clean dataset
    clean_dataset = utils.sorted_nicely(utils.listdir(clean_dataset_path))
    clean_dataset = [f for f in clean_dataset if not f.startswith(".")]
    kwargs = {
        "degraded_dataset_name": degraded_dataset_name,
        "images": [os.path.basename(f).split(".")[0] for f in clean_dataset], # remove ext
        "kernel_name": kernel_name, 
        "n_channels": n_channels, 
        "noise_level_img": noise_level_img, 
        "seed": seed
    }
    utils.archive_kwargs(kwargs, os.path.join(degraded_dataset_path, "dataset_info.json"))    
    
    for img in clean_dataset:
        create_blurred_and_noised_image(
            kernel, 
            os.path.join(clean_dataset_path, img),
            n_channels,
            noise_level_img,
            degraded_dataset_path,
            seed,
            show_img,
        )
        
    print(f"Blurred dataset '{degraded_dataset_name}' generated.")
        
    return degraded_dataset_path


#### MASKING

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Apply a mask to an image in [0, 1] range. """
    return image * mask


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
        np.save(os.path.join(mask_family_path, f"{mask_family_name}_{i}.npy"), mask)

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
        np.save(os.path.join(save_path, f"{img_name}.npy"), degraded_img) # save as .npy to adapt to diffpir...
        utils_image.imsave(
            utils_image.single2uint(degraded_img), 
            os.path.join(save_path, f"{img_name}{ext}")
        ) # save as .png for visualization
    else:
        return degraded_img, clean_img, img_name, ext


def generate_degraded_dataset_masked(
    degraded_dataset_name: str,
    mask_family_name: str,
    n_channels: int,
    noise_level_img: float,
    seed: int = 0,
    show_img: bool = False,
    ):
    """ Generate a degraded dataset from a clean dataset using the first masks in the passed mask family. """
    print(f"Generating masked dataset '{degraded_dataset_name}' from clean dataset using masks from {mask_family_name}.")
    clean_dataset_path = os.path.join(CLEAN_DATA_PATH)
    degraded_dataset_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name)
    
    # Load clean dataset
    clean_dataset = utils.sorted_nicely(utils.listdir(clean_dataset_path))
    os.makedirs(degraded_dataset_path, exist_ok=True)
    image_names = [os.path.basename(f).split(".")[0] for f in clean_dataset] # remove ext    
    
    # Create image-to-mask correspondance
    image_to_mask = {image: mask_idx for mask_idx, image in enumerate(image_names)}  
    
    kwargs = {
        "degraded_dataset_name": degraded_dataset_name,
        "images": image_names,
        "mask_family_name": mask_family_name,
        "image_to_mask": image_to_mask,
        "n_channels": n_channels, 
        "noise_level_img": noise_level_img, 
        "seed": seed
    }
    utils.archive_kwargs(kwargs, os.path.join(degraded_dataset_path, "dataset_info.json"))    
    
    for i, img in enumerate(clean_dataset):
        mask = load_operator(operator_family=mask_family_name, operator_idx=i)
        create_masked_image(
            mask=mask, 
            img=os.path.join(clean_dataset_path, img),
            n_channels=n_channels,
            noise_level_img=noise_level_img,
            save_path=degraded_dataset_path,
            seed=seed,
            show_img=show_img,
        )
        
    print(f"Masked dataset '{degraded_dataset_name}' generated.")
        
    return degraded_dataset_path


def main():

    parser = argparse.ArgumentParser(description="Generate kernels/masks/datasets.")
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
        "--seed",
        "-s",
        dest="seed",
        type=int,
        default=None, 
        help="specify a random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.kernel: # create blur kernels family
        raise NotImplementedError("Not implemented yet.")
    
    if args.mask: # create inpainting masks family

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

        # plot masks
        utils_plots.plot_operator_family(operator_family="box_masks", n_samples=16)
        utils_plots.plot_operator_family(operator_family="random_masks", n_samples=16)
    
    if not args.kernel and not args.mask: # generate dataset
        
        # GENERATE BLURRED DATASET

        noise_level_img = 12.75/255.0 # 0.05

        kernel_filename = None
        kernel_family = "levin09"
        kernel_idx = 0

        generate_degraded_dataset_blurred(
            degraded_dataset_name="blurred_ffhq_test20", 
            kernel=load_operator(filename=kernel_filename, operator_family=kernel_family, operator_idx=kernel_idx),
            kernel_name=kernel_filename if kernel_filename is not None else f"{kernel_family}_{kernel_idx}", 
            n_channels=3, 
            noise_level_img=noise_level_img, 
            seed=42, 
        )
        
        # GENERATE MASKED DATASET
        
        noise_level_img = 12.75/255.0 # 0.05

        mask_family_name = "box_masks"

        generate_degraded_dataset_masked(
            degraded_dataset_name="masked_ffhq_test20", 
            mask_family_name=mask_family_name, 
            n_channels=3, 
            noise_level_img=noise_level_img, 
            seed=42, 
        )


if __name__ == "__main__":
    main()
    # command example 1: python data.py --kernel -x 20 -s 42 (create 20 blur kernels with seed 42)
    # command example 2: python data.py --mask -x 20 -s 42 (create 20 inpainting masks with seed 42)
    # command example 3: python data.py (generate datasets with existing masks/kernels familiy)