import os
from pathlib import Path
import hdf5storage
import numpy as np
from scipy import ndimage
import torch
from typing import List

import delires.utils.utils as utils
import delires.utils.utils_image as utils_image
from delires.methods.diffpir.utils import utils_sisr as sr
from delires.utils.utils_resizer import Resizer
from delires.methods.diffpir.utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from delires.methods.utils.utils_agem import fft_blur
from delires.methods.utils.utils_inpaint import mask_generator
from delires.params import OPERATORS_PATH, CLEAN_DATA_PATH, DEGRADED_DATA_PATH


def all_files_exist(filenames: list[str], ext: str = None, path: str = None) -> bool:
    """ Check if all files in a list exist. """
    path = "" if path is None else path
    if ext is not None:
        ext = ext[1:] if ext[0] == "." else ext
    else:
        ext = ""
    return all([os.path.isfile(os.path.join(path, f"{f}.{ext}")) for f in filenames])


# BLURRING

def tito_blur(image: np.ndarray, kernel: np.ndarray):
    """
    Apply a blur kernel to an image. 
    
    ARGUMENTS:
        - image: np.ndarray, the image to blur.
        - kernel: np.ndarray with 2 dimension, the blur kernel in float.
        
    RETURNS:
        - np.ndarray, the blurred image.
    """
    # mode='wrap' is important for analytical solution
    blurred_img = ndimage.convolve(image, np.expand_dims(kernel, axis=2), mode='wrap')
    return blurred_img


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


def load_blur_kernel(diy_kernel_name: str|None = None) -> np.ndarray:
    """ Load a blur kernel stored as a .npy file in OPERATORS_PATH. """
    if diy_kernel_name:
        k = np.load(os.path.join(OPERATORS_PATH, f"{diy_kernel_name}.npy"))
    else:
        k_index = 0
        kernels = hdf5storage.loadmat(os.path.join(OPERATORS_PATH, 'Levin09.mat'))['kernels']
        k = kernels[0, k_index].astype(np.float32)

    k = np.squeeze(k) # remove single dimensions

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
    clean_dataset = sorted(os.listdir(clean_dataset_path))
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


# DOWNSAMPLING

def load_downsample_kernel(
    classical_degradation: bool,
    sf: int,
    k_index: int = 0,
    ):
    """ Fetch the downsample kernel. k_index shoyld be 0 for bicubic degradation, in [0, 7] for classical degradation."""
    # kernels = hdf5storage.loadmat(os.path.join('kernels', 'Levin09.mat'))['kernels']
    if classical_degradation:
        kernels = hdf5storage.loadmat(os.path.join(OPERATORS_PATH, 'kernels_12.mat'))['kernels']
    else:
        kernels = hdf5storage.loadmat(os.path.join(OPERATORS_PATH, 'kernels_bicubicx234.mat'))['kernels']
    
    if not classical_degradation:  # for bicubic degradation
        k_index = sf-2 if sf < 5 else 2
    return kernels[0, k_index].astype(np.float64)


def create_downsampled_image(
        kernel: np.ndarray,
        img: str,
        sr_mode: str,
        classical_degradation: bool,
        sf: int,
        n_channels: int,
        noise_level_img: float,
        save_path: str|None = None,
        seed: int = 0,
        show_img: bool = False,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """ Create a downsampled and noised image from a given clean image and a blur/downsample kernel. """
    
    img_name, ext = os.path.splitext(os.path.basename(img))
    clean_img = utils_image.imread_uint(img, n_channels=n_channels)
    clean_img = utils_image.modcrop(clean_img, sf)  # modcrop

    if sr_mode == 'blur':
        if classical_degradation:
            degraded_img = sr.classical_degradation(clean_img, kernel, sf)
            utils_image.imshow(degraded_img) if show_img else None
            degraded_img = utils_image.uint2single(degraded_img)
        else:
            degraded_img = utils_image.imresize_np(utils_image.uint2single(clean_img), 1/sf)
    elif sr_mode == 'cubic':
        clean_img_tensor = np.transpose(clean_img, (2, 0, 1))
        clean_img_tensor = torch.from_numpy(clean_img_tensor)[None,:,:,:].to(device)
        clean_img_tensor = clean_img_tensor / 255
        # set up resizers
        down_sample = Resizer(clean_img_tensor.shape, 1/sf).to(device)
        degraded_img = down_sample(clean_img_tensor)
        degraded_img = degraded_img.cpu().numpy()       #[0,1]
        degraded_img = np.squeeze(degraded_img)
        if degraded_img.ndim == 3:
            degraded_img = np.transpose(degraded_img, (1, 2, 0))

    np.random.seed(seed=seed)  # for reproducibility
    degraded_img = degraded_img * 2 - 1
    degraded_img += np.random.normal(0, noise_level_img * 2, degraded_img.shape) # add AWGN
    degraded_img = degraded_img / 2 + 0.5
    
    if save_path is not None:
        degraded_img = utils_image.single2uint(degraded_img)
        utils_image.imsave(degraded_img, os.path.join(save_path, f"{img_name}{ext}"))
    else:
        return degraded_img, clean_img, img_name, ext


def generate_degraded_dataset_downsampled(
    degraded_dataset_name: str,
    kernel: np.ndarray,
    kernel_name: str,
    n_channels: int,
    sr_mode: str,
    classical_degradation: bool,
    sf: int,
    noise_level_img: float,
    seed: int = 0,
    show_img: bool = False,
    ):
    """ Generate a degraded dataset from a clean dataset using a given downsample kernel. """
    print(f"Generating downsampled dataset '{degraded_dataset_name}' from clean dataset using kernel {kernel_name}.")
    clean_dataset_path = os.path.join(CLEAN_DATA_PATH)
    degraded_dataset_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name)
    
    # Load clean dataset
    clean_dataset = sorted(os.listdir(clean_dataset_path))
    os.makedirs(degraded_dataset_path, exist_ok=True)
    kwargs = {
        "degraded_dataset_name": degraded_dataset_name,
        "images": [os.path.basename(f).split(".")[0] for f in clean_dataset], # remove ext
        "kernel_name": kernel_name, 
        "n_channels": n_channels, 
        "sr_mode": sr_mode, 
        "classical_degradation": classical_degradation, 
        "sf": sf, 
        "noise_level_img": noise_level_img, 
        "seed": seed
    }
    utils.archive_kwargs(kwargs, os.path.join(degraded_dataset_path, "dataset_info.json"))   
    
    for img in clean_dataset:
        create_downsampled_image(
            kernel,
            os.path.join(clean_dataset_path, img),
            sr_mode,
            classical_degradation,
            sf,
            n_channels,
            noise_level_img,
            degraded_dataset_path,
            seed,
            show_img,
        )
        
    print(f"Downsampled dataset '{degraded_dataset_name}' generated.")
        
    return degraded_dataset_path


#### MASKING

def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Apply a mask to an image in [0, 1] range. """
    return image * mask


def create_inpainting_masks(
    mask_type: str,
    mask_len_range: List[int] = None,
    mask_prob_range: List[float] =None,
    image_shape: List[int] = (256, 256),
    n_channels: int = 3,
    margin: List[int] = (16, 16),
    number_masks: int = 20,
    masks_save_name: str|None = None,
    seed: int = 0,
    ):
    np.random.seed(seed=seed) # for reproducibility
    mask_gen = mask_generator(mask_type=mask_type, mask_len_range=mask_len_range, mask_prob_range=mask_prob_range, img_shape=image_shape, n_channels=n_channels, margin=margin)
    masks = []
    for _ in range(number_masks):
        mask = mask_gen()
        masks.append(mask)
    if masks_save_name is not None:
        np.save(os.path.join(OPERATORS_PATH, f"{masks_save_name}.npy"), masks)
    
    return masks


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
    degraded_img = apply_mask(degraded_img, mask)
    utils_image.imshow(degraded_img) if show_img else None

    if save_path is not None:
        np.save(os.path.join(save_path, f"{img_name}.npy"), degraded_img) # save as .npy because diffpir...
        utils_image.imsave(
            utils_image.single2uint(degraded_img), 
            os.path.join(save_path, f"{img_name}{ext}")
        ) # save as .png for visualization
    else:
        return degraded_img, clean_img, img_name, ext
    
    
def load_masks(masks_name: str|None = None) -> np.ndarray:
    """ Load a set of masks stored as a .npy file in KERNELS_PATH. """
    if masks_name:
        masks = np.load(os.path.join(OPERATORS_PATH, f"{masks_name}.npy"))
    else:
        masks = np.load(os.path.join(OPERATORS_PATH, "square_masks.npy"))

    return masks
    
    
def generate_degraded_dataset_masked(
    degraded_dataset_name: str,
    masks: np.ndarray,
    masks_name: str,
    n_channels: int,
    noise_level_img: float,
    seed: int = 0,
    show_img: bool = False,
    ):
    """ Generate a degraded dataset from a clean dataset using given masks. """
    print(f"Generating masked dataset '{degraded_dataset_name}' from clean dataset using masks {masks_name}.")
    clean_dataset_path = os.path.join(CLEAN_DATA_PATH)
    degraded_dataset_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name)
    
    # Load clean dataset
    clean_dataset = sorted(os.listdir(clean_dataset_path))
    os.makedirs(degraded_dataset_path, exist_ok=True)
    kwargs = {
        "degraded_dataset_name": degraded_dataset_name,
        "images": [os.path.basename(f).split(".")[0] for f in clean_dataset], # remove ext
        "masks_name": masks_name, 
        "n_channels": n_channels, 
        "noise_level_img": noise_level_img, 
        "seed": seed
    }
    utils.archive_kwargs(kwargs, os.path.join(degraded_dataset_path, "dataset_info.json"))    
    
    for i, img in enumerate(clean_dataset):
        mask = masks[i]
        create_masked_image(
            mask, 
            os.path.join(clean_dataset_path, img),
            n_channels,
            noise_level_img,
            degraded_dataset_path,
            seed,
            show_img,
        )
        
    print(f"Masked dataset '{degraded_dataset_name}' generated.")
        
    return degraded_dataset_path


def main():

    ### GENERATE BLURRED DATASET
    seed = 0
    # kernel_name = "gaussian_kernel_05"
    kernel_name = "motion_kernel_example"
    noise_level_img = 12.75/255.0 # 0.05
    # blur_kernel = create_blur_kernel("Gaussian", 61, seed, kernel_name, "cpu")
    blur_kernel = load_blur_kernel(kernel_name) # should be a 2D float32 np.ndarray with values in [0,1]
    generate_degraded_dataset_blurred("blurred_ffhq_test20", blur_kernel, kernel_name, 3, noise_level_img, seed, False)

    ### GENERATE DOWNSAMPLED DATASET
    # seed = 0
    # sr_mode = "cubic"
    # classical_degradation = False
    # sf = 4
    # if classical_degradation and sr_mode == "blur":
    #     kernel_name = "kernels_12.mat"
    #     kernel = load_downsample_kernel(classical_degradation, sf, 0)
    # else:
    #     kernel_name = "None"
    #     kernel = None
    # generate_degraded_dataset_downsampled("downsampled_ffhq_test20", kernel, kernel_name, 3, sr_mode, False, 4, 0.05, seed, False)
    
    ### GENERATE MASKED DATASET
    # seed = 0
    # masks_name = "box_masks"
    # noise_level_img = 12.75/255.0 # 0.05
    # Generate masked dataset
    # seed = 0
    # masks_name = "box_masks"
    # noise_level_img = 12.75/255.0 # 0.05
    # masks = create_inpainting_masks(
    #     mask_type = "box",
    #     mask_len_range = [96, 128],
    #     mask_prob_range = None,
    #     image_shape = (256, 256),
    #     n_channels = 3,
    #     margin = (16, 16),
    #     number_masks = 20,
    #     masks_save_name = masks_name,
    #     seed = seed
    #     )
    # masks = load_masks(masks_name)
    # generate_degraded_dataset_masked("masked_ffhq_test20", masks, masks_name, 3, noise_level_img, seed, False)

if __name__ == "__main__":
    main()