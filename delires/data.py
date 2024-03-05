import os
import hdf5storage
import numpy as np
from scipy import ndimage
import torch
import logging

from delires.params import *
import delires.utils.utils as utils
import delires.utils.utils_image as utils_image
from delires.diffusers.diffpir.utils import utils_sisr as sr
from delires.utils.utils_resizer import Resizer
from delires.diffusers.diffpir.utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator



# BLURRING

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
        np.save(os.path.join(KERNELDIR, f"{kernel_save_name}.npy"), k)
    
    return k


def load_blur_kernel(
    diy_kernel_name: str|None = None,
    device: str|torch.DeviceObjType = "cpu",
    ):
    if diy_kernel_name:
        k = np.load(os.path.join(KERNELDIR, diy_kernel_name + ".npy"))
    else:
        k_index = 0
        kernels = hdf5storage.loadmat(os.path.join(KERNELDIR, 'Levin09.mat'))['kernels']
        k = kernels[0, k_index].astype(np.float32)

    # img_name, ext = os.path.splitext(os.path.basename(img))
    # util.imsave(k*255.*200, os.path.join(E_path, f'motion_kernel_{img_name}{ext}'))
    # util.imsave(k*255.*200, os.path.join(E_path, "blur_kernel.jpeg"))
    #np.save(os.path.join(E_path, 'motion_kernel.npy'), k)
    k_4d = torch.from_numpy(k).to(device)
    k_4d = torch.einsum('ab,cd->abcd',torch.eye(3).to(device),k_4d)

    return k, k_4d


def create_blurred_and_noised_image(
        kernel: np.ndarray, 
        img: str,
        n_channels: int,
        noise_level_img: float,
        save_path: str|None = None,
        seed: int = 0,
        show_img: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """ Create a blurred and noised image from a given clean image and a blur kernel. """
    img_name, ext = os.path.splitext(os.path.basename(img))
    clean_img = utils_image.imread_uint(img, n_channels=n_channels)
    clean_img = utils_image.modcrop(clean_img, 8)  # modcrop

    # mode='wrap' is important for analytical solution
    degraded_img = ndimage.convolve(clean_img, np.expand_dims(kernel, axis=2), mode='wrap')
    utils_image.imshow(degraded_img) if show_img else None
    degraded_img = utils_image.uint2single(degraded_img)

    np.random.seed(seed=seed) # for reproducibility
    degraded_img = degraded_img * 2 - 1
    degraded_img += np.random.normal(0, noise_level_img * 2, degraded_img.shape) # add AWGN
    degraded_img = degraded_img / 2 + 0.5

    if save_path is not None:
        degraded_img = utils_image.single2uint(degraded_img)
        utils_image.imsave(degraded_img, os.path.join(save_path, f"{img_name}{ext}"))
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
    print(f"Generating blurred dataset {degraded_dataset_name} from clean dataset using kernel {kernel_name}.")
    clean_dataset_path = os.path.join(CLEAN_DATA_PATH)
    degraded_dataset_path = os.path.join(DEGRAGDED_DATA_PATH, degraded_dataset_name)
    
    # Load clean dataset
    clean_dataset = os.listdir(clean_dataset_path)
    os.makedirs(degraded_dataset_path, exist_ok=True)
    utils.archive_kwargs({"degraded_dataset_name": degraded_dataset_name, "kernel_name": kernel_name, "n_channels": n_channels, "noise_level_img": noise_level_img, "seed": seed}, os.path.join(degraded_dataset_path, "dataset_info.json"))    
    
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
        
    print(f"Blurred dataset {degraded_dataset_name} generated.")
        
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
        kernels = hdf5storage.loadmat(os.path.join(KERNELDIR, 'kernels_12.mat'))['kernels']
    else:
        kernels = hdf5storage.loadmat(os.path.join(KERNELDIR, 'kernels_bicubicx234.mat'))['kernels']
    
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
    print(f"Generating downsampled dataset {degraded_dataset_name} from clean dataset using kernel {kernel_name}.")
    clean_dataset_path = os.path.join(CLEAN_DATA_PATH)
    degraded_dataset_path = os.path.join(DEGRAGDED_DATA_PATH, degraded_dataset_name)
    
    # Load clean dataset
    clean_dataset = os.listdir(clean_dataset_path)
    os.makedirs(degraded_dataset_path, exist_ok=True)
    utils.archive_kwargs({"degraded_dataset_name": degraded_dataset_name, "kernel_name": kernel_name, "n_channels": n_channels, "sr_mode": sr_mode, "classical_degradation": classical_degradation, "sf": sf, "noise_level_img": noise_level_img, "seed": seed}, os.path.join(degraded_dataset_path, "dataset_info.json"))
    
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
        
    print(f"Downsampled dataset {degraded_dataset_name} generated.")
        
    return degraded_dataset_path


if __name__ == "__main__":
    # Generate blurred dataset
    seed = 0
    kernel_name = "gaussian_kernel_05"
    noise_level_img = 0.05
    # blur_kernel = create_blur_kernel("Gaussian", 21, seed, kernel_name, "cpu")
    blur_kernel, _ = load_blur_kernel(kernel_name, "cpu")
    generate_degraded_dataset_blurred("blurred_dataset", blur_kernel, kernel_name, 3, noise_level_img, seed, False)
    
    # Generate downsampled dataset
    seed = 0
    sr_mode = "cubic"
    classical_degradation = False
    sf = 4
    if classical_degradation and sr_mode == "blur":
        kernel_name = "kernels_12.mat"
        kernel = load_downsample_kernel(classical_degradation, sf, 0)
    else:
        kernel_name = "None"
        kernel = None
    generate_degraded_dataset_downsampled("downsampled_dataset", kernel, kernel_name, 3, sr_mode, False, 4, 0.05, seed, False)