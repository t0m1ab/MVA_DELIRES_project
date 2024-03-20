import os
from pathlib import Path
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from delires.params import (
    OPERATORS_PATH, 
    MODELS_PATH, 
    HF_REPO_ID,
    MATLAB_BLUR_KERNELS_FILES,
)


def download_matlab_kernels_from_hf_hub(path: str = None):
    """ 
    Download matlab kernels files listed in MATLAB_BLUR_KERNELS_FILES from HF_REPO_ID
    and stored them in path.
    """

    path = path if path is not None else OPERATORS_PATH
    Path(path).mkdir(parents=True, exist_ok=True)

    print(f"Downloading matlab kernels from {HF_REPO_ID}...")
    for filename in MATLAB_BLUR_KERNELS_FILES:
        if os.path.isfile(os.path.join(path, filename)):
            print(f"File {filename} already exists in {path}")
        else:
            _ = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="model",
                filename=filename,
                local_dir=path,
            )


def matlab2numpy_kernel(matlab_kernels_filename: str, path: str = None, n_kernels: int = None, seed: int = None):
    """ 
    Transform a matlab kernels file into a series of numpy arrays 
    (one for each kernel) and save them as .npy files in path. 

    If n_kernels is None then it will extract all kernels,
    otherwise it will randomly extract n_kernels.

    Possible matlab_kernel_filename are:
    - 'Levin09.mat' [https://github.com/yuanzhi-zhu/DiffPIR/tree/main/kernels]
    - 'kernel_12.mat' [https://github.com/yuanzhi-zhu/DiffPIR/tree/main/kernels]
    - 'custom_blur_centered.mat' [https://drive.google.com/drive/folders/1aUuQa_6xyw6OVNDQClrsnY4ucvcPm6gn]
    """

    path = OPERATORS_PATH if path is None else path

    if not matlab_kernels_filename.endswith(".mat"):
        raise ValueError("matlab_kernel_filename must end with '.mat'")
    filepath = os.path.join(path, matlab_kernels_filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{matlab_kernels_filename} not found in: {path}")

    # kernels_list = loadmat(filepath)['kernels'][0] # doesn't work for 'custom_blur_centered.mat' (NotImplementedError: Please use HDF reader for matlab v7.3 files, e.g. h5py)
    kernels_list = hdf5storage.loadmat(filepath)['kernels']

    # choose n_kernels kernels to extract (all kernels if n_kernels is None)
    if kernels_list.shape[1] > 20 and n_kernels is None:
        raise ValueError(f"There are many kernels to extract {kernels_list.shape[1]}, you should only extract a subset by specifying n_kernels")
    kernels_indexes = np.arange(kernels_list.shape[1])
    if n_kernels is not None:
        if seed is not None:
            np.random.seed(seed)
        kernels_indexes = np.random.choice(kernels_indexes, n_kernels, replace=False)
    else:
        n_kernels = kernels_indexes.shape[0]

    # create subfolder in OPERATORS_PATH for each matlab kernel
    kernel_basename = matlab_kernels_filename.split(".")[0].lower()
    Path(os.path.join(path, kernel_basename)).mkdir(parents=True, exist_ok=True)

    # save individual kernels as .npy files
    for kidx in kernels_indexes:
        kernel = kernels_list[0, kidx].astype(np.float32)
        assert kernel.ndim == 2
        assert kernel.dtype == np.float32
        assert kernel.min() >= 0
        assert kernel.max() <= 1
        kernel_name = f"{kernel_basename}_{kidx}.npy"
        np.save(os.path.join(path, kernel_basename, kernel_name), kernel)
    
    print(f"Extracted {n_kernels} kernels from {matlab_kernels_filename} and saved them in: {path}/{kernel_basename}")


def tito_load_blur_kernel(kernel_filename: str, path: str = None) -> np.ndarray:
    """ Load a blur kernel stored as a .npy file in OPERATORS_PATH. """
    path = OPERATORS_PATH if path is None else path
    kernel_filename = kernel_filename.split(".")[0] # remove extension if present
    
    kernel = np.load(os.path.join(OPERATORS_PATH, f"{kernel_filename}.npy")) # load kernel
    kernel = np.squeeze(kernel) # remove single dimensions

    if not kernel.ndim == 2:
        raise ValueError(f"kernel.ndim must be 2, but kernel has shape {kernel.shape}")

    return kernel


def load_blur_kernel(
        filename: str = None, 
        kernel_family: str = None, 
        kernel_idx: str | int = None, 
        path: str = None
    )-> np.ndarray:
    """ 
    Load a blur kernel stored as a .npy file in a subfolder family in path.
    Check filename first which can be either like f'{family_name}/{family_name}_{kernel_idx}.npy' 
    or f'{familyname}_{kernel_idx}.npy' (with or without .npy extension).
    If filename is None, then kernel_family and kernel_idx must be provided and combined to find the right kernel.
    If path is None, then OPERATORS_PATH is used.
    """
    path = OPERATORS_PATH if path is None else path

    if filename is not None: # use filename
        filename = f"{filename}.npy" if not filename.endswith(".npy") else filename # add extension if not present
    else: # use kernel_family and kernel_idx
        if kernel_family is None or kernel_idx is None:
            raise ValueError("kernel_family and kernel_idx must be provided to load a kernel if no valid filename is provided")
        filename = os.path.join(kernel_family, f"{kernel_family}_{kernel_idx}.npy")
    
    if os.path.isfile(os.path.join(path, filename)):
        kernel = np.load(os.path.join(path, filename))
    else:
        raise ValueError(f"Kernel file {filename} not found in: {path}")

    kernel = np.squeeze(kernel) # remove single dimensions

    if not kernel.ndim == 2:
        raise ValueError(f"kernel.ndim must be 2, but kernel has shape {kernel.shape}")

    return kernel


def visualize_single_kernel(kernel_family: str, kernel_idx: int|str):
    """
    Example: visualize_single_kernel(kernel_family="custom_blur_centered", kernel_idx=823)
    """

    kernel_filename = os.path.join(OPERATORS_PATH, kernel_family, f"{kernel_family}_{kernel_idx}.npy")

    kernel = np.load(kernel_filename)
    # print(kernel.shape, kernel.dtype, kernel.min(), kernel.max())

    plt.imshow(kernel, cmap="gray")
    plt.axis("off")
    plt.title(f"{os.path.basename(kernel_filename)} | shape={kernel.shape}")
    _ = plt.show()


def main():

    ### DOWNLOAD matlab kernels files from HF_REPO_ID and stored them in MODELS_PATH
    download_matlab_kernels_from_hf_hub()

    ### EXTRACT matlab kernels in individual .npy files for each of the following matlab kernels files
    matlab2numpy_kernel(matlab_kernels_filename='Levin09.mat') # diffpir kernels
    matlab2numpy_kernel(matlab_kernels_filename='kernels_12.mat') # diffpir kernels
    matlab2numpy_kernel(matlab_kernels_filename='custom_blur_centered.mat', n_kernels=20, seed=42) # DELIRES TP3 motion kernels

    ### VISUALIZE a single kernel
    # visualize_single_kernel(kernel_family="levin09", kernel_idx=1)
    # visualize_single_kernel(kernel_family="kernels_12", kernel_idx=8)
    # visualize_single_kernel(kernel_family="custom_blur_centered", kernel_idx=30277)


if __name__ == "__main__":
    main()