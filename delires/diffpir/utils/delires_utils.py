import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import hdf5storage
from scipy import ndimage

from utils import utils_image
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator


def create_blur_kernel(
        use_DIY_kernel: bool,
        blur_mode: str,
        kernel_size: int,
        seed: int = 0,
        cwd: str = "",
        device: str = "cpu",
    ) -> tuple[np.ndarray, torch.Tensor]:
    """ Create and return a blur kernel <k> and its associated <k_4d>. """

    if use_DIY_kernel:
        np.random.seed(seed=seed) # for reproducibility
        kernel_std = 3.0 if blur_mode == 'Gaussian' else 0.5
        if blur_mode == 'Gaussian':
            kernel_std_i = kernel_std * np.abs(np.random.rand()*2+1)
            kernel = GaussialBlurOperator(kernel_size=kernel_size, intensity=kernel_std_i, device=device)
        elif blur_mode == 'motion':
            kernel = MotionBlurOperator(kernel_size=kernel_size, intensity=kernel_std, device=device)
        else:
            raise ValueError(f"Unknown blur mode: {blur_mode}")
        k_tensor = kernel.get_kernel().to(device, dtype=torch.float)
        k = k_tensor.clone().detach().cpu().numpy()       #[0,1]
        k = np.squeeze(k)
        k = np.squeeze(k)
    else:
        k_index = 0
        kernels = hdf5storage.loadmat(os.path.join(cwd, 'kernels', 'Levin09.mat'))['kernels']
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
        seed: int = 0,
        show_img: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """ Create a blurred and noised image from a given clean image and a blur kernel. """

    img_name, ext = os.path.splitext(os.path.basename(img))
    img_H = utils_image.imread_uint(img, n_channels=n_channels)
    img_H = utils_image.modcrop(img_H, 8)  # modcrop

    # mode='wrap' is important for analytical solution
    img_L = ndimage.convolve(img_H, np.expand_dims(kernel, axis=2), mode='wrap')
    utils_image.imshow(img_L) if show_img else None
    img_L = utils_image.uint2single(img_L)

    np.random.seed(seed=seed) # for reproducibility
    img_L = img_L * 2 - 1
    img_L += np.random.normal(0, noise_level_img * 2, img_L.shape) # add AWGN
    img_L = img_L / 2 + 0.5

    return img_L, img_H, img_name, ext


def plot_sequence(seq: list, path: str = None, title: str = None):
    path = path if path is not None else os.getcwd()
    title = title if title is not None else "sequence"
    plt.plot(seq)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title(f"{title} (length={len(seq)})")
    plt.savefig(os.path.join(path, title + ".png"))
    plt.close()


def main():
    plot_sequence(seq=[1,2,2,2,3,4,5,5,5,6], path=None, title="my_sequence")


if __name__ == '__main__':
    main()