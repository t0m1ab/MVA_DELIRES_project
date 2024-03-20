import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import delires.utils.utils as utils
from delires.params import OPERATORS_PATH


def visualize_single_operator(operator_family: str, operator_idx: int|str):
    """
    Example: visualize_single_operator(operator_family="custom_blur_centered", operator_idx=823)
    """

    operator_filename = os.path.join(OPERATORS_PATH, operator_family, f"{operator_family}_{operator_idx}.npy")

    operator = np.load(operator_filename)
    # print(operator.shape, operator.dtype, operator.min(), operator.max())

    plt.imshow(operator, cmap="gray")
    plt.axis("off")
    plt.title(f"{os.path.basename(operator_filename)} | shape={operator.shape}")
    _ = plt.show()


def plot_operator_family(operator_family: str, n_samples: int = None, path: str = None):
    """ Create a plot with all operators from an operator family limited to n_samples if specified. """
    
    path = path if path is not None else OPERATORS_PATH
    
    if not os.path.isdir(os.path.join(path, operator_family)):
        raise FileNotFoundError(f"Operator family {operator_family} not found in: {path}")

    operators_filenames = [f for f in os.listdir(os.path.join(path, operator_family)) if f.endswith(".npy")]
    n_samples = 100 if len(operators_filenames) > 100 else n_samples
    operators_filenames = operators_filenames[:n_samples] if n_samples is not None else operators_filenames
    operators_filenames = sorted(operators_filenames)
    operators_indexes = [int(f.split("_")[-1].split(".")[0]) for f in operators_filenames]

    operators = [np.load(os.path.join(path, operator_family, f)) for f in operators_filenames]

    (n_rows, n_cols) = utils.get_best_dimensions_for_plot(len(operators))


    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), dpi=300)
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            ax[row, col].imshow(operators[idx], cmap="gray")
            ax[row, col].axis("off")
            ax[row, col].set_title(f"id={operators_indexes[idx]} | shape={operators[idx].shape}")
    fig.suptitle(f"{operator_family} operators", fontsize=20)
    fig.tight_layout()
    plt.savefig(os.path.join(path, f"{operator_family }_operators.png"))

    print(f"Plot of {operator_family} operators saved in {path}")