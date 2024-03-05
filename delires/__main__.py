import sys
import os
import numpy as np

from delires.utils.utils import *
from delires.params import *



def run_experiment(apply_method, problem_type: str, clean_dataset_path: str, degraded_dataset_path: str, nb_gen: int = 1, exp_name: str = "exp"):
    """ 
    Run a test for a given method.
    """
    # TODO: merge from pseudocode to code

    # Load clean and degraded datasets
    clean_dataset_path = DATADIR + clean_dataset_path
    degraded_dataset = DATADIR + degraded_dataset_path
    
    # Create experiment folder
    os.makedirs(os.path.join(LOGDIR, exp_name), exist_ok=False)
    
    metrics = {"PSNR": []} # TODO: other metrics
    # Apply the method over the dataset
    print("Running test for", apply_method.__name__)
    for i in range(len(clean_dataset)):
        # Create image folder
        os.makedirs(os.path.join(LOGDIR, exp_name, f"img_{i}"), exist_ok=False)  # TODO: check that correspondance between clean and generated images is correct
        
        psnrs_img = []
        
        clean_image = clean_dataset[i]
        degraded_image = degraded_dataset[i]
        for j in range(nb_gen):
            result_image, intermediary_images = apply_method(clean_image, degraded_image)  # TODO: what to do with intermediary images?
            save_img(result_image, os.path.join(LOGDIR, exp_name, f"img_{i}", f"gen_{j}.png"))
            psnr, mse, var_img, var_avg, fid, cov = compute_metrics(result_image, clean_image)
            psnrs_img.append(psnr)
            
        metrics["PSNR"].append(psnrs_img)
        
    # Save metrics
    np.savez(os.path.join(LOGDIR, exp_name, "metrics.npz"), **metrics)
    report_metrics(metrics, os.path.join(LOGDIR, exp_name, "metrics.txt"))        
    

def main():
    print("Hello world!")


if __name__ == "__main__":
    run_experiment()

    
    # """ 
    # Entry point for the application script.
    # (sys.argv = list of arguments given to the program as strings separated by spaces)
    # """

    # if ("--help" in sys.argv) or ("-h" in sys.argv):
    #     print("No help available yet..")

    # elif ("--test" in sys.argv) or ("-t" in sys.argv):
    #     run_tests()

    # else:
    #     print("command 'alphazero' is working: try --help or --test")
