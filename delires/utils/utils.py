import numpy as np
from PIL import Image
import csv


def save_img(arr, path):
    im = Image.fromarray(arr)
    im.save(path)
    
    
def report_metrics(metrics, exp_path):
    with open(exp_path, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fields = ["img", "PSNR", "MSE", "Var_img", "Var_avg", "FID", "Cov"]
        writer.writerow(fields)
        writer.writerow(["Overall"] + [np.mean(metric) for metric in metrics])
        for i in range(len(metrics["PSNR"])):
            writer.writerow([i, np.mean(metrics["PSNR"][i])])