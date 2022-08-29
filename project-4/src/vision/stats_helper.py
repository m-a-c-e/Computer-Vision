import glob
import os
from re import I
from typing import Tuple

import numpy as np
from PIL import Image
from torch import mul


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    # Find the mean of the dataset (cannot store the images, might run out of memory)
    mean = 0.0
    count = 0.0

    for root, dirs, files in os.walk(dir_name):
        for name in files:
            img_path = os.path.join(root, name)                             # get the image path
            img      = Image.open(img_path).convert('L')                    # convert to grayscale
            img      = np.array(img)                                             
            img      = img / 255.0
            count    += img.size
            mean     += np.sum(img)
    mean = mean / count
    # Find the standard deviation of the dataset
    std = 0.0
    count = 0.0

    for root, dirs, files in os.walk(dir_name):
        for name in files:
            img_path = os.path.join(root, name)                             # get the image path
            img      = Image.open(img_path).convert('L')                    # convert to grayscale
            img      = np.array(img)                                             
            img      = img / 255.0
            count    += img.size
            img      = np.sum(np.square(img - mean))
            std      += img
    std = std / (count - 1)
    std = np.sqrt(std)
        
    ############################################################################
    # Student code end
    ############################################################################
    return mean, std



