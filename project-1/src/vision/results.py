from pathlib import Path
from vision.part1 import my_conv2d_numpy
from vision.utils import *

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

class Image:
    def __init__(self, filename, filter):
        self.path = f"{ROOT}\\data\\1a_dog.bmp"
        self.img  =  load_image(f"{ROOT}/data/1a_dog.bmp")
        if filter == 'identity':
            self.filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        if filter == 'box':
            self.filter = np.repeat(np.array([[1, 1, 1]]), 5, 1)

    def show_img(self):
        plt.imshow(self.img)
        plt.show()
    
    def show_filtered_img(self):
        plt.imshow(my_conv2d_numpy(self.img, self.filter), self.filter)
        plt.show()

img = Image("1b_cat.bmp", 'identity')
img.show_img()


