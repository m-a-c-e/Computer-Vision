#!/usr/bin/python3

import os
from pathlib import Path
from re import L
from timeit import repeat

from matplotlib import pyplot as plt

import numpy as np
from vision.part2_datasets import HybridImageDataset
from vision.utils import (
    im2single,
    load_image,
    save_image,
    single2im,
    vis_image_scales_numpy,
)

from vision.part1 import *

ROOT = Path(__file__).resolve().parent.parent  # ../..


def test_vis_image_scales_numpy():
    """Verify that the vis_hybrid_image function is working as anticipated."""
    fpath = f"{ROOT}/data/1a_dog.bmp"
    img = load_image(fpath)
    img_h, img_w, _ = img.shape

    img_scales = vis_image_scales_numpy(img)

    assert img_h == 361
    assert np.allclose(img_scales[:, :img_w, :], img)
    assert img_scales.shape == (361, 813, 3)
    assert isinstance(img_scales, np.ndarray)

    #### For visualization only ####
    # plt.imshow( (img_scales * 255).astype(np.uint8) )
    # plt.show()
    # ################################


# def test_im2single_gray():
#     """ Convert 3-channel RGB image.


#     """
#     rgb_img = im2single(im_uint8)


def test_im2single_rgb():
    """Convert an image with values [0,255] to a single-precision floating
    point data type with values [0,1].

    """
    img = np.array(range(4 * 5 * 3), dtype=np.uint8)
    img = img.reshape(4, 5, 3)
    float_img = im2single(img)

    gt_float_img = np.array(range(4 * 5 * 3), dtype=np.uint8)
    gt_float_img = gt_float_img.reshape(4, 5, 3).astype(np.float32)
    gt_float_img /= 255.0
    assert np.allclose(gt_float_img, float_img)
    assert gt_float_img.dtype == float_img.dtype
    assert gt_float_img.shape == float_img.shape


def test_single2im():
    """
    Test conversion from single-precision floating point in [0,1] to
    uint8 in range [0,255].
    """
    float_img = np.array(range(4 * 5 * 3), dtype=np.uint8)
    float_img = float_img.reshape(4, 5, 3).astype(np.float32)
    float_img /= 255.0
    uint8_img = single2im(float_img)

    gt_uint8_img = np.array(range(4 * 5 * 3), dtype=np.uint8)
    gt_uint8_img = gt_uint8_img.reshape(4, 5, 3)

    assert np.allclose(gt_uint8_img, uint8_img)
    assert gt_uint8_img.dtype == uint8_img.dtype
    assert gt_uint8_img.shape == uint8_img.shape


def test_load_image():
    """Load the dog image in `single` format."""
    fpath = f"{ROOT}/data/1a_dog.bmp"
    img = load_image(fpath)
    plt.imshow(img)
    plt.show()
    
    assert img.dtype == np.float32
    assert img.shape == (361, 410, 3)
    assert np.amin(img) >= 0.0
    assert np.amax(img) <= 1.0


def test_save_image():
    """ """
    save_fpath = "results/temp.png"

    # Create array as single-precision in [0,1]
    img = np.zeros((2, 3, 3), dtype=np.float32)
    img[0, 0, :] = 1
    img[1, 1, :] = 1
    save_image(save_fpath, img)
    assert Path(save_fpath).exists()
    os.remove(save_fpath)

class Image:
    def __init__(self, filename_low, filename_hi, filter):
        self.img_low  =  load_image(f"{ROOT}/data/" + filename_low)
        self.img_hi   =  load_image(f"{ROOT}/data/" + filename_hi)

        if filter == 'identity':
            self.filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        if filter == 'box':
            self.filter = np.repeat(np.array([[1, 1, 1]]), 5, 1) / (5 * 3)
        if filter == 'sobel':
            self.filter = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        if filter == 'laplacian':
            self.filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    def show_img(self):
        plt.imshow(self.img_low)
        plt.show()
    
    def show_filtered_img(self):
        filtered_img = my_conv2d_numpy(self.img_low, self.filter)
        plt.figure(figsize=(3,3)); 
        plt.imshow((filtered_img*255).astype(np.uint8));
        plt.show()
    
    def show_hybrid(self, cutoff_frequency):
        filter          = create_Gaussian_kernel_2D(cutoff_frequency)
        low, hi, hybrid = create_hybrid_image(self.img_low, self.img_hi, filter)
        plt.imshow(vis_image_scales_numpy(hybrid))
        plt.show()
#        plt.imshow(hybrid)
#        plt.show()

if __name__ == '__main__':
    cat     = "1b_cat.bmp"
    dog     = "1a_dog.bmp"
    motor   = "2a_motorcycle.bmp"
    bicycle = "2b_bicycle.bmp"
    plane   = "3a_plane.bmp"
    bird    = "3b_bird.bmp"
    ein     = "4a_einstein.bmp"
    mar     = "4b_marilyn.bmp"
    sub     = "5a_submarine.bmp"
    fish    = "5b_fish.bmp"

#    data = HybridImageDataset(f"{ROOT}/data", f"{ROOT}/cutoff_frequencies.txt")
#    print(data.images_a[0])



    filters = ['identity', 'box', 'sobel', 'laplacian']
    img1 = Image(cat, dog, filters[0])
    img2 = Image(cat, dog, filters[1])
    img3 = Image(cat, dog, filters[2])
    img4 = Image(cat, dog, filters[3]) 

    img3.show_filtered_img()

#   img2.show_filtered_img()    
#   img3.show_filtered_img()
#   img4.show_filtered_img()


    # img1 = Image(cat, dog, filters[0])
    # img2 = Image(motor, bicycle, filters[0])

    # img3 = Image(plane, bird, filters[0])
    # img4 = Image(ein, mar, filters[0])
    # img5 = Image(sub, fish, filters[0])


#    img1.show_hybrid(7)
    # img2.show_hybrid(8)
    # img3.show_hybrid(5)
    # img4.show_hybrid(4)
    # img5.show_hybrid(7)


    


