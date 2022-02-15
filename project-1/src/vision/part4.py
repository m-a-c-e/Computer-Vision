#!/usr/bin/python3

import numpy as np


def my_conv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation. 
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the convolution in the frequency domain, and 
    - the result of the convolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        conv_result_freq: array of shape (m, n)
        conv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the convolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 for how to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """
    # pad the filter to be the same size as image
    row_diff = image.shape[0] - filter.shape[0]
    col_diff = image.shape[1] - filter.shape[1]
    row_pad = row_diff // 2
    col_pad = col_diff // 2
    if row_diff % 2 != 0:
        if col_diff % 2 != 0:
            filter = np.pad(filter, ((row_pad, row_pad + 1), (col_pad, col_pad + 1)), 'constant', constant_values=(0, 0))
        else:
            filter = np.pad(filter, ((row_pad, row_pad + 1), (col_pad, col_pad)), 'constant', constant_values=(0, 0))
    else:
        if col_diff % 2 != 0:
            filter = np.pad(filter, ((row_pad, row_pad), (col_pad, col_pad + 1)), 'constant', constant_values=(0, 0))
        else:
            filter = np.pad(filter, ((row_pad, row_pad), (col_pad, col_pad)), 'constant', constant_values=(0, 0))

    image_freq       = np.fft.fft2(image)
    filter_freq      = np.fft.fft2(filter)
    conv_result_freq = np.multiply(image_freq, filter_freq)
    conv_result      = np.real(np.fft.ifft2(conv_result_freq))
    conv_result_freq = conv_result_freq

    image_freq       = np.fft.fftshift(image_freq)
    filter_freq      = np.fft.fftshift(filter_freq)
    conv_result_freq = np.fft.fftshift(conv_result_freq)
    conv_result      = np.fft.fftshift(conv_result)

    return image_freq, filter_freq, conv_result_freq, conv_result 


def my_deconv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation.
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the deconvolution in the frequency domain, and 
    - the result of the deconvolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        deconv_result_freq: array of shape (m, n)
        deconv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the deconvolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 to see what this means and to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """
    row_diff = image.shape[0] - filter.shape[0]
    col_diff = image.shape[1] - filter.shape[1]
    row_pad = row_diff // 2
    col_pad = col_diff // 2
    if row_diff % 2 != 0:
        if col_diff % 2 != 0:
            filter = np.pad(filter, ((row_pad, row_pad + 1), (col_pad, col_pad + 1)), 'constant', constant_values=(0, 0))
        else:
            filter = np.pad(filter, ((row_pad, row_pad + 1), (col_pad, col_pad)), 'constant', constant_values=(0, 0))
    else:
        if col_diff % 2 != 0:
            filter = np.pad(filter, ((row_pad, row_pad), (col_pad, col_pad + 1)), 'constant', constant_values=(0, 0))
        else:
            filter = np.pad(filter, ((row_pad, row_pad), (col_pad, col_pad)), 'constant', constant_values=(0, 0))

    image_freq       = np.fft.fft2(image)
    filter_freq      = np.fft.fft2(filter)
    deconv_result_freq = np.divide(image_freq, filter_freq)
    deconv_result      = np.real(np.fft.ifft2(deconv_result_freq))
    deconv_result_freq = deconv_result_freq

    image_freq         = np.fft.fftshift(image_freq)
    filter_freq        = np.fft.fftshift(filter_freq)
    deconv_result_freq = np.fft.fftshift(deconv_result_freq)
    deconv_result      = np.fft.fftshift(deconv_result)

    return image_freq, filter_freq, deconv_result_freq, deconv_result


