#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    X = X.astype(dtype=np.int32)
    Y = Y.astype(dtype=np.int32)
    fvs = []
    shift = 0

    print("Image before padding = ", np.shape(image_bw))
    if feature_width % 2 == 0:
        # it is a square of even side
        shift = feature_width // 2 - 1

        # padding
        image_bw = np.pad(image_bw, ((shift, shift + 1), (shift, shift + 1)), 'constant', constant_values=((0,0)))
        print(np.shape(image_bw))

        for (row_idx, col_idx) in zip(Y, X):
            x_start = row_idx - shift + shift
            x_end = row_idx + shift + 1 + 1 + shift
            y_start = col_idx - shift + shift
            y_end = col_idx + shift + 1 + 1 + shift
            img_slice = image_bw[x_start: x_end, y_start : y_end].flatten()
            divisor = np.linalg.norm(img_slice)
            if divisor != 0:
                img_slice = img_slice / divisor
            if len(img_slice) != 256:
                print(np.shape(img_slice))
                # print("x_start = ", x_start, "x_end = ", x_end, " y_start = " , y_start, " y_end = ", y_end)
            fvs.append(img_slice)
    else:
        shift = (feature_width - 1) // 2
        for (x, y) in zip(X, Y):
            img_slice = image_bw[x - shift : x + shift, y - shift : y + shift].flatten()
            divisor = np.linalg.norm(img_slice)
            img_slice = img_slice / divisor
            fvs.append(img_slice)
    fvs = np.array(fvs, dtype=np.float32)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs

# scale_factor = 0.5
# image1 = load_image('./data/1a_notredame.jpg')
# image1 = PIL_resize(image1, (int(image1.shape[1]*scale_factor), int(image1.shape[0]*scale_factor)))
# image1_bw = rgb2gray(image1)

# num_interest_points = 2500
# X1, Y1, _ = get_harris_interest_points( copy.deepcopy(image1_bw), num_interest_points)
# image1_features = compute_normalized_patch_descriptors(np.array(image1_bw), X1, Y1, feature_width=16)