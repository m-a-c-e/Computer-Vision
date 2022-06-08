"""Fundamental matrix utilities."""

from matplotlib.pyplot import axis, rc
import numpy as np
from torch import full


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    mean_x_y = np.mean(points, axis=0)
    std_x_y = 1 / np.std(points, axis=0)

    # append 1
    scale_diag = np.append(std_x_y, 1)
    scale_matrix = np.diag(scale_diag)

    # guild the offset matrix
    offset_matrix = np.diag([1., 1., 1.])
    offset_matrix[0, 2] = -mean_x_y[0]
    offset_matrix[1, 2] = -mean_x_y[1]

    # build the transformaiton matrix
    T = np.matmul(scale_matrix, offset_matrix)
    
    # normalize points
    points_t = np.transpose(points)
    one_vector = np.ones((1, np.shape(points)[0]))
    points_t = np.append(points_t, one_vector, axis=0)

    points_normalized = T @ points_t
    points_normalized = np.delete(points_normalized, -1, axis=0).T


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    F_orig = T_b.transpose() @ (F_norm @ T_a)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    num_pts = 8
    pts_a, T_a = normalize_points(points_a)
    pts_b, T_b = normalize_points(points_b)

    m1 = np.expand_dims(pts_a[0:8, 0] * pts_b[0:8, 0], axis=1)
    m2 = np.expand_dims(pts_a[0:8, 1] * pts_b[0:8, 0], axis=1)
    m3 = np.expand_dims(pts_b[0:8, 0], axis=1)
    m4 = np.expand_dims(pts_a[0:8, 0] * pts_b[0:8, 1], axis=1)
    m5 = np.expand_dims(pts_a[0:8, 1] * pts_b[0:8, 1], axis=1)
    m6 = np.expand_dims(pts_b[0:8, 1], axis=1)
    m7 = np.expand_dims(pts_a[0:8, 0], axis=1)
    m8 = np.expand_dims(pts_a[0:8, 1], axis=1)
    one_vector = np.ones((num_pts, 1))

    # solve using least squares
    A = np.concatenate((m1, m2, m3, m4, m5, m6, m7, m8), axis=1)
    B = -1 * one_vector

    X = np.append(np.linalg.lstsq(A, B, rcond=None)[0], 1)
    X = np.reshape(X, (3, 3))

    # reduce the rank of the matrix
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    s[-1] = 0
    smat = np.diag(s)

    F = u @ smat @ vh

    F = unnormalize_F(F, T_a=T_a, T_b=T_b)
    F = F / F[-1, -1]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
