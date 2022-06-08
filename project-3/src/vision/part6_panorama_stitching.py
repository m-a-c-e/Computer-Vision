from matplotlib import image
import numpy as np
import cv2
from vision.part3_ransac import *
from vision.part2_fundamental_matrix import *


def resize_img(image, scale):
    aspect_ratio = image.shape[1] / image.shape[0]
    height = (int)(image.shape[0] * scale)
    width =  (int)(aspect_ratio * height)
    resized_image = cv2.resize(image, (width, height // 1))
    return resized_image


def estimate_homography_matrix(
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

    F = unnormalize_F(X, T_a=T_a, T_b=T_b)
    F = F / F[-1, -1]

    return F


def ransac_homography_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    prob_success = 0.97
    sample_size = 8.0
    ind_prob_correct = 0.9
    threshold = 0.1

    inliers_a = None
    inliers_b = None
    best_F = None
    max_inliers = 0
    num_inliers = 0
    total_matches = matches_a.shape[0]

    iterations = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct)    # get number of iterations to run

    for i in range(iterations):
        # sample 8 points from matches_a and matches_b and build fundamental matrix
        sample_indices = np.random.choice(np.arange(total_matches), 8, False)
        sample_matches_a = matches_a[sample_indices, :]     # 8 x 2 array
        sample_matches_b = matches_b[sample_indices, :]     # 8 x 2 array

        F = estimate_homography_matrix(sample_matches_a, sample_matches_b)

        one_vector = np.expand_dims(np.ones(total_matches), axis=1)
        x = np.concatenate((matches_a, one_vector), axis=1)
        x_prime = np.concatenate((matches_a, one_vector), axis=1)

        # x @ F @ x_prime.T ~ 0
        values = np.expand_dims(np.sum(x @ F * x_prime, axis=1), axis=1)

        # find the indices where values < threshold
        inlier_indices = np.where(values < threshold)
        num_inliers = np.shape(inlier_indices[0])[0]

        if max_inliers == 0 or num_inliers > max_inliers:
            inliers_a = np.expand_dims(matches_a[inlier_indices[0], inlier_indices[1]], axis=1)
            inliers_b = np.expand_dims(matches_b[inlier_indices[0], inlier_indices[1]], axis=1)
            max_inliers = num_inliers
            best_F = F
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_F, inliers_a, inliers_b


def panorama_stitch(imageA, imageB):
    """
    ImageA and ImageB will be an image pair that you choose to stitch together
    to create your panorama. This can be your own image pair that you believe
    will give you the best stitched panorama. Feel free to play around with 
    different image pairs as a fun exercise!
    
    Please note that you can use your fundamental matrix estimation from part3
    (imported for you above) to compute the homography matrix that you will 
    need to stitch the panorama.
    
    Feel free to reuse your interest point pipeline from project 2, or you may
    choose to use any existing interest point/feature matching functions from
    Opencv2. You may NOT use any pre-existing warping function though.

    Args:
        imageA: first image that we are looking at (from camera view 1) [A x B]
        imageB: second image that we are looking at (from camera view 2) [M x N]

    Returns:
        panorama: stitch of image 1 and image 2 using warp. Ideal dimensions
            are either:
            1. A or M x (B + N)
                    OR
            2. (A + M) x B or N)
    """

    panorama = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    #### SIFT USING OPENCV ###################################
    img1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv2.drawMatches(imageA, keypoints_1, imageB, keypoints_2, matches[:100], imageB, flags=2)
    cv2.imshow("SIFT Matching (Top 100)", img3)
    cv2.imwrite("100_sift_matches.jpg", img3)



    #### ESTIMATE FUNDAMENTAL MATRIX ######################
    # get the x and y co-ordinates in both images
    matchesA = []
    matchesB = []

    # For each match...
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        x1, y1 = keypoints_1[img1_idx].pt
        x2, y2 = keypoints_2[img2_idx].pt

        matchesA.append([x1, y1])
        matchesB.append([x2, y2])
    
    matchesA = np.array(matchesA)
    #matchesA = matchesA.astype(np.int32)
    matchesB = np.array(matchesB)

    H, mask = cv2.findHomography(matchesA, matchesB, 0)

    #### Map points from imageA to imageB's plane using the H
    num_cols_a = imageA.shape[0]
    num_rows_a = imageB.shape[1]

    x_coords = np.expand_dims(np.arange(num_cols_a), axis=0)
    x_coords = np.repeat(x_coords, num_rows_a, axis = 0)
    x_coords = np.ravel(x_coords)

    y_coords = np.expand_dims(np.arange(num_rows_a), axis=1)
    y_coords = np.repeat(y_coords, num_cols_a, axis=1)
    y_coords = np.ravel(y_coords)


    panorama = np.zeros((num_cols_a * 2, num_rows_a, 3))

    vec_a = np.reshape(np.array([0, 0, 1]), (3, 1))
    vec_b = H @ vec_a
    #vec_b = vec_b/vec_b[-1]
    offset = (vec_b).flatten().astype(np.int32)

    for x, y in zip(x_coords, y_coords):
        vec_a = np.reshape(np.array([x, y, 1]), (3, 1))
        vec_b = H @ vec_a
    #    vec_b = (vec_b/vec_b[-1]).flatten().astype(np.int32)
        vec_b = vec_b.flatten().astype(np.int32)
        x_pano = vec_b[0] - offset[0]
        y_pano = vec_b[1] - offset[1]
        brightness = imageA[x][y]
        try:
            panorama[x_pano][y_pano] = brightness 
        except:
            pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cv2.imshow("hello", panorama)
    cv2.waitKey(0)

    return panorama
imageA = resize_img(cv2.imread('data/fall1.jpg'), 0.25)
imageB = resize_img(cv2.imread('data/fall2.jpg'), 0.25) 
panorama_stitch(imageA, imageB)