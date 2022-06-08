import numpy as np
from vision.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    x = ind_prob_correct ** sample_size
    x_not = 1 - x
    y = prob_success
    num_samples = 1 + np.log(1 - y) / np.log(x_not)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
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
        assert sample_matches_a.shape == (8, 2)

        F = estimate_fundamental_matrix(sample_matches_a, sample_matches_b)

        one_vector = np.expand_dims(np.ones(total_matches), axis=1)
        x = np.concatenate((matches_a, one_vector), axis=1)
        x_prime = np.concatenate((matches_a, one_vector), axis=1)

        # x @ F @ x_prime.T ~ 0
        values = np.expand_dims(np.sum(x @ F * x_prime, axis=1), axis=1)

        # find the indices where values < threshold
        inlier_indices = np.where(values < threshold)[0]
        num_inliers = np.shape(inlier_indices)[0]

        if max_inliers == 0 or num_inliers > max_inliers:
            inliers_a = matches_a[inlier_indices]
            inliers_b = matches_b[inlier_indices]
            max_inliers = num_inliers
            best_F = F
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_F, inliers_a, inliers_b


desired     = 0.999
sample_size = 18.0
ind_prob_correct = 0.9

print(calculate_num_ransac_iterations(desired, sample_size, ind_prob_correct))