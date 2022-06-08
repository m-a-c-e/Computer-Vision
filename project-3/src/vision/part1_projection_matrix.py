import numpy as np


def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray
) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_3d: A numpy array of shape (N, 3)

    Returns:
        M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # A @ X = B
    A = []
    B = []
    for i in range(6):
        world_co = points_3d[i].flatten()
        image_co = points_2d[i].flatten()
        pad      = np.zeros(4)

        arr1 = np.append(world_co, 1)
        arr2 = -1 * world_co * image_co[0]
        arr3 = -1 * world_co * image_co[1]

        A.append(np.concatenate((arr1, pad, arr2), axis=0))
        A.append(np.concatenate((pad, arr1, arr3), axis = 0))

        B.append(image_co[0])
        B.append(image_co[1])

    A = np.array(A)
    B = np.array(B)

    M = np.append(np.linalg.lstsq(A, B, rcond=None)[0], 1)
    M = np.reshape(M, (3, 4))
    print(M)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # converting to 2D homogenous co-ordinates
    pad = np.ones((1, np.shape(points_3d)[0]))
    points_3d = np.concatenate((points_3d.T, pad), axis=0)
    conv_mat = P @ points_3d
    conv_mat /= conv_mat[-1]
    conv_mat = (conv_mat[0:conv_mat.shape[0] - 1, :]).T

    projected_points_2d = conv_mat
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    Q = M[:, 0:3]
    col = np.reshape(M[:, 3], (3, 1))
    Q_inv = np.linalg.inv(Q)
    cc = np.reshape(- 1 * Q_inv @ col,  (3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc
