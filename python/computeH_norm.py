# Q2.4

import numpy as np
import cv2 
from computeH import transform_image_to_plane, computeH

def computeH_norm(x1: np.array, x2: np.array):
    """
    Computes H transform from image 2 to image 1 with normalization.
    ## Parameters:
        x1: feature coordinates of image 1, shape(N,2)
        x2: feature coordinates of image 2, shape(N,2)
    ## Returns:
        H transform matrix (3,3) from coordinates 2 to 1
    """
    # Compute centroids of the points
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    x1_shifted = x1 - centroid1
    x2_shifted = x2 - centroid2

    # Normalize the points so that the average distance from the origin is equal to sqrt(2)
    std1 = np.std(x1_shifted, axis=0)
    std2 = np.std(x2_shifted, axis=0)
    x1_normalized = x1_shifted / std1
    x2_normalized = x2_shifted / std2

    # Similarity transform 1
    T1 = np.array([[ 1/std1[0],      0,       -centroid1[0]/std1[0] ],
                   [     0,       1/std1[1],  -centroid1[1]/std1[1] ],
                   [     0,          0,                    1        ]])

    # Similarity transform 2
    T2 = np.array([[ 1/std2[0],      0,       -centroid2[0]/std2[0] ],
                   [     0,       1/std2[1],  -centroid2[1]/std2[1] ],
                   [     0,          0,                    1        ]])

    # Compute Homography
    H21_normalized_points = computeH(x1_normalized, x2_normalized)

    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H21_normalized_points @ T2

    return H2to1


if __name__ == "__main__":
    I1 = cv2.imread('../data/hp_cover.jpg')
    I2 = cv2.imread('../data/hp_desk.png')
    # I1 = cv2.imread('../data/cv_cover.jpg')
    # I2 = cv2.imread('../data/cv_desk.png')
    match_image = transform_image_to_plane(I1, I2, computeH_norm, 10)
    cv2.imshow("Transformed Image Match using Normalized H", match_image)
    cv2.imwrite("../results/2_4_Match_using_normalized_H.jpg",
                match_image, )
    cv2.waitKey()