# Q2.6

import cv2
import numpy as np

from matchPics import matchPics
from computeH_ransac import computeH_ransac
from warpH import warpH
from compositeH import compositeH
from computeH import get_matched_coordinates

# Load images
cv_img = cv2.imread('../data/cv_cover.jpg')
desk_img = cv2.imread('../data/cv_desk.png')
hp_img = cv2.imread('../data/hp_cover.jpg')

# Extract features and match
locs1, locs2, matches = matchPics(cv_img, desk_img)

# Compute homography using RANSAC
pts1, pts2 = get_matched_coordinates(locs1,locs2,matches[:50])
bestH2to1, _, _ = computeH_ransac(pts1, pts2)
bestH1to2 = np.linalg.inv(bestH2to1)
                    
# Scale harry potter image to template size
scaled_hp_img = cv2.resize(hp_img, (cv_img.shape[1], cv_img.shape[0]))

# Display warped image
warped_hp_img = warpH(scaled_hp_img, bestH1to2, desk_img.shape)
cv2.imshow('Warped Image of Harry Potter', warped_hp_img)
cv2.imwrite("../results/2_6_Warped_HarryPotter.jpg", warped_hp_img)
cv2.waitKey(0)

# Display composite image
composite_img = compositeH(bestH1to2, scaled_hp_img, desk_img)
cv2.imshow('Composite Image', composite_img)
cv2.imwrite('../results/2_6_Composite_image.jpg', composite_img)
cv2.waitKey(0)
cv2.destroyAllWindows()