# Q2.1
import cv2
import matplotlib.pyplot as plt

from matchPics import matchPics

# Load the images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

# Match the features
locs1, locs2, matches = matchPics(cv_cover, cv_desk)


# Display the matched features
match_image = cv2.drawMatches(cv_cover, locs1, cv_desk, locs2, matches[:40], None,
                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('show all matchings', match_image)
cv2.imwrite('../results/2_1_FAST_and_BRIEF.jpg', match_image)
cv2.waitKey()