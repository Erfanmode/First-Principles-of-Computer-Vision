# Q2.2
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert to grayscale
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_cover_G = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
# Get the image dimensions
height, width = cv_cover_G.shape[:2]
# Define the rotation center
center = (width // 2, height // 2)

# Create a histogram to store the results
orientation = [i*10 for i in range(37)]
match_count = []

# Compute the features and descriptors for the original image using BRIEF
# Initiate FAST, SIFT, and SURF object with default values for feature detector
fast = cv2.xfeatures2d.StarDetector_create()
sift = cv2.xfeatures2d.SIFT_create()

# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Compute features and descriptors for original image
kp0 = fast.detect(cv_cover_G,None)
kp0, des0 = brief.compute(cv_cover_G, kp0)

# Initiate matching object for two images by using the Brute Force Matcher object
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Loop with FAST detection
for i in range(37):
    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D(center, orientation[i], 1.0)
    rotated_image = cv2.warpAffine(cv_cover_G, rotation_matrix, (width, height))

    # Compute features and descriptors for the rotated image
    kpr = fast.detect(rotated_image,None)
    kpr, desr = brief.compute(rotated_image, kpr)

    # Match features
    matches = bf_matcher.knnMatch(des0,desr,k=2) # Two best matches
    # Apply ratio test
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.7 * m2.distance:
            good_matches.append(m1)
    matches = good_matches

    # Also sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Update histogram
    match_count.append(len(matches))

    if i%10 == 0:
        match_image = cv2.drawMatches(cv_cover, kp0, rotated_image, kpr, matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f'../results/2_2_BRIEF_with_Rot_{orientation[i]}_degree.jpg',
                     match_image)


# Display histogram
plt.bar(orientation,match_count, width= 6)
plt.title("Rotation vs Number of Matches using BRIEF & FAST")
plt.xlabel("Rotation (Degree)")
plt.ylabel("Number of Matches between Two Images")
plt.savefig("../results/2_2_Rot_Hist_FAST.jpg")
plt.show()




# Loop with SIFT detection
# Compute features and descriptors for original image

bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# kp0 = sift.detect(cv_cover_G,None)
# kp0, des0 = brief.compute(cv_cover_G, kp0)
kp0, des0 = sift.detectAndCompute(cv_cover_G,None)
match_count = []
for i in range(37):
    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D(center, orientation[i], 1.0)
    rotated_image = cv2.warpAffine(cv_cover_G, rotation_matrix, (width, height))

    # Compute features and descriptors for the rotated image
    # kpr = sift.detect(rotated_image,None)
    # kpr, desr = brief.compute(rotated_image, kpr)
    kpr, desr = sift.detectAndCompute(rotated_image,None)

    # Match features
    matches = bf_matcher.knnMatch(des0,desr,k=2) # Two best matches
    # Apply ratio test
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.7 * m2.distance:
            good_matches.append(m1)
    matches = good_matches
    
    # Also sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Update histogram
    match_count.append(len(matches))


# Display histogram
plt.bar(orientation,match_count, width= 6)
plt.title("Rotation vs Number of Matches using BRIEF & SIFT")
plt.xlabel("Rotation (Degree)")
plt.ylabel("Number of Matches between Two Images")
plt.savefig("../results/2_2_Rot_Hist_SIFT.jpg")
plt.show()
