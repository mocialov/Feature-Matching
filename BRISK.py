import cv2
import numpy as np
from matplotlib import pyplot as plt

def BRISK(img1, img2, n=100):
    # Initiate BRISK descriptor
    BRISK = cv2.BRISK_create()

    # Find the keypoints and compute the descriptors for input and training-set image
    keypoints1, descriptors1 = BRISK.detectAndCompute(img1, None)
    keypoints2, descriptors2 = BRISK.detectAndCompute(img2, None)

    # create BFMatcher object
    BFMatcher = cv2.BFMatcher(normType = cv2.NORM_HAMMING,
                            crossCheck = True)

    # Matching descriptor vectors using Brute Force Matcher
    matches = BFMatcher.match(queryDescriptors = descriptors1,
                            trainDescriptors = descriptors2)

    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x: x.distance)

    return keypoints1, keypoints2, matches[:min(n, len(matches))]
