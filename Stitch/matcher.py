import cv2
import numpy as np

# Author: Junhwan Lee
# CSC181: Programming Assignment 1
# Date: 02/05/2023
# Resource: https://www.youtube.com/@firstprinciplesofcomputerv3258
#           https://kushalvyas.github.io/stitching.html
#           https://www.youtube.com/watch?v=ToldvnUtBh0
#           https://www.youtube.com/watch?v=uMABRY8QPe0


# matcher
class matcher:
    
    # create SIFT
    def __init__(self):
        self.sift = cv2.SIFT_create()
    
     # get SIFT keypoints and features
    def getSIFTFeatures(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return (kp,des)
    
    # begin matching keypoints
    def match(self, i1, i2, direction=None):
        
        # get keypoints and features
        (kp1, des1) = self.getSIFTFeatures(i1)
        (kp2, des2) = self.getSIFTFeatures(i2)
        
        # brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, False)
        matches = bf.knnMatch(des2, des1, k=2)
        
        # good match points
        good = []
        for i , (m, n) in enumerate(matches):
            if m.distance < 0.5*n.distance:
                good.append((m.trainIdx, m.queryIdx))

        # if enough matching points
        MIN_INDEX= 4
        if len(good) > MIN_INDEX:

            train_pts = np.float32([kp2[i].pt for (__, i) in good])
            query_pts = np.float32([kp1[i].pt for (i, __) in good])

            H, s = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 4)
            return H
        
        # if not return none
        return None
