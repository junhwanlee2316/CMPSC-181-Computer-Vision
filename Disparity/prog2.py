# Author: Junhwan Lee
# CSC181: Programming Assignment 2
# Date: 02/26/2023
# Resource: https://vision.middlebury.edu/stereo/data/

import cv2
import numpy as np
import math
import sys


# sys.argv
left_image_name = sys.argv[1]
right_image_name =  sys.argv[2]
scale_factor = sys.argv[3]
output_disparity_name = sys.argv[4]
scale_factor = int(scale_factor)
a = cv2.imread(left_image_name)
a1 = cv2.imread(right_image_name)

# convert to gray scale
a  = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
a1  = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)

# convert to numpy arrays
left_i = np.array(a, dtype= float)
right_i = np.array(a1, dtype = float)

# Window Size = n x n
W = 13

# center pixel in the window
c = math.ceil(W/2)

# the size of the image (row (height), column (width), color (channel))
h,w = left_i.shape

# number of pixels serch
ser = 64

# create an empty array for output
out = np.zeros((h,w))

# begin SAD
# iterate through height
for i in range(0,h):
    x = i
    x_end = i+c-1
    
    # iterate through width
    for j in range(0,w):
        y = j
        y_end = j+c-1 
        
        # window on left image
        px_c = left_i[x:x_end,y:y_end]
        
        # parameters
        minsub = None
        pos = 0
        first = True
        
        # iterate through the right image search times
        for j1 in range(j,min([j+ser,w-c])):
            y2 = max(j1,j1-c + 1)
            
            # window on the right image
            px_c1 = right_i[x:x_end,y2:j1+c-1]
            
            # SAD
            disparity = np.abs(px_c-px_c1)
            SAD = np.sum(disparity)
            
            # update minimum SAD and displacement
            if first:
                minsub = SAD
                pos = j1-j
                first = False
            else:
                if(SAD < minsub):
                    minsub = SAD
                    pos = j1 - j
        
        # plot disparity divided by the scale_factor
        out[i,j] = pos/scale_factor 


# final output image
out1 = out/ser
out1 = out1 * 250

# write the image
cv2.imwrite(output_disparity_name + ".jpg",out1)

# end of program