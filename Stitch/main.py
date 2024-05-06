import cv2
import numpy as np
import glob
from matcher import matcher
import sys

# Author: Junhwan Lee
# CSC181: Programming Assignment 1
# Date: 02/05/2023
# Resource: https://www.youtube.com/@firstprinciplesofcomputerv3258
#           https://kushalvyas.github.io/stitching.html
#           https://www.youtube.com/watch?v=ToldvnUtBh0
#           https://www.youtube.com/watch?v=uMABRY8QPe0


# trim black using recursion
def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

# scale given image with given scale percentage
def scale_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # return ressized scale
    result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return result

# match pixels
def match_pixels(img1, img2):
    # img1 regular photo, img2 is warped
    # set of pixels in img1 and img2
    y1, x1 = img1.shape[:2]
    y2, x2 = img2.shape[:2]

    # loop through all pixels 
    for i in range(0, x1):
        for j in range(0, y1):
            try:
                # if pixels match
                if(np.array_equal(img1[j,i],np.array([0,0,0])) and  np.array_equal(img2[j,i],np.array([0,0,0]))):
                    # take average
                    img2[j,i] = [0, 0, 0]
                else:
                    # if pixels don't match, but pixel exists
                    if(np.array_equal(img2[j,i],[0,0,0])):
                        img2[j,i] = img1[j,i]   # bring pixel from original image and set it to the warp image
                    else:
                        if not np.array_equal(img1[j,i], [0,0,0]):
                            img2[j,i] = img1[j,i]   # bring pixel from original image and set it to the warp image
            except:
                pass
    # return warped image with corresponding pixels from original image
    return img2

path = sys.argv[1]
filename = sys.argv[2]

# Declare image path
image_paths = glob.glob(path + "/*.JPG")

# Declare set of images
images=[]

# Initialize images
for image in image_paths:
    img = cv2.imread(image)
    img = cv2.resize(img, (480,300))
    img = scale_img(img,50)
    images.append(img)

# Find the center image for reference
center =round(len(images)/2)

# Declare two lists: right and left
left_list = []
right_list = []

# sort images => right side: right , left side: left
for i in range(len(images)-1):
    if(i < center):
        left_list.append(images[i])
    else:
        right_list.append(images[i])

# initiate matcher object
i_matcher = matcher()

##############################################################################################
# start left stitch from first image to the center image
##############################################################################################
a= left_list[0]
for b in left_list[1:len(images)-1]:
    
    # Homography Matrix
    H = i_matcher.match(a,b, 'left')
    
    # inverse homography matrix for left stitch
    xh = np.linalg.inv(H)
    
    # staring pixel
    ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
    ds = ds/ds[-1]
    
    # transforming the matrix 
    f1 = np.dot(xh, np.array([0,0,1]))
    f1 = f1/f1[-1]
    xh[0][-1] += abs(f1[0])
    xh[1][-1] += abs(f1[1])
    ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
    y_off = abs(int(f1[1]))
    x_off = abs(int(f1[0]))
    
    # warped image x,y dimensions
    width = int(ds[0]) + x_off
    height = int(ds[1]) + y_off
    
    # warp image
    tmp = cv2.warpPerspective(a, xh, (width,height))
    
    # insert the warped image using the offset in x,y coordinate
    tmp[y_off:b.shape[0]+y_off, x_off:b.shape[1]+x_off] = b
    
    # new reference image 
    # trim to remove the black screen and for more precise matching
    a = trim(tmp)

# left stitched image
left_img= trim(tmp)

##############################################################################################
# start right stitch from right images to the created left image
##############################################################################################

# begin to stitch images to the right of center image
for each in right_list:
    
    # Homography Matrix
    H = i_matcher.match(left_img, each, 'right')
    
    # transforming the matrix
    txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
    txyz = txyz/txyz[-1]
    
    # warped image x,y dimensions
    dsize = (int(txyz[0])+left_img.shape[1], int(txyz[1])+left_img.shape[0])
    
    # warp image
    tmp = cv2.warpPerspective(each, H, dsize)
    
    # bring pixels
    tmp = match_pixels(left_img, tmp)
    
    # update reference image
    left_img= trim(tmp)


cv2.imwrite(filename + ".jpg",left_img)

# end of program