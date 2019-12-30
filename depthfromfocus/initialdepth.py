import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt
#import imutils
def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts




# task 2: Initial depth from focus measure
# find the distance from camera to object

def find_marker(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_size = (5, 5)
    img_gray_gaussian = cv2.GaussianBlur(img_gray, blur_size, 0)
    edged_img = cv2.Canny(img_gray_gaussian, 35, 125) # 2 thresholds

    Contours = cv2.findContours(edged_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Contours = grab_contours(Contours)
    con_max = max(Contours, key=cv2.contourArea)

    return cv2.minAreaRect(con_max)

def distance_to_camera(k_width, f_length, p_width):
    return (k_width*f_length)/p_width

# read images
img_path = './PA1_Dataset/05/*.JPG'
# 05, 07 -> jpg
# mobo7 -> png
img_list = glob.glob(img_path)

#print(img_list)
# img size = 1920(w) * 1280(h)