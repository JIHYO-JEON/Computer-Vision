import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt


# Task 4: Graph-cuts and weighted median filter
def graphcut(img):
    img = cv2.imread(img)
    mask = np.zeros(img.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    #RoI
    rectangle = (0,0,1280,1920) # left upper x, left upper y, height, width
    cv2.grabCut(img, mask, rectangle, background, foreground, 3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()

# read images
img_path = './PA1_Dataset/05/*.JPG'
# 05, 07 -> jpg
# mobo7 -> png
img_list = glob.glob(img_path)
print(img_list)
#print(img_list)
# img size = 1920(w) * 1280(h)

graphcut(img_list[0])