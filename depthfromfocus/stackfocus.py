import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt

# task 3: All in focus image

def Laplacian(img):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_size = (5, 5)
    blurred_img = cv2.GaussianBlur(img, blur_size, 0)
    lap_blurred_img = cv2.Laplacian(blurred_img, cv2.CV_64F, ksize=kernel_size)
    return lap_blurred_img

def focus_stack(img_list):
    lap_list = []
    for img in img_list:
        lap_img = Laplacian(img)
        lap_list.append(lap_img)
    laps = np.asarray(lap_list)

    output = np.zeros(shape = cv2.imread(img_list[0]).shape, dtype=cv2.imread(img_list[0]).dtype)
    abs_laps = np.absolute(laps)
    max = abs_laps.max(axis=0)
    check = abs_laps == max
    final_laps = check.astype(np.uint8)

    for i in range(0, len(img_list)):
        output = cv2.bitwise_not(cv2.imread(img_list[i]), output, mask=final_laps[i])

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)), plt.show()
    plt.imshow(cv2.cvtColor(255-output, cv2.COLOR_BGR2RGB)), plt.show()
    return 255-output


# read images
img_path = './PA1_Dataset/05/*.JPG'
# 05, 07 -> jpg
# mobo2 -> png
img_list = glob.glob(img_path)
"""
for j in img_list:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)
"""
print(img_list)
focus_stack(img_list)