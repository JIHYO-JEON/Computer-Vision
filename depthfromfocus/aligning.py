import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt

# align the images
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    #detect = cv2.xfeatures2d.SURF_create(MAX_FEATURES)
    detect = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = detect.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = detect.detectAndCompute(im2Gray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Draw top matches
    # 힘내라 지효야...........................................................................
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    plt.imshow(cv2.cvtColor(imMatches, cv2.COLOR_BGR2RGB)), plt.show()
    cv2.imwrite("matches.jpg", imMatches)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

def img_alignment(img_list):
    output_image, _ = alignImages(cv2.imread(img_list[0], cv2.IMREAD_COLOR), cv2.imread(img_list[1], cv2.IMREAD_COLOR))
    for i in range(2, len(img_list)):
        output_image, _ = alignImages(output_image, cv2.imread(img_list[i]))


# read images
img_path = './PA1_Dataset/05/*.JPG'
# 05, 07 -> jpg
# mobo7 -> png
img_list = glob.glob(img_path)

# print(img_list)
# img size = 1920(w) * 1280(h)

new_img = img_alignment(img_list)
plt.imshow(new_img), plt.show()
