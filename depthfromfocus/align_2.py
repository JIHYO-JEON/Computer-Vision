import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

path = './PA1_Dataset/05/'
save_path = './PA1_test/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
image = glob.glob(path + '*')

MIN_MATCH_COUNT = 100


def feature_matching(img1, img2, savefig=True):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches2to1 = flann.knnMatch(des2, des1, k=2)

    matchesMask_ratio = [[0, 0] for i in range(len(matches2to1))]
    match_dict = {}
    for i, (m, n) in enumerate(matches2to1):
        if m.distance < 0.7 * n.distance:
            matchesMask_ratio[i] = [1, 0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1, des2, k=2)
    matchesMask_ratio_recip = [[0, 0] for i in range(len(recip_matches))]

    for i, (m, n) in enumerate(recip_matches):
        if m.distance < 0.7 * n.distance:  # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx:  # reciprocal
                good.append(m)
                matchesMask_ratio_recip[i] = [1, 0]

    if savefig:
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask_ratio_recip,
                           flags=0)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, recip_matches, None, **draw_params)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        plt.figure(), plt.xticks([]), plt.yticks([])
        plt.imshow(img3)
        plt.show()

    return [kp1[m.queryIdx].pt for m in good], [kp2[m.trainIdx].pt for m in good]


if __name__ == '__main__':
    # Read reference image
    refFile = image[0]
    print('Reading reference image: ', refFile)
    imReference = cv2.imread(refFile, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFile = image[1]
    print('Reading image to align: ', imFile)
    im = cv2.imread(imFile, cv2.IMREAD_COLOR)

    print('Aligning images ...')
    # Registered image will be resorted in imReg.
    # The estimated homography will be stored in h.
    imReg, h = feature_matching(im, imReference)