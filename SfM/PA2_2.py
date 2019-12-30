import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt
import PA2.utils as structure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# from pyntcloud import PyntCloud


def fileopen(path):
    # read images
    img_path_list = glob.glob(path)
    img_list = []
    for i in img_path_list:
        img_list.append(cv2.imread(i))
    return img_list

def readmatrices(path):
    f = open(path, 'r')
    lines = f.readlines()
    m = []
    for line in lines:
        m.append([float(i) for i in line[:-1].split(' ') if i != ''])
    return m

def matcher(img1, img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
    print(F)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    return pts1, pts2, F

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def epilines(img1, img2):
    pts1, pts2, F = matcher(img1, img2)

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    plt.subplot(122), plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

def reconstruction3d(img1,img2):
    points1, points2, _ = matcher(img1, img2)
    points1 = points1.T
    points2 = points2.T

    points1 = structure.cart2hom(points1)
    points2 = structure.cart2hom(points2)

    height, width, ch = img1.shape

    intrinsic = np.array(readmatrices('./PA2_SfM/data_2_sfm/K.txt'))

    points1n = np.dot(np.linalg.inv(intrinsic), points1)
    points2n = np.dot(np.linalg.inv(intrinsic), points2)

    E = structure.compute_essential_normalized(points1n, points2n)

    print('Computed essential matrix:', (-E / E[0][1]))

    # Given we are at camera 1, calculate the parameters for camera 2
    # Using the essential matrix returns 4 possible camera paramters
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = structure.compute_P_from_essential(E)

    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = structure.reconstruct_one_point(
            points1n[:, 0], points2n[:, 0], P1, P2)

        # Convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)

    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = Axes3D(fig)
    ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()

# image_load ---------------------------------------------------------------------
img_list = fileopen('./PA2_SfM/data_2_sfm/*.JPG')
left_image = img_list[0]
right_image = img_list[1]

# main ----------------------------------------------------------------------------
reconstruction3d(left_image, right_image)