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

def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])
	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

def main(img1, img2):
    K = np.array(readmatrices('./PA2_SfM/data_2_sfm/K.txt'))
    # focal length
    f = K[0][0]
    # distance between cameras
    d = K[0][2] - K[1][2]
    h, w = img2.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))

    img_1_undistorted = cv2.undistort(img1, K, d, None, new_camera_matrix)
    img_2_undistorted = cv2.undistort(img2, K, d, None, new_camera_matrix)

    img_1_downsampled = downsample_image(img_1_undistorted, 3)
    img_2_downsampled = downsample_image(img_2_undistorted, 3)

    win_size = 5
    min_disp = -1
    max_disp = 63  # min_disp * 9
    num_disp = max_disp - min_disp  # Needs to be divisible by 16

    # Create Block matching object.
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=5,
                                   uniquenessRatio=5,
                                   speckleWindowSize=5,
                                   speckleRange=5,
                                   disp12MaxDiff=2,
                                   P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                   P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)

    # Compute disparity map
    print("\nComputing the disparity  map...")
    disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)
    plt.imshow(disparity_map, 'gray')
    plt.show()

    # Generate  point cloud.
    # h, w = img_2_downsampled.shape[:2]
    print("\nGenerating the 3D map...")
    Q = np.float32([[1, 0, 0, -w / 2.0],
                    [0, -1, 0, h / 2.0],
                    [0, 0, 0, -f],
                    [0, 0, 1, 0]])
    Q2 = np.float32([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, f * 0.05, 0],  # Focal length multiplication obtained experimentally.
                     [0, 0, 0, 1]])
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
    colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)
    mask_map = disparity_map > disparity_map.min()

    # Mask colors and points.
    output_points = points_3D[mask_map]
    print(output_points)
    output_colors = colors[mask_map]
    print(output_colors)

    # Generate point cloud
    print("\n Creating the output file... \n")
    output_file = 'reconstructed_2_q2.ply'
    print(type(output_points))
    create_output(output_points, output_colors, output_file)

# image_load ---------------------------------------------------------------------
img_list = fileopen('./PA2_SfM/data_2_sfm/*.JPG')
left_image = img_list[0]
right_image = img_list[1]

# main ----------------------------------------------------------------------------
main(left_image, right_image)