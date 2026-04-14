import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 10
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data2/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = img[0].shape
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
# Vr = np.array(rvecs)
# Tr = np.array(tvecs)
# extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""
Write your code here
"""
# =========================
# Step 1: Compute Homography for each image
# =========================
def compute_homography(objp, imgp):
    A = []
    for i in range(len(objp)):
        X, Y = objp[i][0], objp[i][1]
        u, v = imgp[i][0][0], imgp[i][0][1]

        # u = (h11 X + h12 Y + h13) / (h31 X + h32 Y + h33)
        # v = (h21 X + h22 Y + h23) / (h31 X + h32 Y + h33)
        # ->
        # -Xh11 -Yh12 -h13 + uXh31 + uYh32 + u h33 = 0
        # -Xh21 -Yh22 -h23 + vXh31 + vYh32 + v h33 = 0
        A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
        # 每一個點會產生 2 條 equation

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3) # 取最後一個向量(最小 singular value 對應的向量)

    return H / H[-1,-1] # normalization


# =========================
# Step 2: Compute intrinsic matrix K (Zhang method)
# =========================
def compute_intrinsics(H_list): # p.82
    def v_ij(H, i, j):
        return np.array([
            H[0,i]*H[0,j],
            H[0,i]*H[1,j] + H[1,i]*H[0,j],
            H[1,i]*H[1,j],
            H[2,i]*H[0,j] + H[0,i]*H[2,j],
            H[2,i]*H[1,j] + H[1,i]*H[2,j],
            H[2,i]*H[2,j]
        ])

    V = []
    for H in H_list:
        V.append(v_ij(H,0,1))
        V.append(v_ij(H,0,0) - v_ij(H,1,1))

    V = np.array(V)
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1]

    # construct B matrix
    B11, B12, B22, B13, B23, B33 = b

    cy = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lam = B33 - (B13**2 + cy*(B12*B13 - B11*B23)) / B11
    fx = np.sqrt(lam / B11)
    fy = np.sqrt(lam*B11 / (B11*B22 - B12**2))
    skew = -B12 * fx**2 * fy / lam
    cx = skew*cy/fy - B13*fx**2/lam

    K = np.array([
        [fx, skew, cx],
        [0,  fy,   cy],
        [0,   0,    1]
    ])
    print(K)
    return K


# =========================
# Step 3: Compute extrinsics
# =========================
def compute_extrinsics(K, H_list):
    extrinsics = []

    K_inv = np.linalg.inv(K)

    for H in H_list:
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]

        lam = 1 / np.linalg.norm(np.dot(K_inv, h1))

        r1 = lam * np.dot(K_inv, h1)
        r2 = lam * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)
        t  = lam * np.dot(K_inv, h3)

        R = np.stack((r1, r2, r3), axis=1)

        # enforce orthogonality (SVD correction)
        U, _, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)

        # convert R to Rodrigues vector
        rvec, _ = cv2.Rodrigues(R)

        extrinsics.append(np.hstack((rvec.flatten(), t)))

    return np.array(extrinsics)


# =========================
# MAIN calibration pipeline
# =========================
H_list = []
for i in range(len(objpoints)):
    H = compute_homography(objpoints[i], imgpoints[i])
    H_list.append(H)

# intrinsic
mtx = compute_intrinsics(H_list)

# extrinsic (rvec + tvec → same format as original code)
extrinsics = compute_extrinsics(mtx, H_list)
"""
End of your code
"""
# show the camera extrinsics
print('Show the camera extrinsics')

# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d') # matplotlib version issue
ax = fig.add_subplot(111, projection='3d')

# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
