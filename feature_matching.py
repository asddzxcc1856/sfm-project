import cv2
import numpy as np

def detect_and_match_features(img1, img2, ratio_thresh=0.75):
    """
    Detect SIFT keypoints and descriptors, match them using BFMatcher,
    and apply Lowe's ratio test and RANSAC.
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(f"Detected {len(kp1)} keypoints in image 1")
    print(f"Detected {len(kp2)} keypoints in image 2")

    # Match features using BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print(f"Number of good matches after ratio test: {len(good_matches)}")

    # Extract coordinates of matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Use RANSAC to filter outliers via Fundamental Matrix
    # This is Step 1's requirement "Use RANSAC to filter out outliers"
    # Note: We will compute F properly in Step 2, here we just use it for filtering.
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,ransacReprojThreshold=1.0,confidence=0.99)
    
    # We select only inlier points
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    
    print(f"Number of inliers after RANSAC: {len(pts1_inliers)}")

    return pts1_inliers, pts2_inliers, kp1, kp2, good_matches, mask

def draw_matches(img1, kp1, img2, kp2, matches, mask=None, filename="matches.png"):
    """
    Visualize matches.
    """
    if mask is not None:
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=mask.ravel().tolist(),
                           flags=cv2.DrawMatchesFlags_DEFAULT)
    else:
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           flags=cv2.DrawMatchesFlags_DEFAULT)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imwrite(filename, img_matches)
    print(f"Matches visualization saved to {filename}")
