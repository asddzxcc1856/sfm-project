import cv2
import numpy as np
import os
from feature_matching import detect_and_match_features, draw_matches
from geometry import (estimate_fundamental_matrix, compute_essential_matrix, 
                      decompose_essential_matrix, select_correct_pose)
from visualization import draw_epipolar_lines, visualize_3d, export_to_obj

def main():
    # 0. Configuration
    img1_path = "data/IMG_0694.jpg"
    img2_path = "data/IMG_0695.jpg"

    # Provided K (normalized version)
    K_orig = np.array([
        [1.4219, 0.0005, 0.5092],
        [0.0,    1.4219, 0.3802],
        [0.0,    0.0,    0.0010]
    ])
    K_orig = np.array([
        [5426.566895, 0.678017, 387.430023],
        [0.000000, 5423.133301, 620.616699],
        [0.000000, 0.000000, 1.000000]
    ])
    K_orig = np.array([
        [3116.54043, -8.83247006, 1526.58090],
        [0.0, 3105.58852, 1976.07831],
        [0.0, 0.0, 1.0]
    ])
    
    # Standardize K (make K[2,2] = 1)
    K = K_orig / K_orig[2, 2]
    print("Camera Intrinsic Matrix K:\n", K)

    # 1. Load Images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("Error: Could not load images. Check paths.")
        return

    # Step 1: Feature Detection & Matching
    pts1, pts2, kp1, kp2, matches, mask = detect_and_match_features(img1, img2)
    draw_matches(img1, kp1, img2, kp2, matches, mask, "matches_ransac.png")

    # Step 2: Fundamental Matrix Estimation
    F_custom = estimate_fundamental_matrix(pts1, pts2)
    print("\nCustom Fundamental Matrix F:\n", F_custom)
    
    # OpenCV baseline for comparison
    F_cv, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    print("\nOpenCV Fundamental Matrix F (8-point):\n", F_cv)

    # Step 3: Epipolar Geometry Visualization
    draw_epipolar_lines(img1, img2, pts1, pts2, F_custom, "epipolar_lines.png")

    # Step 4: Compute Essential Matrix
    E = compute_essential_matrix(F_custom, K)
    print("\nEssential Matrix E:\n", E)

    # Step 5: Decompose Essential Matrix
    solutions = decompose_essential_matrix(E)

    # Step 6 & 7: Select Correct Pose & Triangulation
    R, t, pts3d = select_correct_pose(solutions, pts1, pts2, K)
    print("\nCorrect Camera Pose:")
    print("Rotation R:\n", R)
    print("Translation t:\n", t)

    # Step 8: 3D Visualization & Export
    os.makedirs("output", exist_ok=True)
    visualize_3d(pts3d, title="Reconstructed 3D Points")
    export_to_obj(pts3d, "output/model.obj")

    print("\nSfM Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
