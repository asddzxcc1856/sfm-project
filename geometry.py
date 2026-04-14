import numpy as np
import cv2

def normalize_points(pts):
    """
    Normalize 2D points such that the centroid is at the origin
    and the average distance from the origin is sqrt(2).
    """
    centroid = np.mean(pts, axis=0)
    shifted_pts = pts - centroid
    avg_dist = np.mean(np.sqrt(np.sum(shifted_pts**2, axis=1)))
    scale = np.sqrt(2) / avg_dist
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1))))
    # normalized_pts_hom = (T @ pts_hom.T).T
    # return normalized_pts_hom[:, :2], T
    
    pts_hom = np.column_stack((pts, np.ones(len(pts))))
    normalized_pts = (T @ pts_hom.T).T
    return normalized_pts[:, :2], T

def estimate_fundamental_matrix(pts1, pts2):
    """
    Step 2: Fundamental Matrix Estimation (Normalized 8-point algorithm)
    """
    # 1. Normalize coordinates
    n_pts1, T1 = normalize_points(pts1)
    n_pts2, T2 = normalize_points(pts2)
    
    # 2. Construct matrix A
    num_pts = n_pts1.shape[0]
    A = np.zeros((num_pts, 9))
    for i in range(num_pts):
        x1, y1 = n_pts1[i]
        x2, y2 = n_pts2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        
    # 3. Solve F using SVD
    _, _, Vh = np.linalg.svd(A)
    F = Vh[-1].reshape(3, 3)
    
    # 4. Enforce rank-2 constraint
    U, S, Vh = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vh
    
    # 5. Denormalize F
    F = T2.T @ F @ T1
    
    # Final scaling to make F[2,2] = 1
    F = F / F[2, 2]
    
    return F

def compute_essential_matrix(F, K):
    """
    Step 4: Compute Essential Matrix
    E = K^T * F * K
    """
    E = K.T @ F @ K
    
    # Enforce rank-2 constraint (singular values should be [s, s, 0])
    U, S, Vh = np.linalg.svd(E)
    m = (S[0] + S[1]) / 2.0
    E = U @ np.diag([m, m, 0]) @ Vh
    
    return E

def decompose_essential_matrix(E):
    """
    Step 5: Decompose Essential Matrix into R and t.
    Returns 4 possible solutions.
    """
    U, _, Vh = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ Vh
    R2 = U @ W.T @ Vh
    t = U[:, 2]
    
    # Ensure rotation matrices have det(R) = 1
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        
    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    return solutions

def triangulate_points(P1, P2, pts1, pts2):
    """
    Step 7: Triangulation
    Triangulate points using the Linear Least Squares method.
    """
    num_pts = pts1.shape[0]
    pts3d = np.zeros((num_pts, 3))
    
    for i in range(num_pts):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        
        A = np.zeros((4, 4))
        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]
        
        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        X = X / X[3]
        pts3d[i] = X[:3]
        
    return pts3d

def select_correct_pose(solutions, pts1, pts2, K):
    """
    Step 6: Select Correct Pose (Cheirality Check)
    """
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    best_count = -1
    best_pose = None
    best_pts3d = None
    
    for R, t in solutions:
        t = t.reshape(3, 1)
        P2 = K @ np.hstack((R, t))
        
        pts3d = triangulate_points(P1, P2, pts1, pts2)
        
        # Check cheirality: point must be in front of both cameras
        # Point in front of camera 1: Z > 0
        # Point in front of camera 2: R_3 * (X - t_C) > 0 or simpler: transform point to cam2 coord
        
        count = 0
        transformed_pts3d = (R @ pts3d.T + t).T
        
        for i in range(len(pts3d)):
            if pts3d[i, 2] > 0 and transformed_pts3d[i, 2] > 0:
                count += 1
        
        if count > best_count:
            best_count = count
            best_pose = (R, t)
            best_pts3d = pts3d
            
    print(f"Best solution has {best_count}/{len(pts1)} points in front of cameras.")
    return best_pose[0], best_pose[1], best_pts3d
