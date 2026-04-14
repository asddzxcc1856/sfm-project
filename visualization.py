import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_epipolar_lines(img1, img2, pts1, pts2, F, filename="epipolar_lines.png"):
    """
    Step 3: Epipolar Geometry Visualization.
    Draws points in img1 and corresponding epipolar lines in img2.
    """
    # Compute epipolar lines in img2 for points in img1
    # l' = F * x
    pts1_hom = np.column_stack((pts1, np.ones(len(pts1))))
    lines2 = (F @ pts1_hom.T).T
    
    # Create copies to draw on
    img1_out = img1.copy()
    img2_out = img2.copy()
    
    r, c, _ = img2.shape
    
    for r1, r2, l2 in zip(pts1, pts2, lines2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw point in img1
        x1, y1 = map(int, r1)
        cv2.circle(img1_out, (x1, y1), 5, color, -1)
        
        # Draw line and point in img2
        x0, y0 = map(int, [0, -l2[2]/l2[1]])
        x1, y1 = map(int, [c, -(l2[2] + l2[0]*c)/l2[1]])
        cv2.line(img2_out, (x0, y0), (x1, y1), color, 2)
        
        x2, y2 = map(int, r2)
        cv2.circle(img2_out, (x2, y2), 5, color, -1)
        
    # Combine images side by side
    combined = np.hstack((img1_out, img2_out))
    cv2.imwrite(filename, combined)
    print(f"Epipolar lines visualization saved to {filename}")

def visualize_3d(points, title="3D Point Cloud"):
    """
    Step 8: 3D Visualization using Matplotlib.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([points[:,0].max()-points[:,0].min(), 
                          points[:,1].max()-points[:,1].min(), 
                          points[:,2].max()-points[:,2].min()]).max() / 2.0
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig("output/pointcloud_plot.png")
    print("3D point cloud plot saved to output/pointcloud_plot.png")
    # plt.show() # Can't show in this environment easily

def export_to_obj(points, filename="output/model.obj"):
    """
    Step 8: Export 3D point cloud to .obj file.
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            
    print(f"3D points exported to {filename}")
    
    # Bonus: Export as .ply
    ply_filename = filename.replace(".obj", ".ply")
    with open(ply_filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    print(f"3D points exported to {ply_filename}")
