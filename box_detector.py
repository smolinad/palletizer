import open3d as o3d
import numpy as np
import pyvista as pv
from pathlib import Path
import glob
import os
import json
import cv2

def load_point_cloud(path):
    """Load a point cloud from a PLY file."""
    print(f"Loading {path}...")
    pcd = o3d.io.read_point_cloud(path)
    return pcd

def preprocess_point_cloud(pcd, voxel_size=0.005):
    """Downsample and remove outliers."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean = pcd_down.select_by_index(ind)
    return pcd_clean

def remove_floor(pcd, distance_threshold=0.005, ransac_n=3, num_iterations=1000):
    """Segment the floor plane using RANSAC."""
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    
    # Inliers are the floor, outliers are the objects
    floor_pcd = pcd.select_by_index(inliers)
    objects_pcd = pcd.select_by_index(inliers, invert=True)
    
    return objects_pcd, floor_pcd, plane_model

def cluster_boxes(pcd, floor_plane, eps=0.08, min_points=50):
    """
    Cluster points into individual box tops using normals and DBSCAN.
    1. Compute normals.
    2. Filter points with normals aligned with the floor normal.
    3. Cluster the remaining points.
    """
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Floor normal from plane model [a, b, c, d]
    floor_normal = np.array(floor_plane[:3])
    # Normalize just in case
    floor_normal = floor_normal / np.linalg.norm(floor_normal)
    
    print(f"Floor normal: {floor_normal}")
    
    normals = np.asarray(pcd.normals)
    
    # Check alignment with Floor Normal
    # Top faces should be parallel to the floor, so their normals should be parallel to floor normal.
    # We take absolute value of dot product to handle normal flipping.
    dots = np.dot(normals, floor_normal)
    
    # Filter: keep points where normal is roughly parallel to floor normal
    # Threshold: 0.9 is about 25 degrees.
    top_face_indices = np.where(np.abs(dots) > 0.9)[0]
    
    if len(top_face_indices) == 0:
        print("No top face points found.")
        return []
        
    top_face_pcd = pcd.select_by_index(top_face_indices)
    print(f"Filtered {len(top_face_indices)} points aligned with floor normal")
    
    # Now cluster these top face points
    labels = np.array(top_face_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    print(f"Found {max_label + 1} potential box tops")
    
    boxes = []
    for i in range(max_label + 1):
        # Extract points for this cluster
        cluster_indices = np.where(labels == i)[0]
        
        # Filter out small clusters
        if len(cluster_indices) < 50: 
            continue
            
        cluster_pcd = top_face_pcd.select_by_index(cluster_indices)
        boxes.append(cluster_pcd)
        
    return boxes

def get_box_details(box_pcd, floor_plane):
    """
    Calculate the centroid and 3D bounding rectangle of the top face.
    The rectangle is constrained to be parallel to the floor plane.
    """
    if len(box_pcd.points) < 4:
        return None, None

    points = np.asarray(box_pcd.points)
    
    # 1. Align points with Z-axis based on floor normal
    floor_normal = np.array(floor_plane[:3])
    floor_normal = floor_normal / np.linalg.norm(floor_normal)
    
    # Create rotation matrix to align floor_normal with Z=[0,0,1]
    z_axis = np.array([0, 0, 1])
    
    # Rotation calculation
    # v = cross(a, b)
    # s = ||v||
    # c = dot(a, b)
    # R = I + [v]x + [v]x^2 * (1-c)/s^2
    
    a = floor_normal
    b = z_axis
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    if np.linalg.norm(v) < 1e-6:
        # Already aligned or opposite
        if c > 0:
            R = np.eye(3)
        else:
            # 180 degree rotation around X
            R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
        
    # Rotate points
    points_rot = points @ R.T
    
    # 2. Project to 2D (flatten Z)
    z_mean = np.mean(points_rot[:, 2])
    points_flat = points_rot.copy()
    points_flat[:, 2] = 0
    
    # 3. Compute 2D OBB using OpenCV minAreaRect (Rotating Calipers)
    # This guarantees the minimum area rectangle, unlike PCA-based OBB
    points_2d = points_flat[:, :2].astype(np.float32)
    
    # cv2.minAreaRect returns ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(points_2d)
    ((cx, cy), (w, h), angle) = rect
    
    # 4. Filter by dimensions
    # Sort dimensions to be consistent
    dims = sorted([w, h])
    dim1, dim2 = dims[0], dims[1]
    
    if dim1 < 0.08 or dim2 < 0.08:
        print(f"  Rejected candidate with dimensions {dim1:.3f} x {dim2:.3f}")
        return None, None
        
    # 5. Get Corners
    box_corners_2d = cv2.boxPoints(rect)
    # box_corners_2d is (4, 2)
    
    # Add Z coordinate (z_mean)
    corners_2d_with_z = np.hstack([box_corners_2d, np.full((4, 1), z_mean)])
    
    # 6. Rotate back to 3D
    # points_rot = points @ R.T  => points = points_rot @ R
    corners_3d = corners_2d_with_z @ R
    
    # Centroid is the center from minAreaRect transformed back
    center_2d_with_z = np.array([cx, cy, z_mean])
    centroid_3d = center_2d_with_z @ R
    
    return centroid_3d, corners_3d

def visualize_results(original_pcd, boxes, centroids, corners_list):
    """Visualize the results using PyVista."""
    
    plotter = pv.Plotter()
    
    # Add original point cloud
    original_points = np.asarray(original_pcd.points)
    original_colors = np.asarray(original_pcd.colors)
    
    cloud = pv.PolyData(original_points)
    if len(original_colors) > 0:
        cloud['RGB'] = original_colors
        plotter.add_mesh(cloud, scalars='RGB', rgb=True, point_size=1, style='points_gaussian', opacity=0.3)
    else:
        plotter.add_mesh(cloud, color='gray', point_size=1, style='points_gaussian', opacity=0.3)

    # Add detected boxes
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange']
    
    for i, box in enumerate(boxes):
        box_points = np.asarray(box.points)
        box_cloud = pv.PolyData(box_points)
        
        color = colors[i % len(colors)]
        plotter.add_mesh(box_cloud, color=color, point_size=3, render_points_as_spheres=True, label=f"Box {i}")
        
        # Add centroid
        if i < len(centroids) and centroids[i] is not None:
            plotter.add_mesh(pv.Sphere(radius=0.01, center=centroids[i]), color='white')
            
        # Add wireframe rectangle
        if i < len(corners_list) and corners_list[i] is not None:
            corners = corners_list[i]
            # Create lines: 0-1, 1-2, 2-3, 3-0
            # PyVista lines format: [n_points, p1, p2, ..., n_points, p3, p4, ...]
            # But we can use pv.Line or just construct a mesh
            
            # Order is 0,1,2,3 around the center from our construction
            # But let's verify order. Our construction: (+,+), (-,+), (-,-), (+,-).
            # This is circular order (0->1->2->3->0).
            
            lines = np.array([
                [2, 0, 1],
                [2, 1, 2],
                [2, 2, 3],
                [2, 3, 0]
            ]).flatten()
            
            # Need to adjust indices to be absolute? No, relative to the points we pass
            # Wait, PyVista PolyData constructor takes (points, lines)
            # lines array: [2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0]
            
            rect_poly = pv.PolyData(corners)
            rect_poly.lines = lines
            
            plotter.add_mesh(rect_poly, color='white', line_width=3, style='wireframe')

    if len(boxes) > 0:
        plotter.add_legend()
    plotter.add_axes()
    plotter.show()

def main():
    # Find all PLY files in subdirectories of 'data'
    data_dir = Path("data")
    ply_files = list(data_dir.glob("**/point_cloud_PLY_*.ply"))
    
    if not ply_files:
        print("No PLY files found in data directory.")
        return

    print(f"Found {len(ply_files)} PLY files.")

    for ply_path in ply_files:
        print(f"\nProcessing {ply_path}...")
        
        # 1. Load
        pcd = load_point_cloud(str(ply_path))
        if pcd.is_empty():
            print("Point cloud is empty.")
            continue
            
        # 2. Preprocess
        pcd_clean = preprocess_point_cloud(pcd)
        
        # 3. Remove Floor
        objects_pcd, floor_pcd, floor_plane = remove_floor(pcd_clean)
        print(f"Objects point cloud has {len(objects_pcd.points)} points")
        
        # 4. Cluster Boxes
        boxes = cluster_boxes(objects_pcd, floor_plane)
        
        # 5. Calculate Centroids and Filter
        valid_boxes = []
        valid_centroids = []
        valid_corners = []
        
        for box in boxes:
            centroid, corners = get_box_details(box, floor_plane)
            if centroid is not None:
                valid_boxes.append(box)
                valid_centroids.append(centroid)
                valid_corners.append(corners)
                print(f"  Box centroid: {centroid}")
            
        # 6. Save Centroids to JSON
        json_data = []
        for i, (centroid, corners) in enumerate(zip(valid_centroids, valid_corners)):
            json_data.append({
                "box_id": i,
                "centroid": centroid.tolist(),
                "corners": corners.tolist()
            })
        
        json_path = ply_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Saved centroids to {json_path}")

        # 7. Visualize
        # We pass the original clean pcd to show context (floor)
        visualize_results(pcd_clean, valid_boxes, valid_centroids, valid_corners)

if __name__ == "__main__":
    main()
