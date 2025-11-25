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
    """Segment the floor plane using RANSAC and remove points on/below it."""
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    
    # Plane equation: ax + by + cz + d = 0
    a, b, c, d = plane_model
    print(f"Plane model: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    
    # Calculate signed distance for ALL points
    points = np.asarray(pcd.points)
    distances = (points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d) / np.sqrt(a**2 + b**2 + c**2)
    
    # STAGE 1: Pre-Cluster Filter
    # Remove points very close to the floor to break merges.
    # Threshold: 0.005m (5mm). This is strict enough to separate boxes from floor.
    floor_threshold = 0.005 
    
    # Indices of points that are NOT floor (distance > threshold)
    # We use abs() to remove the slab.
    object_indices = np.where(np.abs(distances) > floor_threshold)[0]
    
    objects_pcd = pcd.select_by_index(object_indices)
    floor_pcd = pcd.select_by_index(object_indices, invert=True)
    
    return objects_pcd, floor_pcd, plane_model

def split_cluster_by_height(cluster_pcd, floor_plane, bin_size=0.005, min_peak_prominence=10):
    """
    Analyze the height distribution of a cluster and split it if multiple levels are detected.
    """
    points = np.asarray(cluster_pcd.points)
    if len(points) < 50:
        return [cluster_pcd]

    # 1. Align points with Z-axis based on floor normal
    floor_normal = np.array(floor_plane[:3])
    floor_normal = floor_normal / np.linalg.norm(floor_normal)
    
    z_axis = np.array([0, 0, 1])
    
    a_vec = floor_normal
    b_vec = z_axis
    v = np.cross(a_vec, b_vec)
    c_val = np.dot(a_vec, b_vec)
    
    if np.linalg.norm(v) < 1e-6:
        if c_val > 0:
            R = np.eye(3)
        else:
            R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c_val) / (s ** 2))
        
    points_rot = points @ R.T
    z_values = points_rot[:, 2]
    
    # 2. Compute Histogram
    # Use a fixed range based on min/max Z
    z_min, z_max = np.min(z_values), np.max(z_values)
    if z_max - z_min < 0.02: # If height range is very small (<2cm), don't split
        return [cluster_pcd]
        
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    hist, bin_edges = np.histogram(z_values, bins=bins)
    
    # 3. Find Peaks (Simple local maxima)
    # We look for bins that are significantly higher than neighbors
    # A peak must be separated by a valley.
    
    # Smooth histogram slightly
    hist_smooth = np.convolve(hist, np.ones(3)/3, mode='same')
    
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            if hist_smooth[i] > min_peak_prominence:
                peaks.append(i)
                
    if len(peaks) < 2:
        return [cluster_pcd]
        
    print(f"  Detected {len(peaks)} height levels in cluster. Splitting...")
    
    # 4. Find Split Point (Valley between peaks)
    # For simplicity, if 2 peaks, split at the lowest point between them.
    # If >2 peaks, we might need multiple splits. Let's handle the 2-peak case primarily.
    # Or just iterate through valleys.
    
    sub_clusters = []
    
    # Sort peaks by index
    peaks.sort()
    
    # Define split thresholds (Z values)
    split_z_values = []
    
    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i+1]
        
        # Find minimum between p1 and p2 in the SMOOTHED histogram
        valley_idx = p1 + np.argmin(hist_smooth[p1:p2])
        
        # Check if valley is deep enough (e.g. < 50% of lower peak)
        lower_peak_height = min(hist_smooth[p1], hist_smooth[p2])
        valley_height = hist_smooth[valley_idx]
        
        if valley_height < 0.5 * lower_peak_height:
             split_z = bin_edges[valley_idx]
             split_z_values.append(split_z)
             
    if not split_z_values:
        return [cluster_pcd]
        
    # Perform splitting
    # Add -inf and +inf to thresholds
    thresholds = [-np.inf] + split_z_values + [np.inf]
    
    for i in range(len(thresholds) - 1):
        low = thresholds[i]
        high = thresholds[i+1]
        
        mask = (z_values >= low) & (z_values < high)
        if np.sum(mask) > 10:
            indices = np.where(mask)[0]
            sub_pcd = cluster_pcd.select_by_index(indices)
            sub_clusters.append(sub_pcd)
            
    print(f"  Split into {len(sub_clusters)} sub-clusters.")
    return sub_clusters

def cluster_boxes(pcd, floor_plane, eps=0.015, min_points=10):
    """
    Cluster points into individual box tops using normals and DBSCAN.
    """
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Floor normal from plane model [a, b, c, d]
    floor_normal = np.array(floor_plane[:3])
    floor_normal = floor_normal / np.linalg.norm(floor_normal)
    
    print(f"Floor normal: {floor_normal}")
    
    normals = np.asarray(pcd.normals)
    
    # Check alignment with Floor Normal
    dots = np.dot(normals, floor_normal)
    
    # Filter: keep points where normal is roughly parallel to floor normal
    top_face_indices = np.where(np.abs(dots) > 0.8)[0]
    
    if len(top_face_indices) == 0:
        print("No top face points found.")
        return []
        
    top_face_pcd = pcd.select_by_index(top_face_indices)
    print(f"Filtered {len(top_face_indices)} points aligned with floor normal")
    
    # Clean up
    # Tightened to match clustering scale: radius=0.015 (1.5cm), nb_points=10
    cl, ind = top_face_pcd.remove_radius_outlier(nb_points=10, radius=0.015)
    top_face_pcd = top_face_pcd.select_by_index(ind)
    print(f"Points after radius outlier removal: {len(top_face_pcd.points)}")
    
    if len(top_face_pcd.points) == 0:
        return []
    
    # Cluster
    labels = np.array(top_face_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    print(f"Found {max_label + 1} potential box tops (before height split)")
    
    boxes = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < 50: 
            continue
        cluster_pcd = top_face_pcd.select_by_index(cluster_indices)
        
        # Apply Height Splitting
        sub_clusters = split_cluster_by_height(cluster_pcd, floor_plane)
        boxes.extend(sub_clusters)
        
    return boxes

def get_box_details(box_pcd, floor_plane):
    """
    Calculate the centroid and 3D bounding rectangle of the top face.
    """
    if len(box_pcd.points) < 4:
        return None, None

    points = np.asarray(box_pcd.points)
    
    # STAGE 2: Post-Cluster Filter
    # Check if ANY point in this cluster is too close to the floor.
    # If so, it means this cluster includes floor points (merged) or IS floor noise.
    # Threshold: 0.01m (1cm).
    a, b, c, d = floor_plane
    distances = (points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d) / np.sqrt(a**2 + b**2 + c**2)
    
    min_dist = np.min(np.abs(distances))
    if min_dist < 0.01:
        print(f"  Rejected candidate with min point height {min_dist:.3f}m (contains floor points)")
        return None, None

    # 1. Align points with Z-axis based on floor normal
    floor_normal = np.array(floor_plane[:3])
    floor_normal = floor_normal / np.linalg.norm(floor_normal)
    
    z_axis = np.array([0, 0, 1])
    
    a_vec = floor_normal
    b_vec = z_axis
    v = np.cross(a_vec, b_vec)
    c_val = np.dot(a_vec, b_vec)
    
    if np.linalg.norm(v) < 1e-6:
        if c_val > 0:
            R = np.eye(3)
        else:
            R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c_val) / (s ** 2))
        
    points_rot = points @ R.T
    
    # 2. Project to 2D
    z_mean = np.mean(points_rot[:, 2])
    points_flat = points_rot.copy()
    points_flat[:, 2] = 0
    
    # 3. Compute 2D OBB
    points_2d = points_flat[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(points_2d)
    ((cx, cy), (w, h), angle) = rect
    
    # 4. Filter by Rectangularity
    rect_area = w * h
    if rect_area > 0:
        hull = cv2.convexHull(points_2d)
        hull_area = cv2.contourArea(hull)
        rectangularity = hull_area / rect_area
        
        if rectangularity < 0.7:
            print(f"  Rejected candidate with rectangularity {rectangularity:.3f} (irregular shape)")
            return None, None
            
        density = len(points_2d) / rect_area
        if density < 2000:
             print(f"  Rejected candidate with density {density:.2f} (too sparse)")
             return None, None
    else:
        return None, None

    # 5. Filter by dimensions
    dims = sorted([w, h])
    dim1, dim2 = dims[0], dims[1]
    
    if dim1 < 0.03 or dim2 < 0.03:
        print(f"  Rejected candidate with dimensions {dim1:.3f} x {dim2:.3f}")
        return None, None
        
    # 6. Get Corners
    box_corners_2d = cv2.boxPoints(rect)
    corners_2d_with_z = np.hstack([box_corners_2d, np.full((4, 1), z_mean)])
    
    # 7. Rotate back to 3D
    corners_3d = corners_2d_with_z @ R
    
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
