import open3d as o3d
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class BoxDetectorConfig:
    # Floor Removal
    floor_dist_threshold: float = 0.005
    floor_pre_filter_dist: float = 0.005
    
    # Normal Estimation
    normal_radius: float = 0.1
    normal_max_nn: int = 30
    
    # Clustering (DBSCAN)
    eps: float = 0.015
    min_points: int = 10
    
    # Outlier Removal
    radius_outlier_r: float = 0.015
    radius_outlier_n: int = 10
    
    # Height Splitting
    split_bin_size: float = 0.005
    split_min_prominence: int = 10
    
    # Box Filtering
    min_box_dim: float = 0.03
    min_cluster_points: int = 30

def remove_floor(pcd, config: BoxDetectorConfig):
    """Segment the floor plane using RANSAC and remove points on/below it."""
    plane_model, inliers = pcd.segment_plane(distance_threshold=config.floor_dist_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    
    a, b, c, d = plane_model
    
    # Calculate signed distance for ALL points
    points = np.asarray(pcd.points)
    distances = (points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d) / np.sqrt(a**2 + b**2 + c**2)
    
    # Pre-Cluster Filter: Remove points very close to the floor
    object_indices = np.where(np.abs(distances) > config.floor_pre_filter_dist)[0]
    
    objects_pcd = pcd.select_by_index(object_indices)
    floor_pcd = pcd.select_by_index(object_indices, invert=True)
    
    return objects_pcd, floor_pcd, plane_model

def split_cluster_by_height(cluster_pcd, floor_plane, config: BoxDetectorConfig):
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
    z_min, z_max = np.min(z_values), np.max(z_values)
    if z_max - z_min < 0.02: 
        return [cluster_pcd]
        
    bins = np.arange(z_min, z_max + config.split_bin_size, config.split_bin_size)
    hist, bin_edges = np.histogram(z_values, bins=bins)
    
    # 3. Find Peaks
    hist_smooth = np.convolve(hist, np.ones(3)/3, mode='same')
    
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            if hist_smooth[i] > config.split_min_prominence:
                peaks.append(i)
                
    if len(peaks) < 2:
        return [cluster_pcd]
        
    # 4. Find Split Point
    sub_clusters = []
    peaks.sort()
    split_z_values = []
    
    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i+1]
        valley_idx = p1 + np.argmin(hist_smooth[p1:p2])
        lower_peak_height = min(hist_smooth[p1], hist_smooth[p2])
        valley_height = hist_smooth[valley_idx]
        
        if valley_height < 0.5 * lower_peak_height:
             split_z = bin_edges[valley_idx]
             split_z_values.append(split_z)
             
    if not split_z_values:
        return [cluster_pcd]
        
    thresholds = [-np.inf] + split_z_values + [np.inf]
    
    for i in range(len(thresholds) - 1):
        low = thresholds[i]
        high = thresholds[i+1]
        mask = (z_values >= low) & (z_values < high)
        if np.sum(mask) > 10:
            indices = np.where(mask)[0]
            sub_pcd = cluster_pcd.select_by_index(indices)
            sub_clusters.append(sub_pcd)
            
    return sub_clusters

def cluster_boxes(pcd, floor_plane, config: BoxDetectorConfig):
    """
    Cluster points into individual box tops using normals and DBSCAN.
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=config.normal_radius, max_nn=config.normal_max_nn))
    
    floor_normal = np.array(floor_plane[:3])
    floor_normal = floor_normal / np.linalg.norm(floor_normal)
    
    normals = np.asarray(pcd.normals)
    dots = np.dot(normals, floor_normal)
    top_face_indices = np.where(np.abs(dots) > 0.8)[0]
    
    if len(top_face_indices) == 0:
        return []
        
    top_face_pcd = pcd.select_by_index(top_face_indices)
    
    cl, ind = top_face_pcd.remove_radius_outlier(nb_points=config.radius_outlier_n, 
                                                 radius=config.radius_outlier_r)
    top_face_pcd = top_face_pcd.select_by_index(ind)
    
    if len(top_face_pcd.points) < config.min_cluster_points:
        return []
    
    labels = np.array(top_face_pcd.cluster_dbscan(eps=config.eps, 
                                                  min_points=config.min_points, 
                                                  print_progress=False))
    
    max_label = labels.max()
    
    boxes = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < config.min_cluster_points: 
            continue
        cluster_pcd = top_face_pcd.select_by_index(cluster_indices)
        
        sub_clusters = split_cluster_by_height(cluster_pcd, floor_plane, config)
        boxes.extend(sub_clusters)
        
    return boxes

def get_box_details(box_pcd, floor_plane, config: BoxDetectorConfig):
    """
    Calculate the centroid and 3D bounding rectangle of the top face.
    """
    if len(box_pcd.points) < 4:
        return None, None

    points = np.asarray(box_pcd.points)
    
    # Post-Cluster Filter
    a, b, c, d = floor_plane
    distances = (points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d) / np.sqrt(a**2 + b**2 + c**2)
    min_dist = np.min(np.abs(distances))
    
    # Use a fixed threshold for post-filter or add to config? 
    # box_detector.py uses 0.01. Let's use config if available or hardcode to match.
    # Config has floor_post_filter_dist = 0.01
    if min_dist < 0.01:
        return None, None

    # Align points with Z-axis
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
    
    # Project to 2D
    z_mean = np.mean(points_rot[:, 2])
    points_flat = points_rot.copy()
    points_flat[:, 2] = 0
    
    # Compute 2D OBB
    points_2d = points_flat[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(points_2d)
    ((cx, cy), (w, h), angle) = rect
    
    # Filter by Rectangularity
    rect_area = w * h
    if rect_area > 0:
        hull = cv2.convexHull(points_2d)
        hull_area = cv2.contourArea(hull)
        rectangularity = hull_area / rect_area
        
        if rectangularity < 0.7:
            return None, None
            
        density = len(points_2d) / rect_area
        if density < 2000:
             return None, None
    else:
        return None, None

    # Filter by dimensions
    dims = sorted([w, h])
    dim1, dim2 = dims[0], dims[1]
    
    if dim1 < config.min_box_dim or dim2 < config.min_box_dim:
        return None, None
        
    # Get Corners
    box_corners_2d = cv2.boxPoints(rect)
    corners_2d_with_z = np.hstack([box_corners_2d, np.full((4, 1), z_mean)])
    
    # Rotate back to 3D
    corners_3d = corners_2d_with_z @ R
    
    center_2d_with_z = np.array([cx, cy, z_mean])
    centroid_3d = center_2d_with_z @ R
    
    return centroid_3d, corners_3d
