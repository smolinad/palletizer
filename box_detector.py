import open3d as o3d
import numpy as np
import pyvista as pv
from pathlib import Path
import glob
import os
import json

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

def get_top_face_centroid(box_pcd):
    """
    Calculate the centroid of the top face using OBB and apply dimension checks.
    """
    if len(box_pcd.points) == 0:
        return None

    # Compute OBB for the cluster
    obb = box_pcd.get_oriented_bounding_box()
    center = obb.center
    R = obb.R # Rotation matrix
    extent = obb.extent # Extent along each axis of the OBB

    # Determine which axis is vertical (aligned with Z-axis)
    # We assume Z-axis is up (0,0,1)
    # Actually, we should probably use the floor normal if we had it passed here,
    # but Z-axis is usually a good enough approximation for the OBB vertical axis 
    # since the OBB is computed on the isolated cluster.
    z_axis = np.array([0, 0, 1])
    
    # Dot product of OBB axes with Z-axis to find the vertical axis
    # The axis with the largest absolute dot product is the vertical one
    dots = np.abs(np.dot(R.T, z_axis)) # R.T gives the OBB's local axes
    vertical_axis_idx = np.argmax(dots)

    # We need to check the sign of the dot product to know if it's + or - direction
    # This determines if the top face is +extent/2 or -extent/2 along the vertical OBB axis
    # Use the original dot product (without abs) to get the sign
    original_dots = np.dot(R.T, z_axis)
    direction_sign = np.sign(original_dots[vertical_axis_idx])
    
    # If dot is 0 (unlikely for argmax > 0), default to 1
    if direction_sign == 0:
        direction_sign = 1
        
    # Filter by planar dimensions (remove small noise clusters)
    planar_indices = [i for i in range(3) if i != vertical_axis_idx]
    dim1 = extent[planar_indices[0]]
    dim2 = extent[planar_indices[1]]
    
    # Minimum dimension threshold (e.g., 8cm)
    if dim1 < 0.08 or dim2 < 0.08:
        print(f"  Rejected candidate with dimensions {dim1:.3f} x {dim2:.3f}")
        return None
        
    # The vector from center to top face center
    # axis_vector is R[:, vertical_axis_idx] (the column of R corresponding to the vertical axis)
    offset_vector = direction_sign * (extent[vertical_axis_idx] / 2.0) * R[:, vertical_axis_idx]
    
    top_face_centroid = center + offset_vector
    
    return top_face_centroid

def visualize_results(original_pcd, boxes, centroids):
    """Visualize the results using PyVista."""
    
    plotter = pv.Plotter()
    
    # Add original point cloud (maybe just the floor or everything faintly)
    # Converting Open3D to PyVista
    original_points = np.asarray(original_pcd.points)
    original_colors = np.asarray(original_pcd.colors)
    
    # Create PyVista point cloud
    cloud = pv.PolyData(original_points)
    if len(original_colors) > 0:
        cloud['RGB'] = original_colors
        plotter.add_mesh(cloud, scalars='RGB', rgb=True, point_size=1, style='points_gaussian', opacity=0.3)
    else:
        plotter.add_mesh(cloud, color='gray', point_size=1, style='points_gaussian', opacity=0.3)

    # Add detected boxes with different colors
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange']
    
    for i, box in enumerate(boxes):
        box_points = np.asarray(box.points)
        box_cloud = pv.PolyData(box_points)
        
        color = colors[i % len(colors)]
        plotter.add_mesh(box_cloud, color=color, point_size=3, render_points_as_spheres=True, label=f"Box {i}")
        
        # Add centroid
        if i < len(centroids) and centroids[i] is not None:
            plotter.add_mesh(pv.Sphere(radius=0.01, center=centroids[i]), color='white', label=f"Centroid {i}")

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
        for box in boxes:
            centroid = get_top_face_centroid(box)
            if centroid is not None:
                valid_boxes.append(box)
                valid_centroids.append(centroid)
                print(f"  Box centroid: {centroid}")
            
        # 6. Save Centroids to JSON
        json_data = []
        for i, centroid in enumerate(valid_centroids):
            json_data.append({
                "box_id": i,
                "centroid": centroid.tolist()
            })
        
        json_path = ply_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Saved centroids to {json_path}")

        # 7. Visualize
        # We pass the original clean pcd to show context (floor)
        visualize_results(pcd_clean, valid_boxes, valid_centroids)

if __name__ == "__main__":
    main()
