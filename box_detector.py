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

def remove_floor(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """Segment the floor plane using RANSAC."""
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    
    # Inliers are the floor, outliers are the objects
    floor_pcd = pcd.select_by_index(inliers)
    objects_pcd = pcd.select_by_index(inliers, invert=True)
    
    return objects_pcd, floor_pcd

def cluster_boxes(pcd, eps=0.1, min_points=100):
    """
    Cluster points into individual box tops using normals and DBSCAN.
    1. Compute normals.
    2. Filter points with normals aligned with Z-axis.
    3. Cluster the remaining points.
    """
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Orient normals to point towards camera or consistent direction if needed
    # But for now, we just check alignment with Z (0, 0, 1)
    normals = np.asarray(pcd.normals)
    
    # Check alignment with Z axis
    # We want normals that are roughly (0, 0, 1) or (0, 0, -1)
    # But since we removed the floor, and boxes are on the floor, top faces should point UP.
    # Assuming Z is up.
    z_axis = np.array([0, 0, 1])
    
    # Dot product
    dots = np.dot(normals, z_axis)
    
    # Filter: keep points where normal is roughly parallel to Z
    # Threshold: cos(angle). 
    # 0.95 is about 18 degrees.
    top_face_indices = np.where(np.abs(dots) > 0.95)[0]
    
    if len(top_face_indices) == 0:
        print("No top face points found.")
        return []
        
    top_face_pcd = pcd.select_by_index(top_face_indices)
    print(f"Filtered {len(top_face_indices)} points aligned with Z-axis")
    
    # Now cluster these top face points
    # We can use a larger eps to ensure we don't split a single face
    labels = np.array(top_face_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    print(f"Found {max_label + 1} potential box tops")
    
    boxes = []
    for i in range(max_label + 1):
        # Extract points for this cluster
        cluster_indices = np.where(labels == i)[0]
        
        # Filter out small clusters
        if len(cluster_indices) < 200: 
            continue
            
        cluster_pcd = top_face_pcd.select_by_index(cluster_indices)
        boxes.append(cluster_pcd)
        
    return boxes

def get_top_face_centroid(box_pcd):
    """
    Calculate the centroid of the top face.
    Since 'box_pcd' is now already just the top face points, 
    we can simply take the mean.
    """
    points = np.asarray(box_pcd.points)
    if len(points) == 0:
        return None
        
    return np.mean(points, axis=0)

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
        objects_pcd, floor_pcd = remove_floor(pcd_clean)
        
        # 4. Cluster Boxes
        boxes = cluster_boxes(objects_pcd)
        
        # 5. Calculate Centroids
        centroids = []
        for box in boxes:
            centroid = get_top_face_centroid(box)
            centroids.append(centroid)
            print(f"  Box centroid: {centroid}")
            
        # 6. Save Centroids to JSON
        json_data = []
        for i, centroid in enumerate(centroids):
            if centroid is not None:
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
        visualize_results(pcd_clean, boxes, centroids)

if __name__ == "__main__":
    main()
