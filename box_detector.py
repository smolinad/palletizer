import open3d as o3d
import numpy as np
import pyvista as pv
from pathlib import Path
import glob
import os

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

def cluster_boxes(pcd, eps=0.02, min_points=50):
    """Cluster points into individual objects using DBSCAN."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    
    boxes = []
    for i in range(max_label + 1):
        # Extract points for this cluster
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)
        boxes.append(cluster_pcd)
        
    return boxes

def get_top_face_centroid(box_pcd):
    """
    Calculate the centroid of the top face of a box.
    Assumes the box is roughly aligned with gravity (Z-axis).
    """
    # 1. Compute Oriented Bounding Box (OBB)
    obb = box_pcd.get_oriented_bounding_box()
    
    # 2. Get points and rotate them to be axis-aligned with the OBB
    # This makes finding the "top" face easier if the box is rotated
    # However, for simple top-face detection based on Z height in world frame,
    # we might just look at the highest points if the boxes are flat on the floor.
    
    # Let's try a simpler approach first: Filter points with high Z values relative to the box's height.
    points = np.asarray(box_pcd.points)
    if len(points) == 0:
        return None

    z_values = points[:, 2]
    max_z = np.max(z_values)
    min_z = np.min(z_values)
    height = max_z - min_z
    
    # Define "top face" as points within top 10% of height (or a fixed threshold)
    # Adjust this threshold as needed
    top_threshold = max_z - 0.02 # 2cm from top
    
    top_indices = np.where(z_values > top_threshold)[0]
    
    if len(top_indices) == 0:
        # Fallback if box is very thin or something
        return np.mean(points, axis=0)
        
    top_points = points[top_indices]
    centroid = np.mean(top_points, axis=0)
    
    return centroid

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
            plotter.add_point_labels([centroids[i]], [f"Top Center\n{centroids[i]}"], point_size=10, font_size=16, always_visible=True)

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
            
        # 6. Visualize
        # We pass the original clean pcd to show context (floor)
        visualize_results(pcd_clean, boxes, centroids)

if __name__ == "__main__":
    main()
