import open3d as o3d
import numpy as np
import pyvista as pv
from pathlib import Path
import json
from src.box_lib import BoxDetectorConfig, remove_floor, cluster_boxes, get_box_details

def load_and_preprocess(path, voxel_size=0.005):
    """Load PLY and preprocess (downsample + statistical outlier removal)."""
    print(f"Loading {path}...")
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        return None
        
    pcd_down = pcd.voxel_down_sample(voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd_down.select_by_index(ind)

def visualize_results(original_pcd, boxes, centroids, corners_list):
    """Visualize results using PyVista."""
    plotter = pv.Plotter()
    
    # 1. Original Point Cloud
    points = np.asarray(original_pcd.points)
    colors = np.asarray(original_pcd.colors)
    cloud = pv.PolyData(points)
    if len(colors) > 0:
        cloud['RGB'] = colors
        plotter.add_mesh(cloud, scalars='RGB', rgb=True, point_size=1, opacity=0.3)
    else:
        plotter.add_mesh(cloud, color='gray', point_size=1, opacity=0.3)

    # 2. Detected Boxes
    box_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange']
    
    for i, (box, centroid, corners) in enumerate(zip(boxes, centroids, corners_list)):
        # Cluster points
        plotter.add_mesh(pv.PolyData(np.asarray(box.points)), 
                        color=box_colors[i % len(box_colors)], 
                        point_size=3, render_points_as_spheres=True)
        
        # Centroid
        plotter.add_mesh(pv.Sphere(radius=0.01, center=centroid), color='white')
        
        # Wireframe (Top Face)
        # corners is (4, 3). Create a loop 0-1-2-3-0
        lines = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]]).flatten()
        rect_poly = pv.PolyData(corners)
        rect_poly.lines = lines
        plotter.add_mesh(rect_poly, color='white', line_width=3, style='wireframe')

    plotter.add_axes()
    plotter.show(title=f"Detected {len(boxes)} Boxes")

def main():
    data_dir = Path("data")
    ply_files = list(data_dir.glob("**/point_cloud_PLY_*.ply"))
    
    if not ply_files:
        print("No PLY files found in data directory.")
        return

    config = BoxDetectorConfig() # Use default config

    for ply_path in ply_files:
        print(f"\nProcessing {ply_path.name}...")
        
        # 1. Load & Preprocess
        pcd = load_and_preprocess(ply_path)
        if pcd is None: continue
            
        # 2. Detect Boxes (using library)
        objects_pcd, floor_pcd, floor_plane = remove_floor(pcd, config)
        clusters = cluster_boxes(objects_pcd, floor_plane, config)
        
        # 3. Extract Details
        valid_results = []
        for cluster in clusters:
            centroid, corners = get_box_details(cluster, floor_plane, config)
            if centroid is not None:
                valid_results.append((cluster, centroid, corners))
        
        print(f"  Found {len(valid_results)} valid boxes.")
        
        # 4. Save & Visualize
        if valid_results:
            boxes, centroids, corners = zip(*valid_results)
            
            # Save JSON
            json_data = [{"box_id": i, "centroid": c.tolist(), "corners": cr.tolist()} 
                         for i, (c, cr) in enumerate(zip(centroids, corners))]
            with open(ply_path.with_suffix('.json'), 'w') as f:
                json.dump(json_data, f, indent=4)
                
            # Visualize
            visualize_results(pcd, boxes, centroids, corners)

if __name__ == "__main__":
    main()
