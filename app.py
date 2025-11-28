import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Box Detector Parameter Tuning
    Adjust parameters to refine clustering.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import open3d as o3d
    import numpy as np
    import matplotlib.pyplot as plt
    import tempfile
    import os
    from pathlib import Path
    from src.box_lib import BoxDetectorConfig, remove_floor, cluster_boxes, get_box_details
    return (
        BoxDetectorConfig,
        Path,
        cluster_boxes,
        get_box_details,
        mo,
        np,
        o3d,
        os,
        plt,
        remove_floor,
        tempfile,
    )


@app.cell
def _(mo):
    file_upload = mo.ui.file(filetypes=[".ply"], label="Upload PLY File")
    return (file_upload,)


@app.cell
def _(mo):
    eps_slider = mo.ui.slider(
        start=0.005, stop=0.05, step=0.001, value=0.015, label="DBSCAN eps (m)"
    )
    min_points_slider = mo.ui.slider(
        start=5, stop=50, step=1, value=10, label="Min Points"
    )
    floor_dist_slider = mo.ui.slider(
        start=0.001,
        stop=0.02,
        step=0.001,
        value=0.005,
        label="Floor Pre-Filter (m)",
    )
    radius_r_slider = mo.ui.slider(
        start=0.005, stop=0.05, step=0.001, value=0.015, label="Radius Outlier R"
    )
    return eps_slider, floor_dist_slider, min_points_slider, radius_r_slider


@app.cell
def _(
    eps_slider,
    file_upload,
    floor_dist_slider,
    min_points_slider,
    mo,
    radius_r_slider,
):
    mo.vstack(
        [
            file_upload,
            mo.hstack([eps_slider, min_points_slider]),
            mo.hstack([floor_dist_slider, radius_r_slider]),
        ]
    )
    return


@app.cell
def _(file_upload, o3d, tempfile, os):
    # Load PLY (Cached by Marimo reactivity)
    if not file_upload.value:
        pcd = None
    else:
        # Get the first uploaded file
        uploaded_file = file_upload.value[0]
        
        # Write to temp file because Open3D needs a path
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp.write(uploaded_file.contents)
            tmp_path = tmp.name
            
        try:
            pcd = o3d.io.read_point_cloud(tmp_path)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    return (pcd,)


@app.cell(hide_code=True)
def _(
    BoxDetectorConfig,
    cluster_boxes,
    eps_slider,
    floor_dist_slider,
    get_box_details,
    min_points_slider,
    pcd,
    radius_r_slider,
    remove_floor,
):
    # Process
    if pcd is None:
        clusters = []
        floor_plane = None
        objects_pcd = None
        valid_boxes = []
    else:
        config = BoxDetectorConfig(
            eps=eps_slider.value,
            min_points=min_points_slider.value,
            floor_pre_filter_dist=floor_dist_slider.value,
            radius_outlier_r=radius_r_slider.value,
            radius_outlier_n=10,
        )

        objects_pcd, floor_pcd, floor_plane = remove_floor(pcd, config)
        clusters = cluster_boxes(objects_pcd, floor_plane, config)
        
        valid_boxes = []
        for _cluster in clusters:
            _centroid, _corners = get_box_details(_cluster, floor_plane, config)
            if _centroid is not None:
                valid_boxes.append({"centroid": _centroid, "corners": _corners, "cluster": _cluster})
                
    return clusters, config, floor_plane, floor_pcd, objects_pcd, valid_boxes


@app.cell(hide_code=True)
def _(clusters, mo, np, plt, valid_boxes):
    # Visualize
    if not clusters:
        viz = mo.md("No clusters found or no file selected.")
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Clusters
        for i, box_data in enumerate(valid_boxes):
            _cluster = box_data["cluster"]
            pts = np.asarray(_cluster.points)
            if len(pts) > 500:
                indices = np.random.choice(len(pts), 500, replace=False)
                pts = pts[indices]
            
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.5)
            
            # Plot Centroid
            c = box_data["centroid"]
            ax.scatter(c[0], c[1], c[2], c='r', s=50, marker='x')
            
            # Plot Wireframe
            _corners = box_data["corners"]
            # Order: 0-1, 1-2, 2-3, 3-0 (top face)
            # But corners_3d has 4 points.
            # Let's draw the loop.
            # Append first point to close loop
            loop = np.vstack([_corners, _corners[0]])
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], c='k', linewidth=2)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Detected Boxes: {len(valid_boxes)}")
        
        # Equal Aspect Ratio Hack
        if valid_boxes:
            all_pts = np.concatenate([np.asarray(b["cluster"].points) for b in valid_boxes])
            max_range = np.array([all_pts[:, 0].max()-all_pts[:, 0].min(), 
                                  all_pts[:, 1].max()-all_pts[:, 1].min(), 
                                  all_pts[:, 2].max()-all_pts[:, 2].min()]).max() / 2.0
            mid_x = (all_pts[:, 0].max()+all_pts[:, 0].min()) * 0.5
            mid_y = (all_pts[:, 1].max()+all_pts[:, 1].min()) * 0.5
            mid_z = (all_pts[:, 2].max()+all_pts[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        viz = mo.mpl.interactive(fig)
            
    viz
    return


if __name__ == "__main__":
    app.run()
