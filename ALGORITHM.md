# Box Detection Algorithm

This document details the mathematical and algorithmic approach used in `box_detector.py` and `src/box_lib.py` to detect boxes from a 3D point cloud.

## 1. Preprocessing

The raw point cloud $P = \{p_1, p_2, \dots, p_N\}$ where $p_i \in \mathbb{R}^3$ is first preprocessed to reduce noise and computational load.

1.  **Voxel Downsampling**: Points are bucketed into voxels of size $v$ (e.g., $0.005m$). For each occupied voxel, the centroid of points inside is kept.
2.  **Statistical Outlier Removal**: For each point $p_i$, the mean distance $\bar{d}_i$ to its $k$ nearest neighbors is computed. Points are removed if:
    $$ \bar{d}_i > \mu_d + \alpha \cdot \sigma_d $$
    where $\mu_d$ and $\sigma_d$ are the mean and standard deviation of neighbor distances across the cloud, and $\alpha$ is a threshold (e.g., 2.0).

## 2. Floor Removal (RANSAC)

We assume the boxes are placed on a planar floor. We use **RANSAC (Random Sample Consensus)** to find the dominant plane.

The plane model is defined as:
$$ ax + by + cz + d = 0 $$
where $\mathbf{n} = [a, b, c]^T$ is the plane normal vector with $\|\mathbf{n}\| = 1$.

Points $p_i$ are classified as inliers (floor) if their perpendicular distance to the plane is within a threshold $\delta_{floor}$:
$$ \frac{|ax_i + by_i + cz_i + d|}{\sqrt{a^2 + b^2 + c^2}} < \delta_{floor} $$

The remaining points $P_{objects} = P \setminus P_{floor}$ are considered potential objects. To prevent merging boxes with the floor, we strictly remove points within a buffer distance $\delta_{buffer}$ from the floor plane.

## 3. Clustering (DBSCAN)

We identify individual objects using **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**.

1.  **Normal Estimation**: For each point $p_i \in P_{objects}$, we estimate the surface normal $\mathbf{n}_i$ using Principal Component Analysis (PCA) on its local neighborhood.
2.  **Top Face Filtering**: We filter for points belonging to the top faces of boxes. Since boxes are horizontal, their top normals should be parallel to the floor normal $\mathbf{n}_{floor}$. We keep points where:
    $$ |\mathbf{n}_i \cdot \mathbf{n}_{floor}| > \cos(\theta_{thresh}) $$
    where $\theta_{thresh}$ is the maximum deviation angle.
3.  **Clustering**: We apply DBSCAN on the filtered top-face points. Two points $p_i, p_j$ are in the same cluster if $\|p_i - p_j\| < \epsilon$ and they are density-reachable.

## 4. Height Splitting (Histogram Analysis)

DBSCAN may merge boxes that are touching side-by-side. However, if they have different heights, we can separate them.

For each cluster $C_k$:
1.  **Alignment**: Rotate points so the $Z$-axis aligns with the floor normal $\mathbf{n}_{floor}$.
2.  **Histogram**: Compute the histogram $H(z)$ of the $z$-coordinates (heights) of points in $C_k$.
3.  **Peak Detection**: Identify local maxima (peaks) in $H(z)$.
4.  **Splitting**: If multiple significant peaks are found, find the "valley" (minimum density) $z_{split}$ between them.
    - Split $C_k$ into sub-clusters $C_{k,1} = \{p \in C_k \mid z < z_{split}\}$ and $C_{k,2} = \{p \in C_k \mid z \ge z_{split}\}$.

## 5. Bounding Box Fitting

For each final cluster, we fit an **Oriented Bounding Box (OBB)**.

1.  **Projection**: Project the cluster points onto the 2D plane parallel to the floor (by dropping the aligned $z$-coordinate).
2.  **2D Minimal Area Rectangle**: Use the **Rotating Calipers** algorithm to find the rectangle of minimum area enclosing the 2D points. This gives the center $(c_x, c_y)$, dimensions $(w, h)$, and rotation angle $\phi$.
3.  **3D Reconstruction**:
    - The 3D centroid is $(c_x, c_y, \bar{z})$, where $\bar{z}$ is the mean height of the cluster.
    - The 3D corners are obtained by rotating the 2D corners back to the original coordinate system using the inverse of the alignment rotation matrix $R^{-1}$.

## 6. Validation

Candidates are rejected if:
-   **Dimensions**: Width or length $< 3cm$.
-   **Rectangularity**: The ratio of the convex hull area to the bounding box area is too low (indicates non-rectangular shapes).
-   **Density**: The number of points per unit area is too low (indicates noise).
