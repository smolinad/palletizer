# Box Detection and Centroid Calculation

This project implements a robust algorithm to detect cardboard boxes from 3D point cloud data, calculate their centroids, and estimate their 3D bounding rectangles.

## Algorithm Description

Let the input point cloud be a set of points $P = \{p_1, p_2, \dots, p_N\} \subset \mathbb{R}^3$. The goal is to identify a set of boxes $B = \{b_1, b_2, \dots, b_M\}$, where each $b_i$ is defined by its centroid $c_i$ and its 3D corners.

### 1. Floor Removal (Two-Stage Filtering)
We assume the floor is the dominant plane in the scene. We model the floor as a plane $\Pi_{\mathrm{floor}}$:
$$ n_{\mathrm{floor}} \cdot x + d_{\mathrm{floor}} = 0 $$
where $n_{\mathrm{floor}}$ is the floor normal and $d_{\mathrm{floor}}$ is the distance from the origin.

**Stage 1: Pre-Cluster Filtering**
Using RANSAC, we find the plane parameters. To strictly separate objects from the floor and break "bridges" caused by noise, we remove all points within a strict threshold $\tau_{\mathrm{floor\_pre}} = 0.005$m (5mm) of the plane:
$$ P_{\mathrm{obj}} = \{p \in P \mid |n_{\mathrm{floor}} \cdot p + d_{\mathrm{floor}}| > \tau_{\mathrm{floor\_pre}}\} $$

### 2. Top Face Detection (Normal Filtering)
For each point $p \in P_{\mathrm{obj}}$, we estimate its surface normal $n_p$. Since boxes are resting on the floor, their top faces should be parallel to the floor plane. We filter points based on the alignment of their normal with the floor normal:
$$ P_{\mathrm{top}} = \{p \in P_{\mathrm{obj}} \mid |n_p \cdot n_{\mathrm{floor}}| > \tau_{\mathrm{angle}}\} $$
where $\tau_{\mathrm{angle}} \approx 0.9$ (corresponding to $\approx 25^\circ$ deviation).

### 3. Pre-Clustering Cleanup (Radius Outlier Removal)
To prevent noise from bridging separate clusters, we apply a Radius Outlier Removal filter to $P_{\mathrm{top}}$:
$$ P_{\mathrm{clean}} = \{p \in P_{\mathrm{top}} \mid |N_r(p)| \ge k_{\mathrm{radius}}\} $$
where $N_r(p)$ is the set of neighbors within radius $r$ ($r=0.02$m, $k_{\mathrm{radius}}=15$).

### 4. Clustering (DBSCAN)
We partition $P_{\mathrm{clean}}$ into clusters $\{C_1, C_2, \dots, C_M\}$ using DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
$$ C_i = \text{DBSCAN}(P_{\mathrm{clean}}, \epsilon=0.05, \text{min\_points}=30) $$

### 5. Cluster Refinement
For each cluster $C_i$, we apply further refinement:

#### a. Statistical Outlier Removal
We remove points that are statistical outliers based on the distribution of neighbor distances:
$$ C_i' = \{p \in C_i \mid \bar{d}_k(p) < \mu_k + \alpha \sigma_k\} $$
where $\bar{d}_k(p)$ is the mean distance to $k$ nearest neighbors, and $\alpha=2.0$.

#### b. 3D Bounding Rectangle (Rotating Calipers)
We project the points $C_i'$ onto the plane parallel to the floor. Let $R$ be the rotation matrix that aligns $n_{\mathrm{floor}}$ with the Z-axis. The projected points are:
$$ P'_{\mathrm{2D}} = \{ (Rp)_x, (Rp)_y \mid p \in C_i' \} $$
We compute the Minimum Area Rectangle using the Rotating Calipers algorithm:
$$ \text{Rect}_i = \text{minAreaRect}(P'_{\mathrm{2D}}) $$
This gives us the center, dimensions $(w, h)$, and rotation angle in the 2D plane.

### 6. Filtering Candidates
We reject candidate clusters based on several geometric criteria:

*   **Stage 2: Post-Cluster Floor Check**: To ensure no floor noise is mistaken for a box, we reject any cluster that contains points too close to the floor:
    $$ \min_{p \in C_i'} |n_{\mathrm{floor}} \cdot p + d_{\mathrm{floor}}| > \tau_{\mathrm{floor\_post}} $$
    where $\tau_{\mathrm{floor\_post}} = 0.01$m (1cm).
*   **Dimensions**: $w > 0.03$m and $h > 0.03$m.
*   **Rectangularity**: To reject irregular shapes (e.g., merged outliers), we compare the area of the Convex Hull to the Minimum Area Rectangle:
    $$ \text{Rectangularity} = \frac{\text{Area}(\text{ConvexHull}(P'_{\mathrm{2D}}))}{\text{Area}(\text{Rect}_i)} $$
    We require $\text{Rectangularity} > 0.7$.
*   **Density**: To reject sparse ghost clusters:
    $$ \text{Density} = \frac{|C_i'|}{\text{Area}(\text{Rect}_i)} > 2000 \text{ pts}/m^2 $$
*   **Height**: The centroid distance to the floor plane must be sufficient:
    $$ \text{dist}(c_i, \Pi_{\mathrm{floor}}) > 0.01 \text{m} $$

### 7. Output
For valid clusters, the 2D rectangle is transformed back to 3D space using $R^{-1}$ to obtain the final 3D corners and centroid.
