# Palletizer Box Detection

This project implements a 3D box detection pipeline using Point Cloud data (PLY files). It includes a standalone script and an interactive Marimo app for parameter tuning.

## 1. Environment Setup

This project uses `uv` for dependency management.

### Initialize Environment
If you haven't already set up the environment:

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
#or 
pip install uv

# 2. Clone the repo 
git clone --depth 1 https://github.com/smolinad/palletizer.git

# 3. Change directory to repo
cd palletizer

# 3. Initialize project (if starting fresh)
uv init
```

## 2. Running the Code

### Simplified Box Detector
Run the simplified script to process all PLY files in the `data` directory:

```bash
uv run box_detector.py
```
This will:
1. Load each PLY file.
2. Detect boxes using the configured algorithm.
3. Save centroids to a `.json` file (e.g., `data/file.json`).
4. Open a 3D visualization window (PyVista) for each file. Close the window to proceed to the next file.

### Interactive Parameter Tuning (Marimo App)
To interactively tune parameters (like clustering radius, floor threshold) and visualize results in real-time:

```bash
uv run marimo edit app.py
```
This will open a web browser where you can:
1. Select a PLY file from the dropdown.
2. Adjust sliders for `DBSCAN eps`, `Min Points`, etc.
3. View the 3D result (Matplotlib) instantly.
   - **Note**: The 3D plot is interactive (zoom/pan enabled).

## 3. Project Structure

- `box_detector.py`: Main script (simplified) for batch processing.
- `app.py`: Marimo notebook for interactive tuning.
- `src/box_lib.py`: Core algorithms (floor removal, clustering, box fitting).
- `data/`: Directory containing `.ply` files.
