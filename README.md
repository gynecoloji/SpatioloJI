# SpatioloJI
## core foundation for Ji Universe

![Ji Universe Logo](./spatioloji-logo.svg)

SpatioloJI is a comprehensive Python library for analyzing spatial transcriptomics data. It provides a robust framework for managing, visualizing, and performing advanced spatial statistics on multi-FOV (Field of View) spatial transcriptomics datasets.

## Overview

SpatioloJI offers an extensive suite of tools and functionalities specifically designed to address the challenges in spatial transcriptomics data analysis:

- **Data Management**: Organize cell polygons, gene expression, metadata, and images across multiple FOVs
- **Quality Control**: Comprehensive QC pipeline for filtering cells and genes
- **Spatial Visualization**: Advanced visualization tools for displaying cells, gene expression, and spatial relationships
- **Spatial Statistics**: Methods for detecting spatial patterns, correlations, and organization of cells and gene expression
- **Network Analysis**: Tools for building and analyzing cell interaction networks

## Main Components

The library consists of three main components:

1. **Quality Control** (`Spatioloji_qc Class`): Tools for quality control and data preprocessing
2. **Spatioloji Class** (`Spatial_Object.py`): Core data structure for managing filtered spatial transcriptomics data
3. **Spatial Analysis Functions** (`Spatial_function.py`): Collection of statistical methods for spatial analysis
4. **Spatial Visualization Functions** (`Plot_Spatial_Image.py`): Functions for visualizing spatial relationships


## Installation

```bash
conda create -n SpatioloJI python=3.12 -y
pip install SpatioloJI
```

## Key Features

### 0. Quality Control

Comprehensive QC pipeline for preprocessing data:

```python
from spatioloji import Spatioloji_qc

# Initialize QC object
spatioloji_qc = Spatioloji_qc(
    expr_matrix=expr_matrix,
    cell_metadata=cell_metadata,
    output_dir="./qc_output/"
)

# Run complete QC pipeline
filtered_cells, filtered_genes = spatioloji_qc.run_qc_pipeline()
```

### 1. Data Management

The `Spatioloji` class provides a unified data structure for spatial transcriptomics data:

```python
from spatioloji import Spatioloji

# Create from existing data
spatioloji_obj = Spatioloji(
    polygons=polygons_df,
    cell_meta=cell_meta_df,
    adata=anndata_obj,
    fov_positions=fov_positions_df,
    images=fov_images
)

# Or load from files
spatioloji_obj = Spatioloji.from_files(
    polygons_path="polygons.csv",
    cell_meta_path="cell_meta.csv",
    adata_path="adata.h5ad",
    fov_positions_path="fov_positions.csv",
    images_folder="images/"
)
```

### 2. Spatial Visualization

SpatioloJI provides multiple functions for visualizing spatial data:

```python
from spatioloji import stitch_fov_images, plot_global_polygon_by_features

# Stitch multiple FOV images into a single view
stitched = stitch_fov_images(
    spatioloji_obj,
    fov_ids=None,  # Use all FOVs
    flip_vertical=True,
    save_path="stitched_image.png"
)

# Visualize features across cell polygons
plot_global_polygon_by_features(
    spatioloji_obj,
    feature="gene_expression",
    background_img=True,
    colormap="viridis"
)

# Or visualize categorical data
plot_global_polygon_by_categorical(
    spatioloji_obj,
    feature="cell_type",
    background_img=True
)
```

### 3. Spatial Statistics

SpatioloJI includes a wide range of spatial statistics methods:

```python
from spatioloji import (
    calculate_nearest_neighbor_distances,
    calculate_ripleys_k,
    perform_neighbor_analysis,
    calculate_hotspot_analysis
)

# Calculate nearest neighbor distances
nn_distances = calculate_nearest_neighbor_distances(
    spatioloji_obj,
    use_global_coords=True
)

# Ripley's K function for spatial point pattern analysis
ripley_k = calculate_ripleys_k(
    spatioloji_obj,
    max_distance=100,
    num_distances=20
)

# Perform comprehensive neighbor analysis
neighbor_results = perform_neighbor_analysis(
    polygon_file=spatioloji_obj.polygons,
    cell_metadata=spatioloji_obj.cell_meta,
    cell_type_column="cell_type"
)

# Identify statistically significant hot spots based on a feature
hotspots = calculate_hotspot_analysis(
    spatioloji_obj,
    attribute_name="gene_expression",
    distance_threshold=50
)
```

### 4. Network Analysis

Build and analyze cell interaction networks:

```python
from spatioloji import calculate_network_statistics

# Create and analyze a cell interaction network
network_results = calculate_network_statistics(
    spatioloji_obj,
    distance_threshold=50,
    cell_type_column="cell_type",
    community_detection=True
)
```


## Example Workflows

### 1. Basic Workflow

```python
# Load data
spatioloji_obj = Spatioloji.from_files(...)

# Run QC
qc = Spatioloji_qc(...)
filtered_cells, filtered_genes = qc.run_qc_pipeline()

# Visualize the data
stitched = stitch_fov_images(spatioloji_obj)
plot_global_polygon_by_features(spatioloji_obj, feature="CD3")

# Perform spatial analysis
ripley_k = calculate_ripleys_k(spatioloji_obj, max_distance=100)
neighbor_results = perform_neighbor_analysis(spatioloji_obj.polygons, spatioloji_obj.cell_meta, "cell_type")
```

### 2. Advanced Spatial Analysis

```python
# Analyze spatial autocorrelation of gene expression
gene_autocorr = calculate_gene_spatial_autocorrelation(
    spatioloji_obj,
    genes=["CD3", "CD8", "PD1"],
    method="moran"
)

# Calculate spatial context profiles
context_profiles = calculate_spatial_context(
    spatioloji_obj,
    distance_threshold=50,
    cell_type_column="cell_type"
)

# Calculate spatial heterogeneity index
heterogeneity = calculate_spatial_heterogeneity(
    spatioloji_obj,
    attribute_name="gene_expression",
    method="quadrat"
)
```

## Analysis Categories

SpatioloJI provides functions for spatial ststs in the following categories:

1. **Neighbor Analysis**
   - perform_neighbor_analysis: Comprehensive analysis of neighboring cells based on polygon geometries
   - calculate_nearest_neighbor_distances: Calculates distances to nearest neighbors for each cell
   - calculate_cell_density: Measures local cell density within a specified radius

2. **Spatial Pattern Analysis**
   - calculate_ripleys_k: Analyzes spatial point patterns using Ripley's K function
   - calculate_cross_k_function: Examines spatial relationships between different cell types
   - calculate_j_function: Uses Baddeley's J-function for spatial pattern analysis
   - calculate_g_function: Analyzes nearest neighbor distance distributions
   - calculate_pair_correlation_function: Measures correlations between cells at different distances

3. **Cell Type Interaction Analysis**
   - calculate_cell_type_correlation: Measures how different cell types correlate in space
   - calculate_colocation_quotient: Quantifies spatial relationships between cell types
   - calculate_proximity_analysis: Measures distances between specific cell types

4. **Heterogeneity and Clustering**
   - calculate_morisita_index: Measures the spatial distribution pattern (clustered vs. uniform)
   - calculate_quadrat_variance: Analyzes how variance changes with grid size
   - calculate_spatial_entropy: Quantifies randomness in spatial distribution
   - calculate_hotspot_analysis: Identifies statistically significant spatial hot/cold spots
   - calculate_spatial_autocorrelation: Measures Moran's I and related statistics
   - calculate_kernel_density: Creates density maps of cell distributions
   - calculate_spatial_heterogeneity: Quantifies and characterizes spatial variation

5. **Network-Based Analysis**
   - calculate_network_statistics: Creates and analyzes cell interaction networks
   - calculate_spatial_context: Analyzes cell neighborhoods and their composition

6. **Gene Expression Spatial Analysis**
   - calculate_gene_spatial_autocorrelation: Examines spatial patterns of gene expression
   - calculate_mark_correlation: Analyzes spatial correlation of cell attributes

## Contributing

Contributions to SpatioloJI are welcome! Please feel free to submit a pull request or open an issue to discuss your ideas.

## License

SpatioloJI is released under the [MIT License](LICENSE).

## Citation

If you use SpatioloJI in your research, please cite:

```
Citation information coming soon
```

## Acknowledgments

SpatioloJI builds upon several established algorithms and methods for spatial analysis, and we thank the community for their contributions to this field.
