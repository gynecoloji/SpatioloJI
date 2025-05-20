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

## Tutorials
Please check [SpatioloJI Documentation](https://spatioloji.readthedocs.io/en/latest/installation.html) for detailed instructions.

## Spatial Stats Analysis Categories

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
