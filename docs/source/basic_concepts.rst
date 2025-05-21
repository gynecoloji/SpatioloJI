
Introduction to Spatioloji Object: Advanced Spatial Transcriptomics Framework
=======================================================================

.. image:: _static/spatioloji-object-animation.svg
   :width: 300px
   :align: center

Spatioloji is a sophisticated Python framework specifically designed for managing and analyzing spatial transcriptomics data across multiple fields of view (FOVs). It provides a comprehensive object-oriented solution that elegantly handles the complex relationships between spatial coordinates, cellular morphologies, and gene expression data.

Key Features and Advantages
-------------------------------

1. Unified Data Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spatioloji excels at integrating diverse data types within a cohesive object structure:

* **Cell morphology**: Stores and manages cell polygon boundaries via the ``polygons`` attribute
* **Gene expression**: Incorporates AnnData objects through the ``adata`` attribute
* **Cell metadata**: Maintains annotations and classifications in the ``cell_meta`` attribute
* **Spatial context**: Preserves field of view positions via ``fov_positions``
* **Visual data**: Directly links microscopy images through the ``images`` dictionary

2. Dual Coordinate System Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of Spatioloji's most powerful features is its simultaneous support for both local and global coordinate systems:

* **Local coordinates** (``gdf_local``): For analyzing relationships within individual fields of view
* **Global coordinates** (``gdf_global``): For examining patterns across the entire tissue sample

This dual approach eliminates the need to continually transform between coordinate systems during analysis.

3. Advanced Spatial Analysis Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spatioloji automatically converts raw polygon data into GeoDataFrames, leveraging the powerful spatial analysis capabilities of the GeoPandas library:

* Automatic polygon construction from vertex coordinates
* Seamless integration with spatial analysis tools
* Built-in validation of spatial data integrity

4. Efficient Data Subsetting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework offers intuitive methods for creating coherent data subsets while maintaining all relational connections:

* ``subset_by_fovs()``: Extract specific fields of view while preserving all data relationships
* ``subset_by_cells()``: Focus on particular cells of interest across all data modalities

5. Flexible Extensibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

Spatioloji provides a ``custom`` dictionary attribute for storing user-defined data, allowing for seamless extension with additional analysis results or specialized data structures.

Comparison with Other Frameworks
----------------------------------

Spatioloji vs. Scanpy
~~~~~~~~~~~~~~~~~~~~~~~

While Scanpy is a powerful tool for single-cell analysis, Spatioloji offers several advantages for spatial transcriptomics:

* **Coordinate awareness**: Scanpy lacks native handling of cell geometries and spatial coordinates, while Spatioloji manages both local and global coordinate systems
* **Image integration**: Scanpy doesn't natively support microscopy image incorporation, whereas Spatioloji directly links images to fields of view
* **Polygon-based analysis**: Spatioloji's GeoDataFrame integration enables sophisticated spatial analyses not available in Scanpy
* **Multi-FOV support**: Scanpy has no built-in concept of fields of view, while Spatioloji was specifically designed for tiled/multi-FOV datasets

Spatioloji vs. Squidpy
~~~~~~~~~~~~~~~~~~~~~~~

Squidpy extends Scanpy with spatial capabilities, but Spatioloji still offers distinct advantages:

* **Cell morphology**: Squidpy focuses primarily on spot-based or pixel-based data, while Spatioloji excels with polygon-based cell boundaries
* **Multiple coordinate frames**: Squidpy typically works in a single coordinate system, whereas Spatioloji maintains both local and global frames
* **Image handling**: Spatioloji offers more direct and flexible image linking compared to Squidpy's image container
* **Field of view management**: Spatioloji provides native multi-FOV support, which is more limited in Squidpy

Spatioloji vs. Seurat (R)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Seurat is the dominant R framework for single-cell analysis with some spatial extensions, but Spatioloji provides distinct benefits:

* **Python ecosystem**: Full integration with Python's data science stack
* **Geometric analysis**: Better support for cell polygons versus Seurat's primarily spot-based approach
* **Coordinate flexibility**: More advanced handling of multiple coordinate systems
* **Image integration**: More direct microscopy image association

Ideal Use Cases
------------------

Spatioloji is particularly well-suited for:

1. High-plex imaging-based spatial transcriptomics technologies (MERFISH, Nanostring CosMx, etc.)
2. Multi-FOV datasets where both local context and global positioning matter
3. Analyses that require cell morphology information alongside gene expression
4. Projects needing seamless integration of microscopy images with transcriptomic data
5. Spatial transcriptomics workflows requiring sophisticated geometric operations

By combining cell morphology, gene expression, and spatial context in a unified framework, Spatioloji offers a powerful solution for researchers working with spatially resolved transcriptomic data.