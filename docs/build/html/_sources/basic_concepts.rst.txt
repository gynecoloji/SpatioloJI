SpatioloJI Object Structure
==========================

Overview
--------

The ``Spatioloji`` class provides a unified container for spatial transcriptomics data analysis, integrating multiple data modalities for comprehensive spatial biology research. It organizes expression data, metadata, spatial coordinates, and images into a cohesive structure for streamlined analysis.


Core Components
--------------

The ``Spatioloji`` object consists of five primary components:

1. **adata**
   A standard AnnData object containing gene expression data and analysis results.

2. **cell_meta**
   DataFrame with cell-level metadata and spatial coordinates.

3. **polygons**
   DataFrame containing cell boundary polygons.

4. **fov_positions**
   DataFrame mapping global coordinates of fields of view.

5. **images**
   Dictionary of microscopy images for each field of view.

Additional attributes include ``image_shapes`` for storing image dimensions and ``custom`` for user-defined data.

Component Details
---------------

adata (AnnData Object)
~~~~~~~~~~~~~~~~~~~~~

The central component is an AnnData object that follows the standard structure:

- **X**: Gene expression matrix (cells × genes)
- **obs**: Cell annotations (incorporated from cell_meta)
- **var**: Gene annotations
- **obsm**: Multi-dimensional observations (e.g., UMAP, spatial coordinates)
- **uns**: Unstructured annotations (analysis results)

cell_meta (DataFrame)
~~~~~~~~~~~~~~~~~~~

Contains metadata for each cell with spatial context:

- Row index: Cell IDs
- Columns include:
  - ``fov``: Field of view ID
  - ``CenterX_local_px``, ``CenterY_local_px``: Coordinates within FOV
  - ``CenterX_global_px``, ``CenterY_global_px``: Coordinates in global space
  - Cell type annotations and other metadata

polygons (DataFrame)
~~~~~~~~~~~~~~~~~~

Defines cell boundaries as polygon coordinates:

- ``cell``: Cell identifier
- ``fov``: Field of view identifier
- ``x_local_px``, ``y_local_px``: Local polygon coordinates
- ``x_global_px``, ``y_global_px``: Global polygon coordinates

fov_positions (DataFrame)
~~~~~~~~~~~~~~~~~~~~~~~

Maps the global positions of each field of view:

- Row index: FOV IDs
- ``x_global_px``, ``y_global_px``: Global coordinates of FOV origins

images (Dictionary)
~~~~~~~~~~~~~~~~

Stores the microscopy images for visualization:

- Keys: FOV IDs
- Values: Image arrays (height × width × channels)

Integration and Relationships
---------------------------

The SpatioloJI object maintains relationships between components:

- Cell IDs serve as common identifiers across ``adata``, ``cell_meta``, and ``polygons``
- FOV IDs connect ``cell_meta``, ``fov_positions``, and ``images``
- Spatial coordinates link cells to their position in tissue context
- Analysis results in ``adata`` can be visualized using the spatial information

Creating a SpatioloJI Object
--------------------------

Objects can be created in several ways:

From Individual Components
~~~~~~~~~~~~~~~~~~~~~~~~~

Provide the polygons DataFrame, cell metadata DataFrame, AnnData object, and FOV positions DataFrame directly to the constructor, with optional image data.

From Files
~~~~~~~~~

Use the ``from_files`` static method to create an object from CSV files for polygons, metadata, and FOV positions, and an H5AD file for the AnnData object.

From Pickle
~~~~~~~~~~

Save an existing object to a pickle file using ``to_pickle`` and load it later with the ``from_pickle`` static method.

Key Methods
---------

The SpatioloJI object provides methods for data access and manipulation:

Data Access
~~~~~~~~~~

- ``get_cells_in_fov(fov_id)``: Get cells within a specific FOV
- ``get_polygon_for_cell(cell_id)``: Get polygon data for a specific cell
- ``get_image(fov_id)``: Get the image for a specific FOV
- ``summary()``: Get a summary of the object's data

Custom Data
~~~~~~~~~~

- ``add_custom(key, value)``: Add custom data to the object
- ``get_custom(key)``: Retrieve custom data

   
Summary
------

The SpatioloJI object provides a comprehensive framework for spatial transcriptomics analysis by integrating gene expression data with spatial information and microscopy images. This unified structure facilitates both high-level exploration and detailed investigation of spatial biology phenomena, enabling researchers to uncover meaningful insights from complex spatial datasets.