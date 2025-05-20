import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.patches import Patch


def stitch_fov_images(
    spatioloji_obj,
    fov_ids: Optional[List[Union[str, int]]] = None,
    flip_vertical: bool = True,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 300,
    verbose: bool = True
) -> np.ndarray:
    """
    Stitch multiple FOV images together based on their global positions using a Spatioloji object.
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        A Spatioloji object containing FOV images and position data.
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to include in the stitched image. If None, all available FOVs are used.
    flip_vertical : bool, optional
        Whether to flip images vertically before stitching, by default True
    save_path : str, optional
        Path to save the stitched image. If None, image is not saved.
    show_plot : bool, optional
        Whether to display the stitched image, by default True
    title : str, optional
        Title for the plot. If None, a default title is used.
    figsize : Tuple[int, int], optional
        Figure size for plotting, by default (12, 12)
    dpi : int, optional
        Resolution for saving the image, by default 300
    verbose : bool, optional
        Whether to print progress information, by default True
    
    Returns
    -------
    np.ndarray
        The stitched image array
    
    Raises
    ------
    ValueError
        If required data is missing or no valid FOVs are found
    """
    # Check if spatioloji_obj has the required attributes
    if not hasattr(spatioloji_obj, 'fov_positions') or spatioloji_obj.fov_positions is None:
        raise ValueError("FOV positions data not found in Spatioloji object")
    
    if not hasattr(spatioloji_obj, 'images') or not spatioloji_obj.images:
        raise ValueError("No images found in Spatioloji object")
    
    # If no FOV IDs provided, use all available FOVs with images
    if fov_ids is None:
        fov_ids = list(spatioloji_obj.images.keys())
    
    if not fov_ids:
        raise ValueError("No FOV IDs provided")
    
    # Check for required columns in fov_positions
    required_cols = ['x_global_px', 'y_global_px']
    
    # Handle the case where 'fov' could be the index or a column
    fov_positions = spatioloji_obj.fov_positions.copy()
    
    if 'fov' in fov_positions.columns and fov_positions.index.name != 'fov':
        # Set fov as index if it's in columns
        fov_positions.set_index('fov', inplace=True)
    
    # Check required columns
    missing_cols = [col for col in required_cols if col not in fov_positions.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in fov_positions: {missing_cols}")
    
    # Convert FOV IDs to match the type in fov_positions (str or int)
    if all(isinstance(fid, str) for fid in fov_ids) and all(isinstance(idx, (int, np.integer)) for idx in fov_positions.index):
        # Convert string FOV IDs to integers for matching
        fov_ids = [int(fid) for fid in fov_ids if fid.isdigit()]
    elif all(isinstance(fid, (int, np.integer)) for fid in fov_ids) and all(isinstance(idx, str) for idx in fov_positions.index):
        # Convert integer FOV IDs to strings for matching
        fov_ids = [str(fid) for fid in fov_ids]
    
    # Filter FOV positions to only include selected FOVs
    fov_positions = fov_positions[fov_positions.index.isin(fov_ids)]
    
    if len(fov_positions) == 0:
        raise ValueError(f"None of the selected FOVs {fov_ids} have position information")
    
    if verbose:
        print(f"Stitching {len(fov_positions)} FOVs: {sorted(fov_positions.index.tolist())}")
    
    # Get image dimensions from the first available image
    first_fov = str(fov_positions.index[0])
    first_image = spatioloji_obj.get_image(first_fov)
    
    if first_image is None:
        raise ValueError(f"Could not find image for FOV {first_fov}")
    
    image_height, image_width = first_image.shape[:2]
    if verbose:
        print(f"Detected FOV image dimensions: {image_width} x {image_height}")
    
    # Calculate offsets
    min_x = fov_positions["x_global_px"].min()
    min_y = fov_positions["y_global_px"].min()
    
    fov_positions["x_offset"] = fov_positions["x_global_px"] - min_x
    fov_positions["y_offset"] = fov_positions["y_global_px"] - min_y
    
    # Calculate stitched image dimensions
    max_x = int(np.ceil((fov_positions["x_offset"].max() + image_width)))
    max_y = int(np.ceil((fov_positions["y_offset"].max() + image_height)))
    
    if verbose:
        print(f"Creating stitched image with dimensions: {max_x} x {max_y}")
    
    # Create empty canvas for the stitched image
    stitched = np.zeros((max_y, max_x, 3), dtype=np.uint8)
    
    # Keep track of successfully added FOVs
    added_fovs = []
    
    # Stitch images
    for fov in fov_positions.index:
        try:
            # Get image directly from Spatioloji object
            img = spatioloji_obj.get_image(str(fov))
            
            if img is None:
                if verbose:
                    print(f"Could not find image for FOV {fov}")
                continue
            
            # Flip image if required
            
            img = cv2.flip(img, 0)
            
            # Calculate position in the stitched image
            row = fov_positions.loc[fov]
            x_offset = int(row["x_offset"])
            y_offset = int(row["y_offset"])
            
            # Place image in the stitched canvas
            h, w = img.shape[:2]
            stitched[y_offset:y_offset+h, x_offset:x_offset+w] = img
            
            added_fovs.append(fov)
            if verbose:
                print(f"Added FOV {fov} at position ({x_offset}, {y_offset})")
            
        except Exception as e:
            if verbose:
                print(f"Error processing FOV {fov}: {e}")
            continue
    
    if not added_fovs:
        print("Warning: No FOVs were successfully added to the stitched image.")
        return None
    
    # Save stitched image if path is provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            plt.figure(figsize=figsize)
            if flip_vertical:
                stitched = cv2.flip(stitched,0)
            plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            if title:
                plt.title(title)
            else:
                plt.title(f"Stitched FOV Image ({len(added_fovs)} FOVs)")
                
            plt.tight_layout()
            
            if not save_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf')):
                save_path = save_path + 'stitched_fov.png'
                
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"Saved stitched image to {save_path}")
                
        except Exception as e:
            print(f"Error saving stitched image: {e}")
    
    # Display the stitched image
    if show_plot:
        plt.figure(figsize=figsize)
        if flip_vertical:
            stitched = cv2.flip(stitched,0)
        plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Stitched FOV Image ({len(added_fovs)} FOVs)")
            
        plt.tight_layout()
        plt.show()
    
    # Store min_x and min_y in the object for later use (if a visualizer object is created)
    spatioloji_obj.custom['min_x'] = min_x
    spatioloji_obj.custom['min_y'] = min_y
    
    # Store the stitched image in the object
    spatioloji_obj.custom['stitched_image'] = stitched
    
    if verbose:
        print(f"Stitched image created successfully with {len(added_fovs)} FOVs.")
    
    return spatioloji_obj


def plot_global_polygon_by_features(
    spatioloji_obj,
    feature: str,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = False,
    save_dir: str = "./",
    colormap: str = "viridis",
    edge_color: str = "none",
    edge_width: float = 0.01,
    figsize: tuple = (12, 12),
    fig_title: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 1.0
) -> plt.Figure:
    """
    Plot cell polygons colored by a continuous feature value using a Spatioloji object.
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        Spatioloji object containing polygon data and optionally a stitched image.
    feature : str
        The name of the feature to visualize:
        - If feature_df is None, assumes this is a column in spatioloji_obj.adata.obs
        - If feature_df is provided, this is a column in feature_df
        - If feature is "metadata" and feature_column is provided, this uses a column from spatioloji_obj.cell_meta
    feature_df : pd.DataFrame, optional
        Optional external DataFrame containing cell-level features.
        Must include a 'cell' column and the specified feature column.
    feature_column : str, optional
        If feature="metadata", specifies which column from cell_meta to use.
    background_img : bool, optional
        Whether to display the stitched image as background, by default False
    save_dir : str, optional
        Directory where the output image will be saved, by default "./"
    colormap : str, optional
        Matplotlib colormap name to use, by default "viridis"
    edge_color : str, optional
        Color of polygon edges, by default "none"
    edge_width : float, optional
        Width of polygon edges, by default 0.01
    figsize : tuple, optional
        Figure size as (width, height) in inches, by default (12, 12)
    fig_title : str, optional
        Figure title, by default None (uses feature name)
    filename : str, optional
        Custom filename for saved figure, by default None (auto-generated)
    dpi : int, optional
        Resolution of saved figure, by default 300
    show_plot : bool, optional
        Whether to display the figure, by default True
    vmin : float, optional
        Minimum value for color normalization, by default None (auto-determined)
    vmax : float, optional
        Maximum value for color normalization, by default None (auto-determined)
    alpha : float, optional
        Transparency of the polygons, by default 1.0
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Check if Spatioloji object has required attributes
    if not hasattr(spatioloji_obj, 'polygons') or spatioloji_obj.polygons is None:
        raise ValueError("Polygon data not found in Spatioloji object")
    
    # Create a copy of polygon data to avoid modifying the original
    polygon_df = spatioloji_obj.polygons.copy()
    
    # Check required columns in polygon data
    required_cols = ['cell', 'x_global_px', 'y_global_px']
    missing_cols = [col for col in required_cols if col not in polygon_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Spatioloji polygons: {missing_cols}")
    
    # Get feature values for each cell
    if feature_df is not None:
        # Using external feature dataframe
        if 'cell' not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain 'cell' and '{feature}' columns")
        
        feature_subset = feature_df[['cell', feature]].copy()
        
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if not hasattr(spatioloji_obj, 'cell_meta') or spatioloji_obj.cell_meta is None:
            raise ValueError("Cell metadata not found in Spatioloji object")
            
        if feature_column not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in Spatioloji cell_meta")
            
        feature_subset = spatioloji_obj.cell_meta[['cell', feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        feature = 'feature_value'  # Rename for later use
        
    else:
        # Using adata observations
        if not hasattr(spatioloji_obj, 'adata') or spatioloji_obj.adata is None:
            raise ValueError("Gene expression data (adata) not found in Spatioloji object")
            
        if feature not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Feature '{feature}' not found in Spatioloji adata observations")
            
        feature_subset = spatioloji_obj.adata.obs[feature].reset_index()
        feature_subset.rename(columns={'index': 'cell'}, inplace=True)
    
    # Apply global coordinate offsets if available
    if hasattr(spatioloji_obj, 'min_x') and hasattr(spatioloji_obj, 'min_y'):
        if spatioloji_obj.min_x is not None and spatioloji_obj.min_y is not None:
            polygon_df['x_global_px_offset'] = polygon_df['x_global_px'] - spatioloji_obj.min_x
            polygon_df['y_global_px_offset'] = polygon_df['y_global_px'] - spatioloji_obj.min_y
    else:
        # Calculate min_x and min_y if not already stored
        min_x = polygon_df['x_global_px'].min()
        min_y = polygon_df['y_global_px'].min()
        polygon_df['x_global_px_offset'] = polygon_df['x_global_px'] - min_x
        polygon_df['y_global_px_offset'] = polygon_df['y_global_px'] - min_y
        
        # Store for future use
        spatioloji_obj.min_x = min_x
        spatioloji_obj.min_y = min_y
    
    # Merge datasets
    merged_df = polygon_df.merge(feature_subset, on='cell')
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between polygon data and feature values")
    
    # Normalize feature values for coloring
    if vmin is None:
        vmin = merged_df[feature].min()
    if vmax is None:
        vmax = merged_df[feature].max()
        
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the stitched image as background if requested
    if background_img:
        if 'stitched_image' in spatioloji_obj.custom.keys() and spatioloji_obj.custom['stitched_image'] is not None:
            import cv2
            ax.imshow(cv2.cvtColor(cv2.flip(spatioloji_obj.custom['stitched_image'],0), cv2.COLOR_BGR2RGB))
        else:
            print("Warning: Stitched image not found in Spatioloji object. Proceeding without background.")
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Calculate plot limits
    max_x = merged_df['x_global_px_offset'].max() * 1.05
    max_y = merged_df['y_global_px_offset'].max() * 1.05
    
    # Create polygons for each cell
    print(f"Creating polygons for {len(merged_df['cell'].unique())} cells...")
    
    # Group by cell to get unique cells
    grouped_cells = merged_df.groupby('cell')
    
    # Create and add polygons
    for cell_id, group in grouped_cells:
        feature_value = group[feature].iloc[0]
        color = cmap(norm(feature_value))
        
        # Get polygon coordinates
        coords = group[['x_global_px_offset', 'y_global_px_offset']].values
        
        # Create and add polygon
        polygon = mpl.patches.Polygon(
            coords, 
            closed=True, 
            facecolor=color, 
            edgecolor=edge_color if edge_color != "none" else None,
            linewidth=edge_width,
            alpha=alpha
        )
        ax.add_patch(polygon)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.axis('off')
    
    # Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for matplotlib >= 3.1
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(fig_title if fig_title is not None else feature)
    
    # Set title
    title = fig_title if fig_title is not None else feature
    plt.title(title, fontsize=14)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        feature_name = feature if feature != 'feature_value' else feature_column
        bg_suffix = "_with_bg" if background_img else ""
        filename = f"polygon_{feature_name}_global{bg_suffix}.png"
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_global_polygon_by_categorical(
    spatioloji_obj,
    feature: str,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = False,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    edge_color: str = "black",
    edge_width: float = 0.5,
    figsize: tuple = (12, 12),
    fig_title: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    alpha: float = 0.8,
    legend_loc: str = 'center left',
    legend_bbox_to_anchor: Tuple[float, float] = (1.01, 0.5)
) -> plt.Figure:
    """
    Plot cell polygons colored by a categorical feature value using a Spatioloji object.
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        Spatioloji object containing polygon data and optionally a stitched image.
    feature : str
        The name of the feature to visualize:
        - If feature_df is None, assumes this is a column in spatioloji_obj.adata.obs
        - If feature_df is provided, this is a column in feature_df
        - If feature is "metadata" and feature_column is provided, this uses a column from spatioloji_obj.cell_meta
    feature_df : pd.DataFrame, optional
        Optional external DataFrame containing cell-level features.
        Must include a 'cell' column and the specified feature column.
    feature_column : str, optional
        If feature="metadata", specifies which column from cell_meta to use.
    background_img : bool, optional
        Whether to display the stitched image as background, by default False
    save_dir : str, optional
        Directory where the output image will be saved, by default "./"
    color_map : Dict[str, tuple], optional
        Mapping of category values to colors, by default None.
        If None, colors will be automatically assigned.
    edge_color : str, optional
        Color of polygon edges, by default "black"
    edge_width : float, optional
        Width of polygon edges, by default 0.5
    figsize : tuple, optional
        Figure size as (width, height) in inches, by default (12, 12)
    fig_title : str, optional
        Figure title, by default None (uses feature name)
    filename : str, optional
        Custom filename for saved figure, by default None (auto-generated)
    dpi : int, optional
        Resolution of saved figure, by default 300
    show_plot : bool, optional
        Whether to display the figure, by default True
    alpha : float, optional
        Transparency of the polygons, by default 0.8
    legend_loc : str, optional
        Location of the legend, by default 'center left'
    legend_bbox_to_anchor : Tuple[float, float], optional
        Position of the legend anchor, by default (1.01, 0.5)
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Check if Spatioloji object has required attributes
    if not hasattr(spatioloji_obj, 'polygons') or spatioloji_obj.polygons is None:
        raise ValueError("Polygon data not found in Spatioloji object")
    
    # Create a copy of polygon data to avoid modifying the original
    polygon_df = spatioloji_obj.polygons.copy()
    
    # Check required columns in polygon data
    required_cols = ['cell', 'x_global_px', 'y_global_px']
    missing_cols = [col for col in required_cols if col not in polygon_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Spatioloji polygons: {missing_cols}")
    
    # Get feature values for each cell
    if feature_df is not None:
        # Using external feature dataframe
        if 'cell' not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain 'cell' and '{feature}' columns")
        
        feature_subset = feature_df[['cell', feature]].copy()
        
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if not hasattr(spatioloji_obj, 'cell_meta') or spatioloji_obj.cell_meta is None:
            raise ValueError("Cell metadata not found in Spatioloji object")
            
        if feature_column not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in Spatioloji cell_meta")
            
        feature_subset = spatioloji_obj.cell_meta[['cell', feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        feature = 'feature_value'  # Rename for later use
        
    else:
        # Using adata observations
        if not hasattr(spatioloji_obj, 'adata') or spatioloji_obj.adata is None:
            raise ValueError("Gene expression data (adata) not found in Spatioloji object")
            
        if feature not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Feature '{feature}' not found in Spatioloji adata observations")
            
        feature_subset = spatioloji_obj.adata.obs[[feature]]
        feature_subset['cell'] = spatioloji_obj.adata.obs['cell']
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:  # Too many unique values for categories
            raise ValueError(f"Feature '{feature}' has {n_unique} unique values, which is too many for categorical plotting. Consider binning the data first.")
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Apply global coordinate offsets if available
    if hasattr(spatioloji_obj, 'min_x') and hasattr(spatioloji_obj, 'min_y'):
        if spatioloji_obj.min_x is not None and spatioloji_obj.min_y is not None:
            polygon_df['x_global_px_offset'] = polygon_df['x_global_px'] - spatioloji_obj.min_x
            polygon_df['y_global_px_offset'] = polygon_df['y_global_px'] - spatioloji_obj.min_y
    else:
        # Calculate min_x and min_y if not already stored
        min_x = polygon_df['x_global_px'].min()
        min_y = polygon_df['y_global_px'].min()
        polygon_df['x_global_px_offset'] = polygon_df['x_global_px'] - min_x
        polygon_df['y_global_px_offset'] = polygon_df['y_global_px'] - min_y
        
        # Store for future use
        spatioloji_obj.min_x = min_x
        spatioloji_obj.min_y = min_y
    
    # Merge datasets
    merged_df = polygon_df.merge(feature_subset, on='cell')
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between polygon data and feature values")
    
    # Get list of categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors  # Default color palette
        color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the stitched image as background if requested
    if background_img:
        if hasattr(spatioloji_obj, 'stitched_image') and spatioloji_obj.stitched_image is not None:
            import cv2
            ax.imshow(cv2.cvtColor(spatioloji_obj.stitched_image, cv2.COLOR_BGR2RGB))
        elif hasattr(spatioloji_obj, 'custom') and 'stitched_image' in spatioloji_obj.custom and spatioloji_obj.custom['stitched_image'] is not None:
            import cv2
            ax.imshow(cv2.cvtColor(cv2.flip(spatioloji_obj.custom['stitched_image'], 0), cv2.COLOR_BGR2RGB))
        else:
            print("Warning: Stitched image not found in Spatioloji object. Proceeding without background.")
    
    # Calculate plot limits
    max_x = merged_df['x_global_px_offset'].max() * 1.05
    max_y = merged_df['y_global_px_offset'].max() * 1.05
    
    # Create a list to track which categories are actually present in the data
    categories_present = []
    
    # Group by cell to get unique cells
    cell_groups = {}
    
    # First, group all cells by category for efficiency
    for category in categories:
        # Get all cells belonging to this category
        category_cells = merged_df[merged_df[feature] == category]['cell'].unique()
        
        if len(category_cells) == 0:
            continue
            
        categories_present.append(category)
        cell_groups[category] = category_cells
    
    # Now process all polygons by category
    print(f"Creating polygons for {len(merged_df['cell'].unique())} cells in {len(categories_present)} categories...")
    
    for category in categories_present:
        category_color = color_map[category]
        
        # Get all cells for this category
        for cell_id in cell_groups[category]:
            # Get all polygon vertices for this cell
            cell_vertices = merged_df[merged_df['cell'] == cell_id]
            
            # Create polygon coordinates
            coords = cell_vertices[['x_global_px_offset', 'y_global_px_offset']].values
            
            # Create and add polygon
            polygon = mpl.patches.Polygon(
                coords, 
                closed=True, 
                facecolor=category_color, 
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=alpha
            )
            ax.add_patch(polygon)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.axis('off')
    
    # Create display name for the feature
    display_name = feature if feature != 'feature_value' else feature_column
    
    # Add legend patches
    legend_patches = [Patch(facecolor=color_map[cat], 
                           edgecolor=edge_color,
                           alpha=alpha,
                           label=str(cat)) 
                     for cat in categories_present]
    
    # Add legend
    if len(categories_present) <= 20:
        ax.legend(
            handles=legend_patches,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            title=display_name,
            fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend
    else:
        plt.tight_layout()
        print(f"Warning: Too many categories ({len(categories_present)}) to display in legend.")
    
    # Set title
    title = fig_title if fig_title is not None else f"{display_name} Distribution"
    plt.title(title, fontsize=14)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        feature_name = display_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f"polygon_{feature_name}_categorical{bg_suffix}.png"
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_global_dot_by_features(
    spatioloji_obj,
    feature: str,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = False,
    save_dir: str = "./",
    colormap: str = "viridis",
    dot_size: int = 20,
    figsize: tuple = (12, 12),
    fig_title: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 1.0,
    edge_color: Optional[str] = None,
    edge_width: float = 0.0
) -> plt.Figure:
    """
    Plot cell dots colored by a continuous feature value using a Spatioloji object.
    Uses global coordinates from cell_meta (CenterX_global_px, CenterY_global_px).
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        Spatioloji object containing cell metadata and optionally a stitched image.
    feature : str
        The name of the feature to visualize:
        - If feature_df is None, assumes this is a column in spatioloji_obj.adata.obs
        - If feature_df is provided, this is a column in feature_df
        - If feature is "metadata" and feature_column is provided, this uses a column from spatioloji_obj.cell_meta
    feature_df : pd.DataFrame, optional
        Optional external DataFrame containing cell-level features.
        Must include a 'cell' column and the specified feature column.
    feature_column : str, optional
        If feature="metadata", specifies which column from cell_meta to use.
    background_img : bool, optional
        Whether to display the stitched image as background, by default False
    save_dir : str, optional
        Directory where the output image will be saved, by default "./"
    colormap : str, optional
        Matplotlib colormap name to use, by default "viridis"
    dot_size : int, optional
        Size of the dots in points squared, by default 20
    figsize : tuple, optional
        Figure size as (width, height) in inches, by default (12, 12)
    fig_title : str, optional
        Figure title, by default None (uses feature name)
    filename : str, optional
        Custom filename for saved figure, by default None (auto-generated)
    dpi : int, optional
        Resolution of saved figure, by default 300
    show_plot : bool, optional
        Whether to display the figure, by default True
    vmin : float, optional
        Minimum value for color normalization, by default None (auto-determined)
    vmax : float, optional
        Maximum value for color normalization, by default None (auto-determined)
    alpha : float, optional
        Transparency of the dots, by default 1.0
    edge_color : str, optional
        Color of dot edges, by default None (same as fill color)
    edge_width : float, optional
        Width of dot edges, by default 0.0
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Check if Spatioloji object has required attributes
    if not hasattr(spatioloji_obj, 'cell_meta') or spatioloji_obj.cell_meta is None:
        raise ValueError("Cell metadata not found in Spatioloji object")
    
    # Create a copy of cell_meta to avoid modifying the original
    cell_meta = spatioloji_obj.cell_meta.copy()
    
    # Check required columns in cell_meta for global coordinates
    required_cols = ['cell', 'CenterX_global_px', 'CenterY_global_px']
    missing_cols = [col for col in required_cols if col not in cell_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Spatioloji cell_meta: {missing_cols}")
    cell_meta = cell_meta[required_cols]
    
    # Get feature values for each cell
    if feature_df is not None:
        # Using external feature dataframe
        if 'cell' not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain 'cell' and '{feature}' columns")
        
        feature_subset = feature_df[['cell', feature]].copy()
        
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if feature_column not in cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in Spatioloji cell_meta")
            
        feature_subset = cell_meta[['cell', feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        feature = 'feature_value'  # Rename for later use
        
    else:
        # Using adata observations
        if not hasattr(spatioloji_obj, 'adata') or spatioloji_obj.adata is None:
            raise ValueError("Gene expression data (adata) not found in Spatioloji object")
            
        if feature not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Feature '{feature}' not found in Spatioloji adata observations")
            
        feature_subset = spatioloji_obj.adata.obs[[feature]]
        feature_subset['cell'] = spatioloji_obj.adata.obs['cell']
    
    # Apply global coordinate offsets if available
    if hasattr(spatioloji_obj, 'min_x') and hasattr(spatioloji_obj, 'min_y'):
        if spatioloji_obj.min_x is not None and spatioloji_obj.min_y is not None:
            cell_meta['CenterX_global_px_offset'] = cell_meta['CenterX_global_px'] - spatioloji_obj.min_x
            cell_meta['CenterY_global_px_offset'] = cell_meta['CenterY_global_px'] - spatioloji_obj.min_y
    else:
        # Calculate min_x and min_y if not already stored
        min_x = cell_meta['CenterX_global_px'].min()
        min_y = cell_meta['CenterY_global_px'].min()
        cell_meta['CenterX_global_px_offset'] = cell_meta['CenterX_global_px'] - min_x
        cell_meta['CenterY_global_px_offset'] = cell_meta['CenterY_global_px'] - min_y
        
        # Store for future use
        spatioloji_obj.min_x = min_x
        spatioloji_obj.min_y = min_y
    
    # Merge datasets
    merged_df = cell_meta.merge(feature_subset, on='cell')
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between cell_meta and feature values")
    
    # Check if the feature is numeric
    if not pd.api.types.is_numeric_dtype(merged_df[feature]):
        raise ValueError(f"Feature '{feature}' must be numeric for continuous coloring")
    
    # Normalize feature values for coloring
    if vmin is None:
        vmin = merged_df[feature].min()
    if vmax is None:
        vmax = merged_df[feature].max()
        
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the stitched image as background if requested
    if background_img:
        if hasattr(spatioloji_obj, 'stitched_image') and spatioloji_obj.stitched_image is not None:
            import cv2
            ax.imshow(cv2.cvtColor(spatioloji_obj.stitched_image, cv2.COLOR_BGR2RGB))
        elif hasattr(spatioloji_obj, 'custom') and 'stitched_image' in spatioloji_obj.custom and spatioloji_obj.custom['stitched_image'] is not None:
            import cv2
            if 'stitched_image' in spatioloji_obj.custom.keys():
                ax.imshow(cv2.cvtColor(cv2.flip(spatioloji_obj.custom['stitched_image'], 0), cv2.COLOR_BGR2RGB))
        else:
            print("Warning: Stitched image not found in Spatioloji object. Proceeding without background.")
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Calculate plot limits
    max_x = merged_df['CenterX_global_px_offset'].max() * 1.05
    max_y = merged_df['CenterY_global_px_offset'].max() * 1.05
    
    # Create scatter plot colored by feature value
    print(f"Creating scatter plot for {len(merged_df)} cells...")
    
    # Create scatter plot with feature-based coloring
    scatter = ax.scatter(
        merged_df['CenterX_global_px_offset'],
        merged_df['CenterY_global_px_offset'],
        c=merged_df[feature],
        cmap=colormap,
        norm=norm,
        s=dot_size,
        alpha=alpha,
        edgecolors=edge_color,
        linewidths=edge_width
    )
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    display_name = feature if feature != 'feature_value' else feature_column
    cbar.set_label(display_name)
    
    # Set title
    title = fig_title if fig_title is not None else display_name
    plt.title(title, fontsize=14)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        feature_name = display_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f"global_dot_{feature_name}{bg_suffix}.png"
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_global_dot_by_categorical(
    spatioloji_obj,
    feature: str,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = False,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    dot_size: int = 20,
    figsize: tuple = (12, 12),
    fig_title: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    alpha: float = 1.0,
    edge_color: Optional[str] = None,
    edge_width: float = 0.0
) -> plt.Figure:
    """
    Plot cell dots colored by a categorical feature using a Spatioloji object.
    Uses global coordinates from cell_meta (CenterX_global_px, CenterY_global_px).
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        Spatioloji object containing cell metadata and optionally a stitched image.
    feature : str
        The name of the feature to visualize:
        - If feature_df is None, assumes this is a column in spatioloji_obj.adata.obs
        - If feature_df is provided, this is a column in feature_df
        - If feature is "metadata" and feature_column is provided, this uses a column from spatioloji_obj.cell_meta
    feature_df : pd.DataFrame, optional
        Optional external DataFrame containing cell-level features.
        Must include a 'cell' column and the specified feature column.
    feature_column : str, optional
        If feature="metadata", specifies which column from cell_meta to use.
    background_img : bool, optional
        Whether to display the stitched image as background, by default False
    save_dir : str, optional
        Directory where the output image will be saved, by default "./"
    color_map : Dict[str, tuple], optional
        Mapping of category values to colors, by default None.
        If None, colors will be automatically assigned.
    dot_size : int, optional
        Size of the dots in points squared, by default 20
    figsize : tuple, optional
        Figure size as (width, height) in inches, by default (12, 12)
    fig_title : str, optional
        Figure title, by default None (uses feature name)
    filename : str, optional
        Custom filename for saved figure, by default None (auto-generated)
    dpi : int, optional
        Resolution of saved figure, by default 300
    show_plot : bool, optional
        Whether to display the figure, by default True
    alpha : float, optional
        Transparency of the dots, by default 1.0
    edge_color : str, optional
        Color of dot edges, by default None (same as fill color)
    edge_width : float, optional
        Width of dot edges, by default 0.0
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Check if Spatioloji object has required attributes
    if not hasattr(spatioloji_obj, 'cell_meta') or spatioloji_obj.cell_meta is None:
        raise ValueError("Cell metadata not found in Spatioloji object")
    
    # Create a copy of cell_meta to avoid modifying the original
    cell_meta = spatioloji_obj.cell_meta.copy()
    
    # Check required columns in cell_meta for global coordinates
    required_cols = ['cell', 'CenterX_global_px', 'CenterY_global_px']
    missing_cols = [col for col in required_cols if col not in cell_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Spatioloji cell_meta: {missing_cols}")
    cell_meta = cell_meta[required_cols]
    
    # Get feature values for each cell
    if feature_df is not None:
        # Using external feature dataframe
        if 'cell' not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain 'cell' and '{feature}' columns")
        
        feature_subset = feature_df[['cell', feature]].copy()
        
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if feature_column not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in Spatioloji cell_meta")
            
        feature_subset = spatioloji_obj.cell_meta[['cell', feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        feature = 'feature_value'  # Rename for later use
        
    else:
        # Using adata observations
        if not hasattr(spatioloji_obj, 'adata') or spatioloji_obj.adata is None:
            raise ValueError("Gene expression data (adata) not found in Spatioloji object")
            
        if feature not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Feature '{feature}' not found in Spatioloji adata observations")
            
        feature_subset = spatioloji_obj.adata.obs[[feature]]
        feature_subset['cell'] = spatioloji_obj.adata.obs_names
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:  # Too many unique values for categories
            raise ValueError(f"Feature '{feature}' has {n_unique} unique values, which is too many for categorical plotting. Consider binning the data first.")
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Apply global coordinate offsets if available
    if hasattr(spatioloji_obj, 'min_x') and hasattr(spatioloji_obj, 'min_y'):
        if spatioloji_obj.min_x is not None and spatioloji_obj.min_y is not None:
            cell_meta['CenterX_global_px_offset'] = cell_meta['CenterX_global_px'] - spatioloji_obj.min_x
            cell_meta['CenterY_global_px_offset'] = cell_meta['CenterY_global_px'] - spatioloji_obj.min_y
    else:
        # Calculate min_x and min_y if not already stored
        min_x = cell_meta['CenterX_global_px'].min()
        min_y = cell_meta['CenterY_global_px'].min()
        cell_meta['CenterX_global_px_offset'] = cell_meta['CenterX_global_px'] - min_x
        cell_meta['CenterY_global_px_offset'] = cell_meta['CenterY_global_px'] - min_y
        
        # Store for future use
        spatioloji_obj.min_x = min_x
        spatioloji_obj.min_y = min_y
    
    # Merge datasets
    merged_df = cell_meta.merge(feature_subset, on='cell')
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between cell_meta and feature values")
    
    # Get list of categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors  # Default color palette
        color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the stitched image as background if requested
    if background_img:
        if hasattr(spatioloji_obj, 'stitched_image') and spatioloji_obj.stitched_image is not None:
            import cv2
            ax.imshow(cv2.cvtColor(spatioloji_obj.stitched_image, cv2.COLOR_BGR2RGB))
        elif hasattr(spatioloji_obj, 'custom') and 'stitched_image' in spatioloji_obj.custom and spatioloji_obj.custom['stitched_image'] is not None:
            import cv2
            if 'stitched_image' in spatioloji_obj.custom.keys():
                ax.imshow(cv2.cvtColor(cv2.flip(spatioloji_obj.custom['stitched_image'], 0), cv2.COLOR_BGR2RGB))
        else:
            print("Warning: Stitched image not found in Spatioloji object. Proceeding without background.")
    
    # Calculate plot limits
    max_x = merged_df['CenterX_global_px_offset'].max() * 1.05
    max_y = merged_df['CenterY_global_px_offset'].max() * 1.05
    
    # Create a list to track which categories are actually present in the data
    categories_present = []
    
    # Plot each category separately for better control over legend
    for category in categories:
        category_cells = merged_df[merged_df[feature] == category]
        
        if len(category_cells) == 0:
            continue
            
        categories_present.append(category)
        
        # Create scatter plot for this category
        ax.scatter(
            category_cells['CenterX_global_px_offset'],
            category_cells['CenterY_global_px_offset'],
            c=[color_map[category]],  # Use consistent color for this category
            s=dot_size,
            alpha=alpha,
            edgecolors=edge_color,
            linewidths=edge_width,
            label=str(category)  # Add label for legend
        )
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.axis('off')
    
    # Create display name for the feature
    display_name = feature if feature != 'feature_value' else feature_column
    
    # Add legend if not too many categories
    if len(categories_present) <= 20:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            title=display_name,
            fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend
    else:
        plt.tight_layout()
        print(f"Warning: Too many categories ({len(categories_present)}) to display in legend.")
    
    # Set title
    title = fig_title if fig_title is not None else f"{display_name} Distribution"
    plt.title(title, fontsize=14)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        feature_name = display_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f"global_dot_{feature_name}_categorical{bg_suffix}.png"
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_local_polygon_by_categorical(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create polygon plots colored by categorical features across multiple FOVs using a Spatioloji object.
    Uses local coordinates from polygon data to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        Spatioloji object containing polygon data, cell_meta, and FOV images.
    feature : str
        The name of the feature to visualize:
        - If feature_df is None, assumes this is a column in spatioloji_obj.adata.obs
        - If feature_df is provided, this is a column in feature_df
        - If feature is "metadata" and feature_column is provided, this uses a column from spatioloji_obj.cell_meta
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images will be used.
    feature_df : pd.DataFrame, optional
        Optional external DataFrame containing cell-level features.
        Must include a 'cell' column and the specified feature column.
    feature_column : str, optional
        If feature="metadata", specifies which column from cell_meta to use.
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save the output figure, by default "./"
    color_map : Dict[str, tuple], optional
        Mapping of category values to colors, by default None.
        If None, colors will be automatically assigned.
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout as (rows, columns), by default None (auto-determined)
    edge_color : str, optional
        Color of polygon edges, by default 'black'
    edge_width : float, optional
        Width of polygon edges, by default 0.5
    alpha : float, optional
        Transparency of the polygons, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for the main title, by default 24
    filename : str, optional
        Custom filename for the saved figure, by default None (auto-generated)
    dpi : int, optional
        Resolution of the saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    
    Raises
    ------
    ValueError
        If required data is missing or incompatible
    """
    # Check if Spatioloji object has required attributes
    if not hasattr(spatioloji_obj, 'polygons') or spatioloji_obj.polygons is None:
        raise ValueError("Polygon data not found in Spatioloji object")
    
    # Create a copy of polygon data to avoid modifying the original
    polygon_df = spatioloji_obj.polygons.copy()
    
    # Check required columns in polygon data
    required_cols = ['cell', 'fov', 'x_local_px', 'y_local_px']
    missing_cols = [col for col in required_cols if col not in polygon_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Spatioloji polygons: {missing_cols}")
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if hasattr(spatioloji_obj, 'images') and spatioloji_obj.images:
            fov_ids = list(spatioloji_obj.images.keys())
        else:
            # Fall back to all FOVs in polygon_df
            fov_ids = sorted(polygon_df['fov'].unique())
    
    # Convert FOV IDs to match the type in polygon_df
    if all(isinstance(fid, str) for fid in fov_ids) and all(isinstance(fid, (int, np.integer)) for fid in polygon_df['fov'].unique()):
        # Convert string FOV IDs to integers for matching
        fov_ids = [int(fid) for fid in fov_ids if fid.isdigit()]
    elif all(isinstance(fid, (int, np.integer)) for fid in fov_ids) and all(isinstance(fid, str) for fid in polygon_df['fov'].unique()):
        # Convert integer FOV IDs to strings for matching
        fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature values for each cell
    if feature_df is not None:
        # Using external feature dataframe
        if 'cell' not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain 'cell' and '{feature}' columns")
        
        feature_subset = feature_df[['cell', feature]].copy()
        
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if not hasattr(spatioloji_obj, 'cell_meta') or spatioloji_obj.cell_meta is None:
            raise ValueError("Cell metadata not found in Spatioloji object")
            
        if feature_column not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in Spatioloji cell_meta")
            
        feature_subset = spatioloji_obj.cell_meta[['cell', feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        feature = 'feature_value'  # Rename for later use
        
    else:
        # Using adata observations
        if not hasattr(spatioloji_obj, 'adata') or spatioloji_obj.adata is None:
            raise ValueError("Gene expression data (adata) not found in Spatioloji object")
            
        if feature not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Feature '{feature}' not found in Spatioloji adata observations")
            
        feature_subset = spatioloji_obj.adata.obs[[feature]]
        feature_subset['cell'] = spatioloji_obj.adata.obs_names
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:  # Too many unique values for categories
            raise ValueError(f"Feature '{feature}' has {n_unique} unique values, which is too many for categorical plotting. Consider binning the data first.")
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Merge datasets
    merged_df = polygon_df.merge(feature_subset, on='cell')
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between polygon data and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df['fov'].isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Get list of categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors  # Default color palette
        color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Create display name for the feature
    display_name = feature if feature != 'feature_value' else feature_column
    
    # Keep track of categories actually present in the data
    categories_present = set()
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
            
        ax = axs[idx]
        
        # Display the FOV image as background if requested
        image_width = None
        image_height = None
        
        import cv2
        if background_img:
            if hasattr(spatioloji_obj, 'get_image'):
                img = cv2.flip(spatioloji_obj.get_image(str(fov_id)),0)
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    image_height, image_width = img.shape[:2]
                else:
                    print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df['fov'] == fov_id]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If we don't have image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data["x_local_px"].max() * 1.1
            image_height = fov_data["y_local_px"].max() * 1.1
        
        # Group cells by category for this FOV
        cell_categories = {}
        
        # First, identify all cells by category
        for category in categories:
            category_cells = fov_data[fov_data[feature] == category]['cell'].unique()
            
            if len(category_cells) == 0:
                continue
                
            categories_present.add(category)
            cell_categories[category] = category_cells
        
        # Process all cells by category
        for category, cells in cell_categories.items():
            category_color = color_map[category]
            
            # Process each cell in this category
            for cell_id in cells:
                # Get polygon vertices for this cell
                cell_vertices = fov_data[fov_data['cell'] == cell_id]
                
                # Skip cells with fewer than 3 vertices (can't form a polygon)
                if len(cell_vertices) < 3:
                    print(f"Warning: Cell {cell_id} in FOV {fov_id} has fewer than 3 vertices. Skipping.")
                    continue
                
                # Create polygon coordinates
                coords = cell_vertices[['x_local_px', 'y_local_px']].values
                
                # Create and add polygon
                polygon = mpl.patches.Polygon(
                    coords, 
                    closed=True, 
                    facecolor=category_color, 
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    alpha=alpha
                )
                ax.add_patch(polygon)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add legend for all actually present categories
    legend_patches = [Patch(facecolor=color_map[cat], 
                          edgecolor=edge_color,
                          alpha=alpha,
                          label=str(cat)) 
                     for cat in sorted(categories_present)]
    
    fig.legend(
        handles=legend_patches, 
        loc='center right', 
        bbox_to_anchor=(1.02, 0.5), 
        title=display_name,
        fontsize=20
    )
    
    # Set title and adjust layout
    plt.suptitle(display_name, fontsize=suptitle_fontsize)
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])  # Adjust for legend space
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        feature_name = display_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_polygon_{feature_name}_categorical{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_local_polygon_by_features(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    colormap: str = "viridis",
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_position: str = 'right'
) -> plt.Figure:
    """
    Create polygon plots colored by continuous features across multiple FOVs using a Spatioloji object.
    Uses local coordinates from polygon data to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        Spatioloji object containing polygon data, cell_meta, and FOV images.
    feature : str
        The name of the feature to visualize:
        - If feature_df is None, assumes this is a column in spatioloji_obj.adata.obs
        - If feature_df is provided, this is a column in feature_df
        - If feature is "metadata" and feature_column is provided, this uses a column from spatioloji_obj.cell_meta
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images will be used.
    feature_df : pd.DataFrame, optional
        Optional external DataFrame containing cell-level features.
        Must include a 'cell' column and the specified feature column.
    feature_column : str, optional
        If feature="metadata", specifies which column from cell_meta to use.
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save the output figure, by default "./"
    colormap : str, optional
        Matplotlib colormap name to use, by default "viridis"
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout as (rows, columns), by default None (auto-determined)
    edge_color : str, optional
        Color of polygon edges, by default 'black'
    edge_width : float, optional
        Width of polygon edges, by default 0.5
    alpha : float, optional
        Transparency of the polygons, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for the main title, by default 24
    filename : str, optional
        Custom filename for the saved figure, by default None (auto-generated)
    dpi : int, optional
        Resolution of the saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    vmin : float, optional
        Minimum value for color normalization, by default None (auto-determined)
    vmax : float, optional
        Maximum value for color normalization, by default None (auto-determined)
    colorbar_position : str, optional
        Position of the colorbar, by default 'right'
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    
    Raises
    ------
    ValueError
        If required data is missing or incompatible
    """
    # Check if Spatioloji object has required attributes
    if not hasattr(spatioloji_obj, 'polygons') or spatioloji_obj.polygons is None:
        raise ValueError("Polygon data not found in Spatioloji object")
    
    # Create a copy of polygon data to avoid modifying the original
    polygon_df = spatioloji_obj.polygons.copy()
    
    # Check required columns in polygon data
    required_cols = ['cell', 'fov', 'x_local_px', 'y_local_px']
    missing_cols = [col for col in required_cols if col not in polygon_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Spatioloji polygons: {missing_cols}")
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if hasattr(spatioloji_obj, 'images') and spatioloji_obj.images:
            fov_ids = list(spatioloji_obj.images.keys())
        else:
            # Fall back to all FOVs in polygon_df
            fov_ids = sorted(polygon_df['fov'].unique())
    
    # Convert FOV IDs to match the type in polygon_df
    if all(isinstance(fid, str) for fid in fov_ids) and all(isinstance(fid, (int, np.integer)) for fid in polygon_df['fov'].unique()):
        # Convert string FOV IDs to integers for matching
        fov_ids = [int(fid) for fid in fov_ids if fid.isdigit()]
    elif all(isinstance(fid, (int, np.integer)) for fid in fov_ids) and all(isinstance(fid, str) for fid in polygon_df['fov'].unique()):
        # Convert integer FOV IDs to strings for matching
        fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature values for each cell
    if feature_df is not None:
        # Using external feature dataframe
        if 'cell' not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain 'cell' and '{feature}' columns")
        
        feature_subset = feature_df[['cell', feature]].copy()
        
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if not hasattr(spatioloji_obj, 'cell_meta') or spatioloji_obj.cell_meta is None:
            raise ValueError("Cell metadata not found in Spatioloji object")
            
        if feature_column not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in Spatioloji cell_meta")
            
        feature_subset = spatioloji_obj.cell_meta[['cell', feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        feature = 'feature_value'  # Rename for later use
        
    else:
        # Using adata observations
        if not hasattr(spatioloji_obj, 'adata') or spatioloji_obj.adata is None:
            raise ValueError("Gene expression data (adata) not found in Spatioloji object")
            
        if feature not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Feature '{feature}' not found in Spatioloji adata observations")
            
        feature_subset = spatioloji_obj.adata.obs[[feature]]
        feature_subset['cell'] = spatioloji_obj.adata.obs_names
    
    # Check if the feature is numeric
    if not pd.api.types.is_numeric_dtype(feature_subset[feature]):
        raise ValueError(f"Feature '{feature}' must be numeric for continuous coloring")
    
    # Merge datasets
    merged_df = polygon_df.merge(feature_subset, on='cell')
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between polygon data and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df['fov'].isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Determine global color normalization
    if vmin is None:
        vmin = merged_df[feature].min()
    if vmax is None:
        vmax = merged_df[feature].max()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Create display name for the feature
    display_name = feature if feature != 'feature_value' else feature_column
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
            
        ax = axs[idx]
        
        # Display the FOV image as background if requested
        image_width = None
        image_height = None
        
        import cv2
        if background_img:
            if hasattr(spatioloji_obj, 'get_image'):
                img = spatioloji_obj.get_image(str(fov_id))
                img = cv2.flip(img, 0)  # Flip vertically
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_height, image_width = img.shape[:2]
            else:
                print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df['fov'] == fov_id]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If we don't have image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data["x_local_px"].max() * 1.1
            image_height = fov_data["y_local_px"].max() * 1.1
        
        # Group cells by cell ID
        cell_groups = fov_data.groupby('cell')
        
        # Process each cell
        for cell_id, cell_vertices in cell_groups:
            # Get feature value for this cell
            feature_value = cell_vertices[feature].iloc[0]
            
            # Get color for this value
            color = cmap(norm(feature_value))
            
            # Skip cells with fewer than 3 vertices (can't form a polygon)
            if len(cell_vertices) < 3:
                print(f"Warning: Cell {cell_id} in FOV {fov_id} has fewer than 3 vertices. Skipping.")
                continue
            
            # Create polygon coordinates
            coords = cell_vertices[['x_local_px', 'y_local_px']].values
            
            # Create and add polygon
            polygon = mpl.patches.Polygon(
                coords, 
                closed=True, 
                facecolor=color, 
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=alpha
            )
            ax.add_patch(polygon)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add colorbar to the figure
    if colorbar_position == 'right':
        # Add colorbar to the right of the figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap)), 
                           cax=cbar_ax)
        cbar.set_label(display_name)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust for colorbar space
    else:
        # Add colorbar at the bottom
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap)), 
                           cax=cbar_ax, orientation='horizontal')
        cbar.set_label(display_name)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for colorbar space
    
    # Set title
    plt.suptitle(display_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        feature_name = display_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_polygon_{feature_name}_continuous{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_local_dots_by_categorical(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    dot_size: float = 20,  # New parameter for dot size
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create dot plots colored by categorical features across multiple FOVs using a Spatioloji object.
    Uses center coordinates from cell metadata to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        Spatioloji object containing cell_meta with center coordinates and FOV images.
    feature : str
        The name of the feature to visualize:
        - If feature_df is None, assumes this is a column in spatioloji_obj.adata.obs
        - If feature_df is provided, this is a column in feature_df
        - If feature is "metadata" and feature_column is provided, this uses a column from spatioloji_obj.cell_meta
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images will be used.
    feature_df : pd.DataFrame, optional
        Optional external DataFrame containing cell-level features.
        Must include a 'cell' column and the specified feature column.
    feature_column : str, optional
        If feature="metadata", specifies which column from cell_meta to use.
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save the output figure, by default "./"
    color_map : Dict[str, tuple], optional
        Mapping of category values to colors, by default None.
        If None, colors will be automatically assigned.
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout as (rows, columns), by default None (auto-determined)
    dot_size : float, optional
        Size of the dots representing cells, by default 20
    edge_color : str, optional
        Color of dot edges, by default 'black'
    edge_width : float, optional
        Width of dot edges, by default 0.5
    alpha : float, optional
        Transparency of the dots, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for the main title, by default 24
    filename : str, optional
        Custom filename for the saved figure, by default None (auto-generated)
    dpi : int, optional
        Resolution of the saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    
    Raises
    ------
    ValueError
        If required data is missing or incompatible
    """
    # Check if Spatioloji object has required attributes
    if not hasattr(spatioloji_obj, 'cell_meta') or spatioloji_obj.cell_meta is None:
        raise ValueError("Cell metadata not found in Spatioloji object")
    
    # Create a copy of cell metadata to avoid modifying the original
    cell_meta = spatioloji_obj.cell_meta.copy()
    
    # Check required columns in cell metadata
    required_cols = ['cell', 'fov', 'CenterX_local_px', 'CenterY_local_px']
    missing_cols = [col for col in required_cols if col not in cell_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Spatioloji cell_meta: {missing_cols}")
    
    cell_meta = cell_meta[required_cols]
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if hasattr(spatioloji_obj, 'images') and spatioloji_obj.images:
            fov_ids = list(spatioloji_obj.images.keys())
        else:
            # Fall back to all FOVs in cell_meta
            fov_ids = sorted(cell_meta['fov'].unique())
    
    # Convert FOV IDs to match the type in cell_meta
    if all(isinstance(fid, str) for fid in fov_ids) and all(isinstance(fid, (int, np.integer)) for fid in cell_meta['fov'].unique()):
        # Convert string FOV IDs to integers for matching
        fov_ids = [int(fid) for fid in fov_ids if fid.isdigit()]
    elif all(isinstance(fid, (int, np.integer)) for fid in fov_ids) and all(isinstance(fid, str) for fid in cell_meta['fov'].unique()):
        # Convert integer FOV IDs to strings for matching
        fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature values for each cell
    if feature_df is not None:
        # Using external feature dataframe
        if 'cell' not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain 'cell' and '{feature}' columns")
        
        feature_subset = feature_df[['cell', feature]].copy()
        
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if feature_column not in cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in Spatioloji cell_meta")
            
        feature_subset = cell_meta[['cell', feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        feature = 'feature_value'  # Rename for later use
        
    else:
        # Using adata observations
        if not hasattr(spatioloji_obj, 'adata') or spatioloji_obj.adata is None:
            raise ValueError("Gene expression data (adata) not found in Spatioloji object")
            
        if feature not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Feature '{feature}' not found in Spatioloji adata observations")
            
        feature_subset = spatioloji_obj.adata.obs[[feature]]
        feature_subset['cell'] = spatioloji_obj.adata.obs['cell']
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:  # Too many unique values for categories
            raise ValueError(f"Feature '{feature}' has {n_unique} unique values, which is too many for categorical plotting. Consider binning the data first.")
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Merge datasets
    merged_df = cell_meta.merge(feature_subset, on='cell')
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between cell metadata and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df['fov'].isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Get list of categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors  # Default color palette
        color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Create display name for the feature
    display_name = feature if feature != 'feature_value' else feature_column
    
    # Keep track of categories actually present in the data
    categories_present = set()
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
            
        ax = axs[idx]
        
        # Display the FOV image as background if requested
        image_width = None
        image_height = None
        
        import cv2
        if background_img:
            if hasattr(spatioloji_obj, 'get_image'):
                img = cv2.flip(spatioloji_obj.get_image(str(fov_id)), 0)
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    image_height, image_width = img.shape[:2]
                else:
                    print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df['fov'] == fov_id]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If we don't have image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data["CenterX_local_px"].max() * 1.1
            image_height = fov_data["CenterY_local_px"].max() * 1.1
        
        # Plot dots for each category
        for category in categories:
            category_cells = fov_data[fov_data[feature] == category]
            
            if len(category_cells) == 0:
                continue
                
            categories_present.add(category)
            
            # Plot dots for this category
            ax.scatter(
                category_cells['CenterX_local_px'],
                category_cells['CenterY_local_px'],
                s=dot_size,
                c=[color_map[category]],
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=alpha,
                label=category
            )
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add legend for all actually present categories
    legend_patches = [Patch(facecolor=color_map[cat], 
                          edgecolor=edge_color,
                          alpha=alpha,
                          label=str(cat)) 
                     for cat in sorted(categories_present)]
    
    fig.legend(
        handles=legend_patches, 
        loc='center right', 
        bbox_to_anchor=(1.02, 0.5), 
        title=display_name,
        fontsize=20
    )
    
    # Set title and adjust layout
    plt.suptitle(display_name, fontsize=suptitle_fontsize)
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])  # Adjust for legend space
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        feature_name = display_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_dots_{feature_name}_categorical{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_local_dots_by_features(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    colormap: str = "viridis",
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    dot_size: float = 20,  # New parameter for dot size
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_position: str = 'right'
) -> plt.Figure:
    """
    Create dot plots colored by continuous features across multiple FOVs using a Spatioloji object.
    Uses center coordinates from cell metadata to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : Spatioloji
        Spatioloji object containing cell_meta with center coordinates and FOV images.
    feature : str
        The name of the feature to visualize:
        - If feature_df is None, assumes this is a column in spatioloji_obj.adata.obs
        - If feature_df is provided, this is a column in feature_df
        - If feature is "metadata" and feature_column is provided, this uses a column from spatioloji_obj.cell_meta
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images will be used.
    feature_df : pd.DataFrame, optional
        Optional external DataFrame containing cell-level features.
        Must include a 'cell' column and the specified feature column.
    feature_column : str, optional
        If feature="metadata", specifies which column from cell_meta to use.
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save the output figure, by default "./"
    colormap : str, optional
        Matplotlib colormap name to use, by default "viridis"
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout as (rows, columns), by default None (auto-determined)
    dot_size : float, optional
        Size of the dots representing cells, by default 20
    edge_color : str, optional
        Color of dot edges, by default 'black'
    edge_width : float, optional
        Width of polygon edges, by default 0.5
    alpha : float, optional
        Transparency of the dots, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for the main title, by default 24
    filename : str, optional
        Custom filename for the saved figure, by default None (auto-generated)
    dpi : int, optional
        Resolution of the saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    vmin : float, optional
        Minimum value for color normalization, by default None (auto-determined)
    vmax : float, optional
        Maximum value for color normalization, by default None (auto-determined)
    colorbar_position : str, optional
        Position of the colorbar, by default 'right'
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    
    Raises
    ------
    ValueError
        If required data is missing or incompatible
    """
    # Check if Spatioloji object has required attributes
    if not hasattr(spatioloji_obj, 'cell_meta') or spatioloji_obj.cell_meta is None:
        raise ValueError("Cell metadata not found in Spatioloji object")
    
    # Create a copy of cell metadata to avoid modifying the original
    cell_meta = spatioloji_obj.cell_meta.copy()
    
    # Check required columns in cell metadata
    required_cols = ['cell', 'fov', 'CenterX_local_px', 'CenterY_local_px']
    missing_cols = [col for col in required_cols if col not in cell_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Spatioloji cell_meta: {missing_cols}")
    cell_meta = cell_meta[required_cols]
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if hasattr(spatioloji_obj, 'images') and spatioloji_obj.images:
            fov_ids = list(spatioloji_obj.images.keys())
        else:
            # Fall back to all FOVs in cell_meta
            fov_ids = sorted(cell_meta['fov'].unique())
    
    # Convert FOV IDs to match the type in cell_meta
    if all(isinstance(fid, str) for fid in fov_ids) and all(isinstance(fid, (int, np.integer)) for fid in cell_meta['fov'].unique()):
        # Convert string FOV IDs to integers for matching
        fov_ids = [int(fid) for fid in fov_ids if fid.isdigit()]
    elif all(isinstance(fid, (int, np.integer)) for fid in fov_ids) and all(isinstance(fid, str) for fid in cell_meta['fov'].unique()):
        # Convert integer FOV IDs to strings for matching
        fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature values for each cell
    if feature_df is not None:
        # Using external feature dataframe
        if 'cell' not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain 'cell' and '{feature}' columns")
        
        feature_subset = feature_df[['cell', feature]].copy()
        
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if feature_column not in cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in Spatioloji cell_meta")
            
        feature_subset = cell_meta[['cell', feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        feature = 'feature_value'  # Rename for later use
        
    else:
        # Using adata observations
        if not hasattr(spatioloji_obj, 'adata') or spatioloji_obj.adata is None:
            raise ValueError("Gene expression data (adata) not found in Spatioloji object")
            
        if feature not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Feature '{feature}' not found in Spatioloji adata observations")
            
        feature_subset = spatioloji_obj.adata.obs[[feature]]
        feature_subset['cell'] = spatioloji_obj.adata.obs['cell']
    
    # Check if the feature is numeric
    if not pd.api.types.is_numeric_dtype(feature_subset[feature]):
        raise ValueError(f"Feature '{feature}' must be numeric for continuous coloring")
    
    # Merge datasets
    merged_df = cell_meta.merge(feature_subset, on='cell')
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between cell metadata and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df['fov'].isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Determine global color normalization
    if vmin is None:
        vmin = merged_df[feature].min()
    if vmax is None:
        vmax = merged_df[feature].max()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Create display name for the feature
    display_name = feature if feature != 'feature_value' else feature_column
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
            
        ax = axs[idx]
        
        # Display the FOV image as background if requested
        image_width = None
        image_height = None
        
        import cv2
        if background_img:
            if hasattr(spatioloji_obj, 'get_image'):
                img = spatioloji_obj.get_image(str(fov_id))
                img = cv2.flip(img, 0)  # Flip vertically
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_height, image_width = img.shape[:2]
            else:
                print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df['fov'] == fov_id]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If we don't have image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data["CenterX_local_px"].max() * 1.1
            image_height = fov_data["CenterY_local_px"].max() * 1.1
        
        # Create a scatter plot with points colored by the feature value
        scatter = ax.scatter(
            fov_data['CenterX_local_px'],
            fov_data['CenterY_local_px'],
            s=dot_size,
            c=fov_data[feature],
            cmap=colormap,
            norm=norm,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=alpha
        )
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add colorbar to the figure
    if colorbar_position == 'right':
        # Add colorbar to the right of the figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap)), 
                           cax=cbar_ax)
        cbar.set_label(display_name)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust for colorbar space
    else:
        # Add colorbar at the bottom
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap)), 
                           cax=cbar_ax, orientation='horizontal')
        cbar.set_label(display_name)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for colorbar space
    
    # Set title
    plt.suptitle(display_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        feature_name = display_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_dots_{feature_name}_continuous{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig





