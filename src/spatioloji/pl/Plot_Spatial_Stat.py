import numpy as np
import re
from shapely.geometry import Polygon
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

def plot_polygons_with_gradient(gdf, 
                               category_column, 
                               gradient_column,
                               edge_color_map=None,
                               cmap_name='viridis',
                               figsize=(10, 8),
                               xlim=None,
                               ylim=None,
                               title="Gradient Facecolor with Categorical Edgecolor",
                               legend_title="Edge Category",
                               legend_loc="lower right",
                               legend_bbox_to_anchor=(1.5, 0.5),
                               save_dir=None,
                               filename='polygon_plot.png',
                               dpi=300,
                               font_size=12,
                               title_size=14,
                               legend_size=10,
                               colorbar_size=12):
    """
    Plot polygons with custom edge colors based on categories and gradient fill based on a numeric attribute.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing polygons to plot
    category_column : str
        Column name with categorical values to determine edge colors
    gradient_column : str
        Column name with numeric values to determine gradient colors
    edge_color_map : dict, optional
        Dictionary mapping category values to edge colors
    cmap_name : str, optional
        Matplotlib colormap name for the gradient
    figsize : tuple, optional
        Figure size as (width, height)
    xlim : tuple, optional
        x-axis limits as (min, max)
    ylim : tuple, optional
        y-axis limits as (min, max)
    title : str, optional
        Plot title
    legend_title : str, optional
        Title for the legend
    legend_loc : str, optional
        Legend location
    legend_bbox_to_anchor : tuple, optional
        Legend bbox_to_anchor parameter
    save_dir : str, optional
        Directory to save the figure. If None, figure is not saved.
    filename : str, optional
        Filename to save the figure
    dpi : int, optional
        Resolution of the saved figure
    font_size : int, optional
        Base font size for axis labels
    title_size : int, optional
        Font size for the title
    legend_size : int, optional
        Font size for the legend
    colorbar_size : int, optional
        Font size for the colorbar
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import numpy as np
    import os
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create a copy to avoid modifying the original
    gdf_plot = gdf.copy()
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': title_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': legend_size,
        'legend.title_fontsize': legend_size + 2
    })
    
    # Default edge color map if none provided
    if edge_color_map is None:
        # Get unique values from category column
        categories = gdf_plot[category_column].unique()
        # Create default colors
        default_colors = ['black', 'red', 'blue', 'green', 'purple', 'orange']
        # Map categories to colors
        edge_color_map = {cat: default_colors[i % len(default_colors)] 
                          for i, cat in enumerate(categories)}
    
    # Map edge colors
    gdf_plot['edgecolor'] = gdf_plot[category_column].map(edge_color_map)
    
    # Colormap normalization
    vmin = gdf_plot[gradient_column].min()
    vmax = gdf_plot[gradient_column].max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    
    # Prepare the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    patches = []
    facecolors = []
    edgecolors = []
    
    # Loop over polygons
    for _, row in gdf_plot.iterrows():
        geom = row.geometry
        if geom and hasattr(geom, 'exterior') and geom.exterior:
            coords = np.array(geom.exterior.coords)
            patch = MplPolygon(coords, closed=True)
            patches.append(patch)
            facecolors.append(cmap(norm(row[gradient_column])))
            edgecolors.append(row['edgecolor'])
    
    # Create patch collection
    pc = PatchCollection(patches, facecolor=facecolors, edgecolor=edgecolors, linewidth=1.2)
    ax.add_collection(pc)
    
    # Add colorbar
    # Create a divider for the existing axes
    divider = make_axes_locatable(ax)

    # Append a new axes to the right of the current one (same height)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # size = width of colorbar

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, cax=cax)
    cbar.set_label(gradient_column, size=colorbar_size)
    cbar.ax.tick_params(labelsize=colorbar_size)
    
    # Add legend for edge categories
    legend_patches = [mpatches.Patch(facecolor='white', edgecolor=color, label=cat)
                      for cat, color in edge_color_map.items()]
    

    ax.legend(handles=legend_patches, title=legend_title, loc=legend_loc,
              bbox_to_anchor=legend_bbox_to_anchor, fontsize=legend_size)
    
    # Set axis limits
    if xlim is None:
        xlim = gdf_plot.total_bounds[[0, 2]]
    if ylim is None:
        ylim = gdf_plot.total_bounds[[1, 3]]
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    
    # Increase text sizes for tick labels
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    # Finalize plot
    plt.title(title, fontsize=title_size, pad=20)
    plt.tight_layout()
    
    # Save the figure if save_dir is provided
    if save_dir is not None:
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Full path to save the figure
        save_path = os.path.join(save_dir, filename)
        
        # Save figure
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax





