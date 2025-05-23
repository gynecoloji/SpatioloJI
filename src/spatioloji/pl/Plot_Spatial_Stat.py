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

def plot_dots_with_gradient_shapes(df, 
                           x_column='CenterX_local_px',
                           y_column='CenterY_local_px',
                           category_column=None, 
                           gradient_column=None,
                           marker_size=50,
                           category_marker_map=None,
                           cmap_name='viridis',
                           figsize=(10, 8),
                           xlim=None,
                           ylim=None,
                           title="Dots with Gradient Color",
                           legend_title="Category",
                           legend_loc="lower right",
                           legend_bbox_to_anchor=(1.5, 0.5),
                           save_dir=None,
                           filename='dot_plot.png',
                           dpi=300,
                           font_size=12,
                           title_size=14,
                           legend_size=10,
                           colorbar_size=12):
    """
    Plot points/dots with colors based on a gradient and optional categorization by marker shape.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing point coordinates and attributes
    x_column : str
        Column name for x-coordinates (default: 'CenterX_local_px')
    y_column : str
        Column name for y-coordinates (default: 'CenterY_local_px')
    category_column : str, optional
        Column name with categorical values to determine marker shapes
    gradient_column : str
        Column name with numeric values to determine gradient colors
    marker_size : int or str, optional
        Size of markers or column name for variable sizes
    category_marker_map : dict, optional
        Dictionary mapping category values to marker shapes
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
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    import numpy as np
    import os
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create a copy to avoid modifying the original
    df_plot = df.copy()
    
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
    
    # Extract point coordinates from the specified columns
    x = df_plot[x_column].values
    y = df_plot[y_column].values
    
    # Handle marker sizes - either fixed or from a column
    if isinstance(marker_size, str) and marker_size in df_plot.columns:
        sizes = df_plot[marker_size].values
    else:
        sizes = marker_size
    
    # Prepare categorical markers if needed
    if category_column is not None:
        # Default marker map if none provided
        if category_marker_map is None:
            # Standard matplotlib markers
            default_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
            categories = df_plot[category_column].unique()
            category_marker_map = {cat: default_markers[i % len(default_markers)] 
                              for i, cat in enumerate(categories)}
        
        # Map categories to markers
        markers = df_plot[category_column].map(category_marker_map)
    else:
        # Use default marker for all points
        markers = ['o'] * len(df_plot)
        category_marker_map = {'All points': 'o'}
    
    # Colormap normalization
    vmin = df_plot[gradient_column].min()
    vmax = df_plot[gradient_column].max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    
    # Prepare the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each category separately to create proper legend
    if category_column is not None:
        for category, marker in category_marker_map.items():
            category_mask = df_plot[category_column] == category
            if not any(category_mask):
                continue
            
            # Plot this category's points
            scatter = ax.scatter(
                x[category_mask],
                y[category_mask],
                c=df_plot.loc[category_mask, gradient_column],
                s=sizes if isinstance(sizes, (int, float)) else sizes[category_mask],
                marker=marker,
                cmap=cmap,
                norm=norm,
                label=category,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )
    else:
        # Plot all points with same marker
        scatter = ax.scatter(
            x, y, 
            c=df_plot[gradient_column],
            s=sizes,
            marker='o',
            cmap=cmap, 
            norm=norm,
            label='All points',
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, cax=cax)
    cbar.set_label(gradient_column, size=colorbar_size)
    cbar.ax.tick_params(labelsize=colorbar_size)
    
    # Add category legend if needed
    if category_column is not None:
        ax.legend(title=legend_title, loc=legend_loc,
                  bbox_to_anchor=legend_bbox_to_anchor, fontsize=legend_size)
    
    # Set axis limits
    if xlim is None:
        # Buffer the x limits a bit
        x_range = max(x) - min(x)
        xlim = (min(x) - 0.05 * x_range, max(x) + 0.05 * x_range)
    if ylim is None:
        # Buffer the y limits a bit
        y_range = max(y) - min(y)
        ylim = (min(y) - 0.05 * y_range, max(y) + 0.05 * y_range)
        
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




def plot_dots_with_gradient_edgecolors(df, 
                           x_column='CenterX_local_px',
                           y_column='CenterY_local_px',
                           category_column=None, 
                           gradient_column=None,
                           marker_size=50,
                           marker_shape='o',
                           category_edge_color_map=None,
                           cmap_name='viridis',
                           figsize=(10, 8),
                           xlim=None,
                           ylim=None,
                           title="Dots with Gradient Color",
                           legend_title="Category",
                           legend_loc="lower right",
                           legend_bbox_to_anchor=(1.5, 0.5),
                           save_dir=None,
                           filename='dot_plot.png',
                           dpi=300,
                           font_size=12,
                           title_size=14,
                           legend_size=10,
                           colorbar_size=12,
                           edge_linewidth=1.5):
    """
    Plot points/dots with colors based on a gradient and categories represented by edge colors.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing point coordinates and attributes
    x_column : str
        Column name for x-coordinates (default: 'CenterX_local_px')
    y_column : str
        Column name for y-coordinates (default: 'CenterY_local_px')
    category_column : str, optional
        Column name with categorical values to determine edge colors
    gradient_column : str
        Column name with numeric values to determine gradient fill colors
    marker_size : int or str, optional
        Size of markers or column name for variable sizes
    marker_shape : str, optional
        Shape of markers (default: 'o' for circle)
    category_edge_color_map : dict, optional
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
    edge_linewidth : float, optional
        Width of the marker edges
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    import matplotlib.patches as mpatches
    import numpy as np
    import os
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create a copy to avoid modifying the original
    df_plot = df.copy()
    
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
    
    # Extract point coordinates from the specified columns
    x = df_plot[x_column].values
    y = df_plot[y_column].values
    
    # Handle marker sizes - either fixed or from a column
    if isinstance(marker_size, str) and marker_size in df_plot.columns:
        sizes = df_plot[marker_size].values
    else:
        sizes = marker_size
    
    # Default edge color map if none provided
    if category_column is not None and category_edge_color_map is None:
        # Default colors for categories
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        categories = df_plot[category_column].unique()
        category_edge_color_map = {cat: default_colors[i % len(default_colors)] 
                              for i, cat in enumerate(categories)}
    
    # Colormap normalization for gradient fill colors
    vmin = df_plot[gradient_column].min()
    vmax = df_plot[gradient_column].max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    
    # Prepare the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points based on categories
    if category_column is not None:
        for category, edge_color in category_edge_color_map.items():
            category_mask = df_plot[category_column] == category
            if not any(category_mask):
                continue
            
            # Plot this category's points with specific edge color
            scatter = ax.scatter(
                x[category_mask],
                y[category_mask],
                c=df_plot.loc[category_mask, gradient_column],
                s=sizes if isinstance(sizes, (int, float)) else sizes[category_mask],
                marker=marker_shape,
                cmap=cmap,
                norm=norm,
                edgecolors=edge_color,
                linewidths=edge_linewidth,
                alpha=0.85
            )
    else:
        # Plot all points with default edge color (black)
        scatter = ax.scatter(
            x, y, 
            c=df_plot[gradient_column],
            s=sizes,
            marker=marker_shape,
            cmap=cmap, 
            norm=norm,
            edgecolors='black',
            linewidths=edge_linewidth,
            alpha=0.85
        )
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, cax=cax)
    cbar.set_label(gradient_column, size=colorbar_size)
    cbar.ax.tick_params(labelsize=colorbar_size)
    
    # Add category legend if needed
    if category_column is not None:
        # Create custom legend handles
        legend_patches = [
            mpatches.Patch(
                facecolor='lightgray',
                edgecolor=color,
                linewidth=edge_linewidth,
                label=category
            ) for category, color in category_edge_color_map.items()
            if any(df_plot[category_column] == category)
        ]
        
        ax.legend(
            handles=legend_patches,
            title=legend_title,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=legend_size
        )
    
    # Set axis limits
    if xlim is None:
        # Buffer the x limits a bit
        x_range = max(x) - min(x)
        xlim = (min(x) - 0.05 * x_range, max(x) + 0.05 * x_range)
    if ylim is None:
        # Buffer the y limits a bit
        y_range = max(y) - min(y)
        ylim = (min(y) - 0.05 * y_range, max(y) + 0.05 * y_range)
        
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

def visualize_ripleys_k(ripley_results, title=None, figsize=(10, 6), save_path=None):
    """
    Visualize Ripley's K function results with confidence envelopes.
    
    Args:
        ripley_results: DataFrame returned by calculate_ripleys_k function
        title: Optional title for the plot (default: "Ripley's L Function Analysis")
        figsize: Tuple specifying figure size (width, height) in inches
        save_path: Optional path to save the figure (e.g., 'ripley_plot.png')
        
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the observed L function
    ax.plot(ripley_results['distance'], ripley_results['L'], 
            color='blue', linewidth=2, label='Observed L(r)')
    
    # Plot confidence envelopes if available
    if 'L_low' in ripley_results.columns and 'L_high' in ripley_results.columns:
        ax.fill_between(
            ripley_results['distance'],
            ripley_results['L_low'],
            ripley_results['L_high'],
            color='gray', alpha=0.3,
            label='Confidence Envelope'
        )
    
    # Add horizontal line at y=0 (theoretical expectation under CSR)
    ax.axhline(y=0, color='red', linestyle='--', 
               label='Theoretical L(r) under CSR')
    
    # Identify regions of significant clustering or dispersion
    if 'L_high' in ripley_results.columns:
        # Mark regions of significant clustering (above upper envelope)
        clustering_mask = ripley_results['L'] > ripley_results['L_high']
        significant_distances = ripley_results.loc[clustering_mask, 'distance']
        
        if len(significant_distances) > 0:
            # Get continuous ranges of significant distances
            ranges = []
            start = None
            
            for i, d in enumerate(significant_distances):
                if start is None:
                    start = d
                elif i == len(significant_distances) - 1 or significant_distances.iloc[i] != significant_distances.iloc[i-1] + 1:
                    ranges.append((start, d))
                    start = None
            
            # Add text annotations for significant clustering
            y_pos = max(ripley_results['L']) * 0.9
            for r_start, r_end in ranges:
                mid_point = (r_start + r_end) / 2
                ax.annotate(
                    '',
                    xy=(mid_point, y_pos),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='green')
                )
    
    # Add labels and title
    ax.set_xlabel('Distance (r)', fontsize=12)
    ax.set_ylabel('L(r)', fontsize=12)
    ax.set_title(title or "Ripley's L Function Analysis", fontsize=14)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    # Add interpretation guide
    interpretation_text = (
        "Interpretation:\n"
        "• L(r) > 0: Clustering at distance r\n"
        "• L(r) < 0: Dispersion at distance r\n"
        "• L(r) within envelope: Pattern consistent with CSR"
    )
    
    # Position the text in the lower right corner
    plt.figtext(1.05, -0.2, interpretation_text, 
                horizontalalignment='right',
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add summary statistics
    max_l = ripley_results['L'].max()
    max_l_distance = ripley_results.loc[ripley_results['L'].idxmax(), 'distance']
    min_l = ripley_results['L'].min()
    min_l_distance = ripley_results.loc[ripley_results['L'].idxmin(), 'distance']
    
    stats_text = (
        f"Max clustering at r = {max_l_distance:.2f} (L = {max_l:.3f})\n"
        f"Max dispersion at r = {min_l_distance:.2f} (L = {min_l:.3f})"
    )
    
    plt.figtext(0.03, -0.1, stats_text,
                horizontalalignment='left',
                verticalalignment='bottom',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_ripleys_multitype(results_dict, plot_type='L', figsize=(12, 8), 
                          title=None, cell_types=None, legend_loc='best',
                          show_confidence=True, show_csr=True, 
                          colors=None, line_styles=None,
                          highlight_significant=True, y_label=None,
                          save_path=None, dpi=300):
    """
    Plot Ripley's K or L function results for multiple cell types.
    
    Args:
        results_dict: Dictionary from calculate_ripleys_k_by_cell_type mapping cell types to results
        plot_type: Which function to plot ('K' or 'L')
        figsize: Figure size tuple (width, height)
        title: Custom title for the plot (None for default)
        cell_types: List of cell types to include (None for all)
        legend_loc: Location of the legend ('best', 'upper right', etc.)
        show_confidence: Whether to show confidence envelopes
        show_csr: Whether to show the CSR expectation line
        colors: Dictionary mapping cell types to colors (or None to use default colormap)
        line_styles: Dictionary mapping cell types to line styles (or None for default)
        highlight_significant: Whether to highlight regions above/below confidence envelopes
        y_label: Custom y-axis label (None for default)
        save_path: Path to save the figure (None to not save)
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter cell types if specified
    if cell_types:
        plot_results = {k: results_dict[k] for k in cell_types if k in results_dict}
    else:
        plot_results = results_dict
    
    # Set default colors if not provided
    if colors is None:
        # Use tab10 colormap by default
        cmap = cm.get_cmap('tab10')
        colors = {cell_type: cmap(i % 10) for i, cell_type in enumerate(plot_results.keys())}
    else:
        # For any cell types without specified colors, assign default colors
        cmap = cm.get_cmap('tab10')
        for i, cell_type in enumerate(plot_results.keys()):
            if cell_type not in colors:
                colors[cell_type] = cmap(i % 10)
    
    # Set default line styles if not provided
    if line_styles is None:
        line_styles = {cell_type: '-' for cell_type in plot_results.keys()}
    
    # Track legend elements
    legend_elements = []
    
    # For storing significant regions
    sig_regions = {}
    
    # Plot each cell type
    for cell_type, result in plot_results.items():
        color = colors[cell_type]
        line_style = line_styles.get(cell_type, '-')
        
        # Plot the main function line
        line = ax.plot(
            result['distance'], 
            result[plot_type], 
            color=color, 
            linestyle=line_style,
            linewidth=2, 
            label=cell_type
        )[0]
        
        # Add to legend elements
        legend_elements.append(Line2D([0], [0], color=color, lw=2, 
                                      linestyle=line_style,
                                      label=cell_type))
        
        # Plot confidence envelopes if available and requested
        has_confidence = (f'{plot_type}_low' in result.columns and 
                          f'{plot_type}_high' in result.columns)
        
        if show_confidence and has_confidence:
            ax.fill_between(
                result['distance'],
                result[f'{plot_type}_low'],
                result[f'{plot_type}_high'],
                color=color, 
                alpha=0.15
            )
            
            # Identify significant regions
            if highlight_significant:
                # Find where observed values exceed confidence envelope
                above_high = result[plot_type] > result[f'{plot_type}_high']
                below_low = result[plot_type] < result[f'{plot_type}_low']
                
                # Store regions
                sig_regions[cell_type] = {
                    'above': (above_high, color),
                    'below': (below_low, color)
                }
    
    # Highlight significant regions if requested
    if highlight_significant:
        for cell_type, regions in sig_regions.items():
            # Get result for this cell type
            result = plot_results[cell_type]
            distances = result['distance']
            
            # Above high confidence (clustering)
            if regions['above'][0].any():
                above_mask = regions['above'][0]
                color = regions['above'][1]
                
                # Draw markers at significant points
                ax.plot(
                    distances[above_mask],
                    result[plot_type][above_mask],
                    'o', 
                    color=color, 
                    markersize=4,
                    alpha=0.7
                )
                
                # Calculate midpoint of first significant region for annotation
                if above_mask.any():
                    runs = np.where(np.diff(np.concatenate(([False], above_mask, [False]))))[0].reshape(-1, 2)
                    if len(runs) > 0:
                        start_idx, end_idx = runs[0]
                        mid_idx = start_idx + (end_idx - start_idx) // 2
                        mid_point = distances[mid_idx]
                        
                        # Add arrowhead annotation
                        y_pos = result[plot_type][mid_idx]
                        ax.annotate(
                            f"{cell_type}\nclustering",
                            xy=(mid_point, y_pos),
                            xytext=(0, 15),
                            textcoords="offset points",
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            color=color,
                            arrowprops=dict(
                                arrowstyle="->",
                                color=color,
                                alpha=0.7
                            )
                        )
            
            # Below low confidence (dispersion)
            if regions['below'][0].any():
                below_mask = regions['below'][0]
                color = regions['below'][1]
                
                # Draw markers at significant points
                ax.plot(
                    distances[below_mask],
                    result[plot_type][below_mask],
                    's', 
                    color=color, 
                    markersize=4,
                    alpha=0.7
                )
                
                # Calculate midpoint of first significant region for annotation
                if below_mask.any():
                    runs = np.where(np.diff(np.concatenate(([False], below_mask, [False]))))[0].reshape(-1, 2)
                    if len(runs) > 0:
                        start_idx, end_idx = runs[0]
                        mid_idx = start_idx + (end_idx - start_idx) // 2
                        mid_point = distances[mid_idx]
                        
                        # Add arrowhead annotation
                        y_pos = result[plot_type][mid_idx]
                        ax.annotate(
                            f"{cell_type}\ndispersion",
                            xy=(mid_point, y_pos),
                            xytext=(0, -15),
                            textcoords="offset points",
                            ha='center',
                            va='top',
                            fontsize=8,
                            color=color,
                            arrowprops=dict(
                                arrowstyle="->",
                                color=color,
                                alpha=0.7
                            )
                        )
    
    # Add reference line for CSR expectation
    if show_csr:
        if plot_type == 'L':
            csr_line = ax.axhline(
                y=0, 
                color='black', 
                linestyle='--', 
                linewidth=1.5,
                alpha=0.7,
                zorder=1
            )
            legend_elements.append(Line2D([0], [0], color='black', lw=1.5, 
                                         linestyle='--', label='CSR expectation'))
        elif plot_type == 'K':
            # For K function, the CSR expectation is πr²
            distances = next(iter(plot_results.values()))['distance']
            theoretical_k = np.pi * distances**2
            csr_line = ax.plot(
                distances, 
                theoretical_k,
                color='black', 
                linestyle='--', 
                linewidth=1.5,
                alpha=0.7,
                zorder=1,
                label='CSR expectation'
            )
            legend_elements.append(Line2D([0], [0], color='black', lw=1.5, 
                                         linestyle='--', label='CSR expectation (πr²)'))
    
    # Set axis labels
    ax.set_xlabel('Distance (r)', fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    else:
        ax.set_ylabel(f'{plot_type}(r)', fontsize=12)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Ripley's {plot_type} Function by Cell Type", fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc=legend_loc, framealpha=0.9)
    
    # Add interpretation guide
    if plot_type == 'L':
        interpretation_text = (
            "Interpretation:\n"
            "• L(r) > 0: Clustering at distance r\n"
            "• L(r) < 0: Dispersion at distance r\n"
            "• L(r) within envelope: Pattern consistent with CSR"
        )
    else:  # K function
        interpretation_text = (
            "Interpretation:\n"
            "• K(r) > πr²: Clustering at distance r\n"
            "• K(r) < πr²: Dispersion at distance r\n"
            "• K(r) within envelope: Pattern consistent with CSR"
        )
    
    # Position the text in the lower right corner
    plt.figtext(0.97, -0.1, interpretation_text, 
               horizontalalignment='right',
               verticalalignment='bottom',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def visualize_cross_functions(result_df, plot_type='L', figsize=(10, 6), 
                             cell_type1=None, cell_type2=None, save_path=None):
    """
    Visualize the results of Cross-K/L function analysis.
    
    Args:
        result_df: DataFrame returned by calculate_cross_k_function
        plot_type: 'K' or 'L' to specify which function to plot (default: 'L')
        figsize: Figure size as (width, height) tuple (default: (10, 6))
        cell_type1: Name of the first cell type (for title)
        cell_type2: Name of the second cell type (for title)
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Determine which columns to plot based on plot_type
    if plot_type == 'K':
        y_col = 'cross_k'
        theoretical_col = 'theoretical_k'
        y_label = 'Cross-K(r)'
        title_prefix = 'Cross-K Function'
        ci_low = 'cross_k_low' if 'cross_k_low' in result_df.columns else None
        ci_high = 'cross_k_high' if 'cross_k_high' in result_df.columns else None
    elif plot_type == 'L':
        y_col = 'cross_l'
        theoretical_col = None  # For L function, theoretical is just 0
        y_label = 'Cross-L(r)'
        title_prefix = 'Cross-L Function'
        ci_low = 'cross_l_low' if 'cross_l_low' in result_df.columns else None
        ci_high = 'cross_l_high' if 'cross_l_high' in result_df.columns else None
    else:
        raise ValueError("plot_type must be 'K' or 'L'")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the cross function
    ax.plot(result_df['distance'], result_df[y_col], '-', color='blue', linewidth=2, label='Observed')
    
    # Plot theoretical line
    if theoretical_col:
        ax.plot(result_df['distance'], result_df[theoretical_col], '--', color='black', 
                linewidth=1, label='Theoretical (CSR)')
    else:
        # For L function, theoretical line is at y=0
        ax.axhline(y=0, linestyle='--', color='black', linewidth=1, label='Theoretical (CSR)')
    
    # Plot confidence intervals if available
    if ci_low and ci_high and ci_low in result_df.columns and ci_high in result_df.columns:
        ax.fill_between(result_df['distance'], result_df[ci_low], result_df[ci_high], 
                       color='gray', alpha=0.3, label='95% Confidence Envelope')
    
    # Add shading for interpretation
    if plot_type == 'L':
        # Add light red/blue shading for attraction/repulsion areas
        ax.fill_between(result_df['distance'], 0, result_df['cross_l'].max() + 0.1, 
                       where=result_df['cross_l'] > 0, color='red', alpha=0.1, label='Attraction')
        ax.fill_between(result_df['distance'], result_df['cross_l'].min() - 0.1, 0, 
                       where=result_df['cross_l'] < 0, color='blue', alpha=0.1, label='Repulsion')
    
    # Highlight significant regions if available
    if 'is_significant' in result_df.columns:
        significant_regions = result_df[result_df['is_significant']]
        if not significant_regions.empty:
            ax.scatter(significant_regions['distance'], significant_regions[y_col], 
                      color='red', s=30, marker='o', label='Significant')
    
    # Set labels and title
    ax.set_xlabel('Distance (pixels)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    # Create title based on provided cell types
    if cell_type1 and cell_type2:
        title = f'{title_prefix}: {cell_type1} to {cell_type2}'
    else:
        title = f'{title_prefix}'
    ax.set_title(title, fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)
    
    # Add annotations explaining interpretations
    if plot_type == 'L':
        interpretation_text = """
        • L(r) > 0: Attraction (cell types tend to be near each other)
        • L(r) = 0: Spatial randomness (no interaction)
        • L(r) < 0: Repulsion (cell types tend to avoid each other)
        """
    else:  # K function
        interpretation_text = """
        • K(r) > πr²: Attraction (cell types tend to be near each other)
        • K(r) = πr²: Spatial randomness (no interaction)
        • K(r) < πr²: Repulsion (cell types tend to avoid each other)
        """
    
    # Place text in the figure
    plt.figtext(0.5, -0.2, interpretation_text, fontsize=9,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add summary statistics
    if 'interpretation' in result_df.columns:
        # Get the most common interpretation
        interpretation_counts = result_df['interpretation'].value_counts()
        most_common = interpretation_counts.index[0]
        ratio = interpretation_counts[0] / len(result_df) * 100
        
        summary_text = f"Overall pattern: {most_common} ({ratio:.1f}% of distances)"
        plt.figtext(0.5, -0.05, summary_text, fontsize=10, 
                   ha='center', bbox=dict(facecolor='yellow', alpha=0.2, boxstyle='round'))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_j_function(result_df, figsize=(10, 8), cell_type=None, 
                         color='blue', show_components=False, save_path=None):
    """
    Create a comprehensive visualization of Baddeley's J-function results.
    
    Args:
        result_df: DataFrame returned by calculate_j_function
        figsize: Figure size as (width, height) tuple (default: (10, 8))
        cell_type: Name of the cell type analyzed (for title)
        color: Color to use for the J-function line (default: 'blue')
        show_components: Whether to show G and F functions in additional panels (default: False)
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    
    # Determine if we need a multi-panel or single-panel figure
    if show_components:
        # Create figure with 3 panels using GridSpec for more control
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        
        # Main J-function plot (larger)
        ax_j = plt.subplot(gs[0, :])
        # G-function plot
        ax_g = plt.subplot(gs[1, 0])
        # F-function plot
        ax_f = plt.subplot(gs[1, 1])
    else:
        # Create single panel figure
        fig, ax_j = plt.subplots(figsize=figsize)
    
    # Plot J-function with custom color
    ax_j.plot(result_df['distance'], result_df['j_empirical'], '-', color=color, 
             linewidth=2.5, label=f'Observed J-function{" for " + cell_type if cell_type else ""}')
    
    # Plot theoretical J-function (always 1 for CSR)
    ax_j.axhline(y=1, linestyle='--', color='black', linewidth=1.5, 
                label='CSR (Complete Spatial Randomness)')
    
    # Add confidence envelopes if available
    if 'j_low' in result_df.columns and 'j_high' in result_df.columns:
        ax_j.fill_between(result_df['distance'], result_df['j_low'], result_df['j_high'], 
                         color=color, alpha=0.2, label='95% Confidence Envelope')
    
    # Add shading for clustering/regularity regions
    y_max = max(2, result_df['j_empirical'].max() * 1.1)
    ax_j.fill_between(result_df['distance'], 0, 1, color='red', alpha=0.1, label='Clustering')
    ax_j.fill_between(result_df['distance'], 1, y_max, color='blue', alpha=0.1, label='Regularity')
    
    # Mark significant points if available
    if 'is_significant' in result_df.columns:
        # Points where pattern is significantly clustered
        clustered = result_df[result_df['is_clustered']]
        if not clustered.empty:
            ax_j.scatter(clustered['distance'], clustered['j_empirical'], 
                        color='red', s=40, marker='o', label='Significant Clustering')
        
        # Points where pattern is significantly regular
        regular = result_df[result_df['is_regular']]
        if not regular.empty:
            ax_j.scatter(regular['distance'], regular['j_empirical'], 
                        color='blue', s=40, marker='s', label='Significant Regularity')
    
    # Set labels and title for J-function
    ax_j.set_xlabel('Distance (pixels)', fontsize=12)
    ax_j.set_ylabel('J(r)', fontsize=12)
    
    title = 'Baddeley\'s J-function'
    if cell_type:
        title += f': {cell_type}'
    ax_j.set_title(title, fontsize=14, fontweight='bold')
    
    # Set y-axis limits for J-function
    ax_j.set_ylim(0, y_max)
    
    # Add grid
    ax_j.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend for J-function
    ax_j.legend(loc='best', frameon=True, fancybox=True, framealpha=0.7)
    
    # Add annotation explaining J-function interpretation
    j_text = """
    J-function Interpretation:
    • J(r) < 1: Clustering (cells tend to be grouped together)
    • J(r) = 1: Complete Spatial Randomness (CSR)
    • J(r) > 1: Regularity (cells more evenly spaced than random)
    
    J(r) = (1-G(r))/(1-F(r)) combines:
    • G(r): Nearest neighbor distance distribution
    • F(r): Empty space function
    """
    plt.figtext(0, -0.2, j_text, fontsize=9,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # If showing component functions (G and F)
    if show_components:
        # Get a slightly darker and lighter shade of the same color for components
        component_colors = {'g': color, 'f': color}
        
        # Plot G-function 
        ax_g.plot(result_df['distance'], result_df['g_empirical'], '-', color=component_colors['g'], 
                 linewidth=2, label='G(r)')
        ax_g.plot(result_df['distance'], result_df['theoretical_csr'], '--', color='black', 
                 linewidth=1, label='CSR')
        ax_g.set_xlabel('Distance (pixels)', fontsize=10)
        ax_g.set_ylabel('G(r)', fontsize=10)
        ax_g.set_title('G-function (Nearest Neighbor)', fontsize=11)
        ax_g.grid(True, linestyle='--', alpha=0.5)
        ax_g.legend(loc='best', fontsize=8)
        
        # Add annotation for G-function
        g_text = "G(r): Proportion of cells with\nnearest neighbor ≤ distance r"
        ax_g.text(0.02, 0.95, g_text, transform=ax_g.transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot F-function
        ax_f.plot(result_df['distance'], result_df['f_empirical'], '-', color=component_colors['f'], 
                 linewidth=2, label='F(r)')
        ax_f.plot(result_df['distance'], result_df['theoretical_csr'], '--', color='black', 
                 linewidth=1, label='CSR')
        ax_f.set_xlabel('Distance (pixels)', fontsize=10)
        ax_f.set_ylabel('F(r)', fontsize=10)
        ax_f.set_title('F-function (Empty Space)', fontsize=11)
        ax_f.grid(True, linestyle='--', alpha=0.5)
        ax_f.legend(loc='best', fontsize=8)
        
        # Add annotation for F-function
        f_text = "F(r): Proportion of empty space\npoints with nearest cell ≤ distance r"
        ax_f.text(0.02, 0.95, f_text, transform=ax_f.transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add summary statistics if interpretation column is available
    if 'interpretation' in result_df.columns:
        # Get the most common interpretation
        interpretation_counts = result_df['interpretation'].value_counts()
        most_common = interpretation_counts.index[0]
        ratio = interpretation_counts[0] / len(result_df) * 100
        
        # Calculate average J value
        avg_j = result_df['j_empirical'].mean()
        
        # Determine overall pattern
        if avg_j < 0.95:
            overall = "Clustered"
        elif avg_j > 1.05:
            overall = "Regular"
        else:
            overall = "Random"
        
        # Create summary text
        summary_text = f"Summary: {most_common} ({ratio:.1f}% of distances), Average J(r) = {avg_j:.2f} → {overall} pattern"
        
        # Add summary box
        plt.figtext(0.5, 0.01, summary_text, fontsize=10, ha='center',
                   bbox=dict(facecolor='yellow', alpha=0.2, boxstyle='round'))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



