import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon as sPolygon
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import networkx as nx
from scipy.spatial import Voronoi
from collections import defaultdict


# Neighbor Analysis
def perform_neighbor_analysis(
    spatioloji_obj: 'spatioloji',
    cell_type_column: str,
    distance_threshold: float = 0.0,
    coordinate_type: str = 'local',
    fov_id: Optional[Union[str, List[str]]] = None,
    fov_column: str = 'fov',
    save_dir: str = "./",
    filename_prefix: str = "neighbor_analysis",
    include_plots: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    return_data: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Perform comprehensive neighbor distance analysis for cell polygons using a spatioloji object.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        A spatioloji object containing cell polygons and metadata.
    cell_type_column : str
        Column in spatioloji_obj.adata.obs that defines cell types for analysis.
    distance_threshold : float, optional
        Maximum distance between polygons to be considered neighbors, by default 0.0
        (0.0 means polygons must be touching, positive values include nearby non-touching cells)
    coordinate_type : str, optional
        Type of coordinates to use ('local' or 'global'), by default 'local'
    fov_id : str or List[str], optional
        FOV ID or list of FOV IDs to analyze. If None, analyze all FOVs, by default None
    fov_column : str, optional
        Column in cell_meta containing FOV IDs, by default 'fov'
    save_dir : str, optional
        Directory to save result files, by default "./"
    filename_prefix : str, optional
        Prefix for output filenames, by default "neighbor_analysis"
    include_plots : bool, optional
        Whether to generate and save plots, by default True
    figsize : Tuple[int, int], optional
        Figure size for plots, by default (10, 8)
    dpi : int, optional
        DPI for saved figures, by default 300
    return_data : bool, optional
        Whether to return the analysis data, by default True
    verbose : bool, optional
        Whether to display progress information, by default True
    
    Returns
    -------
    Dict
        Dictionary containing analysis results if return_data is True
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    from collections import defaultdict
    from tqdm import tqdm
    from typing import Optional, Union, List, Dict, Tuple
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Check required columns in adata.obs
    if cell_type_column not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
    
    # Handle FOV selection
    working_obj = spatioloji_obj
    
    if fov_id is not None:
        if verbose:
            print(f"Subsetting data to specific FOV(s): {fov_id}")
            
        # Convert single FOV ID to list
        if isinstance(fov_id, str):
            fov_id = [fov_id]
            
        # Ensure FOV column exists
        if fov_column not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"FOV column '{fov_column}' not found in cell_meta")
            
        # Create subset for specified FOVs
        working_obj = spatioloji_obj.subset_by_fovs(fov_ids=fov_id, fov_column=fov_column)
        
        # Add FOV info to filename
        if len(fov_id) == 1:
            filename_prefix = f"{filename_prefix}_fov_{fov_id[0]}"
        else:
            filename_prefix = f"{filename_prefix}_fovs_{len(fov_id)}"
    
    # Select the appropriate GeoDataFrame based on coordinate type
    if coordinate_type == 'local':
        gdf = working_obj.gdf_local
        if verbose:
            print("Using local coordinate system for spatial analysis")
    elif coordinate_type == 'global':
        gdf = working_obj.gdf_global
        if verbose:
            print("Using global coordinate system for spatial analysis")
    else:
        raise ValueError(f"Invalid coordinate_type: {coordinate_type}. Must be 'local' or 'global'")
    
    # Validate GeoDataFrame
    if gdf is None or gdf.empty or 'geometry' not in gdf.columns:
        raise ValueError(f"GeoDataFrame for {coordinate_type} coordinates is not properly initialized")
    
    if verbose:
        print(f"Processing {len(gdf)} cells")
    
    # Get cell polygons
    cell_ids = gdf['cell'].tolist()
    polygon_set = gdf['geometry'].tolist()
    
    # Map cell IDs to indices
    cell_to_index = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    
    # Get cell types from adata.obs instead of cell_meta
    adata = working_obj.adata
    
    # Create a mapping between cell IDs and cell types from adata
    # First, check if 'cell' is a column in adata.obs
    if 'cell' in adata.obs.columns:
        # Use the 'cell' column to link adata.obs to gdf
        cell_to_type_map = dict(zip(adata.obs['cell'], adata.obs[cell_type_column]))
    else:
        # Assume the index of adata.obs contains cell IDs
        cell_to_type_map = dict(zip(adata.obs.index, adata.obs[cell_type_column]))
    
    # Get FOV information from cell_meta if available
    if fov_column in working_obj.cell_meta.columns:
        fov_dict = dict(zip(working_obj.cell_meta['cell'], working_obj.cell_meta[fov_column]))
    else:
        fov_dict = {cell_id: "Unknown" for cell_id in cell_ids}
    
    # Get cell types for each polygon
    cell_types = []
    for cell_id in cell_ids:
        if cell_id in cell_to_type_map:
            cell_types.append(cell_to_type_map[cell_id])
        else:
            if verbose:
                print(f"Warning: Cell {cell_id} not found in adata.obs. Assigning 'Unknown'.")
            cell_types.append("Unknown")
    
    # Get unique cell types
    unique_cell_types = sorted(set(cell_types))
    
    # Find neighboring cells
    if verbose:
        print("Finding neighboring cells...")
    
    neighbor_pairs = []
    neighbor_distances = []
    
    # Use tqdm for progress tracking
    total_comparisons = len(polygon_set) * (len(polygon_set) - 1) // 2
    progress_bar = tqdm(total=total_comparisons, disable=not verbose)
    
    for i in range(len(polygon_set)):
        for j in range(i+1, len(polygon_set)):
            progress_bar.update(1)
            poly1 = polygon_set[i]
            poly2 = polygon_set[j]
            
            # Check if polygons are neighbors
            distance = poly1.distance(poly2)
            if distance <= distance_threshold:
                neighbor_pairs.append((cell_ids[i], cell_ids[j]))
                neighbor_distances.append(distance)
    
    progress_bar.close()
    
    # Create neighbor dataframe
    neighbor_df = pd.DataFrame({
        'cell1': [pair[0] for pair in neighbor_pairs],
        'cell2': [pair[1] for pair in neighbor_pairs],
        'distance': neighbor_distances
    })
    
    # Add cell types and FOV information
    neighbor_df['cell1_type'] = neighbor_df['cell1'].map(cell_to_type_map)
    neighbor_df['cell2_type'] = neighbor_df['cell2'].map(cell_to_type_map)
    neighbor_df['cell1_fov'] = neighbor_df['cell1'].map(fov_dict)
    neighbor_df['cell2_fov'] = neighbor_df['cell2'].map(fov_dict)
    
    # Add column for cross-FOV interactions
    if fov_column in working_obj.cell_meta.columns:
        neighbor_df['cross_fov'] = neighbor_df['cell1_fov'] != neighbor_df['cell2_fov']
    
    # Save neighbor data
    neighbor_df.to_csv(os.path.join(save_dir, f"{filename_prefix}_pairs.csv"), index=False)
    
    # Create a network graph
    if verbose:
        print("Creating cell interaction network...")
    
    G = nx.Graph()
    
    # Add nodes with cell type and FOV attributes
    for i, cell_id in enumerate(cell_ids):
        G.add_node(cell_id, 
                   cell_type=cell_types[i], 
                   fov=fov_dict.get(cell_id, "Unknown"))
    
    # Add edges with distance attributes
    for _, row in neighbor_df.iterrows():
        G.add_edge(row['cell1'], row['cell2'], distance=row['distance'])
    
    # Compute interaction counts between cell types
    interaction_counts = defaultdict(int)
    interaction_distances = defaultdict(list)
    
    for _, row in neighbor_df.iterrows():
        type_pair = tuple(sorted([row['cell1_type'], row['cell2_type']]))
        interaction_counts[type_pair] += 1
        interaction_distances[type_pair].append(row['distance'])
    
    # Create interaction matrix
    interaction_matrix = np.zeros((len(unique_cell_types), len(unique_cell_types)))
    avg_distance_matrix = np.zeros((len(unique_cell_types), len(unique_cell_types)))
    
    for i, type1 in enumerate(unique_cell_types):
        for j, type2 in enumerate(unique_cell_types):
            type_pair = tuple(sorted([type1, type2]))
            interaction_matrix[i, j] = interaction_counts[type_pair]
            avg_distance_matrix[i, j] = np.mean(interaction_distances[type_pair]) if interaction_distances[type_pair] else np.nan
    
    # Make sure diagonal reflects interactions within same cell type
    for i, cell_type in enumerate(unique_cell_types):
        same_type_pair = (cell_type, cell_type)
        interaction_matrix[i, i] = interaction_counts[same_type_pair]
        avg_distance_matrix[i, i] = np.mean(interaction_distances[same_type_pair]) if interaction_distances[same_type_pair] else np.nan
    
    # Store interaction data
    interaction_df = pd.DataFrame(interaction_matrix, index=unique_cell_types, columns=unique_cell_types)
    avg_distance_df = pd.DataFrame(avg_distance_matrix, index=unique_cell_types, columns=unique_cell_types)
    
    # Save interaction data
    interaction_df.to_csv(os.path.join(save_dir, f"{filename_prefix}_interaction_counts.csv"))
    avg_distance_df.to_csv(os.path.join(save_dir, f"{filename_prefix}_avg_distances.csv"))
    
    # Compute statistics for each cell
    cell_stats = {}
    
    for cell_id in cell_ids:
        neighbors = list(G.neighbors(cell_id))
        neighbor_types = [G.nodes[n]['cell_type'] for n in neighbors]
        neighbor_fovs = [G.nodes[n]['fov'] for n in neighbors]
        type_counts = {cell_type: neighbor_types.count(cell_type) for cell_type in unique_cell_types}
        
        # Count neighbors from same/different FOVs
        same_fov_neighbors = sum(1 for n_fov in neighbor_fovs if n_fov == G.nodes[cell_id]['fov'])
        diff_fov_neighbors = len(neighbors) - same_fov_neighbors
        
        cell_stats[cell_id] = {
            'degree': len(neighbors),
            'cell_type': G.nodes[cell_id]['cell_type'],
            'fov': G.nodes[cell_id]['fov'],
            'same_fov_neighbors': same_fov_neighbors,
            'diff_fov_neighbors': diff_fov_neighbors,
            **{f'neighbor_{ct}_count': count for ct, count in type_counts.items()}
        }
    
    cell_stats_df = pd.DataFrame.from_dict(cell_stats, orient='index')
    cell_stats_df.index.name = 'cell'
    cell_stats_df.reset_index(inplace=True)
    
    # Save cell statistics
    cell_stats_df.to_csv(os.path.join(save_dir, f"{filename_prefix}_cell_stats.csv"), index=False)
    
    # Calculate enrichment scores (observed/expected ratios)
    cell_type_counts = {cell_type: cell_types.count(cell_type) for cell_type in unique_cell_types}
    total_cells = len(cell_ids)
    
    enrichment_matrix = np.zeros((len(unique_cell_types), len(unique_cell_types)))
    
    for i, type1 in enumerate(unique_cell_types):
        for j, type2 in enumerate(unique_cell_types):
            # Calculate expected interactions based on frequency
            if type1 == type2:
                expected = (cell_type_counts[type1] * (cell_type_counts[type1] - 1)) / 2
            else:
                expected = cell_type_counts[type1] * cell_type_counts[type2]
            
            # Scale by total possible interactions
            total_possible = (total_cells * (total_cells - 1)) / 2
            expected = expected / total_possible * sum(interaction_counts.values())
            
            # Calculate enrichment (observed/expected)
            observed = interaction_matrix[i, j]
            enrichment = observed / expected if expected > 0 else np.nan
            enrichment_matrix[i, j] = enrichment
    
    # Store enrichment data
    enrichment_df = pd.DataFrame(enrichment_matrix, index=unique_cell_types, columns=unique_cell_types)
    
    # Save enrichment data
    enrichment_df.to_csv(os.path.join(save_dir, f"{filename_prefix}_enrichment.csv"))
    
    # Generate plots if requested
    if include_plots:
        if verbose:
            print("Generating plots...")
            
        # Interaction heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(interaction_df, annot=True, cmap="YlGnBu", fmt=".0f")
        plt.title(f"Cell Type Interaction Counts ({coordinate_type.capitalize()} Coordinates)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_interaction_heatmap.png"), dpi=dpi)
        plt.close()
        
        # Average distance heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(avg_distance_df, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f"Average Distance Between Cell Types ({coordinate_type.capitalize()} Coordinates)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_distance_heatmap.png"), dpi=dpi)
        plt.close()
        
        # Enrichment heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(enrichment_df, annot=True, cmap="coolwarm", center=1.0, fmt=".2f")
        plt.title(f"Cell Type Interaction Enrichment ({coordinate_type.capitalize()} Coordinates)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_enrichment_heatmap.png"), dpi=dpi)
        plt.close()
        
        # Degree distribution by cell type
        plt.figure(figsize=figsize)
        sns.boxplot(x=cell_stats_df['cell_type'], y=cell_stats_df['degree'])
        plt.title(f"Number of Neighbors by Cell Type ({coordinate_type.capitalize()} Coordinates)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_degree_boxplot.png"), dpi=dpi)
        plt.close()
        
        # If we have FOV information, plot cross-FOV vs. same-FOV interactions
        if fov_column in working_obj.cell_meta.columns:
            # Neighbor counts by FOV type
            plt.figure(figsize=figsize)
            cell_stats_melted = pd.melt(
                cell_stats_df, 
                id_vars=['cell', 'cell_type', 'fov'],
                value_vars=['same_fov_neighbors', 'diff_fov_neighbors'],
                var_name='neighbor_location',
                value_name='count'
            )
            sns.boxplot(x='cell_type', y='count', hue='neighbor_location', data=cell_stats_melted)
            plt.title(f"Neighbors by Location ({coordinate_type.capitalize()} Coordinates)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{filename_prefix}_neighbor_location_boxplot.png"), dpi=dpi)
            plt.close()
        
        # Network visualization (if fewer than 1000 cells for clarity)
        if len(cell_ids) < 1000:
            plt.figure(figsize=(12, 12))
            
            # Use spring layout for positioning
            pos = nx.spring_layout(G)
            
            # Create a color map for cell types
            color_map = {ct: plt.cm.tab10(i % 10) for i, ct in enumerate(unique_cell_types)}
            node_colors = [color_map[G.nodes[node]['cell_type']] for node in G.nodes()]
            
            # Draw nodes and edges
            nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8)
            
            # Color edges differently for cross-FOV interactions if we have FOV info
            if fov_column in working_obj.cell_meta.columns:
                same_fov_edges = [(u, v) for u, v in G.edges() if G.nodes[u]['fov'] == G.nodes[v]['fov']]
                diff_fov_edges = [(u, v) for u, v in G.edges() if G.nodes[u]['fov'] != G.nodes[v]['fov']]
                
                # Draw same-FOV edges in light gray
                nx.draw_networkx_edges(G, pos, edgelist=same_fov_edges, width=0.5, alpha=0.3, edge_color='gray')
                
                # Draw cross-FOV edges in red
                if diff_fov_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=diff_fov_edges, width=0.8, alpha=0.5, edge_color='red')
                
                plt.title(f"Cell Interaction Network ({coordinate_type.capitalize()} Coordinates)\nRed = Cross-FOV Interactions")
            else:
                # Draw all edges the same
                nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
                plt.title(f"Cell Interaction Network ({coordinate_type.capitalize()} Coordinates)")
            
            # Add a legend for cell types
            handles = []
            labels = []
            for ct in unique_cell_types:
                handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[ct], markersize=10))
                labels.append(ct)
            
            # Add legend for edge types if we have FOV info
            if fov_column in working_obj.cell_meta.columns:
                handles.append(plt.Line2D([0], [0], color='gray', lw=2, alpha=0.3))
                labels.append('Same FOV')
                if diff_fov_edges:
                    handles.append(plt.Line2D([0], [0], color='red', lw=2, alpha=0.5))
                    labels.append('Cross FOV')
                
            plt.legend(handles, labels, loc='best')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{filename_prefix}_network.png"), dpi=dpi)
            plt.close()
    
    # Prepare return data
    if return_data:
        return {
            'neighbor_df': neighbor_df,
            'interaction_df': interaction_df,
            'avg_distance_df': avg_distance_df,
            'enrichment_df': enrichment_df,
            'cell_stats_df': cell_stats_df,
            'network': G,
            'polygons': {cell_ids[i]: polygon_set[i] for i in range(len(cell_ids))},
            'coordinate_type': coordinate_type,
            'fov_id': fov_id,
            'cell_type_column': cell_type_column
        }
    else:
        return None

def calculate_nearest_neighbor_distances(spatioloji_obj, fov_id=None, use_global_coords=True, use_polygons=False):
    """
    Calculate the nearest neighbor distance for each cell.
    
    Args:
        spatioloji_obj: A Spatioloji object
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        use_polygons: Whether to use polygon boundaries (True) or cell centers (False) for distance calculation
        
    Returns:
        DataFrame with cell IDs and their nearest neighbor distances in pixels
    """
    from shapely.geometry import Point, Polygon
    import numpy as np
    
    # Determine which coordinates to use
    if use_global_coords:
        x_center_col = 'CenterX_global_px'
        y_center_col = 'CenterY_global_px'
        x_poly_col = 'x_global_px'
        y_poly_col = 'y_global_px'
    else:
        x_center_col = 'CenterX_local_px'
        y_center_col = 'CenterY_local_px'
        x_poly_col = 'x_local_px'
        y_poly_col = 'y_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns for centroid-based calculation
    if not use_polygons:
        if x_center_col not in cells.columns or y_center_col not in cells.columns:
            raise ValueError(f"Required columns {x_center_col} and {y_center_col} not found in cell metadata")
    
    # Check if we have polygons data if using polygons
    if use_polygons and (spatioloji_obj.polygons is None or len(spatioloji_obj.polygons) == 0):
        raise ValueError("Polygon data not available in the Spatioloji object")
    
    # Create a list to store results
    nn_distances = []
    
    # If using polygons, create a dictionary of cell polygons
    if use_polygons:
        # Group polygons by cell
        cell_polygons = {}
        for cell_id in cells['cell'].unique():
            cell_poly_data = spatioloji_obj.get_polygon_for_cell(cell_id)
            
            # Skip if no polygon data found for this cell
            if len(cell_poly_data) == 0:
                continue
                
            # Create a polygon from the points
            try:
                points = list(zip(cell_poly_data[x_poly_col], cell_poly_data[y_poly_col]))
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    cell_polygons[cell_id] = Polygon(points)
            except Exception as e:
                print(f"Warning: Could not create polygon for cell {cell_id}: {e}")
    
    # For each cell, calculate the distance to all other cells and find the minimum
    for idx, cell in cells.iterrows():
        cell_id = cell['cell']
        
        if use_polygons:
            # Skip if this cell doesn't have polygon data
            if cell_id not in cell_polygons:
                continue
                
            current_polygon = cell_polygons[cell_id]
            min_distance = float('inf')
            
            # Calculate distance to all other cell polygons
            for other_cell_id, other_polygon in cell_polygons.items():
                if other_cell_id != cell_id:
                    # Calculate distance between polygons
                    distance = current_polygon.distance(other_polygon)
                    min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                nn_distances.append({
                    'cell': cell_id,
                    'nearest_neighbor_distance': min_distance
                })
        else:
            # Using cell centers for distance calculation
            cell_x = cell[x_center_col]
            cell_y = cell[y_center_col]
            
            # Calculate distances to all other cells
            others = cells[cells['cell'] != cell_id]
            if len(others) == 0:
                # Skip if there are no other cells
                continue
                
            distances = np.sqrt((others[x_center_col] - cell_x)**2 + (others[y_center_col] - cell_y)**2)
            min_distance = distances.min()
            
            nn_distances.append({
                'cell': cell_id,
                'nearest_neighbor_distance': min_distance
            })
    
    return pd.DataFrame(nn_distances)

def calculate_cell_density(spatioloji_obj, radius, fov_id=None, use_global_coords=True, 
                          normalize_by_area=True, use_polygons=False, n_threads=None):
    """
    Calculate the cell density/crowding for each cell by counting neighbors within a specified radius.
    
    Args:
        spatioloji_obj: A Spatioloji object
        radius: Radius in pixels to search for neighboring cells
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        normalize_by_area: Whether to normalize count by the search area (True) or return raw counts (False)
        use_polygons: Whether to use polygon boundaries (True) or cell centers (False) for density calculation
        n_threads: Number of threads to use for parallel processing. If None, uses all available processors.
        
    Returns:
        DataFrame with cell IDs and their density measurements
    """
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point, Polygon
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    
    # Determine number of threads if not specified
    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    
    # Determine which coordinates to use
    if use_global_coords:
        x_center_col = 'CenterX_global_px'
        y_center_col = 'CenterY_global_px'
        x_poly_col = 'x_global_px'
        y_poly_col = 'y_global_px'
    else:
        x_center_col = 'CenterX_local_px'
        y_center_col = 'CenterY_local_px'
        x_poly_col = 'x_local_px'
        y_poly_col = 'y_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns for centroid-based calculation
    if not use_polygons:
        if x_center_col not in cells.columns or y_center_col not in cells.columns:
            raise ValueError(f"Required columns {x_center_col} and {y_center_col} not found in cell metadata")
    
    # Check if we have polygons data if using polygons
    if use_polygons and (spatioloji_obj.polygons is None or len(spatioloji_obj.polygons) == 0):
        raise ValueError("Polygon data not available in the Spatioloji object")
    
    # Calculate the search area (for normalization)
    search_area = np.pi * radius**2 if normalize_by_area else 1
    
    # If using polygons, create a dictionary of cell polygons (shared among threads)
    cell_polygons = {}
    cell_centers = {}
    
    if use_polygons:
        # Group polygons by cell
        for cell_id in cells['cell'].unique():
            cell_poly_data = spatioloji_obj.get_polygon_for_cell(cell_id)
            
            # Skip if no polygon data found for this cell
            if len(cell_poly_data) == 0:
                continue
                
            # Create a polygon from the points
            try:
                points = list(zip(cell_poly_data[x_poly_col], cell_poly_data[y_poly_col]))
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    poly = Polygon(points)
                    cell_polygons[cell_id] = poly
                    # Store the centroid for creating the buffer
                    cell_centers[cell_id] = (poly.centroid.x, poly.centroid.y)
            except Exception as e:
                print(f"Warning: Could not create polygon for cell {cell_id}: {e}")
    
    # Define worker functions for each calculation method
    def process_cell_with_polygons(cell_id):
        # Skip if this cell doesn't have polygon data
        if cell_id not in cell_polygons:
            return None
            
        current_polygon = cell_polygons[cell_id]
        center_x, center_y = cell_centers[cell_id]
        
        # Create a circular buffer around the cell centroid
        search_circle = Point(center_x, center_y).buffer(radius)
        
        # Count neighboring cells whose polygons intersect with the search circle
        neighbors_count = sum(1 for other_cell_id, other_polygon in cell_polygons.items() 
                          if other_cell_id != cell_id and search_circle.intersects(other_polygon))
        
        # Normalize by area if requested
        density = neighbors_count / search_area if normalize_by_area else neighbors_count
        
        return {
            'cell': cell_id,
            'density': density,
            'neighbors_count': neighbors_count
        }
    
    def process_cell_with_centers(cell_id):
        # Get the cell data
        cell = cells[cells['cell'] == cell_id].iloc[0]
        cell_x = cell[x_center_col]
        cell_y = cell[y_center_col]
        
        # Calculate distances to all other cells
        others = cells[cells['cell'] != cell_id]
        if len(others) == 0:
            # Skip if there are no other cells
            return None
            
        distances = np.sqrt((others[x_center_col] - cell_x)**2 + (others[y_center_col] - cell_y)**2)
        neighbors_count = np.sum(distances <= radius)
        
        # Normalize by area if requested
        density = neighbors_count / search_area if normalize_by_area else neighbors_count
        
        return {
            'cell': cell_id,
            'density': density,
            'neighbors_count': neighbors_count
        }
    
    # Get list of all cell IDs to process
    cell_ids = cells['cell'].unique()
    
    # Process cells in parallel
    densities = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Choose the appropriate processing function
        process_func = process_cell_with_polygons if use_polygons else process_cell_with_centers
        
        # Submit all tasks and collect results
        future_to_cell = {executor.submit(process_func, cell_id): cell_id for cell_id in cell_ids}
        
        # Gather results as they complete
        for future in future_to_cell:
            result = future.result()
            if result is not None:
                densities.append(result)
    
    # Convert results to DataFrame
    return pd.DataFrame(densities)


# Spatial Pattern Analysis
def calculate_ripleys_k(spatioloji_obj, max_distance, num_distances=20, fov_id=None, 
                        use_global_coords=True, permutations=0, n_jobs=-1):
    """
    Calculate Ripley's K function to analyze spatial point patterns with multithreaded permutations.
    
    Args:
        spatioloji_obj: A Spatioloji object
        max_distance: Maximum distance to calculate K function
        num_distances: Number of distance points to evaluate (default: 20)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        permutations: Number of Monte Carlo simulations for confidence envelope (0 for none)
        n_jobs: Number of parallel jobs for permutations (-1 for all processors)
        
    Returns:
        DataFrame with distances, K values, L values (normalized K), and optionally confidence envelopes
    """
    import numpy as np
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Extract point coordinates
    points = cells[[x_col, y_col]].values
    n_points = len(points)
    
    if n_points < 2:
        raise ValueError("Need at least 2 points to calculate Ripley's K function")
    
    # Calculate the area of the study region
    # For simplicity, use the bounding box of the points as the study area
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    area = (max_x - min_x) * (max_y - min_y)
    
    # Define the distances at which to evaluate the K function
    distances = np.linspace(0, max_distance, num_distances)
    
    # Calculate pairwise distances between all points
    dist_matrix = distance.pdist(points)
    dist_matrix = distance.squareform(dist_matrix)
    
    # Initialize arrays to store K values and L values
    k_values = np.zeros(len(distances))
    l_values = np.zeros(len(distances))
    
    # Calculate K function for each distance
    for i, d in enumerate(distances):
        if d == 0:
            k_values[i] = 0
            l_values[i] = 0
            continue
        
        # Count points within distance d of each point
        points_within_d = (dist_matrix <= d).sum() - n_points  # Subtract n_points to exclude self-counts
        
        # Calculate K value: (area/n_points^2) * sum(I(dij <= d))
        k_values[i] = (area / (n_points * (n_points - 1))) * points_within_d
        
        # Calculate L value: sqrt(K/pi) - d
        l_values[i] = np.sqrt(k_values[i] / np.pi) - d
    
    # Create result DataFrame
    result = pd.DataFrame({
        'distance': distances,
        'K': k_values,
        'L': l_values
    })
    
    # If permutations are requested, calculate confidence envelopes using parallelization
    if permutations > 0:
        # Define a function to calculate K function for a single permutation
        def calculate_k_for_permutation(seed, n_points, min_x, min_y, max_x, max_y, area, distances):
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Generate random points within the bounding box
            random_points = np.random.uniform(
                low=[min_x, min_y],
                high=[max_x, max_y],
                size=(n_points, 2)
            )
            
            # Calculate distance matrix for random points
            random_dist_matrix = distance.pdist(random_points)
            random_dist_matrix = distance.squareform(random_dist_matrix)
            
            # Calculate K and L functions for random pattern
            random_k = np.zeros(len(distances))
            random_l = np.zeros(len(distances))
            
            for i, d in enumerate(distances):
                if d == 0:
                    continue
                
                # Count points within distance d of each point
                points_within_d = (random_dist_matrix <= d).sum() - n_points
                
                # Calculate K value
                random_k[i] = (area / (n_points * (n_points - 1))) * points_within_d
                
                # Calculate L value
                random_l[i] = np.sqrt(random_k[i] / np.pi) - d
            
            return random_k, random_l
        
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Run permutations in parallel
        permutation_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_k_for_permutation)(
                p, n_points, min_x, min_y, max_x, max_y, area, distances
            ) for p in range(permutations)
        )
        
        # Extract K and L values from permutation results
        random_k_values = np.array([res[0] for res in permutation_results])
        random_l_values = np.array([res[1] for res in permutation_results])
        
        # Calculate confidence envelopes (min and max values across permutations)
        k_low = np.min(random_k_values, axis=0)
        k_high = np.max(random_k_values, axis=0)
        l_low = np.min(random_l_values, axis=0)
        l_high = np.max(random_l_values, axis=0)
        
        # Add confidence envelopes to result
        result['K_low'] = k_low
        result['K_high'] = k_high
        result['L_low'] = l_low
        result['L_high'] = l_high
    
    return result

def calculate_ripleys_k_by_cell_type(spatioloji_obj, max_distance, cell_type_column, 
                                     cell_types=None, num_distances=20, fov_id=None, 
                                     use_global_coords=True, permutations=0, n_jobs=-1):
    """
    Calculate Ripley's K function for multiple cell types.
    
    Args:
        spatioloji_obj: A Spatioloji object
        max_distance: Maximum distance to calculate K function
        cell_type_column: Column name in spatioloji.adata.obs containing cell type information
        cell_types: List of cell types to analyze (None = all types)
        num_distances: Number of distance points to evaluate (default: 20)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        permutations: Number of Monte Carlo simulations for confidence envelope (0 for none)
        n_jobs: Number of parallel jobs for permutations (-1 for all processors)
        
    Returns:
        Dictionary mapping cell types to DataFrames with K function results
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Ensure cell type information is available
    if cell_type_column not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_column}' not found in spatioloji.adata.obs")
    
    # Merge cell type information with cell metadata
    cell_ids = cells.index.tolist()
    cell_types_series = spatioloji_obj.adata.obs[cell_type_column]
    cells_with_types = cells.copy()
    cells_with_types['cell_type'] = [cell_types_series.get(cell_id, None) for cell_id in cell_ids]
    
    # Remove cells with missing type information
    cells_with_types = cells_with_types.dropna(subset=['cell_type'])
    
    # Get unique cell types if not specified
    if cell_types is None:
        cell_types = cells_with_types['cell_type'].unique().tolist()
    
    # Dictionary to store results
    results = {}
    
    # Define the distances at which to evaluate the K function
    distances = np.linspace(0, max_distance, num_distances)
    
    # Calculate area of the study region (use full dataset bounds)
    all_points = cells_with_types[[x_col, y_col]].values
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    area = (max_x - min_x) * (max_y - min_y)
    
    # Analyze each cell type separately
    for cell_type in cell_types:
        # Get coordinates for current cell type
        type_cells = cells_with_types[cells_with_types['cell_type'] == cell_type]
        type_points = type_cells[[x_col, y_col]].values
        n_points = len(type_points)
        
        if n_points < 2:
            print(f"Warning: Skipping type '{cell_type}' with fewer than 2 points")
            continue
        
        # Calculate pairwise distances between all points of this type
        dist_matrix = distance.pdist(type_points)
        dist_matrix = distance.squareform(dist_matrix)
        
        # Initialize arrays to store K values and L values
        k_values = np.zeros(len(distances))
        l_values = np.zeros(len(distances))
        
        # Calculate K function for each distance
        for i, d in enumerate(distances):
            if d == 0:
                k_values[i] = 0
                l_values[i] = 0
                continue
            
            # Count points within distance d of each point
            points_within_d = (dist_matrix <= d).sum() - n_points  # Subtract n_points to exclude self-counts
            
            # Calculate K value: (area/n_points^2) * sum(I(dij <= d))
            k_values[i] = (area / (n_points * (n_points - 1))) * points_within_d
            
            # Calculate L value: sqrt(K/pi) - d
            l_values[i] = np.sqrt(k_values[i] / np.pi) - d
        
        # Create result DataFrame
        result = pd.DataFrame({
            'distance': distances,
            'K': k_values,
            'L': l_values
        })
        
        # If permutations are requested, calculate confidence envelopes
        if permutations > 0:
            # Define a function to calculate K function for a single permutation
            def calculate_k_for_permutation(seed, n_points, min_x, min_y, max_x, max_y, area, distances):
                # Set random seed for reproducibility
                np.random.seed(seed)
                
                # Generate random points within the bounding box
                random_points = np.random.uniform(
                    low=[min_x, min_y],
                    high=[max_x, max_y],
                    size=(n_points, 2)
                )
                
                # Calculate distance matrix for random points
                random_dist_matrix = distance.pdist(random_points)
                random_dist_matrix = distance.squareform(random_dist_matrix)
                
                # Calculate K and L functions for random pattern
                random_k = np.zeros(len(distances))
                random_l = np.zeros(len(distances))
                
                for i, d in enumerate(distances):
                    if d == 0:
                        continue
                    
                    # Count points within distance d of each point
                    points_within_d = (random_dist_matrix <= d).sum() - n_points
                    
                    # Calculate K value
                    random_k[i] = (area / (n_points * (n_points - 1))) * points_within_d
                    
                    # Calculate L value
                    random_l[i] = np.sqrt(random_k[i] / np.pi) - d
                
                return random_k, random_l
            
            # Determine number of cores to use
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            
            # Run permutations in parallel
            permutation_results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_k_for_permutation)(
                    p, n_points, min_x, min_y, max_x, max_y, area, distances
                ) for p in range(permutations)
            )
            
            # Extract K and L values from permutation results
            random_k_values = np.array([res[0] for res in permutation_results])
            random_l_values = np.array([res[1] for res in permutation_results])
            
            # Calculate confidence envelopes (min and max values across permutations)
            k_low = np.min(random_k_values, axis=0)
            k_high = np.max(random_k_values, axis=0)
            l_low = np.min(random_l_values, axis=0)
            l_high = np.max(random_l_values, axis=0)
            
            # Add confidence envelopes to result
            result['K_low'] = k_low
            result['K_high'] = k_high
            result['L_low'] = l_low
            result['L_high'] = l_high
        
        # Store the result for this cell type
        results[cell_type] = result
    
    return results


def calculate_cross_k_function(spatioloji_obj, cell_type_column, cell_type1, cell_type2, 
                              max_distance, num_distances=20, fov_id=None, use_global_coords=True, 
                              edge_correction=True, permutations=999, n_jobs=-1, plot_result=False):
    """
    Calculate Ripley's Cross-K function to analyze spatial relationships between two different cell types.
    
    Args:
        spatioloji_obj: A Spatioloji object
        cell_type_column: Column name in adata.obs containing cell type information
        cell_type1: First cell type for cross-K analysis
        cell_type2: Second cell type for cross-K analysis
        max_distance: Maximum distance to calculate K function (in pixels)
        num_distances: Number of distance points to evaluate (default: 20)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        edge_correction: Whether to apply edge correction (True) or not (False)
        permutations: Number of Monte Carlo simulations for confidence envelope (0 for none)
        n_jobs: Number of parallel jobs for permutations (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        DataFrame with distances, Cross-K values, and confidence envelopes if requested
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from shapely.geometry import Point, Polygon
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Get cell type information from AnnData object
    if cell_type_column not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
    
    # Create a dictionary mapping cell IDs to cell types
    cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
    
    # Add cell type information to the cells DataFrame
    cells['cell_type'] = cells['cell'].map(cell_to_type)
    
    # Extract cells of the specific types
    cells_type1 = cells[cells['cell_type'] == cell_type1]
    cells_type2 = cells[cells['cell_type'] == cell_type2]
    
    # Check if we have enough cells
    if len(cells_type1) == 0:
        raise ValueError(f"No cells of type '{cell_type1}' found")
    
    if len(cells_type2) == 0:
        raise ValueError(f"No cells of type '{cell_type2}' found")
    
    # Extract coordinates for each cell type
    coords1 = cells_type1[[x_col, y_col]].values
    coords2 = cells_type2[[x_col, y_col]].values
    
    # Calculate study area
    all_cells = pd.concat([cells_type1, cells_type2])
    x_min, y_min = all_cells[[x_col, y_col]].min().values
    x_max, y_max = all_cells[[x_col, y_col]].max().values
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    # Create study area polygon for edge correction
    study_area = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    
    # Calculate cell densities
    density1 = len(coords1) / area
    density2 = len(coords2) / area
    
    # Create distance bins
    distances = np.linspace(0, max_distance, num_distances + 1)
    distance_centers = 0.5 * (distances[1:] + distances[:-1])
    
    # Calculate Cross-K function
    cross_k = np.zeros(num_distances)
    
    # Count points of type 2 within distance r of each point of type 1
    for i in range(len(coords1)):
        point1 = coords1[i]
        
        # Create shapely point for edge correction
        shapely_point = Point(point1)
        
        for j in range(num_distances):
            r = distances[j+1]  # Upper bound of current distance bin
            
            # Count points of type 2 within distance r of point1
            dist_to_point1 = np.sqrt(np.sum((coords2 - point1)**2, axis=1))
            points_within_r = np.sum(dist_to_point1 <= r)
            
            # Apply edge correction if requested
            if edge_correction:
                # Create a circle buffer around the point
                buffer = shapely_point.buffer(r)
                
                # Calculate the fraction of the circle that falls within the study area
                if not buffer.within(study_area):
                    # Calculate intersection area
                    intersection = buffer.intersection(study_area)
                    # Calculate edge correction factor
                    edge_correction_factor = buffer.area / intersection.area
                else:
                    edge_correction_factor = 1.0
                
                # Apply edge correction
                points_within_r *= edge_correction_factor
            
            # Add to Cross-K function
            cross_k[j] += points_within_r
    
    # Normalize Cross-K function
    cross_k = cross_k / (len(coords1) * density2)
    
    # Calculate Cross-L function (normalized version of Cross-K)
    cross_l = np.sqrt(cross_k / np.pi) - distance_centers
    
    # Calculate theoretical Cross-K under CSR (Complete Spatial Randomness)
    # For CSR, K(r) = πr²
    theoretical_k = np.pi * distance_centers**2
    
    # Calculate difference from CSR
    diff_from_csr = cross_k - theoretical_k
    
    # Create result DataFrame
    result = pd.DataFrame({
        'distance': distance_centers,
        'cross_k': cross_k,
        'cross_l': cross_l,
        'theoretical_k': theoretical_k,
        'diff_from_csr': diff_from_csr
    })
    
    # Calculate Monte Carlo confidence envelope if requested
    if permutations > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate Cross-K for random cell distributions
        def calculate_random_cross_k(seed, coords1, n_points2, x_min, y_min, width, height, distances, edge_correction, study_area):
            np.random.seed(seed)
            
            # Generate random coordinates for type 2 cells
            random_coords2 = np.random.uniform(
                low=[x_min, y_min],
                high=[x_min + width, y_min + height],
                size=(n_points2, 2)
            )
            
            # Calculate Cross-K function for random distribution
            random_cross_k = np.zeros(len(distances) - 1)
            
            for i in range(len(coords1)):
                point1 = coords1[i]
                
                # Create shapely point for edge correction
                shapely_point = Point(point1)
                
                for j in range(len(distances) - 1):
                    r = distances[j+1]  # Upper bound of current distance bin
                    
                    # Count random points within distance r of point1
                    dist_to_point1 = np.sqrt(np.sum((random_coords2 - point1)**2, axis=1))
                    points_within_r = np.sum(dist_to_point1 <= r)
                    
                    # Apply edge correction if requested
                    if edge_correction:
                        # Create a circle buffer around the point
                        buffer = shapely_point.buffer(r)
                        
                        # Calculate the fraction of the circle that falls within the study area
                        if not buffer.within(study_area):
                            # Calculate intersection area
                            intersection = buffer.intersection(study_area)
                            # Calculate edge correction factor
                            edge_correction_factor = buffer.area / intersection.area
                        else:
                            edge_correction_factor = 1.0
                        
                        # Apply edge correction
                        points_within_r *= edge_correction_factor
                    
                    # Add to Cross-K function
                    random_cross_k[j] += points_within_r
            
            # Normalize Cross-K function
            random_cross_k = random_cross_k / (len(coords1) * (n_points2 / area))
            
            return random_cross_k
        
        # Run Monte Carlo simulations in parallel
        mc_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_random_cross_k)(
                i, coords1, len(coords2), x_min, y_min, width, height, distances, edge_correction, study_area
            ) for i in tqdm(range(permutations), desc="Monte Carlo simulations")
        )
        
        # Stack results for easier processing
        mc_cross_k = np.vstack(mc_results)
        
        # Calculate confidence envelopes (2.5% and 97.5% percentiles)
        cross_k_low = np.percentile(mc_cross_k, 2.5, axis=0)
        cross_k_high = np.percentile(mc_cross_k, 97.5, axis=0)
        
        # Calculate Cross-L function for envelopes
        cross_l_low = np.sqrt(cross_k_low / np.pi) - distance_centers
        cross_l_high = np.sqrt(cross_k_high / np.pi) - distance_centers
        
        # Add to result DataFrame
        result['cross_k_low'] = cross_k_low
        result['cross_k_high'] = cross_k_high
        result['cross_l_low'] = cross_l_low
        result['cross_l_high'] = cross_l_high
        
        # Determine if observed Cross-K is significantly different from random
        result['is_clustered'] = result['cross_k'] > result['cross_k_high']
        result['is_dispersed'] = result['cross_k'] < result['cross_k_low']
        result['is_significant'] = result['is_clustered'] | result['is_dispersed']
    
    # Interpret the Cross-K function
    # For Cross-K:
    # - K(r) > πr²: Cells of type 2 are more clustered around cells of type 1 than expected (attraction)
    # - K(r) = πr²: Cells of type 2 are randomly distributed around cells of type 1 (no interaction)
    # - K(r) < πr²: Cells of type 2 are more dispersed around cells of type 1 than expected (repulsion)
    # 
    # For Cross-L:
    # - L(r) > 0: Attraction
    # - L(r) = 0: No interaction
    # - L(r) < 0: Repulsion
    
    # Add interpretation column
    result['interpretation'] = 'Random'
    result.loc[result['cross_l'] > 0, 'interpretation'] = 'Attraction'
    result.loc[result['cross_l'] < 0, 'interpretation'] = 'Repulsion'
    
    # If we have significance testing, refine the interpretation
    if permutations > 0:
        result['interpretation'] = 'Random (Not Significant)'
        result.loc[result['is_clustered'], 'interpretation'] = 'Attraction (Significant)'
        result.loc[result['is_dispersed'], 'interpretation'] = 'Repulsion (Significant)'
    
    # Generate plot configuration if requested
    if plot_result:
        # Define plot configuration
        plot_config = {
            'x': 'distance',
            'y': 'cross_l',
            'title': f'Cross-L Function: {cell_type1} to {cell_type2}',
            'xlabel': 'Distance (pixels)',
            'ylabel': 'Cross-L(r)',
            'reference_line': 0,  # Add a horizontal line at L(r) = 0
            'confidence_intervals': permutations > 0,  # Add confidence intervals if available
            'ci_low': 'cross_l_low',
            'ci_high': 'cross_l_high'
        }
        
        return result, plot_config
    
    return result

def calculate_j_function(spatioloji_obj, max_distance, num_bins=50, cell_type=None,
                        cell_type_column=None, fov_id=None, use_global_coords=True,
                        edge_correction=True, monte_carlo_points=1000,
                        permutations=999, n_jobs=-1, plot_result=False):
    """
    Calculate Baddeley's J-function, which combines G-function and F-function for spatial pattern analysis.
    
    Args:
        spatioloji_obj: A Spatioloji object
        max_distance: Maximum distance to consider (in pixels)
        num_bins: Number of distance bins for the functions
        cell_type: Optional cell type to filter cells (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        edge_correction: Whether to apply edge correction (True) or not (False)
        monte_carlo_points: Number of random points to generate for F-function calculation
        permutations: Number of Monte Carlo simulations for confidence envelope (0 for none)
        n_jobs: Number of parallel jobs for permutations (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        DataFrame with distances and J-function values
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from shapely.geometry import Point, Polygon
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Filter by cell type if specified
    if cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
        
        # Filter to only include cells of the specified type
        cells = cells[cells['cell_type'] == cell_type]
        
        if len(cells) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    
    # Extract coordinates
    coords = cells[[x_col, y_col]].values
    
    # Calculate study area
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    # Create study area polygon for edge correction
    study_area = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    
    # Calculate distance bins
    bins = np.linspace(0, max_distance, num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Calculate G-function (nearest neighbor distance distribution)
    # --------------------------------------------------------------
    
    # Calculate nearest neighbor distances for each cell
    nn_distances = []
    
    for i in range(len(coords)):
        point = coords[i]
        
        # Calculate distances to all other cells (excluding self)
        distances = []
        for j in range(len(coords)):
            if i != j:  # Exclude self
                dist = np.sqrt(np.sum((coords[j] - point)**2))
                distances.append(dist)
        
        # Find minimum distance (nearest neighbor)
        if len(distances) > 0:
            min_dist = np.min(distances)
            nn_distances.append(min_dist)
    
    # Apply edge correction for G-function if requested
    if edge_correction:
        # Border method edge correction
        # Adjust distances for points near the boundary
        nn_distances_corrected = []
        
        for i, dist in enumerate(nn_distances):
            point = coords[i]
            
            # Calculate distance to nearest boundary
            distance_to_edge = min(
                point[0] - x_min,  # Distance to left edge
                x_max - point[0],  # Distance to right edge
                point[1] - y_min,  # Distance to bottom edge
                y_max - point[1]   # Distance to top edge
            )
            
            # If the point is closer to the edge than to its nearest neighbor,
            # this could be due to edge effects
            if distance_to_edge < dist:
                # Apply edge correction factor
                correction_factor = 1 + (dist - distance_to_edge) / distance_to_edge
                dist_corrected = dist / correction_factor
                nn_distances_corrected.append(dist_corrected)
            else:
                nn_distances_corrected.append(dist)
        
        nn_distances = nn_distances_corrected
    
    # Calculate empirical G-function
    g_empirical = np.zeros(num_bins)
    
    for i in range(num_bins):
        # Count proportion of nearest neighbor distances <= r
        g_empirical[i] = np.mean(np.array(nn_distances) <= bins[i+1])
    
    # Calculate F-function (empty space function)
    # ------------------------------------------
    
    # Generate random points in the study area for F-function
    np.random.seed(42)  # For reproducibility
    random_points = np.random.uniform(
        low=[x_min, y_min],
        high=[x_max, y_max],
        size=(monte_carlo_points, 2)
    )
    
    # Calculate distance from each random point to the nearest cell
    es_distances = []
    
    for i in range(len(random_points)):
        point = random_points[i]
        
        # Calculate distances to all cells
        distances = np.sqrt(np.sum((coords - point)**2, axis=1))
        
        # Find minimum distance (nearest cell)
        if len(distances) > 0:
            min_dist = np.min(distances)
            es_distances.append(min_dist)
    
    # Apply edge correction for F-function if requested
    if edge_correction:
        # Reduced sample edge correction
        # Only keep random points that are at least max_distance away from the boundary
        filtered_es_distances = []
        filtered_points = []
        
        for i, dist in enumerate(es_distances):
            point = random_points[i]
            
            # Calculate distance to nearest boundary
            distance_to_edge = min(
                point[0] - x_min,  # Distance to left edge
                x_max - point[0],  # Distance to right edge
                point[1] - y_min,  # Distance to bottom edge
                y_max - point[1]   # Distance to top edge
            )
            
            # Only keep points that are far enough from the edge
            if distance_to_edge >= max_distance:
                filtered_es_distances.append(dist)
                filtered_points.append(point)
        
        # If we have enough points after filtering, use them
        if len(filtered_es_distances) >= 100:
            es_distances = filtered_es_distances
            random_points = np.array(filtered_points)
        else:
            # If too few points remain, use original points but apply correction
            es_distances_corrected = []
            
            for i, dist in enumerate(es_distances):
                point = random_points[i]
                
                # Calculate distance to nearest boundary
                distance_to_edge = min(
                    point[0] - x_min,  # Distance to left edge
                    x_max - point[0],  # Distance to right edge
                    point[1] - y_min,  # Distance to bottom edge
                    y_max - point[1]   # Distance to top edge
                )
                
                # Apply correction if needed
                if distance_to_edge < dist:
                    es_distances_corrected.append(distance_to_edge)
                else:
                    es_distances_corrected.append(dist)
            
            es_distances = es_distances_corrected
    
    # Calculate empirical F-function
    f_empirical = np.zeros(num_bins)
    
    for i in range(num_bins):
        # Count proportion of empty space distances <= r
        f_empirical[i] = np.mean(np.array(es_distances) <= bins[i+1])
    
    # Calculate theoretical functions for CSR
    # For CSR, G(r) = F(r) = 1 - exp(-λπr²)
    lambda_cells = len(coords) / area
    theoretical_csr = 1 - np.exp(-lambda_cells * np.pi * bin_centers**2)
    
    # Calculate J-function
    # J(r) = (1 - G(r)) / (1 - F(r))
    j_empirical = np.zeros(num_bins)
    
    for i in range(num_bins):
        g_value = g_empirical[i]
        f_value = f_empirical[i]
        
        # Avoid division by zero
        if f_value < 1:
            j_empirical[i] = (1 - g_value) / (1 - f_value)
        else:
            j_empirical[i] = 1  # Default to J(r) = 1 (CSR) if F(r) = 1
    
    # Calculate theoretical J-function for CSR
    # For CSR, J(r) = 1 for all r
    j_theoretical = np.ones(num_bins)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'distance': bin_centers,
        'g_empirical': g_empirical,
        'f_empirical': f_empirical,
        'j_empirical': j_empirical,
        'theoretical_csr': theoretical_csr,
        'j_theoretical': j_theoretical
    })
    
    # Calculate Monte Carlo confidence envelope if requested
    if permutations > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate J-function for random cell distributions
        def calculate_random_j(seed, n_points, x_min, y_min, width, height, bins, monte_carlo_points, edge_correction):
            np.random.seed(seed)
            
            # Generate random coordinates for cells
            random_coords = np.random.uniform(
                low=[x_min, y_min],
                high=[x_min + width, y_min + height],
                size=(n_points, 2)
            )
            
            # Calculate G-function for random distribution
            nn_distances = []
            
            for i in range(len(random_coords)):
                point = random_coords[i]
                
                # Calculate distances to all other points (excluding self)
                distances = []
                for j in range(len(random_coords)):
                    if i != j:  # Exclude self
                        dist = np.sqrt(np.sum((random_coords[j] - point)**2))
                        distances.append(dist)
                
                # Find minimum distance (nearest neighbor)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    nn_distances.append(min_dist)
            
            # Apply edge correction for G-function if requested
            if edge_correction:
                # Simple correction for random simulation
                nn_distances = [min(d, max_distance) for d in nn_distances]
            
            # Calculate empirical G-function
            g_random = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                # Count proportion of nearest neighbor distances <= r
                g_random[i] = np.mean(np.array(nn_distances) <= bins[i+1])
            
            # Generate random points in the study area for F-function
            np.random.seed(seed + 10000)  # Different seed for F-function
            random_points = np.random.uniform(
                low=[x_min, y_min],
                high=[x_max, y_max],
                size=(monte_carlo_points, 2)
            )
            
            # Calculate F-function for random distribution
            es_distances = []
            
            for i in range(len(random_points)):
                point = random_points[i]
                
                # Calculate distances to all random cells
                distances = np.sqrt(np.sum((random_coords - point)**2, axis=1))
                
                # Find minimum distance (nearest cell)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    es_distances.append(min_dist)
            
            # Apply edge correction for F-function if requested
            if edge_correction:
                # Simple correction for random simulation
                es_distances = [min(d, max_distance) for d in es_distances]
            
            # Calculate empirical F-function
            f_random = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                # Count proportion of empty space distances <= r
                f_random[i] = np.mean(np.array(es_distances) <= bins[i+1])
            
            # Calculate J-function
            j_random = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                g_value = g_random[i]
                f_value = f_random[i]
                
                # Avoid division by zero
                if f_value < 1:
                    j_random[i] = (1 - g_value) / (1 - f_value)
                else:
                    j_random[i] = 1
            
            return j_random
        
        # Run Monte Carlo simulations in parallel
        mc_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_random_j)(
                i, len(coords), x_min, y_min, width, height, bins, monte_carlo_points, edge_correction
            ) for i in tqdm(range(permutations), desc="Monte Carlo simulations")
        )
        
        # Stack results for easier processing
        mc_j_values = np.vstack(mc_results)
        
        # Calculate confidence envelopes (2.5% and 97.5% percentiles)
        j_low = np.percentile(mc_j_values, 2.5, axis=0)
        j_high = np.percentile(mc_j_values, 97.5, axis=0)
        
        # Add to result DataFrame
        result['j_low'] = j_low
        result['j_high'] = j_high
        
        # Determine if observed J-function is significantly different from random
        result['is_clustered'] = result['j_empirical'] < result['j_low']
        result['is_regular'] = result['j_empirical'] > result['j_high']
        result['is_significant'] = result['is_clustered'] | result['is_regular']
    
    # Interpret the J-function
    # For J-function:
    # - J(r) = 1: Complete Spatial Randomness (CSR)
    # - J(r) < 1: Clustering (attraction between points)
    # - J(r) > 1: Regularity/dispersion (repulsion between points)
    
    # Add interpretation column
    result['interpretation'] = 'Random'
    result.loc[result['j_empirical'] < 0.95, 'interpretation'] = 'Clustered'
    result.loc[result['j_empirical'] > 1.05, 'interpretation'] = 'Regular'
    
    # If we have significance testing, refine the interpretation
    if permutations > 0:
        result['interpretation'] = 'Random (Not Significant)'
        result.loc[result['is_clustered'], 'interpretation'] = 'Clustered (Significant)'
        result.loc[result['is_regular'], 'interpretation'] = 'Regular (Significant)'
    
    # Generate plot configuration if requested
    if plot_result:
        # Define plot configuration
        plot_config = {
            'x': 'distance',
            'y': 'j_empirical',
            'title': 'J-function (Baddeley\'s)',
            'subtitle': f'Cell type: {cell_type or "All cells"}',
            'xlabel': 'Distance (pixels)',
            'ylabel': 'J(r)',
            'reference_line': 1,  # Add a horizontal line at J(r) = 1
            'confidence_intervals': permutations > 0,  # Add confidence intervals if available
            'ci_low': 'j_low',
            'ci_high': 'j_high',
            'y_min': 0,  # Start y-axis at 0
            'y_max': max(2, result['j_empirical'].max() * 1.1)  # Adjust y-axis based on data
        }
        
        return result, plot_config
    
    return result


def calculate_g_function(spatioloji_obj, max_distance, num_bins=50, cell_type=None, 
                        reference_cell_type=None, cell_type_column=None, fov_id=None,
                        use_global_coords=True, edge_correction=True, 
                        permutations=999, n_jobs=-1, plot_result=False):
    """
    Calculate the G-function (nearest neighbor distance distribution function).
    
    Args:
        spatioloji_obj: A Spatioloji object
        max_distance: Maximum distance to consider (in pixels)
        num_bins: Number of distance bins for the G-function
        cell_type: Optional cell type to filter cells (None for all cells)
        reference_cell_type: Optional reference cell type for cross-type G-function (None for same type)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        edge_correction: Whether to apply edge correction (True) or not (False)
        permutations: Number of Monte Carlo simulations for confidence envelope (0 for none)
        n_jobs: Number of parallel jobs for permutations (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        DataFrame with distances and G-function values
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from shapely.geometry import Point, Polygon
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Get cell type information if needed
    cross_type = reference_cell_type is not None
    
    if cell_type is not None or reference_cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type or reference_cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
    
    # Filter cells based on cell type
    if cell_type is not None:
        cells_target = cells[cells['cell_type'] == cell_type]
        if len(cells_target) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    else:
        cells_target = cells
    
    # For cross-type G-function, extract reference cells
    if reference_cell_type is not None:
        cells_reference = cells[cells['cell_type'] == reference_cell_type]
        if len(cells_reference) == 0:
            raise ValueError(f"No cells of type '{reference_cell_type}' found")
    else:
        cells_reference = cells_target
    
    # Extract coordinates
    coords_target = cells_target[[x_col, y_col]].values
    coords_reference = cells_reference[[x_col, y_col]].values
    
    # Calculate study area
    x_min, y_min = cells[[x_col, y_col]].min().values
    x_max, y_max = cells[[x_col, y_col]].max().values
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    # Create study area polygon for edge correction
    study_area = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    
    # Calculate distance bins
    bins = np.linspace(0, max_distance, num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Calculate nearest neighbor distances for each target cell
    nn_distances = []
    
    for i in range(len(coords_target)):
        point = coords_target[i]
        
        if cross_type:
            # Calculate distances to all reference cells
            distances = np.sqrt(np.sum((coords_reference - point)**2, axis=1))
            
            # Find minimum distance (nearest neighbor)
            if len(distances) > 0:
                min_dist = np.min(distances)
                nn_distances.append(min_dist)
        else:
            # Calculate distances to all other target cells (excluding self)
            distances = []
            for j in range(len(coords_target)):
                if i != j:  # Exclude self
                    dist = np.sqrt(np.sum((coords_target[j] - point)**2))
                    distances.append(dist)
            
            # Find minimum distance (nearest neighbor)
            if len(distances) > 0:
                min_dist = np.min(distances)
                nn_distances.append(min_dist)
    
    # Apply edge correction if requested
    if edge_correction:
        # Border method edge correction
        # Adjust distances for points near the boundary
        nn_distances_corrected = []
        
        for i, dist in enumerate(nn_distances):
            point = coords_target[i]
            
            # Calculate distance to nearest boundary
            distance_to_edge = min(
                point[0] - x_min,  # Distance to left edge
                x_max - point[0],  # Distance to right edge
                point[1] - y_min,  # Distance to bottom edge
                y_max - point[1]   # Distance to top edge
            )
            
            # If the point is closer to the edge than to its nearest neighbor,
            # this could be due to edge effects (nearest neighbor might be outside the study area)
            if distance_to_edge < dist:
                # Apply edge correction factor
                # This is a simplified approach; more sophisticated methods exist
                correction_factor = 1 + (dist - distance_to_edge) / distance_to_edge
                dist_corrected = dist / correction_factor
                nn_distances_corrected.append(dist_corrected)
            else:
                nn_distances_corrected.append(dist)
        
        nn_distances = nn_distances_corrected
    
    # Calculate empirical G-function
    g_empirical = np.zeros(num_bins)
    
    for i in range(num_bins):
        # Count proportion of nearest neighbor distances <= r
        g_empirical[i] = np.mean(np.array(nn_distances) <= bins[i+1])
    
    # Calculate theoretical G-function for CSR (Complete Spatial Randomness)
    # For CSR, G(r) = 1 - exp(-λπr²)
    # where λ is the density of points
    
    if cross_type:
        lambda_reference = len(coords_reference) / area
        g_theoretical = 1 - np.exp(-lambda_reference * np.pi * bin_centers**2)
    else:
        lambda_target = len(coords_target) / area
        g_theoretical = 1 - np.exp(-lambda_target * np.pi * bin_centers**2)
    
    # Calculate the difference from CSR
    g_diff = g_empirical - g_theoretical
    
    # Create result DataFrame
    result = pd.DataFrame({
        'distance': bin_centers,
        'g_empirical': g_empirical,
        'g_theoretical': g_theoretical,
        'g_diff': g_diff
    })
    
    # Calculate Monte Carlo confidence envelope if requested
    if permutations > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate G-function for random cell distributions
        def calculate_random_g(seed, n_points_target, n_points_reference, x_min, y_min, width, height, bins, cross_type):
            np.random.seed(seed)
            
            # Generate random coordinates
            if cross_type:
                random_coords_target = coords_target  # Keep target cells fixed
                random_coords_reference = np.random.uniform(
                    low=[x_min, y_min],
                    high=[x_min + width, y_min + height],
                    size=(n_points_reference, 2)
                )
            else:
                random_coords_target = np.random.uniform(
                    low=[x_min, y_min],
                    high=[x_min + width, y_min + height],
                    size=(n_points_target, 2)
                )
                random_coords_reference = random_coords_target
            
            # Calculate nearest neighbor distances for each random target cell
            random_nn_distances = []
            
            for i in range(len(random_coords_target)):
                point = random_coords_target[i]
                
                if cross_type:
                    # Calculate distances to all reference cells
                    distances = np.sqrt(np.sum((random_coords_reference - point)**2, axis=1))
                    
                    # Find minimum distance (nearest neighbor)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        random_nn_distances.append(min_dist)
                else:
                    # Calculate distances to all other target cells (excluding self)
                    distances = []
                    for j in range(len(random_coords_target)):
                        if i != j:  # Exclude self
                            dist = np.sqrt(np.sum((random_coords_target[j] - point)**2))
                            distances.append(dist)
                    
                    # Find minimum distance (nearest neighbor)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        random_nn_distances.append(min_dist)
            
            # Calculate empirical G-function for random distribution
            g_random = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                # Count proportion of nearest neighbor distances <= r
                g_random[i] = np.mean(np.array(random_nn_distances) <= bins[i+1])
            
            return g_random
        
        # Run Monte Carlo simulations in parallel
        mc_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_random_g)(
                i, len(coords_target), len(coords_reference), x_min, y_min, width, height, bins, cross_type
            ) for i in tqdm(range(permutations), desc="Monte Carlo simulations")
        )
        
        # Stack results for easier processing
        mc_g_values = np.vstack(mc_results)
        
        # Calculate confidence envelopes (2.5% and 97.5% percentiles)
        g_low = np.percentile(mc_g_values, 2.5, axis=0)
        g_high = np.percentile(mc_g_values, 97.5, axis=0)
        
        # Add to result DataFrame
        result['g_low'] = g_low
        result['g_high'] = g_high
        
        # Determine if observed G-function is significantly different from random
        result['is_clustered'] = result['g_empirical'] > result['g_high']
        result['is_dispersed'] = result['g_empirical'] < result['g_low']
        result['is_significant'] = result['is_clustered'] | result['is_dispersed']
    
    # Interpret the G-function
    # For G-function:
    # - G(r) > theoretical G(r): Cells are more clustered than expected (shorter nearest neighbor distances)
    # - G(r) = theoretical G(r): Cells follow CSR (random distribution)
    # - G(r) < theoretical G(r): Cells are more dispersed than expected (longer nearest neighbor distances)
    
    # Add interpretation column
    result['interpretation'] = 'Random'
    result.loc[result['g_diff'] > 0, 'interpretation'] = 'Clustered'
    result.loc[result['g_diff'] < 0, 'interpretation'] = 'Dispersed'
    
    # If we have significance testing, refine the interpretation
    if permutations > 0:
        result['interpretation'] = 'Random (Not Significant)'
        result.loc[result['is_clustered'], 'interpretation'] = 'Clustered (Significant)'
        result.loc[result['is_dispersed'], 'interpretation'] = 'Dispersed (Significant)'
    
    # Calculate summary statistics
    mean_nn_distance = np.mean(nn_distances)
    median_nn_distance = np.median(nn_distances)
    min_nn_distance = np.min(nn_distances) if len(nn_distances) > 0 else np.nan
    max_nn_distance = np.max(nn_distances) if len(nn_distances) > 0 else np.nan
    std_nn_distance = np.std(nn_distances)
    
    # Add summary statistics to the result
    result.attrs['mean_nn_distance'] = mean_nn_distance
    result.attrs['median_nn_distance'] = median_nn_distance
    result.attrs['min_nn_distance'] = min_nn_distance
    result.attrs['max_nn_distance'] = max_nn_distance
    result.attrs['std_nn_distance'] = std_nn_distance
    
    # Generate plot configuration if requested
    if plot_result:
        # Define plot configuration
        plot_config = {
            'x': 'distance',
            'y': 'g_empirical',
            'y2': 'g_theoretical',
            'title': 'G-function (Nearest Neighbor Distance Distribution)',
            'subtitle': f'Cell type: {cell_type or "All"}, Reference: {reference_cell_type or "Same"}',
            'xlabel': 'Distance (pixels)',
            'ylabel': 'G(r)',
            'confidence_intervals': permutations > 0,  # Add confidence intervals if available
            'ci_low': 'g_low',
            'ci_high': 'g_high',
            'legend': ['Empirical G(r)', 'Theoretical G(r) for CSR']
        }
        
        return result, plot_config
    
    return result



def calculate_pair_correlation_function(spatioloji_obj, max_distance, num_bins=50, 
                                       cell_type1=None, cell_type2=None, cell_type_column=None,
                                       fov_id=None, use_global_coords=True, edge_correction=True, 
                                       n_jobs=-1, monte_carlo=0):
    """
    Calculate the Pair Correlation Function (PCF) to analyze spatial relationships between cells at different distances.
    
    Args:
        spatioloji_obj: A Spatioloji object
        max_distance: Maximum distance to consider for the PCF (in pixels)
        num_bins: Number of distance bins to use
        cell_type1: Optional first cell type for type-specific PCF (None for all cells)
        cell_type2: Optional second cell type for type-specific PCF (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type1/2 is provided)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        edge_correction: Whether to apply edge correction (True) or not (False)
        n_jobs: Number of parallel jobs (-1 for all processors)
        monte_carlo: Number of Monte Carlo simulations for confidence envelope (0 for none)
        
    Returns:
        DataFrame with distances and PCF values
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm  # Optional, for progress tracking
    from shapely.geometry import Polygon
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Get cell type information if needed
    if cell_type1 is not None or cell_type2 is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type1 or cell_type2 is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
    
    # Filter by cell types if specified
    points1 = cells
    points2 = cells
    
    if cell_type1 is not None:
        points1 = cells[cells['cell_type'] == cell_type1]
        if len(points1) == 0:
            raise ValueError(f"No cells of type '{cell_type1}' found")
    
    if cell_type2 is not None:
        points2 = cells[cells['cell_type'] == cell_type2]
        if len(points2) == 0:
            raise ValueError(f"No cells of type '{cell_type2}' found")
    
    # Extract coordinates
    coords1 = points1[[x_col, y_col]].values
    coords2 = points2[[x_col, y_col]].values
    
    # Create distance bins
    bins = np.linspace(0, max_distance, num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_widths = bins[1:] - bins[:-1]
    
    # Calculate study area
    x_min, y_min = np.min(cells[[x_col, y_col]].values, axis=0)
    x_max, y_max = np.max(cells[[x_col, y_col]].values, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    # Create study area polygon for edge correction
    study_area = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    
    # Determine if we're doing a cross-type or same-type analysis
    same_points = cell_type1 is None and cell_type2 is None or cell_type1 == cell_type2
    
    # Calculate density (points per unit area)
    density1 = len(coords1) / area
    density2 = len(coords2) / area
    
    # Function to count pairs in each distance bin
    def count_pairs(i, same_points, coords1, coords2, bins, width, height, edge_correction):
        counts = np.zeros(len(bins) - 1)
        weights = np.zeros(len(bins) - 1)
        
        # Get coordinates of the reference point
        x_i, y_i = coords1[i]
        
        # Calculate distances to all other points
        if same_points:
            # Skip self-comparisons for same-type PCF
            dists = np.sqrt((coords2[:, 0] - x_i)**2 + (coords2[:, 1] - y_i)**2)
            dists = dists[dists > 0]  # Remove self-distance (which is 0)
        else:
            dists = np.sqrt((coords2[:, 0] - x_i)**2 + (coords2[:, 1] - y_i)**2)
        
        # Count pairs in each distance bin
        for j, dist in enumerate(dists):
            if dist > bins[-1]:
                continue
                
            bin_idx = np.digitize(dist, bins) - 1
            if bin_idx < 0 or bin_idx >= len(bins) - 1:
                continue
                
            # Apply edge correction if requested
            if edge_correction:
                # Border method edge correction
                # Calculate the fraction of the circle with radius 'dist' that falls within the study area
                
                # For simplicity, use a geometric approximation:
                # If the distance to any edge is less than the current distance,
                # the circle is partially outside the study area
                dist_to_left = x_i - x_min
                dist_to_right = x_max - x_i
                dist_to_bottom = y_i - y_min
                dist_to_top = y_max - y_i
                
                min_edge_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                
                if min_edge_dist < dist:
                    # Approximate the fraction of circle inside the study area
                    # This is a simplified approach; for high accuracy, one would need to
                    # calculate the actual intersection area
                    if min_edge_dist <= 0:
                        # Point is on or outside the boundary - skip
                        continue
                        
                    # Simple geometric approximation
                    fraction_inside = 0.5 + 0.5 * (min_edge_dist / dist)
                    weight = 1.0 / fraction_inside
                else:
                    weight = 1.0
            else:
                weight = 1.0
            
            counts[bin_idx] += 1
            weights[bin_idx] += weight
        
        return counts, weights
    
    # Process all points in parallel
    num_workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    results = Parallel(n_jobs=num_workers)(
        delayed(count_pairs)(
            i, same_points, coords1, coords2, bins, width, height, edge_correction
        ) for i in tqdm(range(len(coords1)), desc="Calculating PCF")
    )
    
    # Combine results
    all_counts = np.zeros(len(bins) - 1)
    all_weights = np.zeros(len(bins) - 1)
    
    for counts, weights in results:
        all_counts += counts
        all_weights += weights
    
    # Calculate PCF
    # For a completely random (Poisson) process, PCF = 1 at all distances
    # PCF > 1 indicates clustering at that distance
    # PCF < 1 indicates repulsion at that distance
    
    # Expected number of points in annular ring for CSR:
    # E[n(r)] = lambda * 2πr * dr
    # where lambda is intensity (points per unit area)
    
    # Calculate PCF = observed / expected
    if same_points:
        # For same-type PCF, expected counts are based on intensity of single pattern
        intensity_product = density1 * density1
    else:
        # For cross-type PCF, expected counts are based on product of intensities
        intensity_product = density1 * density2
    
    # Calculate areas of annular rings
    ring_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
    
    # Expected counts in each bin under CSR
    expected_counts = intensity_product * ring_areas * len(coords1)
    
    # Calculate PCF values, handling edge correction weights
    pcf_values = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        if expected_counts[i] > 0:
            if edge_correction and all_weights[i] > 0:
                # Apply edge correction weights
                pcf_values[i] = all_counts[i] / all_weights[i] / expected_counts[i]
            else:
                pcf_values[i] = all_counts[i] / expected_counts[i]
        else:
            pcf_values[i] = np.nan
    
    # Create result DataFrame
    result = pd.DataFrame({
        'distance': bin_centers,
        'pcf': pcf_values,
        'observed_count': all_counts,
        'expected_count': expected_counts,
        'bin_width': bin_widths
    })
    
    # Calculate Monte Carlo confidence envelope if requested
    if monte_carlo > 0:
        # Function to simulate a random point pattern
        def simulate_random_pattern(n_points, x_min, y_min, width, height, seed):
            np.random.seed(seed)
            return np.random.uniform(low=[x_min, y_min], high=[x_min + width, y_min + height], size=(n_points, 2))
        
        # Function to calculate PCF for a random pattern
        def calculate_random_pcf(seed, n_points1, n_points2, x_min, y_min, width, height, bins, same_points):
            # Generate random patterns
            random_coords1 = simulate_random_pattern(n_points1, x_min, y_min, width, height, seed)
            
            if same_points:
                random_coords2 = random_coords1
            else:
                random_coords2 = simulate_random_pattern(n_points2, x_min, y_min, width, height, seed + 10000)
            
            # Count pairs for this random pattern
            all_counts = np.zeros(len(bins) - 1)
            
            for i in range(len(random_coords1)):
                x_i, y_i = random_coords1[i]
                
                # Calculate distances to all other points
                if same_points:
                    # Skip self-comparisons for same-type PCF
                    dists = np.sqrt((random_coords2[:, 0] - x_i)**2 + (random_coords2[:, 1] - y_i)**2)
                    dists = dists[dists > 0]  # Remove self-distance (which is 0)
                else:
                    dists = np.sqrt((random_coords2[:, 0] - x_i)**2 + (random_coords2[:, 1] - y_i)**2)
                
                # Count pairs in each distance bin
                for dist in dists:
                    if dist <= bins[-1]:
                        bin_idx = np.digitize(dist, bins) - 1
                        if 0 <= bin_idx < len(bins) - 1:
                            all_counts[bin_idx] += 1
            
            # Calculate PCF values
            pcf_values = np.zeros(len(bins) - 1)
            
            # Expected counts in each bin under CSR
            ring_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
            
            if same_points:
                expected_counts = (n_points1 / area) * (n_points1 / area) * ring_areas * n_points1
            else:
                expected_counts = (n_points1 / area) * (n_points2 / area) * ring_areas * n_points1
            
            for i in range(len(bins) - 1):
                if expected_counts[i] > 0:
                    pcf_values[i] = all_counts[i] / expected_counts[i]
                else:
                    pcf_values[i] = np.nan
            
            return pcf_values
        
        # Run Monte Carlo simulations in parallel
        mc_results = Parallel(n_jobs=num_workers)(
            delayed(calculate_random_pcf)(
                i, len(coords1), len(coords2), x_min, y_min, width, height, bins, same_points
            ) for i in tqdm(range(monte_carlo), desc="Monte Carlo simulations")
        )
        
        # Combine Monte Carlo results
        mc_pcf_values = np.array(mc_results)
        
        # Calculate confidence envelope (5th and 95th percentiles)
        pcf_low = np.percentile(mc_pcf_values, 2.5, axis=0)
        pcf_high = np.percentile(mc_pcf_values, 97.5, axis=0)
        
        # Add to result DataFrame
        result['pcf_low'] = pcf_low
        result['pcf_high'] = pcf_high
        result['significant_clustering'] = result['pcf'] > result['pcf_high']
        result['significant_repulsion'] = result['pcf'] < result['pcf_low']
    
    return result




# Cell Type Interaction Analysis
def calculate_cell_type_correlation(spatioloji_obj, cell_type_column, max_distance, 
                                   distance_bins=10, fov_id=None, use_global_coords=True,
                                   n_jobs=-1, batch_size=100):
    """
    Calculate spatial correlation between different cell types as a function of distance,
    using multithreading for performance optimization.
    
    Args:
        spatioloji_obj: A Spatioloji object
        cell_type_column: Column name in adata.obs containing cell type information
        max_distance: Maximum distance to consider for correlation analysis (in pixels)
        distance_bins: Number of distance bins to divide the analysis into
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        n_jobs: Number of parallel jobs (-1 for all processors)
        batch_size: Number of cells to process in each parallel batch
        
    Returns:
        Dict containing:
            - pair_counts: DataFrame with counts of cell type pairs in each distance bin
            - correlation_matrix: Dict of correlation matrices for each distance bin
            - distance_bins: Array of distance bin edges
    """
    import numpy as np
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm  # Optional, for progress tracking
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Get cell type information from AnnData object
    if cell_type_column not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
    
    # Create a dictionary mapping cell IDs to cell types
    cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
    
    # Add cell type information to the cells DataFrame
    cells['cell_type'] = cells['cell'].map(cell_to_type)
    
    # Drop cells without cell type information
    cells = cells.dropna(subset=['cell_type'])
    
    # Get the unique cell types
    cell_types = cells['cell_type'].unique()
    n_types = len(cell_types)
    
    # Create a mapping from cell type to index
    type_to_idx = {cell_type: i for i, cell_type in enumerate(cell_types)}
    
    # Create distance bins
    distance_bin_edges = np.linspace(0, max_distance, distance_bins + 1)
    distance_bin_centers = 0.5 * (distance_bin_edges[1:] + distance_bin_edges[:-1])
    
    # Extract coordinates and cell type indices
    coords = cells[[x_col, y_col]].values
    type_indices = np.array([type_to_idx[t] for t in cells['cell_type']])
    
    # Determine number of cores to use
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Function to process a batch of cells
    def process_cell_batch(start_idx, end_idx, coords, type_indices, distance_bin_edges, n_types):
        # Initialize local pair counts matrix
        local_pair_counts = np.zeros((distance_bins, n_types, n_types))
        
        # Get the total number of cells
        n_cells = len(coords)
        
        # Process each cell in the batch
        for i in range(start_idx, min(end_idx, n_cells)):
            # Calculate distances from this cell to all other cells
            distances = np.sqrt(np.sum((coords[i] - coords) ** 2, axis=1))
            
            # Process each other cell
            for j in range(n_cells):
                # Skip self-comparisons and pairs we've already counted (to avoid double counting)
                if i >= j:
                    continue
                    
                d = distances[j]
                
                # Skip if distance is greater than max_distance
                if d > max_distance:
                    continue
                    
                # Determine which distance bin this pair falls into
                bin_idx = np.digitize(d, distance_bin_edges) - 1
                
                # Clip to ensure within valid bin range
                if bin_idx >= distance_bins:
                    continue
                    
                # Get cell types of the pair
                type_i = type_indices[i]
                type_j = type_indices[j]
                
                # Increment the count for this cell type pair
                local_pair_counts[bin_idx, type_i, type_j] += 1
                local_pair_counts[bin_idx, type_j, type_i] += 1  # Count in both directions
        
        return local_pair_counts
    
    # Create batches of cells to process
    n_cells = len(cells)
    batch_starts = list(range(0, n_cells, batch_size))
    
    # Process batches in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell_batch)(
            start_idx,
            start_idx + batch_size,
            coords,
            type_indices,
            distance_bin_edges,
            n_types
        ) for start_idx in tqdm(batch_starts, desc="Processing cell batches")
    )
    
    # Combine results from all batches
    pair_counts = np.zeros((distance_bins, n_types, n_types))
    for batch_result in results:
        pair_counts += batch_result
    
    # Calculate correlation matrices for each distance bin
    correlation_matrices = []
    
    for bin_idx in range(distance_bins):
        # Create a correlation matrix for this distance bin
        bin_corr = np.zeros((n_types, n_types))
        
        # Calculate expected counts based on overall frequency
        total_counts = pair_counts[bin_idx].sum()
        if total_counts > 0:
            # Get marginal counts for each cell type
            type_counts = pair_counts[bin_idx].sum(axis=1)
            
            # Calculate expected counts for each pair assuming independence
            expected = np.outer(type_counts, type_counts) / total_counts
            
            # Calculate correlation as (observed - expected) / expected
            # Add a small value to avoid division by zero
            observed = pair_counts[bin_idx]
            with np.errstate(divide='ignore', invalid='ignore'):
                bin_corr = (observed - expected) / (expected + 1e-10)
                
            # Replace NaN values with 0
            bin_corr = np.nan_to_num(bin_corr)
        
        correlation_matrices.append(bin_corr)
    
    # Convert pair_counts to a DataFrame for easier interpretation
    pair_counts_df = []
    
    for bin_idx in range(distance_bins):
        for type_i_idx, type_i in enumerate(cell_types):
            for type_j_idx, type_j in enumerate(cell_types):
                if pair_counts[bin_idx, type_i_idx, type_j_idx] > 0:
                    pair_counts_df.append({
                        'distance_bin': bin_idx,
                        'distance_min': distance_bin_edges[bin_idx],
                        'distance_max': distance_bin_edges[bin_idx + 1],
                        'distance_center': distance_bin_centers[bin_idx],
                        'cell_type_1': type_i,
                        'cell_type_2': type_j,
                        'count': pair_counts[bin_idx, type_i_idx, type_j_idx],
                        'correlation': correlation_matrices[bin_idx][type_i_idx, type_j_idx]
                    })
    
    pair_counts_df = pd.DataFrame(pair_counts_df)
    
    # Create a dictionary mapping bin index to correlation matrix for easier access
    corr_matrix_dict = {
        f'bin_{i}_dist_{distance_bin_centers[i]:.1f}px': pd.DataFrame(
            correlation_matrices[i],
            index=cell_types,
            columns=cell_types
        ) for i in range(distance_bins)
    }
    
    return {
        'pair_counts': pair_counts_df,
        'correlation_matrices': corr_matrix_dict,
        'distance_bins': distance_bin_edges,
        'cell_types': cell_types
    }

def calculate_colocation_quotient(spatioloji_obj, cell_type_column, distance_threshold,
                                 fov_id=None, use_global_coords=True, normalize=True,
                                 bootstrap=0, confidence=0.95, n_jobs=-1, plot_result=False):
    """
    Calculate the Colocation Quotient (CLQ) to measure spatial relationships between different cell types.
    
    Args:
        spatioloji_obj: A Spatioloji object
        cell_type_column: Column name in adata.obs containing cell type information
        distance_threshold: Maximum distance (in pixels) to consider cells as co-located
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        normalize: Whether to normalize the CLQ by the overall cell type frequencies
        bootstrap: Number of bootstrap iterations for confidence interval (0 for no bootstrapping)
        confidence: Confidence level for bootstrap interval (default: 0.95)
        n_jobs: Number of parallel jobs for bootstrap calculation (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        DataFrame with CLQ values for each pair of cell types
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy import stats
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Get cell type information from AnnData object
    if cell_type_column not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
    
    # Create a dictionary mapping cell IDs to cell types
    cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
    
    # Add cell type information to the cells DataFrame
    cells['cell_type'] = cells['cell'].map(cell_to_type)
    
    # Drop cells without cell type information
    cells = cells.dropna(subset=['cell_type'])
    
    # Get the unique cell types
    cell_types = cells['cell_type'].unique()
    n_types = len(cell_types)
    
    # Extract coordinates and cell types
    coords = cells[[x_col, y_col]].values
    types = cells['cell_type'].values
    
    # Calculate distance matrix
    dist_matrix = distance.squareform(distance.pdist(coords))
    
    # Create a matrix to count co-located pairs
    coloc_counts = np.zeros((n_types, n_types))
    
    # Calculate co-location for each cell type pair
    for i, type_i in enumerate(cell_types):
        # Get indices of cells of type i
        indices_i = np.where(types == type_i)[0]
        
        for j, type_j in enumerate(cell_types):
            # Get indices of cells of type j
            indices_j = np.where(types == type_j)[0]
            
            # Skip if either cell type has no cells
            if len(indices_i) == 0 or len(indices_j) == 0:
                continue
            
            # Count co-located pairs
            if i == j:  # Same cell type
                # Subtract the number of cells to account for self-pairs
                n_pairs = 0
                for idx_i in indices_i:
                    n_pairs += np.sum(dist_matrix[idx_i, indices_j] <= distance_threshold) - 1
                coloc_counts[i, j] = n_pairs / 2  # Divide by 2 to avoid double counting
            else:  # Different cell types
                # Count pairs where distance <= threshold
                n_pairs = 0
                for idx_i in indices_i:
                    n_pairs += np.sum(dist_matrix[idx_i, indices_j] <= distance_threshold)
                coloc_counts[i, j] = n_pairs
    
    # Calculate Colocation Quotient
    # CLQ = observed co-locations / expected co-locations
    
    # Calculate overall cell type frequencies
    type_counts = np.array([np.sum(types == t) for t in cell_types])
    type_freqs = type_counts / np.sum(type_counts)
    
    # Calculate expected co-locations based on cell type frequencies
    expected_coloc = np.zeros((n_types, n_types))
    
    if normalize:
        # Calculate total number of valid cell pairs
        total_pairs = np.sum(coloc_counts)
        
        # Calculate expected co-locations based on random mixing
        for i in range(n_types):
            for j in range(n_types):
                if i == j:
                    # Expected number of same-type pairs
                    expected_coloc[i, j] = total_pairs * type_freqs[i] * type_freqs[j]
                else:
                    # Expected number of different-type pairs
                    expected_coloc[i, j] = total_pairs * type_freqs[i] * type_freqs[j]
    else:
        # Simple counts without normalization
        expected_coloc = np.ones_like(coloc_counts)
    
    # Calculate CLQ
    with np.errstate(divide='ignore', invalid='ignore'):
        clq = coloc_counts / expected_coloc
    
    # Replace NaN values with 0 (when expected count is 0)
    clq = np.nan_to_num(clq)
    
    # Create result DataFrame
    clq_df = pd.DataFrame(clq, index=cell_types, columns=cell_types)
    
    # Calculate log2 CLQ for more intuitive visualization
    log2_clq = np.log2(clq)
    log2_clq_df = pd.DataFrame(log2_clq, index=cell_types, columns=cell_types)
    
    # Replace infinite values with a large value for visualization
    log2_clq_df = log2_clq_df.replace([np.inf, -np.inf], [10, -10])
    
    # Calculate bootstrap confidence intervals if requested
    if bootstrap > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate CLQ for a bootstrap sample
        def bootstrap_clq(seed, coords, types, cell_types, n_types, distance_threshold, normalize):
            np.random.seed(seed)
            
            # Resample cells with replacement
            bootstrap_indices = np.random.choice(range(len(coords)), size=len(coords), replace=True)
            bootstrap_coords = coords[bootstrap_indices]
            bootstrap_types = types[bootstrap_indices]
            
            # Calculate distance matrix for bootstrap sample
            dist_matrix = distance.squareform(distance.pdist(bootstrap_coords))
            
            # Create a matrix to count co-located pairs
            coloc_counts = np.zeros((n_types, n_types))
            
            # Calculate co-location for each cell type pair
            for i, type_i in enumerate(cell_types):
                # Get indices of cells of type i
                indices_i = np.where(bootstrap_types == type_i)[0]
                
                for j, type_j in enumerate(cell_types):
                    # Get indices of cells of type j
                    indices_j = np.where(bootstrap_types == type_j)[0]
                    
                    # Skip if either cell type has no cells
                    if len(indices_i) == 0 or len(indices_j) == 0:
                        continue
                    
                    # Count co-located pairs
                    if i == j:  # Same cell type
                        # Subtract the number of cells to account for self-pairs
                        n_pairs = 0
                        for idx_i in indices_i:
                            n_pairs += np.sum(dist_matrix[idx_i, indices_j] <= distance_threshold) - 1
                        coloc_counts[i, j] = n_pairs / 2  # Divide by 2 to avoid double counting
                    else:  # Different cell types
                        # Count pairs where distance <= threshold
                        n_pairs = 0
                        for idx_i in indices_i:
                            n_pairs += np.sum(dist_matrix[idx_i, indices_j] <= distance_threshold)
                        coloc_counts[i, j] = n_pairs
            
            # Calculate overall cell type frequencies
            type_counts = np.array([np.sum(bootstrap_types == t) for t in cell_types])
            type_freqs = type_counts / np.sum(type_counts)
            
            # Calculate expected co-locations based on cell type frequencies
            expected_coloc = np.zeros((n_types, n_types))
            
            if normalize:
                # Calculate total number of valid cell pairs
                total_pairs = np.sum(coloc_counts)
                
                # Calculate expected co-locations based on random mixing
                for i in range(n_types):
                    for j in range(n_types):
                        if i == j:
                            # Expected number of same-type pairs
                            expected_coloc[i, j] = total_pairs * type_freqs[i] * type_freqs[j]
                        else:
                            # Expected number of different-type pairs
                            expected_coloc[i, j] = total_pairs * type_freqs[i] * type_freqs[j]
            else:
                # Simple counts without normalization
                expected_coloc = np.ones_like(coloc_counts)
            
            # Calculate CLQ
            with np.errstate(divide='ignore', invalid='ignore'):
                clq = coloc_counts / expected_coloc
            
            # Replace NaN values with 0
            clq = np.nan_to_num(clq)
            
            return clq
        
        # Run bootstrap iterations in parallel
        bootstrap_results = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_clq)(
                i, coords, types, cell_types, n_types, distance_threshold, normalize
            ) for i in tqdm(range(bootstrap), desc="Bootstrap iterations")
        )
        
        # Convert list of arrays to 3D array
        bootstrap_clqs = np.array(bootstrap_results)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        clq_lower = np.zeros((n_types, n_types))
        clq_upper = np.zeros((n_types, n_types))
        
        for i in range(n_types):
            for j in range(n_types):
                clq_lower[i, j] = np.percentile(bootstrap_clqs[:, i, j], lower_percentile)
                clq_upper[i, j] = np.percentile(bootstrap_clqs[:, i, j], upper_percentile)
        
        # Create DataFrames for confidence intervals
        clq_lower_df = pd.DataFrame(clq_lower, index=cell_types, columns=cell_types)
        clq_upper_df = pd.DataFrame(clq_upper, index=cell_types, columns=cell_types)
        
        # Calculate statistical significance
        # CLQ is significantly different from 1 if confidence interval doesn't include 1
        significance = np.zeros((n_types, n_types), dtype=bool)
        
        for i in range(n_types):
            for j in range(n_types):
                # Check if confidence interval includes 1
                significance[i, j] = not (clq_lower[i, j] <= 1 <= clq_upper[i, j])
        
        significance_df = pd.DataFrame(significance, index=cell_types, columns=cell_types)
    else:
        clq_lower_df = None
        clq_upper_df = None
        significance_df = None
    
    # Interpret CLQ values
    interpretation = np.zeros((n_types, n_types), dtype=object)
    
    for i in range(n_types):
        for j in range(n_types):
            clq_value = clq[i, j]
            
            if clq_value > 1.5:
                interpretation[i, j] = "Strong co-location"
            elif clq_value > 1.2:
                interpretation[i, j] = "Moderate co-location"
            elif clq_value > 1.0:
                interpretation[i, j] = "Slight co-location"
            elif clq_value >= 0.8:
                interpretation[i, j] = "Random mixing"
            elif clq_value >= 0.5:
                interpretation[i, j] = "Slight avoidance"
            elif clq_value > 0:
                interpretation[i, j] = "Strong avoidance"
            else:
                interpretation[i, j] = "No interaction detected"
    
    interpretation_df = pd.DataFrame(interpretation, index=cell_types, columns=cell_types)
    
    # Create a composite result object
    result = {
        'clq': clq_df,
        'log2_clq': log2_clq_df,
        'interpretation': interpretation_df,
        'distance_threshold': distance_threshold,
        'normalize': normalize,
        'n_cell_types': n_types,
        'cell_types': cell_types,
        'cell_type_frequencies': pd.Series(type_freqs, index=cell_types),
        'clq_lower': clq_lower_df,
        'clq_upper': clq_upper_df,
        'significance': significance_df
    }
    
    # Generate plot configuration if requested
    if plot_result:
        # Custom diverging colormap: blue (avoidance) -> white (random) -> red (co-location)
        cmap = LinearSegmentedColormap.from_list(
            'custom_diverging',
            ['#0000FF', '#AAAAFF', '#FFFFFF', '#FFAAAA', '#FF0000'],
            N=256
        )
        
        # Define plot configuration
        plot_config = {
            'data': log2_clq_df,
            'colormap': cmap,
            'vmin': -2,  # log2(0.25) - strong avoidance
            'vmax': 2,   # log2(4) - strong co-location
            'center': 0, # log2(1) - random mixing
            'title': 'Co-location Quotient (log2 scale)',
            'subtitle': f'Distance threshold: {distance_threshold} pixels',
            'xlabel': 'Cell Type',
            'ylabel': 'Cell Type',
            'text_format': '.2f'
        }
        
        # Plot annotations for statistical significance if available
        if significance_df is not None:
            plot_config['annotations'] = significance_df
            plot_config['annotation_text'] = '*'
        
        return result, plot_config
    
    return result

def calculate_proximity_analysis(spatioloji_obj, reference_cell_type, target_cell_types=None,
                               max_distance=None, distance_bins=10, fov_id=None, 
                               use_global_coords=True, use_polygons=False, cell_type_column=None,
                               permutations=999, edge_correction=True, n_jobs=-1, plot_result=False):
    """
    Calculate proximity analysis to quantify spatial relationships between specific cell types.
    
    Args:
        spatioloji_obj: A Spatioloji object
        reference_cell_type: Cell type to use as reference points
        target_cell_types: List of cell types to analyze proximity to (None for all cell types)
        max_distance: Maximum distance to consider (in pixels) (None for auto-determination)
        distance_bins: Number of distance bins for proximity function
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        use_polygons: Whether to use cell polygons (True) or cell centers (False) for distance calculations
        cell_type_column: Column name in adata.obs containing cell type information
        permutations: Number of Monte Carlo simulations for significance testing (0 for none)
        edge_correction: Whether to apply edge correction for boundary effects (True) or not (False)
        n_jobs: Number of parallel jobs for permutations (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        Dict containing proximity statistics and spatial relationships between cell types
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy import stats
    from shapely.geometry import Point, Polygon
    
    # Validate input parameters
    if cell_type_column is None:
        raise ValueError("cell_type_column must be provided for proximity analysis")
        
    if cell_type_column not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
    
    # Determine which coordinates to use
    if use_global_coords:
        x_center_col = 'CenterX_global_px'
        y_center_col = 'CenterY_global_px'
        x_poly_col = 'x_global_px'
        y_poly_col = 'y_global_px'
    else:
        x_center_col = 'CenterX_local_px'
        y_center_col = 'CenterY_local_px'
        x_poly_col = 'x_local_px'
        y_poly_col = 'y_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_center_col not in cells.columns or y_center_col not in cells.columns:
        raise ValueError(f"Required columns {x_center_col} and {y_center_col} not found in cell metadata")
    
    # Check if we have polygons data if using polygons
    if use_polygons and (spatioloji_obj.polygons is None or len(spatioloji_obj.polygons) == 0):
        raise ValueError("Polygon data not available in the Spatioloji object")
    
    # Get cell type information
    cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
    
    # Add cell type information to the cells DataFrame
    cells['cell_type'] = cells['cell'].map(cell_to_type)
    
    # Drop cells without cell type information
    cells = cells.dropna(subset=['cell_type'])
    
    # Validate reference cell type
    if reference_cell_type not in cells['cell_type'].unique():
        raise ValueError(f"Reference cell type '{reference_cell_type}' not found in the data")
    
    # Determine target cell types
    if target_cell_types is None:
        target_cell_types = sorted(cells['cell_type'].unique())
    else:
        # Validate target cell types
        invalid_types = set(target_cell_types) - set(cells['cell_type'].unique())
        if invalid_types:
            raise ValueError(f"The following target cell types were not found in the data: {invalid_types}")
    
    # Extract reference and target cells
    reference_cells = cells[cells['cell_type'] == reference_cell_type]
    target_cells_by_type = {ct: cells[cells['cell_type'] == ct] for ct in target_cell_types}
    
    # Check if we have enough reference cells
    if len(reference_cells) < 3:
        raise ValueError(f"Not enough cells of reference type '{reference_cell_type}' (need at least 3)")
    
    # Check if we have enough target cells for each type
    for ct, target_cells in target_cells_by_type.items():
        if len(target_cells) < 3:
            raise ValueError(f"Not enough cells of target type '{ct}' (need at least 3)")
    
    # Extract coordinates
    reference_coords = reference_cells[[x_center_col, y_center_col]].values
    target_coords_by_type = {ct: tc[[x_center_col, y_center_col]].values 
                           for ct, tc in target_cells_by_type.items()}
    
    # Create polygons if using polygons
    reference_polygons = {}
    target_polygons_by_type = {ct: {} for ct in target_cell_types}
    
    if use_polygons:
        # Create polygons for reference cells
        for idx, cell in reference_cells.iterrows():
            cell_id = cell['cell']
            cell_poly_data = spatioloji_obj.get_polygon_for_cell(cell_id)
            
            # Skip if no polygon data for this cell
            if len(cell_poly_data) == 0:
                continue
                
            # Create polygon
            try:
                points = list(zip(cell_poly_data[x_poly_col], cell_poly_data[y_poly_col]))
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    reference_polygons[cell_id] = Polygon(points)
            except Exception as e:
                print(f"Warning: Could not create polygon for reference cell {cell_id}: {e}")
        
        # Create polygons for target cells
        for ct, tc in target_cells_by_type.items():
            for idx, cell in tc.iterrows():
                cell_id = cell['cell']
                cell_poly_data = spatioloji_obj.get_polygon_for_cell(cell_id)
                
                # Skip if no polygon data for this cell
                if len(cell_poly_data) == 0:
                    continue
                    
                # Create polygon
                try:
                    points = list(zip(cell_poly_data[x_poly_col], cell_poly_data[y_poly_col]))
                    if len(points) >= 3:  # Need at least 3 points for a polygon
                        target_polygons_by_type[ct][cell_id] = Polygon(points)
                except Exception as e:
                    print(f"Warning: Could not create polygon for target cell {cell_id}: {e}")
    
    # Determine study area for edge correction and random point generation
    x_min, y_min = cells[[x_center_col, y_center_col]].min().values
    x_max, y_max = cells[[x_center_col, y_center_col]].max().values
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    # Create study area polygon for edge correction
    study_area = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    
    # Determine maximum distance if not specified
    if max_distance is None:
        # Use one-quarter of the study area diagonal as default
        max_distance = np.sqrt(width**2 + height**2) / 4
    
    # Create distance bins
    distance_edges = np.linspace(0, max_distance, distance_bins + 1)
    distance_centers = 0.5 * (distance_edges[1:] + distance_edges[:-1])
    
    # Function to calculate distances between reference and target cells
    def calculate_distances(reference_cells, target_cells, use_polygons, reference_polygons, target_polygons):
        distances = []
        
        if use_polygons:
            # Calculate distances between polygons
            for ref_id, ref_polygon in reference_polygons.items():
                for target_id, target_polygon in target_polygons.items():
                    try:
                        # Calculate distance between polygons
                        dist = ref_polygon.distance(target_polygon)
                        distances.append(dist)
                    except Exception as e:
                        print(f"Warning: Error calculating distance between {ref_id} and {target_id}: {e}")
        else:
            # Calculate distances between cell centers
            for i in range(len(reference_cells)):
                ref_point = reference_cells[i]
                
                for j in range(len(target_cells)):
                    target_point = target_cells[j]
                    
                    # Calculate Euclidean distance
                    dist = np.sqrt(np.sum((ref_point - target_point)**2))
                    distances.append(dist)
        
        return np.array(distances)
    
    # Calculate proximity function for each target cell type
    proximity_stats = {}
    
    for ct in target_cell_types:
        # Skip self-comparisons if reference and target are the same
        if ct == reference_cell_type:
            continue
            
        # Get target coordinates
        target_coords = target_coords_by_type[ct]
        
        # Calculate distances
        if use_polygons:
            distances = calculate_distances(
                reference_cells['cell'].values,
                target_cells_by_type[ct]['cell'].values,
                True,
                reference_polygons,
                target_polygons_by_type[ct]
            )
        else:
            distances = calculate_distances(
                reference_coords,
                target_coords,
                False,
                None,
                None
            )
        
        # Calculate proximity function
        # This is the cumulative distribution of distances (similar to G-function)
        hist, _ = np.histogram(distances, bins=distance_edges)
        cumulative_hist = np.cumsum(hist)
        
        # Normalize by the number of reference-target pairs
        n_ref = len(reference_cells)
        n_target = len(target_cells_by_type[ct])
        n_pairs = n_ref * n_target
        
        # Apply edge correction if requested
        if edge_correction and n_pairs > 0:
            # Calculate area of buffer at each distance
            buffer_areas = np.pi * distance_edges[1:]**2
            
            # Calculate proportion of buffer area inside study area for edge distances
            # This is a simple approximation; more sophisticated methods exist
            edge_correction_factors = np.ones(len(buffer_areas))
            
            for i, d in enumerate(distance_edges[1:]):
                # Simple correction based on buffer area and study area overlap
                buffer_area = buffer_areas[i]
                # Approximate correction factor as ratio of areas
                edge_correction_factors[i] = min(1.0, area / buffer_area)
            
            # Apply correction to cumulative histogram
            cumulative_hist = cumulative_hist / edge_correction_factors
            
        # Normalize by total count
        if n_pairs > 0:
            normalized_function = cumulative_hist / n_pairs
        else:
            normalized_function = np.zeros_like(cumulative_hist)
        
        # Calculate theoretical function for Complete Spatial Randomness (CSR)
        # For CSR, G(r) = 1 - exp(-lambda * pi * r^2)
        # where lambda is the intensity of the target point process
        intensity = n_target / area
        theoretical_function = 1 - np.exp(-intensity * np.pi * distance_centers**2)
        
        # Calculate difference from CSR
        diff_from_csr = normalized_function - theoretical_function
        
        # Calculate summary statistics
        nearest_distances = np.sort(distances)[:min(5, len(distances))]
        mean_nearest = np.mean(nearest_distances)
        min_distance = np.min(distances) if len(distances) > 0 else np.nan
        
        # Store results
        proximity_stats[ct] = {
            'reference_type': reference_cell_type,
            'target_type': ct,
            'n_reference': n_ref,
            'n_target': n_target,
            'distances': distances,
            'distance_edges': distance_edges,
            'distance_centers': distance_centers,
            'proximity_function': normalized_function,
            'theoretical_function': theoretical_function,
            'diff_from_csr': diff_from_csr,
            'min_distance': min_distance,
            'mean_nearest': mean_nearest
        }
    
    # Calculate Monte Carlo significance if requested
    if permutations > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to generate random points
        def generate_random_points(n, x_min, y_min, width, height):
            return np.column_stack([
                np.random.uniform(x_min, x_min + width, n),
                np.random.uniform(y_min, y_min + height, n)
            ])
        
        # Function to calculate proximity function for random points
        def calculate_random_proximity(seed, n_reference, n_target, x_min, y_min, width, height, 
                                    distance_edges, area, edge_correction, study_area):
            np.random.seed(seed)
            
            # Generate random coordinates
            random_ref = generate_random_points(n_reference, x_min, y_min, width, height)
            random_target = generate_random_points(n_target, x_min, y_min, width, height)
            
            # Calculate distances between random points
            distances = calculate_distances(random_ref, random_target, False, None, None)
            
            # Calculate proximity function
            hist, _ = np.histogram(distances, bins=distance_edges)
            cumulative_hist = np.cumsum(hist)
            
            # Apply edge correction if requested
            if edge_correction and n_reference * n_target > 0:
                # Calculate area of buffer at each distance
                buffer_areas = np.pi * distance_edges[1:]**2
                
                # Calculate proportion of buffer area inside study area for edge distances
                edge_correction_factors = np.ones(len(buffer_areas))
                
                for i, d in enumerate(distance_edges[1:]):
                    # Simple correction based on buffer area and study area overlap
                    buffer_area = buffer_areas[i]
                    # Approximate correction factor as ratio of areas
                    edge_correction_factors[i] = min(1.0, area / buffer_area)
                
                # Apply correction to cumulative histogram
                cumulative_hist = cumulative_hist / edge_correction_factors
            
            # Normalize by total count
            if n_reference * n_target > 0:
                normalized_function = cumulative_hist / (n_reference * n_target)
            else:
                normalized_function = np.zeros_like(cumulative_hist)
            
            return normalized_function
        
        # Calculate random proximity functions for each target cell type
        for ct in target_cell_types:
            # Skip self-comparisons
            if ct == reference_cell_type:
                continue
                
            # Get counts
            n_ref = proximity_stats[ct]['n_reference']
            n_target = proximity_stats[ct]['n_target']
            
            # Run permutations in parallel
            random_functions = Parallel(n_jobs=n_jobs)(
                delayed(calculate_random_proximity)(
                    i, n_ref, n_target, x_min, y_min, width, height, 
                    distance_edges, area, edge_correction, study_area
                ) for i in tqdm(range(permutations), desc=f"Monte Carlo simulations for {ct}")
            )
            
            # Convert to numpy array
            random_functions = np.array(random_functions)
            
            # Calculate confidence envelopes
            low_envelope = np.percentile(random_functions, 2.5, axis=0)
            high_envelope = np.percentile(random_functions, 97.5, axis=0)
            
            # Determine if observed function is significantly different from CSR
            observed = proximity_stats[ct]['proximity_function']
            is_clustered = observed > high_envelope
            is_dispersed = observed < low_envelope
            
            # Add to results
            proximity_stats[ct]['low_envelope'] = low_envelope
            proximity_stats[ct]['high_envelope'] = high_envelope
            proximity_stats[ct]['is_clustered'] = is_clustered
            proximity_stats[ct]['is_dispersed'] = is_dispersed
            proximity_stats[ct]['is_significant'] = np.logical_or(is_clustered, is_dispersed)
    
    # Calculate attraction/repulsion scores
    attraction_scores = {}
    
    for ct, stats in proximity_stats.items():
        # Calculate attraction score as average difference from CSR
        # Positive scores indicate attraction, negative scores indicate repulsion
        attraction_score = np.mean(stats['diff_from_csr'])
        
        # Determine if attraction/repulsion is significant
        if 'is_significant' in stats:
            is_significant = np.any(stats['is_significant'])
        else:
            # If no Monte Carlo simulations, use heuristic
            is_significant = abs(attraction_score) > 0.1
        
        # Store attraction score
        attraction_scores[ct] = {
            'reference_type': reference_cell_type,
            'target_type': ct,
            'attraction_score': attraction_score,
            'is_significant': is_significant,
            'pattern': 'Random' if not is_significant else ('Attraction' if attraction_score > 0 else 'Repulsion')
        }
    
    # Generate plot configuration if requested
    plot_config = None
    
    if plot_result:
        # Create a multi-line plot of proximity functions
        plot_config = {
            'type': 'line',
            'title': f'Proximity Analysis from {reference_cell_type} to Other Cell Types',
            'xlabel': 'Distance (pixels)',
            'ylabel': 'Proximity Function G(r)',
            'reference_line': None,  # No reference line
            'multi_line': True,  # Multiple lines on one plot
            'lines': []
        }
        
        # Add a line for each target cell type
        for ct, stats in proximity_stats.items():
            line = {
                'x': stats['distance_centers'],
                'y': stats['proximity_function'],
                'label': f'{reference_cell_type} to {ct}',
                'theoretical': stats['theoretical_function']  # Add theoretical line
            }
            
            # Add confidence envelopes if available
            if 'low_envelope' in stats:
                line['low_envelope'] = stats['low_envelope']
                line['high_envelope'] = stats['high_envelope']
            
            plot_config['lines'].append(line)
    
    # Create summary DataFrame of attraction scores
    attraction_df = pd.DataFrame(list(attraction_scores.values()))
    
    # Return results
    return {
        'proximity_stats': proximity_stats,
        'attraction_scores': attraction_df,
        'max_distance': max_distance,
        'reference_cell_type': reference_cell_type,
        'target_cell_types': target_cell_types,
        'plot_config': plot_config
    }





# Heterogeneity and Clustering
def calculate_morisita_index(spatioloji_obj, grid_size, fov_id=None, use_global_coords=True, 
                            cell_type=None, cell_type_column=None, bootstrap=0, confidence=0.95,
                            n_jobs=-1):
    """
    Calculate Morisita's Index of Dispersion to analyze the spatial distribution pattern,
    with multithreaded bootstrap confidence interval calculation.
    
    Args:
        spatioloji_obj: A Spatioloji object
        grid_size: Size of the grid cells in pixels for quadrat counting
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        cell_type: Optional cell type to filter cells (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        bootstrap: Number of bootstrap iterations for confidence interval (0 for no bootstrapping)
        confidence: Confidence level for bootstrap interval (default: 0.95)
        n_jobs: Number of parallel jobs for bootstrap calculation (-1 for all processors)
        
    Returns:
        Dict containing Morisita's index and related statistics
    """
    import numpy as np
    from scipy import stats
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm  # Optional, for progress tracking
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Filter by cell type if specified
    if cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
        
        # Filter to only include cells of the specified type
        cells = cells[cells['cell_type'] == cell_type]
        
        if len(cells) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    
    # Extract coordinates
    coords = cells[[x_col, y_col]].values
    
    # Determine the boundaries of the study area
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    # Create grid cells (quadrats)
    x_grid = np.arange(x_min, x_max + grid_size, grid_size)
    y_grid = np.arange(y_min, y_max + grid_size, grid_size)
    
    # Count cells in each quadrat
    x_indices = np.digitize(coords[:, 0], x_grid) - 1
    y_indices = np.digitize(coords[:, 1], y_grid) - 1
    
    # Create a 2D histogram of cell counts per quadrat
    quadrat_counts = np.zeros((len(y_grid), len(x_grid)))
    for i in range(len(coords)):
        x_idx = x_indices[i]
        y_idx = y_indices[i]
        if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
            quadrat_counts[y_idx, x_idx] += 1
    
    # Flatten the array for calculations
    counts = quadrat_counts.flatten()
    
    # Remove empty quadrats (Morisita's index calculation should only use non-empty quadrats)
    counts = counts[counts > 0]
    
    # Calculate Morisita's index
    num_points = len(coords)
    num_quadrats = len(counts)
    
    if num_quadrats <= 1 or num_points <= 1:
        return {
            'morisita_index': np.nan,
            'interpretation': 'Not enough data for calculation',
            'num_points': num_points,
            'num_quadrats': num_quadrats,
            'bootstrap_results': None
        }
    
    # Calculate Morisita's index: Id = Q * (∑n_i(n_i-1)) / (N(N-1))
    # where Q is number of quadrats, n_i is count in quadrat i, and N is total count
    sum_ni_ni_minus_1 = np.sum(counts * (counts - 1))
    morisita_index = num_quadrats * sum_ni_ni_minus_1 / (num_points * (num_points - 1))
    
    # Calculate standardized Morisita index (Ip)
    # This scales the index to a range from -1 to +1
    # with 0 representing a random distribution
    mu = num_points / num_quadrats  # Expected count per quadrat under CSR
    variance = np.var(counts)
    
    # Calculate chi-square statistic
    chi_square = num_quadrats * variance / mu
    
    # Calculate standardized Morisita index
    if morisita_index >= 1.0:
        # Pattern is clumped
        mc = (chi_square - num_quadrats) / ((num_points * num_quadrats) - num_quadrats)
        standardized_index = 0.5 + 0.5 * (morisita_index - 1) / (mc - 1)
    else:
        # Pattern is uniform
        mu = (chi_square - 1) / (num_quadrats - 1)
        standardized_index = 0.5 * (morisita_index - 1) / (mu - 1)
    
    # Interpret the index
    if morisita_index < 1:
        interpretation = "Uniform distribution"
    elif morisita_index == 1:
        interpretation = "Random distribution"
    else:
        interpretation = "Clustered distribution"
    
    # Calculate bootstrap confidence interval if requested, using multithreading
    bootstrap_results = None
    if bootstrap > 0:
        # Define a function to compute a single bootstrap sample
        def compute_bootstrap_sample(seed, coords, x_grid, y_grid):
            np.random.seed(seed)  # Set seed for reproducibility
            
            # Resample points with replacement
            bootstrap_sample = np.random.choice(range(len(coords)), size=len(coords), replace=True)
            bootstrap_coords = coords[bootstrap_sample]
            
            # Count cells in each quadrat for this bootstrap sample
            x_indices = np.digitize(bootstrap_coords[:, 0], x_grid) - 1
            y_indices = np.digitize(bootstrap_coords[:, 1], y_grid) - 1
            
            # Create a 2D histogram of cell counts per quadrat
            bootstrap_counts = np.zeros((len(y_grid), len(x_grid)))
            for i in range(len(bootstrap_coords)):
                x_idx = x_indices[i]
                y_idx = y_indices[i]
                if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
                    bootstrap_counts[y_idx, x_idx] += 1
            
            # Flatten and remove empty quadrats
            b_counts = bootstrap_counts.flatten()
            b_counts = b_counts[b_counts > 0]
            
            if len(b_counts) <= 1:
                return None
                
            # Calculate Morisita's index for this bootstrap sample
            b_num_points = len(bootstrap_coords)
            b_num_quadrats = len(b_counts)
            b_sum_ni_ni_minus_1 = np.sum(b_counts * (b_counts - 1))
            
            # Check for division by zero
            if b_num_points <= 1 or b_num_quadrats <= 0:
                return None
                
            b_index = b_num_quadrats * b_sum_ni_ni_minus_1 / (b_num_points * (b_num_points - 1))
            
            return b_index
        
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Run bootstrap iterations in parallel
        bootstrap_indices = Parallel(n_jobs=n_jobs)(
            delayed(compute_bootstrap_sample)(
                i, coords, x_grid, y_grid
            ) for i in tqdm(range(bootstrap), desc="Bootstrap iterations")
        )
        
        # Filter out None values
        bootstrap_indices = [idx for idx in bootstrap_indices if idx is not None]
        
        # Calculate confidence interval
        if bootstrap_indices:
            alpha = 1 - confidence
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            ci_lower = np.percentile(bootstrap_indices, lower_percentile)
            ci_upper = np.percentile(bootstrap_indices, upper_percentile)
            
            bootstrap_results = {
                'bootstrap_samples': len(bootstrap_indices),
                'confidence_level': confidence,
                'confidence_interval': (ci_lower, ci_upper),
                'bootstrap_mean': np.mean(bootstrap_indices),
                'bootstrap_std': np.std(bootstrap_indices)
            }
    
    return {
        'morisita_index': morisita_index,
        'standardized_index': standardized_index,
        'interpretation': interpretation,
        'chi_square': chi_square,
        'num_points': num_points,
        'num_quadrats': num_quadrats,
        'grid_size': grid_size,
        'bootstrap_results': bootstrap_results
    }

def calculate_quadrat_variance(spatioloji_obj, min_grid_size, max_grid_size, num_sizes=10, 
                              fov_id=None, use_global_coords=True, cell_type=None, 
                              cell_type_column=None, n_jobs=-1):
    """
    Calculate how variance of cell counts changes with quadrat size (quadrat variance analysis).
    
    Args:
        spatioloji_obj: A Spatioloji object
        min_grid_size: Minimum size of the grid cells in pixels
        max_grid_size: Maximum size of the grid cells in pixels
        num_sizes: Number of different grid sizes to test
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        cell_type: Optional cell type to filter cells (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        n_jobs: Number of parallel jobs (-1 for all processors)
        
    Returns:
        DataFrame with grid sizes and corresponding variance metrics
    """
    import numpy as np
    import pandas as pd
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm  # Optional, for progress tracking
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Filter by cell type if specified
    if cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
        
        # Filter to only include cells of the specified type
        cells = cells[cells['cell_type'] == cell_type]
        
        if len(cells) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    
    # Extract coordinates
    coords = cells[[x_col, y_col]].values
    
    # Determine the boundaries of the study area
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    # Generate grid sizes to test
    grid_sizes = np.linspace(min_grid_size, max_grid_size, num_sizes)
    
    # Determine number of cores to use
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Function to process a single grid size
    def process_grid_size(grid_size, coords, x_min, y_min, x_max, y_max):
        # Create grid for this size
        x_grid = np.arange(x_min, x_max + grid_size, grid_size)
        y_grid = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Count cells in each quadrat
        x_indices = np.digitize(coords[:, 0], x_grid) - 1
        y_indices = np.digitize(coords[:, 1], y_grid) - 1
        
        # Create a 2D histogram of cell counts per quadrat
        quadrat_counts = np.zeros((len(y_grid), len(x_grid)))
        for i in range(len(coords)):
            x_idx = x_indices[i]
            y_idx = y_indices[i]
            if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
                quadrat_counts[y_idx, x_idx] += 1
        
        # Flatten the array for calculations
        counts = quadrat_counts.flatten()
        
        # Calculate metrics
        mean_count = np.mean(counts)
        variance = np.var(counts)
        
        # Calculate variance-to-mean ratio (VMR) or Index of Dispersion
        vmr = variance / mean_count if mean_count > 0 else np.nan
        
        # Calculate other metrics
        # 1. Lloyd's Index of Patchiness
        lloyds_index = 1 + (variance - mean_count) / (mean_count**2) if mean_count > 0 else np.nan
        
        # 2. Green's Index (ranges from -1/(n-1) to 1)
        n = len(counts)
        greens_index = (variance / mean_count - 1) / (n - 1) if mean_count > 0 and n > 1 else np.nan
        
        # 3. Quadrat occupancy (proportion of quadrats with at least one cell)
        occupancy = np.sum(counts > 0) / len(counts)
        
        return {
            'grid_size': grid_size,
            'mean_count': mean_count,
            'variance': variance,
            'vmr': vmr,  # Variance-to-Mean Ratio
            'lloyds_index': lloyds_index,
            'greens_index': greens_index,
            'occupancy': occupancy,
            'num_quadrats': len(counts),
            'num_occupied_quadrats': np.sum(counts > 0),
            'max_count': np.max(counts),
            'quadrat_area': grid_size**2
        }
    
    # Process all grid sizes in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_grid_size)(
            grid_size, coords, x_min, y_min, x_max, y_max
        ) for grid_size in tqdm(grid_sizes, desc="Processing grid sizes")
    )
    
    # Combine results into a DataFrame
    result_df = pd.DataFrame(results)
    
    # Calculate normalized variance (to help compare across scales)
    result_df['normalized_variance'] = result_df['variance'] / result_df['mean_count']
    
    # Calculate expected Poisson variance (which equals the mean)
    result_df['expected_variance'] = result_df['mean_count']
    
    # Calculate the Log-Log relationship (often used to identify fractal patterns)
    result_df['log_grid_size'] = np.log(result_df['grid_size'])
    result_df['log_variance'] = np.log(result_df['variance'])
    result_df['log_mean'] = np.log(result_df['mean_count'])
    result_df['log_vmr'] = np.log(result_df['vmr'])
    
    # Add interpretation column
    def interpret_vmr(vmr):
        if np.isnan(vmr):
            return "Unknown"
        elif vmr < 0.9:  # Allow for some error in random patterns
            return "Uniform"
        elif vmr <= 1.1:  # Allow for some error in random patterns
            return "Random (Poisson)"
        else:
            return "Clustered"
    
    result_df['pattern'] = result_df['vmr'].apply(interpret_vmr)
    
    # Create a summary of the quadrat variance analysis
    slope = None
    if len(result_df) >= 2:
        # Calculate slope of log-log relationship between variance and grid size
        # This is related to the fractal dimension of the pattern
        valid_indices = ~(np.isnan(result_df['log_grid_size']) | np.isnan(result_df['log_variance']))
        if np.sum(valid_indices) >= 2:
            x = result_df.loc[valid_indices, 'log_grid_size'].values
            y = result_df.loc[valid_indices, 'log_variance'].values
            
            # Simple linear regression
            slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else None
            
            result_df.loc[:, 'slope'] = slope
    
    # Sort by grid size
    result_df = result_df.sort_values('grid_size')
    
    return result_df

def calculate_spatial_entropy(spatioloji_obj, grid_size, fov_id=None, use_global_coords=True,
                             cell_type=None, cell_type_column=None, entropy_type='shannon',
                             normalization=True, bootstrap=0, n_jobs=-1):
    """
    Calculate spatial entropy to quantify the randomness in spatial distribution of cells.
    
    Args:
        spatioloji_obj: A Spatioloji object
        grid_size: Size of the grid cells in pixels for quadrat-based entropy calculation
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        cell_type: Optional cell type to filter cells (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        entropy_type: Type of entropy to calculate ('shannon', 'renyi', 'tsallis')
        normalization: Whether to normalize entropy by the maximum possible value
        bootstrap: Number of bootstrap iterations for confidence interval (0 for no bootstrapping)
        n_jobs: Number of parallel jobs for bootstrap calculation (-1 for all processors)
        
    Returns:
        Dict containing entropy values and related statistics
    """
    import numpy as np
    import pandas as pd
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy import stats
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Filter by cell type if specified
    if cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
        
        # Filter to only include cells of the specified type
        cells = cells[cells['cell_type'] == cell_type]
        
        if len(cells) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    
    # Extract coordinates
    coords = cells[[x_col, y_col]].values
    
    # Determine the boundaries of the study area
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    # Create grid cells (quadrats)
    x_grid = np.arange(x_min, x_max + grid_size, grid_size)
    y_grid = np.arange(y_min, y_max + grid_size, grid_size)
    
    # Count cells in each quadrat
    x_indices = np.digitize(coords[:, 0], x_grid) - 1
    y_indices = np.digitize(coords[:, 1], y_grid) - 1
    
    # Create a 2D histogram of cell counts per quadrat
    quadrat_counts = np.zeros((len(y_grid), len(x_grid)))
    for i in range(len(coords)):
        x_idx = x_indices[i]
        y_idx = y_indices[i]
        if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
            quadrat_counts[y_idx, x_idx] += 1
    
    # Flatten the array for calculations
    counts = quadrat_counts.flatten()
    
    # Calculate total number of cells
    total_cells = np.sum(counts)
    
    # Calculate probability for each quadrat (cell proportion)
    probs = counts / total_cells
    
    # Remove quadrats with zero counts
    probs = probs[probs > 0]
    
    # Function to calculate different types of entropy
    def calculate_entropy(probs, entropy_type, normalization=True):
        if entropy_type == 'shannon':
            # Shannon entropy: -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs))
            
            # Maximum possible entropy for normalization
            if normalization:
                n = len(probs)
                max_entropy = np.log(n) if n > 0 else 0
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                norm_entropy = entropy
                
        elif entropy_type == 'renyi':
            # Renyi entropy (alpha=2): -log(sum(p^2))
            alpha = 2  # Can be parameterized if needed
            entropy = -np.log(np.sum(probs**alpha)) / (1 - alpha)
            
            # Maximum possible entropy for normalization
            if normalization:
                n = len(probs)
                max_entropy = np.log(n) if n > 0 else 0
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                norm_entropy = entropy
                
        elif entropy_type == 'tsallis':
            # Tsallis entropy: (1 - sum(p^q)) / (q - 1)
            q = 2  # Can be parameterized if needed
            entropy = (1 - np.sum(probs**q)) / (q - 1)
            
            # Maximum possible entropy for normalization
            if normalization:
                n = len(probs)
                max_entropy = (1 - 1/n**(q-1)) / (q - 1) if n > 0 else 0
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                norm_entropy = entropy
                
        else:
            raise ValueError(f"Unknown entropy type: {entropy_type}")
            
        return entropy, norm_entropy
    
    # Calculate entropy
    entropy, norm_entropy = calculate_entropy(probs, entropy_type, normalization)
    
    # Calculate additional spatial metrics
    # 1. Moran's I (spatial autocorrelation)
    moran_i = None
    
    # 2. Spatial heterogeneity (coefficient of variation)
    cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
    
    # 3. Occupancy (proportion of occupied quadrats)
    occupancy = np.sum(counts > 0) / len(counts)
    
    # 4. Spatial information (mutual information between x and y coordinates)
    # This measures how much knowing a cell's x position tells you about its y position
    spatial_info = None
    
    try:
        # Create 2D histogram for x and y coordinates
        x_hist = np.histogram(coords[:, 0], bins=x_grid)[0]
        y_hist = np.histogram(coords[:, 1], bins=y_grid)[0]
        
        # Calculate marginal probabilities
        p_x = x_hist / np.sum(x_hist)
        p_y = y_hist / np.sum(y_hist)
        
        # Calculate joint probabilities
        p_xy = quadrat_counts / np.sum(quadrat_counts)
        
        # Calculate mutual information
        spatial_info = 0
        
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[j, i] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    spatial_info += p_xy[j, i] * np.log(p_xy[j, i] / (p_x[i] * p_y[j]))
    except Exception as e:
        spatial_info = None
        print(f"Error calculating spatial information: {e}")
    
    # Calculate bootstrap confidence interval if requested
    bootstrap_results = None
    
    if bootstrap > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate entropy for a bootstrap sample
        def bootstrap_entropy(seed, coords, x_grid, y_grid, entropy_type, normalization):
            np.random.seed(seed)
            
            # Resample points with replacement
            bootstrap_sample = np.random.choice(range(len(coords)), size=len(coords), replace=True)
            bootstrap_coords = coords[bootstrap_sample]
            
            # Count cells in each quadrat for this bootstrap sample
            x_indices = np.digitize(bootstrap_coords[:, 0], x_grid) - 1
            y_indices = np.digitize(bootstrap_coords[:, 1], y_grid) - 1
            
            # Create a 2D histogram of cell counts per quadrat
            bootstrap_counts = np.zeros((len(y_grid), len(x_grid)))
            for i in range(len(bootstrap_coords)):
                x_idx = x_indices[i]
                y_idx = y_indices[i]
                if 0 <= x_idx < len(x_grid) and 0 <= y_idx < len(y_grid):
                    bootstrap_counts[y_idx, x_idx] += 1
            
            # Flatten the array for calculations
            counts = bootstrap_counts.flatten()
            
            # Calculate total number of cells
            total_cells = np.sum(counts)
            
            # Calculate probability for each quadrat (cell proportion)
            probs = counts / total_cells
            
            # Remove quadrats with zero counts
            probs = probs[probs > 0]
            
            # Calculate entropy
            entropy, norm_entropy = calculate_entropy(probs, entropy_type, normalization)
            
            return entropy, norm_entropy
        
        # Run bootstrap iterations in parallel
        bootstrap_results_raw = Parallel(n_jobs=n_jobs)(
            delayed(bootstrap_entropy)(
                i, coords, x_grid, y_grid, entropy_type, normalization
            ) for i in tqdm(range(bootstrap), desc="Bootstrap iterations")
        )
        
        # Extract raw and normalized entropy values
        bootstrap_entropies = np.array([res[0] for res in bootstrap_results_raw])
        bootstrap_norm_entropies = np.array([res[1] for res in bootstrap_results_raw])
        
        # Calculate confidence intervals
        alpha = 0.05  # 95% confidence interval
        entropy_ci = stats.scoreatpercentile(bootstrap_entropies, [alpha * 100 / 2, 100 - alpha * 100 / 2])
        norm_entropy_ci = stats.scoreatpercentile(bootstrap_norm_entropies, [alpha * 100 / 2, 100 - alpha * 100 / 2])
        
        bootstrap_results = {
            'entropy_mean': np.mean(bootstrap_entropies),
            'entropy_std': np.std(bootstrap_entropies),
            'entropy_ci': entropy_ci,
            'norm_entropy_mean': np.mean(bootstrap_norm_entropies),
            'norm_entropy_std': np.std(bootstrap_norm_entropies),
            'norm_entropy_ci': norm_entropy_ci
        }
    
    # Generate a qualitative interpretation
    if norm_entropy < 0.3:
        interpretation = "Highly ordered spatial pattern"
    elif norm_entropy < 0.7:
        interpretation = "Moderately ordered spatial pattern"
    else:
        interpretation = "Highly disordered spatial pattern"
    
    # Compare to a random distribution
    # For a completely random (Poisson) distribution, the normalized entropy should be close to 1
    if norm_entropy > 0.95:
        random_comparison = "Pattern is consistent with complete spatial randomness"
    elif norm_entropy > 0.8:
        random_comparison = "Pattern is close to spatial randomness"
    else:
        random_comparison = "Pattern shows significant spatial structure (non-random)"
    
    # Return results
    return {
        'entropy': entropy,
        'normalized_entropy': norm_entropy,
        'entropy_type': entropy_type,
        'interpretation': interpretation,
        'random_comparison': random_comparison,
        'coefficient_of_variation': cv,
        'occupancy': occupancy,
        'spatial_information': spatial_info,
        'grid_size': grid_size,
        'num_quadrats': len(counts),
        'num_occupied_quadrats': np.sum(counts > 0),
        'bootstrap_results': bootstrap_results
    }

def calculate_hotspot_analysis(spatioloji_obj, attribute_name, distance_threshold, 
                              p_value_threshold=0.05, fov_id=None, use_global_coords=True,
                              permutations=999, n_jobs=-1, plot_result=False):
    """
    Calculate Getis-Ord Gi* statistic to identify statistically significant spatial hot spots and cold spots.
    
    Args:
        spatioloji_obj: A Spatioloji object
        attribute_name: Name of the attribute/column in adata.obs to analyze for clustering
        distance_threshold: Maximum distance (in pixels) to consider neighbors
        p_value_threshold: Significance threshold for hot/cold spots (default: 0.05)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        permutations: Number of random permutations for significance testing
        n_jobs: Number of parallel jobs (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        DataFrame with Gi* statistics and hot/cold spot classifications
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy import stats
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Get the attribute values from AnnData object
    if attribute_name not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Attribute '{attribute_name}' not found in adata.obs")
    
    # Create a dictionary mapping cell IDs to attribute values
    cell_to_attr = spatioloji_obj.adata.obs[attribute_name].to_dict()
    
    # Add attribute values to the cells DataFrame
    cells['attribute'] = cells['cell'].map(cell_to_attr)
    
    # Drop cells with missing attribute values
    cells = cells.dropna(subset=['attribute'])
    
    # Check if attribute is numeric
    if not np.issubdtype(cells['attribute'].dtype, np.number):
        raise ValueError(f"Attribute '{attribute_name}' must be numeric for hotspot analysis")
    
    # Extract coordinates and attribute values
    coords = cells[[x_col, y_col]].values
    values = cells['attribute'].values
    cell_ids = cells['cell'].values
    
    # Calculate distance matrix
    dist_matrix = distance.squareform(distance.pdist(coords))
    
    # Create spatial weights matrix (binary weights based on distance threshold)
    weights_matrix = np.zeros_like(dist_matrix)
    weights_matrix[dist_matrix <= distance_threshold] = 1
    weights_matrix[dist_matrix == 0] = 0  # Exclude self-connections
    
    # Row-normalize the weights matrix
    row_sums = weights_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights_matrix = weights_matrix / row_sums[:, np.newaxis]
    
    # Calculate global mean and standard deviation
    global_mean = np.mean(values)
    global_std = np.std(values)
    
    # Function to calculate Gi* statistic for a single cell
    def calculate_gi_star(i, values, weights_matrix, global_mean, global_std, n):
        # Get weights for this cell's neighbors
        weights = weights_matrix[i]
        
        # Calculate weighted sum
        weighted_sum = np.sum(weights * values)
        
        # Calculate sum of weights
        sum_weights = np.sum(weights)
        
        # Calculate Gi* statistic
        numerator = weighted_sum - global_mean * sum_weights
        
        # Calculate denominator
        denominator_term1 = np.sum(weights**2)
        denominator_term2 = (sum_weights**2) / n
        denominator = global_std * np.sqrt((n * denominator_term1 - denominator_term2) / (n - 1))
        
        # Handle division by zero
        if denominator == 0:
            return 0
        
        # Calculate Gi* statistic
        gi_star = numerator / denominator
        
        return gi_star
    
    # Calculate Gi* statistic for each cell
    n = len(values)
    gi_star_values = np.array([
        calculate_gi_star(i, values, weights_matrix, global_mean, global_std, n)
        for i in range(n)
    ])
    
    # Calculate p-values for each Gi* statistic
    # Under the null hypothesis, Gi* follows a standard normal distribution
    p_values = 2 * (1 - stats.norm.cdf(np.abs(gi_star_values)))  # Two-tailed test
    
    # Determine if each cell is a hot spot or cold spot
    hot_spots = (gi_star_values > 0) & (p_values < p_value_threshold)
    cold_spots = (gi_star_values < 0) & (p_values < p_value_threshold)
    not_significant = ~(hot_spots | cold_spots)
    
    # Calculate statistical significance level
    significance = np.zeros(n, dtype=object)
    significance[not_significant] = 'Not Significant'
    
    # Hot spots at different significance levels
    significance[(gi_star_values > 0) & (p_values < 0.1) & (p_values >= 0.05)] = 'Hot Spot 90%'
    significance[(gi_star_values > 0) & (p_values < 0.05) & (p_values >= 0.01)] = 'Hot Spot 95%'
    significance[(gi_star_values > 0) & (p_values < 0.01) & (p_values >= 0.001)] = 'Hot Spot 99%'
    significance[(gi_star_values > 0) & (p_values < 0.001)] = 'Hot Spot 99.9%'
    
    # Cold spots at different significance levels
    significance[(gi_star_values < 0) & (p_values < 0.1) & (p_values >= 0.05)] = 'Cold Spot 90%'
    significance[(gi_star_values < 0) & (p_values < 0.05) & (p_values >= 0.01)] = 'Cold Spot 95%'
    significance[(gi_star_values < 0) & (p_values < 0.01) & (p_values >= 0.001)] = 'Cold Spot 99%'
    significance[(gi_star_values < 0) & (p_values < 0.001)] = 'Cold Spot 99.9%'
    
    # Create result DataFrame
    result = pd.DataFrame({
        'cell': cell_ids,
        'x': coords[:, 0],
        'y': coords[:, 1],
        'attribute': values,
        'gi_star': gi_star_values,
        'p_value': p_values,
        'significance': significance,
        'is_hot_spot': hot_spots,
        'is_cold_spot': cold_spots
    })
    
    # Optional: Calculate Monte Carlo based p-values using permutation test
    if permutations > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate Gi* statistics for a random permutation
        def calculate_permutation(seed, values, weights_matrix, n):
            np.random.seed(seed)
            
            # Randomly permute the attribute values
            shuffled_values = np.random.permutation(values)
            
            # Calculate global mean and standard deviation for shuffled values
            shuffled_mean = np.mean(shuffled_values)
            shuffled_std = np.std(shuffled_values)
            
            # Calculate Gi* for each location
            gi_stars = np.zeros(n)
            
            for i in range(n):
                # Get weights for this cell's neighbors
                weights = weights_matrix[i]
                
                # Calculate weighted sum
                weighted_sum = np.sum(weights * shuffled_values)
                
                # Calculate sum of weights
                sum_weights = np.sum(weights)
                
                # Calculate Gi* statistic
                numerator = weighted_sum - shuffled_mean * sum_weights
                
                # Calculate denominator
                denominator_term1 = np.sum(weights**2)
                denominator_term2 = (sum_weights**2) / n
                denominator = shuffled_std * np.sqrt((n * denominator_term1 - denominator_term2) / (n - 1))
                
                # Handle division by zero
                if denominator == 0:
                    gi_stars[i] = 0
                else:
                    gi_stars[i] = numerator / denominator
            
            return gi_stars
        
        # Run permutations in parallel
        permutation_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_permutation)(
                i, values, weights_matrix, n
            ) for i in tqdm(range(permutations), desc="Monte Carlo permutations")
        )
        
        # Combine permutation results
        permutation_gi_stars = np.array(permutation_results)
        
        # Calculate empirical p-values
        p_values_mc = np.zeros(n)
        
        for i in range(n):
            observed_gi = gi_star_values[i]
            
            if observed_gi >= 0:
                # For positive Gi* (hot spots), count how many permutations have Gi* >= observed
                p_values_mc[i] = np.mean(permutation_gi_stars[:, i] >= observed_gi)
            else:
                # For negative Gi* (cold spots), count how many permutations have Gi* <= observed
                p_values_mc[i] = np.mean(permutation_gi_stars[:, i] <= observed_gi)
        
        # Update result DataFrame with Monte Carlo p-values
        result['p_value_mc'] = p_values_mc
        
        # Update significance classifications based on Monte Carlo p-values
        significance_mc = np.array(['Not Significant'] * n, dtype=object)
        
        # Hot spots at different significance levels
        significance_mc[(gi_star_values > 0) & (p_values_mc < 0.1) & (p_values_mc >= 0.05)] = 'Hot Spot 90%'
        significance_mc[(gi_star_values > 0) & (p_values_mc < 0.05) & (p_values_mc >= 0.01)] = 'Hot Spot 95%'
        significance_mc[(gi_star_values > 0) & (p_values_mc < 0.01) & (p_values_mc >= 0.001)] = 'Hot Spot 99%'
        significance_mc[(gi_star_values > 0) & (p_values_mc < 0.001)] = 'Hot Spot 99.9%'
        
        # Cold spots at different significance levels
        significance_mc[(gi_star_values < 0) & (p_values_mc < 0.1) & (p_values_mc >= 0.05)] = 'Cold Spot 90%'
        significance_mc[(gi_star_values < 0) & (p_values_mc < 0.05) & (p_values_mc >= 0.01)] = 'Cold Spot 95%'
        significance_mc[(gi_star_values < 0) & (p_values_mc < 0.01) & (p_values_mc >= 0.001)] = 'Cold Spot 99%'
        significance_mc[(gi_star_values < 0) & (p_values_mc < 0.001)] = 'Cold Spot 99.9%'
        
        result['significance_mc'] = significance_mc
        result['is_hot_spot_mc'] = (gi_star_values > 0) & (p_values_mc < p_value_threshold)
        result['is_cold_spot_mc'] = (gi_star_values < 0) & (p_values_mc < p_value_threshold)
    
    # Generate plot configuration if requested
    if plot_result:
        plot_config = {
            'x': x_col,
            'y': y_col,
            'color': 'significance',
            'color_map': {
                'Not Significant': '#BBBBBB',  # Grey
                'Hot Spot 90%': '#FFDC73',     # Light yellow
                'Hot Spot 95%': '#FFA319',     # Orange
                'Hot Spot 99%': '#FF5E00',     # Dark orange
                'Hot Spot 99.9%': '#FF0000',   # Red
                'Cold Spot 90%': '#B3DBFF',    # Light blue
                'Cold Spot 95%': '#73B2FF',    # Blue
                'Cold Spot 99%': '#1975FF',    # Dark blue
                'Cold Spot 99.9%': '#0039C6'   # Very dark blue
            },
            'title': f'Hot Spot Analysis of {attribute_name}',
            'subtitle': f'Distance threshold: {distance_threshold} pixels'
        }
        
        return result, plot_config
    
    return result

def calculate_spatial_autocorrelation(spatioloji_obj, attribute_name, distance_threshold=None,
                                     k_nearest=None, fov_id=None, use_global_coords=True,
                                     local=False, permutations=999, n_jobs=-1, plot_result=False):
    """
    Calculate spatial autocorrelation using Moran's I statistic.
    
    Args:
        spatioloji_obj: A Spatioloji object
        attribute_name: Name of the attribute/column in adata.obs to analyze
        distance_threshold: Maximum distance (in pixels) to consider neighbors (None if using k_nearest)
        k_nearest: Number of nearest neighbors to use (None if using distance_threshold)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        local: Whether to calculate Local Moran's I (True) or Global Moran's I (False)
        permutations: Number of random permutations for significance testing
        n_jobs: Number of parallel jobs (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        Global Moran's I result or DataFrame with Local Moran's I values and quadrant classifications
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy import stats
    
    # Check if either distance_threshold or k_nearest is provided
    if distance_threshold is None and k_nearest is None:
        raise ValueError("Either distance_threshold or k_nearest must be provided")
        
    if distance_threshold is not None and k_nearest is not None:
        raise ValueError("Only one of distance_threshold or k_nearest should be provided")
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Get the attribute values from AnnData object
    if attribute_name not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Attribute '{attribute_name}' not found in adata.obs")
    
    # Create a dictionary mapping cell IDs to attribute values
    cell_to_attr = spatioloji_obj.adata.obs[attribute_name].to_dict()
    
    # Add attribute values to the cells DataFrame
    cells['attribute'] = cells['cell'].map(cell_to_attr)
    
    # Drop cells with missing attribute values
    cells = cells.dropna(subset=['attribute'])
    
    # Check if attribute is numeric
    if not np.issubdtype(cells['attribute'].dtype, np.number):
        raise ValueError(f"Attribute '{attribute_name}' must be numeric for spatial autocorrelation")
    
    # Extract coordinates, attribute values, and cell IDs
    coords = cells[[x_col, y_col]].values
    values = cells['attribute'].values
    cell_ids = cells['cell'].values
    
    # Standardize values for easier interpretation
    values_std = (values - np.mean(values)) / np.std(values)
    
    # Calculate distance matrix
    dist_matrix = distance.squareform(distance.pdist(coords))
    
    # Create spatial weights matrix
    if distance_threshold is not None:
        # Binary weights based on distance threshold
        weights_matrix = np.zeros_like(dist_matrix)
        weights_matrix[dist_matrix <= distance_threshold] = 1
        weights_matrix[dist_matrix == 0] = 0  # Exclude self-connections
    else:
        # K-nearest neighbors
        weights_matrix = np.zeros_like(dist_matrix)
        
        for i in range(len(coords)):
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(dist_matrix[i])[1:k_nearest+1]  # Skip self (index 0)
            weights_matrix[i, nearest_indices] = 1
    
    # Row-normalize the weights matrix
    row_sums = weights_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights_matrix = weights_matrix / row_sums[:, np.newaxis]
    
    # Calculate spatial lag (weighted average of neighboring values)
    spatial_lag = weights_matrix.dot(values_std)
    
    # Calculate Global Moran's I
    n = len(values_std)
    sum_weights = np.sum(weights_matrix)
    
    # Calculate the numerator (cross-product of centered values)
    numerator = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += weights_matrix[i, j] * values_std[i] * values_std[j]
    
    # Calculate the denominator (sum of squared deviations)
    denominator = np.sum(values_std**2)
    
    # Calculate Moran's I
    moran_i = (n / sum_weights) * (numerator / denominator)
    
    # Calculate expected value of Moran's I under the null hypothesis
    expected_i = -1 / (n - 1)
    
    # Calculate variance of Moran's I under the null hypothesis
    # This is a simplified formula; the exact formula is more complex
    s1 = 0.5 * np.sum((weights_matrix + np.transpose(weights_matrix))**2)
    s2 = np.sum((np.sum(weights_matrix, axis=1) + np.sum(weights_matrix, axis=0))**2)
    
    var_i = (n**2 * s1 - n * s2 + 3 * sum_weights**2) / ((n**2 - 1) * sum_weights**2)
    
    # Calculate z-score
    z_score = (moran_i - expected_i) / np.sqrt(var_i)
    
    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
    
    # Determine if spatial pattern is clustered, random, or dispersed
    if p_value < 0.05:
        if moran_i > expected_i:
            pattern = "Clustered"
        else:
            pattern = "Dispersed"
    else:
        pattern = "Random"
    
    # Calculate p-value using Monte Carlo permutation
    if permutations > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate Moran's I for a random permutation
        def calculate_permutation(seed, values_std, weights_matrix, n, sum_weights):
            np.random.seed(seed)
            
            # Randomly permute the attribute values
            shuffled_values = np.random.permutation(values_std)
            
            # Calculate the numerator for shuffled values
            numerator = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        numerator += weights_matrix[i, j] * shuffled_values[i] * shuffled_values[j]
            
            # Calculate the denominator for shuffled values
            denominator = np.sum(shuffled_values**2)
            
            # Calculate Moran's I for shuffled values
            return (n / sum_weights) * (numerator / denominator)
        
        # Run permutations in parallel
        permutation_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_permutation)(
                i, values_std, weights_matrix, n, sum_weights
            ) for i in tqdm(range(permutations), desc="Monte Carlo permutations")
        )
        
        # Calculate p-value based on permutation distribution
        if moran_i >= 0:
            # For positive Moran's I, count how many permutations have I >= observed
            p_value_mc = np.mean(np.array(permutation_results) >= moran_i)
        else:
            # For negative Moran's I, count how many permutations have I <= observed
            p_value_mc = np.mean(np.array(permutation_results) <= moran_i)
        
        # Update pattern classification based on Monte Carlo p-value
        if p_value_mc < 0.05:
            if moran_i > expected_i:
                pattern_mc = "Clustered"
            else:
                pattern_mc = "Dispersed"
        else:
            pattern_mc = "Random"
    else:
        p_value_mc = None
        pattern_mc = None
    
    # If only interested in global Moran's I, return the result
    if not local:
        global_result = {
            'moran_i': moran_i,
            'expected_i': expected_i,
            'variance': var_i,
            'z_score': z_score,
            'p_value': p_value,
            'pattern': pattern,
            'p_value_mc': p_value_mc,
            'pattern_mc': pattern_mc,
            'n': n,
            'attribute': attribute_name,
            'distance_threshold': distance_threshold,
            'k_nearest': k_nearest
        }
        
        return global_result
    
    # If local Moran's I is requested, calculate LISA (Local Indicators of Spatial Association)
    local_moran_i = np.zeros(n)
    
    for i in range(n):
        # Calculate Local Moran's I for each cell
        local_moran_i[i] = values_std[i] * spatial_lag[i] * n / sum_weights
    
    # Calculate expected value of Local Moran's I under the null hypothesis
    local_expected_i = -1 / (n - 1)
    
    # Calculate standardized Local Moran's I (z-scores)
    # This is a simplification; exact calculation of variance for Local Moran's I is complex
    local_z_scores = (local_moran_i - local_expected_i) / np.sqrt(var_i / n)
    
    # Calculate p-values for Local Moran's I
    local_p_values = 2 * (1 - stats.norm.cdf(np.abs(local_z_scores)))  # Two-tailed test
    
    # Calculate quadrant for each cell (High-High, Low-Low, High-Low, Low-High)
    # These correspond to:
    # HH: High values surrounded by high values (positive spatial autocorrelation)
    # LL: Low values surrounded by low values (positive spatial autocorrelation)
    # HL: High values surrounded by low values (negative spatial autocorrelation)
    # LH: Low values surrounded by high values (negative spatial autocorrelation)
    quadrants = []
    
    for i in range(n):
        if values_std[i] >= 0 and spatial_lag[i] >= 0:
            quadrants.append("High-High")
        elif values_std[i] < 0 and spatial_lag[i] < 0:
            quadrants.append("Low-Low")
        elif values_std[i] >= 0 and spatial_lag[i] < 0:
            quadrants.append("High-Low")
        else:  # values_std[i] < 0 and spatial_lag[i] >= 0
            quadrants.append("Low-High")
    
    # Calculate Local Moran's I p-values using Monte Carlo permutation
    local_p_values_mc = None
    
    if permutations > 0:
        # Array to store permutation results for each location
        permutation_local_i = np.zeros((permutations, n))
        
        # Function to calculate Local Moran's I for a random permutation
        def calculate_local_permutation(seed, values_std, weights_matrix, n, sum_weights):
            np.random.seed(seed)
            
            # Randomly permute the attribute values
            shuffled_values = np.random.permutation(values_std)
            
            # Calculate spatial lag for shuffled values
            shuffled_lag = weights_matrix.dot(shuffled_values)
            
            # Calculate Local Moran's I for each cell
            local_i = shuffled_values * shuffled_lag * n / sum_weights
            
            return local_i
        
        # Run permutations in parallel
        permutation_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_local_permutation)(
                i, values_std, weights_matrix, n, sum_weights
            ) for i in tqdm(range(permutations), desc="Monte Carlo permutations for Local Moran's I")
        )
        
        # Stack results for easier processing
        permutation_local_i = np.vstack(permutation_results)
        
        # Calculate p-values based on permutation distribution
        local_p_values_mc = np.zeros(n)
        
        for i in range(n):
            if local_moran_i[i] >= 0:
                # For positive Local Moran's I, count how many permutations have I >= observed
                local_p_values_mc[i] = np.mean(permutation_local_i[:, i] >= local_moran_i[i])
            else:
                # For negative Local Moran's I, count how many permutations have I <= observed
                local_p_values_mc[i] = np.mean(permutation_local_i[:, i] <= local_moran_i[i])
    
    # Create result DataFrame for Local Moran's I
    local_result = pd.DataFrame({
        'cell': cell_ids,
        'x': coords[:, 0],
        'y': coords[:, 1],
        'attribute': values,
        'attribute_std': values_std,
        'spatial_lag': spatial_lag,
        'local_moran_i': local_moran_i,
        'z_score': local_z_scores,
        'p_value': local_p_values,
        'quadrant': quadrants,
        'is_significant': local_p_values < 0.05
    })
    
    if local_p_values_mc is not None:
        local_result['p_value_mc'] = local_p_values_mc
        local_result['is_significant_mc'] = local_p_values_mc < 0.05
    
    # Generate plot configuration if requested
    if plot_result:
        # Define color map for quadrants
        quadrant_colors = {
            'High-High': '#FF0000',    # Red
            'Low-Low': '#0000FF',      # Blue
            'High-Low': '#FFA500',     # Orange
            'Low-High': '#00FFFF',     # Cyan
            'Not Significant': '#BBBBBB'  # Grey
        }
        
        # Create plot configuration
        plot_config = {
            'x': x_col,
            'y': y_col,
            'color': 'plot_category',
            'color_map': quadrant_colors,
            'title': f'Local Moran\'s I for {attribute_name}',
            'subtitle': f'Distance threshold: {distance_threshold} pixels' if distance_threshold is not None else f'K-nearest: {k_nearest}'
        }
        
        # Add a column for plotting (only significant quadrants get colored)
        local_result['plot_category'] = 'Not Significant'
        
        # Use Monte Carlo p-values if available
        if 'is_significant_mc' in local_result.columns:
            significant_mask = local_result['is_significant_mc']
        else:
            significant_mask = local_result['is_significant']
            
        local_result.loc[significant_mask, 'plot_category'] = local_result.loc[significant_mask, 'quadrant']
        
        return local_result, plot_config
    
    return local_result

def calculate_kernel_density(spatioloji_obj, bandwidth=None, grid_size=100, 
                            cell_type=None, cell_type_column=None, fov_id=None,
                            use_global_coords=True, kernel='gaussian',
                            normalize=True, plot_result=False, density_threshold=None):
    """
    Calculate Kernel Density Estimation to visualize the spatial density of cells.
    
    Args:
        spatioloji_obj: A Spatioloji object
        bandwidth: Bandwidth for the kernel density estimator (None for automatic selection)
        grid_size: Number of grid points in each dimension for density estimation
        cell_type: Optional cell type to filter cells (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        kernel: Type of kernel to use ('gaussian', 'epanechnikov', 'tophat', etc.)
        normalize: Whether to normalize the density to sum to 1
        plot_result: Whether to return a plot configuration (True) or not (False)
        density_threshold: Optional threshold to identify high-density regions (quantile between 0 and 1)
        
    Returns:
        Dict containing KDE results, including density grid, coordinates, and metrics
    """
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KernelDensity
    from scipy import stats
    from sklearn.cluster import DBSCAN
    from scipy.ndimage import label
    from skimage import measure
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Filter by cell type if specified
    if cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
        
        # Filter to only include cells of the specified type
        cells = cells[cells['cell_type'] == cell_type]
        
        if len(cells) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    
    # Extract coordinates
    coords = cells[[x_col, y_col]].values
    
    # Determine the boundaries of the study area
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    
    # Add a small buffer to the boundaries
    buffer = max((x_max - x_min), (y_max - y_min)) * 0.05
    x_min -= buffer
    y_min -= buffer
    x_max += buffer
    y_max += buffer
    
    # Create a grid for density estimation
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Normalize coordinates for KDE
    x_range = x_max - x_min
    y_range = y_max - y_min
    coords_norm = np.copy(coords)
    coords_norm[:, 0] = (coords[:, 0] - x_min) / x_range
    coords_norm[:, 1] = (coords[:, 1] - y_min) / y_range
    
    # Scale grid points similarly
    grid_points_norm = np.copy(grid_points)
    grid_points_norm[:, 0] = (grid_points[:, 0] - x_min) / x_range
    grid_points_norm[:, 1] = (grid_points[:, 1] - y_min) / y_range
    
    # Determine bandwidth if not provided
    if bandwidth is None:
        # Scott's rule of thumb for bandwidth selection
        n = len(coords)
        bandwidth = 1.06 * min(np.std(coords_norm, axis=0)) * n**(-1/5)
    else:
        # Scale bandwidth to the normalized coordinates
        bandwidth = bandwidth / max(x_range, y_range)
    
    # Perform KDE
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, metric='euclidean')
    kde.fit(coords_norm)
    
    # Calculate log density on the grid
    log_density = kde.score_samples(grid_points_norm)
    
    # Convert back to density
    density = np.exp(log_density)
    
    # Normalize if requested
    if normalize:
        density = density / np.sum(density)
    
    # Reshape to grid
    density_grid = density.reshape(xx.shape)
    
    # Calculate high-density regions if threshold is provided
    hotspots = None
    
    if density_threshold is not None:
        # Apply threshold to identify high-density regions
        threshold_value = np.quantile(density, density_threshold)
        binary_map = density_grid > threshold_value
        
        # Label connected components
        labeled_map, num_features = label(binary_map)
        
        # Extract properties of each hotspot
        if num_features > 0:
            # Find contours
            contours = measure.find_contours(density_grid, threshold_value)
            
            # Initialize list to store hotspot data
            hotspots = []
            
            # Calculate properties for each labeled region
            for region_id in range(1, num_features + 1):
                region_mask = labeled_map == region_id
                
                # Calculate region properties
                y_indices, x_indices = np.where(region_mask)
                region_density = density_grid[region_mask]
                
                # Map indices back to original coordinates
                region_x = x_grid[x_indices]
                region_y = y_grid[y_indices]
                
                # Find cells within the region
                cells_in_region = []
                for i, (x, y) in enumerate(coords):
                    if (x_min <= x <= x_max and y_min <= y <= y_max):
                        # Find closest grid point
                        x_idx = np.argmin(np.abs(x_grid - x))
                        y_idx = np.argmin(np.abs(y_grid - y))
                        
                        # Check if this point is in the region
                        if region_mask[y_idx, x_idx]:
                            cells_in_region.append(cells.iloc[i]['cell'])
                
                # Find contour for this region
                region_contour = None
                for contour in contours:
                    # Convert contour coordinates to grid indices
                    contour_x_idx = np.array([np.argmin(np.abs(x_grid - x)) for x in contour[:, 1]])
                    contour_y_idx = np.array([np.argmin(np.abs(y_grid - y)) for y in contour[:, 0]])
                    
                    # Check if contour belongs to this region
                    if np.any(region_mask[contour_y_idx[0], contour_x_idx[0]]):
                        # Convert contour to original coordinates
                        region_contour = np.column_stack((
                            x_grid[contour_x_idx],
                            y_grid[contour_y_idx]
                        ))
                        break
                
                # Store region data
                hotspots.append({
                    'id': region_id,
                    'area': region_mask.sum() * (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0]),
                    'mean_density': np.mean(region_density),
                    'max_density': np.max(region_density),
                    'centroid_x': np.mean(region_x),
                    'centroid_y': np.mean(region_y),
                    'num_cells': len(cells_in_region),
                    'cell_ids': cells_in_region,
                    'contour': region_contour
                })
    
    # Calculate density metrics
    min_density = np.min(density_grid)
    max_density = np.max(density_grid)
    mean_density = np.mean(density_grid)
    std_density = np.std(density_grid)
    
    # Calculate coefficient of variation to measure heterogeneity
    cv = std_density / mean_density if mean_density > 0 else 0
    
    # Calculate entropy of the density distribution
    if normalize:
        p = density_grid / np.sum(density_grid)
        entropy = -np.sum(p * np.log(p + 1e-10))
    else:
        entropy = None
    
    # Create result dictionary
    result = {
        'density_grid': density_grid,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'bandwidth': bandwidth * max(x_range, y_range),  # Convert back to original scale
        'min_density': min_density,
        'max_density': max_density,
        'mean_density': mean_density,
        'std_density': std_density,
        'coefficient_of_variation': cv,
        'entropy': entropy,
        'cell_coordinates': coords,
        'grid_size': grid_size,
        'kernel': kernel,
        'hotspots': hotspots,
        'density_threshold': density_threshold
    }
    
    # Generate plot configuration if requested
    if plot_result:
        # Define plot configuration
        plot_config = {
            'type': 'heatmap',
            'data': density_grid,
            'x': x_grid,
            'y': y_grid,
            'colormap': 'viridis',
            'title': f'Kernel Density Estimation',
            'subtitle': f'Bandwidth: {result["bandwidth"]:.2f} pixels, Kernel: {kernel}',
            'xlabel': x_col,
            'ylabel': y_col,
            'points': coords if len(coords) < 1000 else None,  # Plot points if not too many
            'contours': True,  # Add contour lines
            'hotspots': hotspots  # Add hotspot contours if available
        }
        
        return result, plot_config
    
    return result

def calculate_spatial_heterogeneity(spatioloji_obj, attribute_name=None, gene_name=None, 
                                   grid_size=None, kernel_bandwidth=None, fov_id=None,
                                   use_global_coords=True, cell_type=None, cell_type_column=None,
                                   method='quadrat', gradient_detection=True, n_jobs=-1, plot_result=False):
    """
    Calculate spatial heterogeneity index to quantify and characterize spatial variation.
    
    Args:
        spatioloji_obj: A Spatioloji object
        attribute_name: Column name in adata.obs containing attribute values (exclusive with gene_name)
        gene_name: Gene name to analyze expression heterogeneity (exclusive with attribute_name)
        grid_size: Size of grid cells in pixels for quadrat analysis (required for 'quadrat' method)
        kernel_bandwidth: Bandwidth for kernel density estimation (required for 'kernel' method)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        cell_type: Optional cell type to filter cells (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        method: Method for heterogeneity calculation ('quadrat', 'kernel', 'nn', or 'variogram')
        gradient_detection: Whether to detect and characterize spatial gradients (True) or not (False)
        n_jobs: Number of parallel jobs (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        Dict containing heterogeneity metrics and spatial variation patterns
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy import stats
    from sklearn.neighbors import KernelDensity
    from skimage import measure, feature, filters
    from scipy.optimize import curve_fit
    from scipy.ndimage import gaussian_filter
    
    # Validate input parameters
    if attribute_name is not None and gene_name is not None:
        raise ValueError("Only one of attribute_name or gene_name should be provided")
        
    if attribute_name is None and gene_name is None:
        raise ValueError("Either attribute_name or gene_name must be provided")
        
    if method == 'quadrat' and grid_size is None:
        raise ValueError("grid_size is required for 'quadrat' method")
        
    if method == 'kernel' and kernel_bandwidth is None:
        raise ValueError("kernel_bandwidth is required for 'kernel' method")
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Filter by cell type if specified
    if cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
        
        # Filter to only include cells of the specified type
        cells = cells[cells['cell_type'] == cell_type]
        
        if len(cells) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    
    # Extract coordinates and cell IDs
    coords = cells[[x_col, y_col]].values
    cell_ids = cells['cell'].values
    
    # Check if we have enough cells
    if len(cell_ids) < 5:
        raise ValueError("Not enough cells for spatial heterogeneity analysis (need at least 5)")
    
    # Get attribute or gene expression values
    if attribute_name is not None:
        if attribute_name in cells.columns:
            # Attribute is in cell metadata
            values = cells[attribute_name].values
            analysis_name = attribute_name
        elif attribute_name in spatioloji_obj.adata.obs.columns:
            # Attribute is in AnnData object
            attr_dict = spatioloji_obj.adata.obs[attribute_name].to_dict()
            values = np.array([attr_dict.get(cell_id, np.nan) for cell_id in cell_ids])
            analysis_name = attribute_name
        else:
            raise ValueError(f"Attribute '{attribute_name}' not found in cell metadata or adata.obs")
    else:
        # Get gene expression values
        if gene_name not in spatioloji_obj.adata.var_names:
            raise ValueError(f"Gene '{gene_name}' not found in the data")
            
        # Extract gene expression values
        gene_idx = spatioloji_obj.adata.var_names.get_loc(gene_name)
        expr_values = spatioloji_obj.adata.X[:, gene_idx].toarray().flatten()
        
        # Create mapping from cell ID to expression value
        expr_dict = {cell_id: expr_values[i] for i, cell_id in enumerate(spatioloji_obj.adata.obs.index)}
        
        # Get expression for cells in our filtered set
        values = np.array([expr_dict.get(cell_id, np.nan) for cell_id in cell_ids])
        analysis_name = gene_name
    
    # Remove NaN values
    valid_indices = ~np.isnan(values)
    coords_valid = coords[valid_indices]
    values_valid = values[valid_indices]
    cell_ids_valid = cell_ids[valid_indices]
    
    if len(values_valid) < 5:
        raise ValueError(f"Not enough valid values for '{analysis_name}' (need at least 5)")
    
    # Calculate basic statistics
    value_min = np.min(values_valid)
    value_max = np.max(values_valid)
    value_mean = np.mean(values_valid)
    value_median = np.median(values_valid)
    value_std = np.std(values_valid)
    value_cv = value_std / value_mean if value_mean != 0 else np.nan  # Coefficient of variation
    
    # Calculate global heterogeneity metrics
    global_metrics = {
        'name': analysis_name,
        'n_cells': len(values_valid),
        'min': value_min,
        'max': value_max,
        'mean': value_mean,
        'median': value_median,
        'std': value_std,
        'cv': value_cv,
        'range': value_max - value_min,
        'iqr': np.percentile(values_valid, 75) - np.percentile(values_valid, 25)
    }
    
    # Calculate spatial heterogeneity using the selected method
    if method == 'quadrat':
        # Determine the boundaries of the study area
        x_min, y_min = coords_valid.min(axis=0)
        x_max, y_max = coords_valid.max(axis=0)
        
        # Add a small buffer to the boundaries
        buffer = max((x_max - x_min), (y_max - y_min)) * 0.05
        x_min -= buffer
        y_min -= buffer
        x_max += buffer
        y_max += buffer
        
        # Create grid cells (quadrats)
        x_grid = np.arange(x_min, x_max + grid_size, grid_size)
        y_grid = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Count cells in each quadrat and calculate statistics
        quadrat_values = np.zeros((len(y_grid) - 1, len(x_grid) - 1))
        quadrat_counts = np.zeros((len(y_grid) - 1, len(x_grid) - 1))
        
        for i in range(len(coords_valid)):
            x, y = coords_valid[i]
            value = values_valid[i]
            
            # Find which quadrat this cell belongs to
            x_idx = np.digitize(x, x_grid) - 1
            y_idx = np.digitize(y, y_grid) - 1
            
            if 0 <= x_idx < len(x_grid) - 1 and 0 <= y_idx < len(y_grid) - 1:
                quadrat_values[y_idx, x_idx] += value
                quadrat_counts[y_idx, x_idx] += 1
        
        # Calculate average value in each quadrat
        with np.errstate(divide='ignore', invalid='ignore'):
            quadrat_means = quadrat_values / quadrat_counts
        
        # Replace NaNs with 0
        quadrat_means = np.nan_to_fill(quadrat_means, 0)
        
        # Calculate spatial heterogeneity metrics
        quadrat_mean = np.mean(quadrat_means[quadrat_counts > 0])
        quadrat_std = np.std(quadrat_means[quadrat_counts > 0])
        quadrat_cv = quadrat_std / quadrat_mean if quadrat_mean != 0 else np.nan
        
        # Calculate variability between quadrats
        variance_ratio = np.var(quadrat_means[quadrat_counts > 0]) / np.var(values_valid)
        
        # Calculate Moran's I for quadrats
        quadrat_coords = []
        quadrat_values_list = []
        
        for y_idx in range(len(y_grid) - 1):
            for x_idx in range(len(x_grid) - 1):
                if quadrat_counts[y_idx, x_idx] > 0:
                    # Get quadrat center
                    x_center = 0.5 * (x_grid[x_idx] + x_grid[x_idx + 1])
                    y_center = 0.5 * (y_grid[y_idx] + y_grid[y_idx + 1])
                    
                    quadrat_coords.append([x_center, y_center])
                    quadrat_values_list.append(quadrat_means[y_idx, x_idx])
        
        quadrat_coords = np.array(quadrat_coords)
        quadrat_values_list = np.array(quadrat_values_list)
        
        if len(quadrat_coords) > 1:
            # Calculate pairwise distances between quadrats
            quadrat_dists = distance.squareform(distance.pdist(quadrat_coords))
            
            # Calculate Moran's I
            # Standardize values
            z = (quadrat_values_list - np.mean(quadrat_values_list)) / np.std(quadrat_values_list)
            
            # Define weights (inverse distance)
            weights = np.zeros_like(quadrat_dists)
            weights[quadrat_dists > 0] = 1 / quadrat_dists[quadrat_dists > 0]
            np.fill_diagonal(weights, 0)
            
            # Row-normalize weights
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            # Calculate Moran's I
            n = len(quadrat_values_list)
            zw = weights.dot(z)
            morans_i = (z * zw).sum() / (z**2).sum()
            
            # Expected value of Moran's I under randomization
            expected_i = -1 / (n - 1)
            
            # If Moran's I is significantly positive, we have clustering
            is_clustered = morans_i > expected_i
        else:
            morans_i = np.nan
            expected_i = np.nan
            is_clustered = False
        
        # Store results
        heterogeneity_metrics = {
            'method': 'quadrat',
            'grid_size': grid_size,
            'n_quadrats': np.sum(quadrat_counts > 0),
            'quadrat_mean': quadrat_mean,
            'quadrat_std': quadrat_std,
            'quadrat_cv': quadrat_cv,
            'variance_ratio': variance_ratio,
            'morans_i': morans_i,
            'expected_i': expected_i,
            'is_clustered': is_clustered
        }
        
        # Prepare grid data for plotting
        grid_data = {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'quadrat_means': quadrat_means,
            'quadrat_counts': quadrat_counts
        }
        
    elif method == 'kernel':
        # Determine the boundaries of the study area
        x_min, y_min = coords_valid.min(axis=0)
        x_max, y_max = coords_valid.max(axis=0)
        
        # Add a small buffer to the boundaries
        buffer = max((x_max - x_min), (y_max - y_min)) * 0.05
        x_min -= buffer
        y_min -= buffer
        x_max += buffer
        y_max += buffer
        
        # Create a grid for density estimation
        grid_size = min(100, len(coords_valid) // 2)  # Adaptive grid size
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Normalize coordinates for KDE
        x_range = x_max - x_min
        y_range = y_max - y_min
        coords_norm = np.copy(coords_valid)
        coords_norm[:, 0] = (coords_valid[:, 0] - x_min) / x_range
        coords_norm[:, 1] = (coords_valid[:, 1] - y_min) / y_range
        
        # Scale grid points similarly
        grid_points_norm = np.copy(grid_points)
        grid_points_norm[:, 0] = (grid_points[:, 0] - x_min) / x_range
        grid_points_norm[:, 1] = (grid_points[:, 1] - y_min) / y_range
        
        # Use a value-weighted KDE approach
        # Create a KDE model with the given bandwidth
        kde = KernelDensity(bandwidth=kernel_bandwidth / max(x_range, y_range), kernel=kernel)
        
        # Weight points by their values
        # Normalize values to [0, 1]
        normalized_values = (values_valid - value_min) / (value_max - value_min) if value_max > value_min else np.ones_like(values_valid)
        
        # Fit KDE with value weights
        kde.fit(coords_norm, sample_weight=normalized_values)
        
        # Calculate log density on the grid
        log_density = kde.score_samples(grid_points_norm)
        
        # Convert to density
        density = np.exp(log_density)
        
        # Reshape to grid
        density_grid = density.reshape(xx.shape)
        
        # Calculate spatial heterogeneity metrics
        kernel_mean = np.mean(density)
        kernel_std = np.std(density)
        kernel_cv = kernel_std / kernel_mean if kernel_mean != 0 else np.nan
        
        # Calculate coefficient of spatial variation (ratio of spatial to total variance)
        total_variance = np.var(values_valid)
        spatial_variance = np.var(density)
        csv = spatial_variance / total_variance if total_variance > 0 else np.nan
        
        # Store results
        heterogeneity_metrics = {
            'method': 'kernel',
            'bandwidth': kernel_bandwidth,
            'kernel_mean': kernel_mean,
            'kernel_std': kernel_std,
            'kernel_cv': kernel_cv,
            'spatial_variance': spatial_variance,
            'total_variance': total_variance,
            'coef_spatial_variation': csv
        }
        
        # Prepare grid data for plotting
        grid_data = {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'density_grid': density_grid
        }
        
    elif method == 'nn':
        # Calculate nearest neighbor distances
        if len(coords_valid) < 2:
            raise ValueError("Need at least 2 valid cells for nearest neighbor analysis")
            
        # Calculate pairwise distances
        dist_matrix = distance.squareform(distance.pdist(coords_valid))
        
        # For each cell, find nearest neighbor and the difference in values
        nn_diffs = []
        
        for i in range(len(coords_valid)):
            # Get distances to all other cells
            dists = dist_matrix[i]
            
            # Find the nearest neighbor (exclude self)
            nn_idx = np.argsort(dists)[1]  # Second smallest (after self)
            nn_dist = dists[nn_idx]
            
            # Calculate value difference
            value_diff = abs(values_valid[i] - values_valid[nn_idx])
            
            nn_diffs.append({
                'cell_id': cell_ids_valid[i],
                'nn_dist': nn_dist,
                'value_diff': value_diff,
                'value': values_valid[i],
                'nn_value': values_valid[nn_idx]
            })
        
        # Convert to DataFrame
        nn_df = pd.DataFrame(nn_diffs)
        
        # Calculate correlation between distance and value difference
        correlation = stats.pearsonr(nn_df['nn_dist'], nn_df['value_diff'])
        
        # Calculate spatial heterogeneity metrics
        nn_mean_diff = np.mean(nn_df['value_diff'])
        nn_std_diff = np.std(nn_df['value_diff'])
        nn_cv_diff = nn_std_diff / nn_mean_diff if nn_mean_diff != 0 else np.nan
        
        # Calculate spatial autocorrelation
        # Standardize values
        z = (values_valid - np.mean(values_valid)) / np.std(values_valid)
        
        # Create weights matrix based on nearest neighbors
        nn_weights = np.zeros_like(dist_matrix)
        
        for i in range(len(coords_valid)):
            # Find index of nearest neighbor
            nn_idx = np.argsort(dist_matrix[i])[1]  # Second smallest (after self)
            nn_weights[i, nn_idx] = 1
        
        # Calculate Moran's I for nearest neighbors
        n = len(values_valid)
        zw = nn_weights.dot(z)
        morans_i = (z * zw).sum() / (z**2).sum()
        
        # Expected value of Moran's I under randomization
        expected_i = -1 / (n - 1)
        
        # Calculate Geary's C for nearest neighbors
        numerator = 0
        for i in range(n):
            for j in range(n):
                if nn_weights[i, j] > 0:
                    numerator += nn_weights[i, j] * ((z[i] - z[j])**2)
        
        denominator = (z**2).sum()
        geary_c = (n - 1) * numerator / (2 * nn_weights.sum() * denominator)
        
        # Store results
        heterogeneity_metrics = {
            'method': 'nn',
            'nn_mean_diff': nn_mean_diff,
            'nn_std_diff': nn_std_diff,
            'nn_cv_diff': nn_cv_diff,
            'correlation': correlation[0],
            'correlation_pvalue': correlation[1],
            'morans_i': morans_i,
            'expected_i': expected_i,
            'geary_c': geary_c,
            'expected_c': 1.0
        }
        
        # Prepare nearest neighbor data for plotting
        grid_data = {
            'nn_df': nn_df,
            'coords': coords_valid
        }
        
    elif method == 'variogram':
        # Calculate experimental variogram
        # Define distance bins
        max_dist = np.max(distance.pdist(coords_valid)) / 2
        n_bins = 10
        distance_bins = np.linspace(0, max_dist, n_bins + 1)
        distance_centers = 0.5 * (distance_bins[1:] + distance_bins[:-1])
        
        # Calculate pairwise distances and value differences
        dist_matrix = distance.squareform(distance.pdist(coords_valid))
        
        # Initialize arrays for variogram
        gamma_values = np.zeros(n_bins)
        pair_counts = np.zeros(n_bins)
        
        # Calculate semivariance for each distance bin
        for i in range(len(coords_valid)):
            for j in range(i + 1, len(coords_valid)):
                dist = dist_matrix[i, j]
                
                # Find which bin this distance falls into
                bin_idx = np.digitize(dist, distance_bins) - 1
                
                if 0 <= bin_idx < n_bins:
                    # Calculate squared difference
                    value_diff = (values_valid[i] - values_valid[j]) ** 2
                    
                    # Add to variogram
                    gamma_values[bin_idx] += value_diff
                    pair_counts[bin_idx] += 1
        
        # Calculate semivariance
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_values = gamma_values / (2 * pair_counts)
        
        # Replace NaNs with 0
        gamma_values = np.nan_to_num(gamma_values)
        
        # Fit variogram model (spherical model)
        def spherical_model(h, nugget, sill, range_param):
            result = np.zeros_like(h)
            
            # Calculate model values
            mask = h <= range_param
            result[mask] = nugget + (sill - nugget) * (1.5 * h[mask] / range_param - 0.5 * (h[mask] / range_param) ** 3)
            result[~mask] = nugget + (sill - nugget)
            
            return result
        
        # Function to fit
        def fit_func(h, nugget, sill, range_param):
            return spherical_model(h, nugget, sill, range_param)
        
        try:
            # Initial parameter guesses
            p0 = [
                0.1 * np.var(values_valid),  # nugget
                np.var(values_valid),       # sill
                max_dist / 3                # range
            ]
            
            # Fit model to experimental variogram
            valid_bins = pair_counts > 0
            if np.sum(valid_bins) > 3:  # Need at least 4 points for fitting
                popt, _ = curve_fit(
                    fit_func, 
                    distance_centers[valid_bins], 
                    gamma_values[valid_bins], 
                    p0=p0,
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
                )
                
                nugget, sill, range_param = popt
                
                # Calculate fitted values
                fitted_values = spherical_model(distance_centers, nugget, sill, range_param)
            else:
                nugget, sill, range_param = np.nan, np.nan, np.nan
                fitted_values = np.zeros_like(distance_centers)
        except:
            nugget, sill, range_param = np.nan, np.nan, np.nan
            fitted_values = np.zeros_like(distance_centers)
        
        # Calculate spatial heterogeneity metrics
        spatial_dependence = (sill - nugget) / sill if sill > 0 else np.nan
        
        # Store results
        heterogeneity_metrics = {
            'method': 'variogram',
            'nugget': nugget,
            'sill': sill,
            'range': range_param,
            'spatial_dependence': spatial_dependence,
            'n_bins': n_bins,
            'max_dist': max_dist
        }
        
        # Prepare variogram data for plotting
        grid_data = {
            'distance_centers': distance_centers,
            'gamma_values': gamma_values,
            'fitted_values': fitted_values,
            'pair_counts': pair_counts
        }
    
    # Detect and characterize spatial gradients if requested
    gradient_metrics = None
    
    if gradient_detection:
        # Create a grid of values for gradient detection
        if method == 'quadrat':
            # Use already computed quadrat means
            value_grid = quadrat_means
            x_centers = 0.5 * (x_grid[:-1] + x_grid[1:])
            y_centers = 0.5 * (y_grid[:-1] + y_grid[1:])
        elif method == 'kernel':
            # Use kernel density estimation
            value_grid = density_grid
            x_centers = x_grid
            y_centers = y_grid
        else:
            # Create a new grid
            x_min, y_min = coords_valid.min(axis=0)
            x_max, y_max = coords_valid.max(axis=0)
            
            grid_size = 20  # Number of grid cells in each dimension
            x_grid = np.linspace(x_min, x_max, grid_size + 1)
            y_grid = np.linspace(y_min, y_max, grid_size + 1)
            
            # Count cells in each grid cell and calculate average values
            value_grid = np.zeros((grid_size, grid_size))
            count_grid = np.zeros((grid_size, grid_size))
            
            for i in range(len(coords_valid)):
                x, y = coords_valid[i]
                value = values_valid[i]
                
                # Find which grid cell this point belongs to
                x_idx = np.digitize(x, x_grid) - 1
                y_idx = np.digitize(y, y_grid) - 1
                
                if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                    value_grid[y_idx, x_idx] += value
                    count_grid[y_idx, x_idx] += 1
            
            # Calculate average value in each grid cell
            with np.errstate(divide='ignore', invalid='ignore'):
                value_grid = value_grid / count_grid
            
            # Replace NaNs with neighborhood average
            value_grid = np.nan_to_num(value_grid)
            
            # Smooth grid with Gaussian filter
            value_grid = gaussian_filter(value_grid, sigma=1)
            
            x_centers = 0.5 * (x_grid[:-1] + x_grid[1:])
            y_centers = 0.5 * (y_grid[:-1] + y_grid[1:])
        
        # Calculate gradients
        grad_y, grad_x = np.gradient(value_grid)
        
        # Calculate gradient magnitude and direction
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        # Calculate average gradient magnitude
        avg_grad_magnitude = np.mean(grad_magnitude)
        max_grad_magnitude = np.max(grad_magnitude)
        
        # Detect major gradient direction
        weighted_direction = np.sum(grad_direction * grad_magnitude) / np.sum(grad_magnitude)
        
        # Convert to degrees
        direction_degrees = np.rad2deg(weighted_direction)
        
        # Fit gradient model
        # Define mesh grid for fitting
        xx, yy = np.meshgrid(x_centers, y_centers)
        
        # Flatten arrays for regression
        X = np.column_stack([xx.flatten(), yy.flatten()])
        y = value_grid.flatten()
        
        # Add constant term
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Fit linear model
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Calculate predicted values
            y_pred = X.dot(beta)
            
            # Reshape to grid
            y_pred_grid = y_pred.reshape(value_grid.shape)
            
            # Calculate R^2
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # Calculate fitted gradient parameters
            gradient_intercept = beta[0]
            gradient_x = beta[1]
            gradient_y = beta[2]
            
            # Calculate fitted gradient magnitude and direction
            fitted_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            fitted_direction = np.arctan2(gradient_y, gradient_x)
            fitted_direction_degrees = np.rad2deg(fitted_direction)
            
            # Determine if gradient is significant
            is_gradient_significant = r_squared > 0.1 and fitted_magnitude > 0.1 * (value_max - value_min)
        except:
            gradient_intercept = np.nan
            gradient_x = np.nan
            gradient_y = np.nan
            fitted_magnitude = np.nan
            fitted_direction_degrees = np.nan
            r_squared = np.nan
            is_gradient_significant = False
            y_pred_grid = np.zeros_like(value_grid)
        
        # Store gradient metrics
        gradient_metrics = {
            'avg_magnitude': avg_grad_magnitude,
            'max_magnitude': max_grad_magnitude,
            'direction_degrees': direction_degrees,
            'fitted_magnitude': fitted_magnitude,
            'fitted_direction_degrees': fitted_direction_degrees,
            'r_squared': r_squared,
            'is_significant': is_gradient_significant,
            'gradient_x': gradient_x,
            'gradient_y': gradient_y,
            'intercept': gradient_intercept
        }
        
        # Add gradient data to grid_data
        grid_data.update({
            'grad_magnitude': grad_magnitude,
            'grad_direction': grad_direction,
            'fitted_grid': y_pred_grid
        })
    
    # Generate plot configuration if requested
    plot_config = None
    
    if plot_result:
        if method == 'quadrat':
            # Heatmap of quadrat values
            plot_config = {
                'type': 'heatmap',
                'data': quadrat_means,
                'x': x_grid,
                'y': y_grid,
                'colormap': 'viridis',
                'title': f'Spatial Heterogeneity of {analysis_name} (Quadrat Analysis)',
                'xlabel': x_col,
                'ylabel': y_col,
                'points': coords_valid,  # Overlay cell positions
                'point_values': values_valid,  # Color points by their values
                'gradient_arrows': gradient_detection  # Show gradient arrows if gradient detection was performed
            }
        elif method == 'kernel':
            # Heatmap of kernel density
            plot_config = {
                'type': 'heatmap',
                'data': density_grid,
                'x': x_grid,
                'y': y_grid,
                'colormap': 'viridis',
                'title': f'Spatial Heterogeneity of {analysis_name} (Kernel Density)',
                'xlabel': x_col,
                'ylabel': y_col,
                'points': coords_valid,  # Overlay cell positions
                'point_values': values_valid,  # Color points by their values
                'gradient_arrows': gradient_detection  # Show gradient arrows if gradient detection was performed
            }
        elif method == 'nn':
            # Scatter plot of nearest neighbor relationships
            plot_config = {
                'type': 'scatter',
                'x': 'nn_dist',
                'y': 'value_diff',
                'data': nn_df,
                'title': f'Nearest Neighbor Analysis of {analysis_name}',
                'xlabel': 'Distance to nearest neighbor (pixels)',
                'ylabel': 'Absolute value difference',
                'trendline': True,  # Add trendline
                'correlation': heterogeneity_metrics['correlation'],  # Show correlation
                'points': True  # Show points
            }
        elif method == 'variogram':
            # Plot experimental variogram and fitted model
            plot_config = {
                'type': 'line',
                'x': distance_centers,
                'y': gamma_values,
                'y2': fitted_values,
                'title': f'Variogram Analysis of {analysis_name}',
                'xlabel': 'Distance (pixels)',
                'ylabel': 'Semivariance',
                'legend': ['Experimental', 'Fitted model'],
                'x_range': [0, max_dist],
                'y_range': [0, None]  # Auto-scale upper limit
            }
    
    # Return results
    return {
        'global_metrics': global_metrics,
        'heterogeneity_metrics': heterogeneity_metrics,
        'gradient_metrics': gradient_metrics,
        'grid_data': grid_data,
        'plot_config': plot_config
    }        



# Network-Based Analysis
def calculate_mark_correlation(spatioloji_obj, attribute_name, max_distance, num_bins=50,
                              cell_type=None, cell_type_column=None, fov_id=None,
                              use_global_coords=True, mark_type='continuous',
                              permutations=999, n_jobs=-1, plot_result=False):
    """
    Calculate the mark correlation function to analyze spatial correlation of cell attributes.
    
    Args:
        spatioloji_obj: A Spatioloji object
        attribute_name: Name of the attribute/column in adata.obs to analyze
        max_distance: Maximum distance to consider (in pixels)
        num_bins: Number of distance bins for the correlation function
        cell_type: Optional cell type to filter cells (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        mark_type: Type of attribute ('continuous' or 'categorical')
        permutations: Number of Monte Carlo simulations for confidence envelope (0 for none)
        n_jobs: Number of parallel jobs for permutations (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        DataFrame with distances and mark correlation values
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Get attribute values from AnnData object
    if attribute_name not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Attribute '{attribute_name}' not found in adata.obs")
    
    # Create a dictionary mapping cell IDs to attribute values
    cell_to_attr = spatioloji_obj.adata.obs[attribute_name].to_dict()
    
    # Filter by cell type if specified
    if cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
        
        # Filter to only include cells of the specified type
        cells = cells[cells['cell_type'] == cell_type]
        
        if len(cells) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    
    # Add attribute values to the cells DataFrame
    cells['attribute'] = cells['cell'].map(cell_to_attr)
    
    # Drop cells with missing attribute values
    cells = cells.dropna(subset=['attribute'])
    
    # Extract coordinates and attribute values
    coords = cells[[x_col, y_col]].values
    marks = cells['attribute'].values
    
    # Check if attribute is categorical and convert if necessary
    if mark_type == 'categorical':
        # Convert categorical marks to numeric labels
        unique_categories = np.unique(marks)
        category_to_idx = {cat: i for i, cat in enumerate(unique_categories)}
        marks = np.array([category_to_idx[m] for m in marks])
    else:
        # Ensure marks are numeric for continuous attributes
        try:
            marks = marks.astype(float)
        except ValueError:
            raise ValueError(f"Attribute '{attribute_name}' contains non-numeric values but mark_type is 'continuous'")
    
    # Normalize marks to [0, 1] for easier interpretation
    if mark_type == 'continuous':
        mark_min = np.min(marks)
        mark_max = np.max(marks)
        
        if mark_min != mark_max:  # Avoid division by zero
            marks = (marks - mark_min) / (mark_max - mark_min)
    
    # Calculate distance bins
    bins = np.linspace(0, max_distance, num_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Calculate pairwise distances
    dist_matrix = distance.squareform(distance.pdist(coords))
    
    # Calculate mark correlation function
    if mark_type == 'continuous':
        # For continuous marks, calculate the mark correlation function
        # This measures whether similar marks are found close together
        
        # Calculate global mean and variance of marks
        global_mean = np.mean(marks)
        global_variance = np.var(marks)
        
        # Initialize arrays for correlation calculation
        numerator = np.zeros(num_bins)
        denominator = np.zeros(num_bins)
        
        # Calculate mark correlation for each distance bin
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):  # Only consider unique pairs
                d = dist_matrix[i, j]
                
                if d <= max_distance:
                    # Determine which bin this distance falls into
                    bin_idx = np.digitize(d, bins) - 1
                    
                    if 0 <= bin_idx < num_bins:
                        # Calculate the product of centered marks
                        mark_product = (marks[i] - global_mean) * (marks[j] - global_mean)
                        
                        # Add to numerator
                        numerator[bin_idx] += mark_product
                        
                        # Add to denominator (squared centered marks)
                        denominator[bin_idx] += 1
        
        # Calculate mark correlation function
        # C(r) = E[(m_i - μ)(m_j - μ)] / σ²
        mark_correlation = np.zeros(num_bins)
        
        for i in range(num_bins):
            if denominator[i] > 0:
                mark_correlation[i] = numerator[i] / (denominator[i] * global_variance)
            else:
                mark_correlation[i] = 0
    
    else:  # Categorical marks
        # For categorical marks, calculate co-occurrence probability
        # This measures whether specific categories tend to be found together
        
        # Get the number of categories
        num_categories = len(np.unique(marks))
        
        # Initialize arrays for co-occurrence calculation
        # Dimension: [distance bins, category 1, category 2]
        mark_co_occurrence = np.zeros((num_bins, num_categories, num_categories))
        pair_counts = np.zeros((num_bins, num_categories, num_categories))
        
        # Calculate co-occurrence for each distance bin
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):  # Only consider unique pairs
                d = dist_matrix[i, j]
                
                if d <= max_distance:
                    # Determine which bin this distance falls into
                    bin_idx = np.digitize(d, bins) - 1
                    
                    if 0 <= bin_idx < num_bins:
                        # Get categories of the two cells
                        cat_i = int(marks[i])
                        cat_j = int(marks[j])
                        
                        # Increment co-occurrence count
                        mark_co_occurrence[bin_idx, cat_i, cat_j] += 1
                        mark_co_occurrence[bin_idx, cat_j, cat_i] += 1  # Symmetric
                        
                        # Increment pair count
                        pair_counts[bin_idx, cat_i, cat_j] += 1
                        pair_counts[bin_idx, cat_j, cat_i] += 1  # Symmetric
        
        # Calculate co-occurrence probability (normalized)
        mark_correlation = np.zeros((num_bins, num_categories, num_categories))
        
        for i in range(num_bins):
            for cat1 in range(num_categories):
                for cat2 in range(num_categories):
                    if pair_counts[i, cat1, cat2] > 0:
                        mark_correlation[i, cat1, cat2] = mark_co_occurrence[i, cat1, cat2] / pair_counts[i, cat1, cat2]
    
    # Create result DataFrame
    if mark_type == 'continuous':
        result = pd.DataFrame({
            'distance': bin_centers,
            'mark_correlation': mark_correlation,
            'n_pairs': denominator
        })
    else:  # Categorical
        # For categorical marks, create a multi-index DataFrame
        rows = []
        
        for i in range(num_bins):
            for cat1 in range(num_categories):
                for cat2 in range(num_categories):
                    if pair_counts[i, cat1, cat2] > 0:
                        rows.append({
                            'distance': bin_centers[i],
                            'category1': unique_categories[cat1],
                            'category2': unique_categories[cat2],
                            'co_occurrence': mark_correlation[i, cat1, cat2],
                            'n_pairs': pair_counts[i, cat1, cat2]
                        })
        
        result = pd.DataFrame(rows)
    
    # Calculate Monte Carlo confidence envelope if requested
    if permutations > 0 and mark_type == 'continuous':
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate mark correlation for randomized marks
        def calculate_random_mark_correlation(seed, coords, dist_matrix, bins, max_distance):
            np.random.seed(seed)
            
            # Randomly permute the marks
            random_marks = np.random.permutation(marks)
            
            # Calculate global mean and variance of random marks
            random_mean = np.mean(random_marks)
            random_variance = np.var(random_marks)
            
            # Initialize arrays for correlation calculation
            numerator = np.zeros(len(bins) - 1)
            denominator = np.zeros(len(bins) - 1)
            
            # Calculate mark correlation for each distance bin
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):  # Only consider unique pairs
                    d = dist_matrix[i, j]
                    
                    if d <= max_distance:
                        # Determine which bin this distance falls into
                        bin_idx = np.digitize(d, bins) - 1
                        
                        if 0 <= bin_idx < len(bins) - 1:
                            # Calculate the product of centered marks
                            mark_product = (random_marks[i] - random_mean) * (random_marks[j] - random_mean)
                            
                            # Add to numerator
                            numerator[bin_idx] += mark_product
                            
                            # Add to denominator (number of pairs)
                            denominator[bin_idx] += 1
            
            # Calculate mark correlation function
            random_correlation = np.zeros(len(bins) - 1)
            
            for i in range(len(bins) - 1):
                if denominator[i] > 0 and random_variance > 0:
                    random_correlation[i] = numerator[i] / (denominator[i] * random_variance)
                else:
                    random_correlation[i] = 0
            
            return random_correlation
        
        # Run Monte Carlo simulations in parallel
        mc_results = Parallel(n_jobs=n_jobs)(
            delayed(calculate_random_mark_correlation)(
                i, coords, dist_matrix, bins, max_distance
            ) for i in tqdm(range(permutations), desc="Monte Carlo simulations")
        )
        
        # Stack results for easier processing
        mc_correlation = np.vstack(mc_results)
        
        # Calculate confidence envelopes (2.5% and 97.5% percentiles)
        correlation_low = np.percentile(mc_correlation, 2.5, axis=0)
        correlation_high = np.percentile(mc_correlation, 97.5, axis=0)
        
        # Add to result DataFrame
        result['correlation_low'] = correlation_low
        result['correlation_high'] = correlation_high
        
        # Determine if observed correlation is significantly different from random
        result['is_positive_correlation'] = result['mark_correlation'] > result['correlation_high']
        result['is_negative_correlation'] = result['mark_correlation'] < result['correlation_low']
        result['is_significant'] = result['is_positive_correlation'] | result['is_negative_correlation']
    
    # Interpret the mark correlation function
    # For continuous marks:
    # - C(r) > 0: Positive spatial correlation (similar marks tend to be close together)
    # - C(r) = 0: No spatial correlation (marks are spatially independent)
    # - C(r) < 0: Negative spatial correlation (similar marks tend to be far apart)
    
    if mark_type == 'continuous':
        # Add interpretation column
        result['interpretation'] = 'No correlation'
        result.loc[result['mark_correlation'] > 0.1, 'interpretation'] = 'Positive correlation'
        result.loc[result['mark_correlation'] < -0.1, 'interpretation'] = 'Negative correlation'
        
        # If we have significance testing, refine the interpretation
        if permutations > 0:
            result['interpretation'] = 'No correlation (Not Significant)'
            result.loc[result['is_positive_correlation'], 'interpretation'] = 'Positive correlation (Significant)'
            result.loc[result['is_negative_correlation'], 'interpretation'] = 'Negative correlation (Significant)'
    
    # Generate plot configuration if requested
    if plot_result:
        if mark_type == 'continuous':
            # Define plot configuration for continuous marks
            plot_config = {
                'x': 'distance',
                'y': 'mark_correlation',
                'title': f'Mark Correlation Function for {attribute_name}',
                'subtitle': f'Cell type: {cell_type or "All cells"}',
                'xlabel': 'Distance (pixels)',
                'ylabel': 'Mark Correlation',
                'reference_line': 0,  # Add a horizontal line at C(r) = 0
                'confidence_intervals': permutations > 0,  # Add confidence intervals if available
                'ci_low': 'correlation_low',
                'ci_high': 'correlation_high'
            }
        else:  # Categorical
            # Define plot configuration for categorical marks
            # This will be a separate plot for each category pair
            plot_config = {
                'x': 'distance',
                'y': 'co_occurrence',
                'hue': ['category1', 'category2'],
                'title': f'Mark Co-occurrence for {attribute_name}',
                'subtitle': f'Cell type: {cell_type or "All cells"}',
                'xlabel': 'Distance (pixels)',
                'ylabel': 'Co-occurrence Probability',
                'legend': True
            }
        
        return result, plot_config
    
    return result

def calculate_spatial_context(spatioloji_obj, distance_threshold, cell_type_column, 
                            fov_id=None, use_global_coords=True, use_polygons=False,
                            weight_by_distance=True, context_types='all',
                            normalize=True, min_neighbors=1, n_jobs=-1, plot_result=False):
    """
    Calculate the spatial context of cells based on their local neighborhood composition.
    
    Args:
        spatioloji_obj: A Spatioloji object
        distance_threshold: Maximum distance (in pixels) to consider cells as neighbors
        cell_type_column: Column name in adata.obs containing cell type information
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        use_polygons: Whether to use cell polygons for spatial relationships (True) or cell centers (False)
        weight_by_distance: Whether to weight neighbors by inverse distance (True) or use binary weights (False)
        context_types: Which cell types to include in context analysis ('all' or list of cell types)
        normalize: Whether to normalize context proportions (True) or use raw counts (False)
        min_neighbors: Minimum number of neighbors required for valid context analysis
        n_jobs: Number of parallel jobs for context calculation (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        DataFrame with cell IDs and their spatial context profiles
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from shapely.geometry import Point, Polygon
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    
    # Determine which coordinates to use
    if use_global_coords:
        x_center_col = 'CenterX_global_px'
        y_center_col = 'CenterY_global_px'
        x_poly_col = 'x_global_px'
        y_poly_col = 'y_global_px'
    else:
        x_center_col = 'CenterX_local_px'
        y_center_col = 'CenterY_local_px'
        x_poly_col = 'x_local_px'
        y_poly_col = 'y_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_center_col not in cells.columns or y_center_col not in cells.columns:
        raise ValueError(f"Required columns {x_center_col} and {y_center_col} not found in cell metadata")
    
    # Check if cell type column exists
    if cell_type_column not in spatioloji_obj.adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
        
    # Create a dictionary mapping cell IDs to cell types
    cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
    
    # Add cell type information to the cells DataFrame
    cells['cell_type'] = cells['cell'].map(cell_to_type)
    
    # Drop cells without cell type information
    cells = cells.dropna(subset=['cell_type'])
    
    # Check if we have polygons data if using polygons
    if use_polygons and (spatioloji_obj.polygons is None or len(spatioloji_obj.polygons) == 0):
        raise ValueError("Polygon data not available in the Spatioloji object")
    
    # Extract coordinates and cell IDs
    coords = cells[[x_center_col, y_center_col]].values
    cell_ids = cells['cell'].values
    cell_types = cells['cell_type'].values
    
    # Determine which cell types to include in context analysis
    if context_types == 'all':
        unique_cell_types = sorted(cells['cell_type'].unique())
        context_cell_types = unique_cell_types
    else:
        # Validate that specified types exist in the data
        unique_cell_types = sorted(cells['cell_type'].unique())
        invalid_types = set(context_types) - set(unique_cell_types)
        
        if invalid_types:
            raise ValueError(f"Specified cell types not found in data: {invalid_types}")
        
        context_cell_types = sorted(context_types)
    
    # Create mapping from cell type to index
    type_to_idx = {cell_type: i for i, cell_type in enumerate(context_cell_types)}
    
    # If using polygons, create cell polygons
    cell_polygons = {}
    
    if use_polygons:
        # Group polygons by cell
        for cell_id in cell_ids:
            cell_poly_data = spatioloji_obj.get_polygon_for_cell(cell_id)
            
            # Skip if no polygon data found for this cell
            if len(cell_poly_data) == 0:
                continue
                
            # Create a polygon from the points
            try:
                points = list(zip(cell_poly_data[x_poly_col], cell_poly_data[y_poly_col]))
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    cell_polygons[cell_id] = Polygon(points)
            except Exception as e:
                print(f"Warning: Could not create polygon for cell {cell_id}: {e}")
    
    # Calculate pairwise distances if not using polygons
    if not use_polygons:
        dist_matrix = distance.squareform(distance.pdist(coords))
    
    # Function to calculate spatial context for a single cell
    def calculate_cell_context(i, cell_id, cell_type):
        # Initialize context vector
        context_vector = np.zeros(len(context_cell_types))
        
        if use_polygons:
            # Skip if no polygon data for this cell
            if cell_id not in cell_polygons:
                return {
                    'cell_id': cell_id,
                    'cell_type': cell_type,
                    'context_vector': context_vector,
                    'n_neighbors': 0,
                    'valid': False
                }
            
            # Get polygon for this cell
            poly1 = cell_polygons[cell_id]
            
            # Calculate distances to all other cell polygons
            neighbors = []
            
            for j, other_id in enumerate(cell_ids):
                if cell_id == other_id:
                    continue
                
                # Skip if no polygon data for other cell
                if other_id not in cell_polygons:
                    continue
                
                # Get polygon for other cell
                poly2 = cell_polygons[other_id]
                
                # Calculate distance between polygons
                distance_between = poly1.distance(poly2)
                
                # Check if neighbor is within threshold
                if distance_between <= distance_threshold:
                    other_type = cell_types[j]
                    
                    # Only include cell types specified in context_cell_types
                    if other_type in type_to_idx:
                        # Calculate weight (inverse of distance or binary)
                        if weight_by_distance:
                            weight = 1 / max(distance_between, 1e-10)
                        else:
                            weight = 1.0
                        
                        # Add to neighbors list
                        neighbors.append((other_type, weight))
        else:
            # Calculate distances to all other cells
            distances = dist_matrix[i]
            
            # Find neighbors within threshold
            neighbors = []
            
            for j, distance_between in enumerate(distances):
                if i == j:
                    continue
                
                if distance_between <= distance_threshold:
                    other_type = cell_types[j]
                    
                    # Only include cell types specified in context_cell_types
                    if other_type in type_to_idx:
                        # Calculate weight (inverse of distance or binary)
                        if weight_by_distance:
                            weight = 1 / max(distance_between, 1e-10)
                        else:
                            weight = 1.0
                        
                        # Add to neighbors list
                        neighbors.append((other_type, weight))
        
        # Check if cell has enough neighbors
        n_neighbors = len(neighbors)
        
        if n_neighbors < min_neighbors:
            return {
                'cell_id': cell_id,
                'cell_type': cell_type,
                'context_vector': context_vector,
                'n_neighbors': n_neighbors,
                'valid': False
            }
        
        # Aggregate neighbors by cell type
        for neighbor_type, weight in neighbors:
            type_idx = type_to_idx[neighbor_type]
            context_vector[type_idx] += weight
        
        # Normalize context vector if requested
        if normalize and np.sum(context_vector) > 0:
            context_vector = context_vector / np.sum(context_vector)
        
        return {
            'cell_id': cell_id,
            'cell_type': cell_type,
            'context_vector': context_vector,
            'n_neighbors': n_neighbors,
            'valid': True
        }
    
    # Calculate spatial context for all cells in parallel
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_cell_context)(
            i, cell_id, cell_type
        ) for i, (cell_id, cell_type) in enumerate(zip(cell_ids, cell_types))
    )
    
    # Convert results to DataFrame
    context_data = []
    
    for result in results:
        if result['valid']:
            # Create row with cell ID and type
            row = {
                'cell_id': result['cell_id'],
                'cell_type': result['cell_type'],
                'n_neighbors': result['n_neighbors']
            }
            
            # Add context proportions for each cell type
            for cell_type, idx in type_to_idx.items():
                col_name = f'context_{cell_type}'
                row[col_name] = result['context_vector'][idx]
            
            # Add to data list
            context_data.append(row)
    
    # Create DataFrame
    if not context_data:
        raise ValueError("No valid spatial context profiles could be calculated. Try increasing distance_threshold or decreasing min_neighbors.")
    
    context_df = pd.DataFrame(context_data)
    
    # Convert context vectors to matrix for clustering
    context_matrix = np.zeros((len(context_df), len(context_cell_types)))
    
    for i, row in enumerate(context_df.itertuples()):
        for j, cell_type in enumerate(context_cell_types):
            col_name = f'context_{cell_type}'
            context_matrix[i, j] = getattr(row, col_name)
    
    # Perform clustering on context profiles
    if len(context_df) >= 5:  # Need at least 5 points for meaningful clustering
        # Determine optimal number of clusters (2-10)
        best_k = 2
        best_score = -1
        
        for k in range(2, min(11, len(context_df))):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(context_matrix)
            
            # Calculate silhouette score
            if len(set(labels)) > 1:  # Ensure more than one cluster
                from sklearn.metrics import silhouette_score
                score = silhouette_score(context_matrix, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # Perform KMeans clustering with optimal k
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        context_df['context_cluster'] = kmeans.fit_predict(context_matrix)
        
        # Calculate cluster centers
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, 
                                     columns=[f'context_{ct}' for ct in context_cell_types])
        cluster_centers['cluster'] = range(len(cluster_centers))
        
        # Add dominant cell type in each cluster
        dominant_types = []
        
        for i, row in cluster_centers.iterrows():
            type_props = {ct: row[f'context_{ct}'] for ct in context_cell_types}
            dominant_type = max(type_props.items(), key=lambda x: x[1])[0]
            dominant_types.append(dominant_type)
        
        cluster_centers['dominant_type'] = dominant_types
        
        # Add descriptive cluster names
        context_df['context_name'] = context_df['context_cluster'].apply(
            lambda x: f"Cluster {x} ({dominant_types[x]} enriched)")
    else:
        # If too few points for clustering, just assign all to one cluster
        context_df['context_cluster'] = 0
        context_df['context_name'] = "All cells"
        cluster_centers = None
    
    # Run t-SNE to visualize context profiles in 2D space
    if len(context_df) >= 5:  # Need at least 5 points for t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_coords = tsne.fit_transform(context_matrix)
            
            context_df['tsne_1'] = tsne_coords[:, 0]
            context_df['tsne_2'] = tsne_coords[:, 1]
        except:
            pass
    
    # Add original coordinates
    for i, row in enumerate(context_df.itertuples()):
        cell_id = row.cell_id
        cell_idx = np.where(cell_ids == cell_id)[0][0]
        
        context_df.loc[row.Index, 'x'] = coords[cell_idx, 0]
        context_df.loc[row.Index, 'y'] = coords[cell_idx, 1]
    
    # Calculate neighborhood diversity for each cell
    context_df['diversity'] = 0.0
    
    for i, row in enumerate(context_df.itertuples()):
        # Get context values
        context_values = np.array([getattr(row, f'context_{ct}') for ct in context_cell_types])
        
        # Calculate entropy (Shannon diversity)
        nonzero = context_values > 0
        
        if np.any(nonzero):
            p = context_values[nonzero]
            entropy = -np.sum(p * np.log(p))
            context_df.loc[row.Index, 'diversity'] = entropy
    
    # Calculate neighborhood specialization (inverse of diversity)
    context_df['specialization'] = 1.0 - (context_df['diversity'] / np.log(len(context_cell_types)))
    
    # Calculate neighborhood composition statistics
    context_stats = {}
    
    for cell_type in sorted(context_df['cell_type'].unique()):
        # Get cells of this type
        type_cells = context_df[context_df['cell_type'] == cell_type]
        
        if len(type_cells) > 0:
            # Calculate average context for this cell type
            type_context = {
                'cell_type': cell_type,
                'count': len(type_cells),
                'avg_neighbors': type_cells['n_neighbors'].mean(),
                'avg_diversity': type_cells['diversity'].mean(),
                'avg_specialization': type_cells['specialization'].mean()
            }
            
            # Add average context proportions
            for ct in context_cell_types:
                col_name = f'avg_context_{ct}'
                type_context[col_name] = type_cells[f'context_{ct}'].mean()
            
            # Calculate standard errors
            type_context['se_neighbors'] = type_cells['n_neighbors'].std() / np.sqrt(len(type_cells))
            type_context['se_diversity'] = type_cells['diversity'].std() / np.sqrt(len(type_cells))
            type_context['se_specialization'] = type_cells['specialization'].std() / np.sqrt(len(type_cells))
            
            for ct in context_cell_types:
                col_name = f'se_context_{ct}'
                type_context[col_name] = type_cells[f'context_{ct}'].std() / np.sqrt(len(type_cells))
            
            context_stats[cell_type] = type_context
    
    # Calculate typical cell type niches
    niche_clusters = None
    
    if cluster_centers is not None:
        # Create descriptive niches
        niches = []
        
        for i, row in cluster_centers.iterrows():
            # Get top 3 cell types in this niche
            type_props = {ct: row[f'context_{ct}'] for ct in context_cell_types}
            sorted_types = sorted(type_props.items(), key=lambda x: x[1], reverse=True)
            
            # Create niche description
            top_types = [f"{ct} ({prop:.1%})" for ct, prop in sorted_types[:3] if prop > 0.05]
            
            # Add cluster info
            niches.append({
                'cluster': i,
                'dominant_type': row['dominant_type'],
                'composition': ", ".join(top_types),
                'cell_count': sum(context_df['context_cluster'] == i)
            })
        
        niche_clusters = pd.DataFrame(niches)
    
    # Generate plot configuration if requested
    plot_config = None
    
    if plot_result:
        # Determine which plot to generate based on what's available
        if 'tsne_1' in context_df.columns and 'tsne_2' in context_df.columns:
            # t-SNE plot of context profiles
            plot_config = {
                'type': 'scatter',
                'x': 'tsne_1',
                'y': 'tsne_2',
                'color_by': 'context_cluster' if 'context_cluster' in context_df.columns else 'cell_type',
                'title': 'Spatial Context Profiles (t-SNE)',
                'size_by': 'n_neighbors',
                'tooltip': ['cell_id', 'cell_type', 'context_name', 'n_neighbors', 'diversity']
            }
        else:
            # Spatial plot with context clusters
            plot_config = {
                'type': 'scatter',
                'x': 'x',
                'y': 'y',
                'color_by': 'context_cluster' if 'context_cluster' in context_df.columns else 'cell_type',
                'title': 'Spatial Context Clusters',
                'size_by': 'n_neighbors',
                'tooltip': ['cell_id', 'cell_type', 'context_name', 'n_neighbors', 'diversity']
            }
    
    # Return results
    return {
        'context_profiles': context_df,
        'context_stats': pd.DataFrame(list(context_stats.values())),
        'cluster_centers': cluster_centers,
        'niche_clusters': niche_clusters,
        'plot_config': plot_config
    }





# Gene Expression Spatial Analysis
def calculate_network_statistics(spatioloji_obj, distance_threshold=None, k_nearest=None,
                                cell_type_column=None, attribute_column=None,
                                fov_id=None, use_global_coords=True, directed=False,
                                weight_by_distance=True, community_detection=True,
                                centrality_measures=True, n_jobs=-1, plot_result=False,
                                use_polygons=False, edge_overlap_threshold=0.0):
    """
    Calculate network-based spatial statistics by constructing a cell interaction network,
    with support for polygon-based cell geometries.
    
    Args:
        spatioloji_obj: A Spatioloji object
        distance_threshold: Maximum distance (in pixels) to consider cells as connected (None if using k_nearest)
        k_nearest: Number of nearest neighbors to connect for each cell (None if using distance_threshold)
        cell_type_column: Optional column name in adata.obs containing cell type information
        attribute_column: Optional column name in adata.obs containing attribute values for node analysis
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        directed: Whether to create a directed network (True) or undirected network (False)
        weight_by_distance: Whether to weight edges by inverse distance (True) or use binary weights (False)
        community_detection: Whether to perform community detection (True) or not (False)
        centrality_measures: Whether to calculate node centrality measures (True) or not (False)
        n_jobs: Number of parallel jobs for community detection and centrality calculation (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        use_polygons: Whether to use cell polygons for spatial relationships (True) or cell centers (False)
        edge_overlap_threshold: Minimum overlap ratio between polygon edges to create a connection (0.0-1.0)
        
    Returns:
        Dict containing network statistics, node metrics, and community structure
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import networkx as nx
    import multiprocessing
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    import community as community_louvain  # python-louvain package
    from shapely.geometry import Point, Polygon, MultiPoint, LineString
    
    # Check if either distance_threshold or k_nearest is provided when not using polygons
    if not use_polygons and distance_threshold is None and k_nearest is None:
        raise ValueError("Either distance_threshold or k_nearest must be provided when not using polygons")
        
    if not use_polygons and distance_threshold is not None and k_nearest is not None:
        raise ValueError("Only one of distance_threshold or k_nearest should be provided when not using polygons")
    
    # Determine which coordinates to use
    if use_global_coords:
        x_center_col = 'CenterX_global_px'
        y_center_col = 'CenterY_global_px'
        x_poly_col = 'x_global_px'
        y_poly_col = 'y_global_px'
    else:
        x_center_col = 'CenterX_local_px'
        y_center_col = 'CenterY_local_px'
        x_poly_col = 'x_local_px'
        y_poly_col = 'y_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_center_col not in cells.columns or y_center_col not in cells.columns:
        raise ValueError(f"Required columns {x_center_col} and {y_center_col} not found in cell metadata")
    
    # Check if we have polygons data if using polygons
    if use_polygons and (spatioloji_obj.polygons is None or len(spatioloji_obj.polygons) == 0):
        raise ValueError("Polygon data not available in the Spatioloji object")
    
    # Add cell type information if requested
    if cell_type_column is not None:
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
    
    # Add attribute information if requested
    if attribute_column is not None:
        if attribute_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Attribute column '{attribute_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to attribute values
        cell_to_attr = spatioloji_obj.adata.obs[attribute_column].to_dict()
        
        # Add attribute information to the cells DataFrame
        cells['attribute'] = cells['cell'].map(cell_to_attr)
    
    # Extract coordinates and cell IDs
    coords = cells[[x_center_col, y_center_col]].values
    cell_ids = cells['cell'].values
    
    # If using polygons, create cell polygons
    if use_polygons:
        # Group polygons by cell
        cell_polygons = {}
        
        for cell_id in cell_ids:
            cell_poly_data = spatioloji_obj.get_polygon_for_cell(cell_id)
            
            # Skip if no polygon data found for this cell
            if len(cell_poly_data) == 0:
                continue
                
            # Create a polygon from the points
            try:
                points = list(zip(cell_poly_data[x_poly_col], cell_poly_data[y_poly_col]))
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    cell_polygons[cell_id] = Polygon(points)
            except Exception as e:
                print(f"Warning: Could not create polygon for cell {cell_id}: {e}")
    
    # Calculate pairwise distances if not using polygons
    if not use_polygons:
        dist_matrix = distance.squareform(distance.pdist(coords))
    
    # Create a network based on spatial relationships
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add nodes with positions and attributes
    for i, cell_id in enumerate(cell_ids):
        node_attrs = {
            'pos': (coords[i, 0], coords[i, 1]),
            'x': coords[i, 0],
            'y': coords[i, 1]
        }
        
        # Add polygon if available
        if use_polygons and cell_id in cell_polygons:
            node_attrs['polygon'] = cell_polygons[cell_id]
        
        # Add cell type if available
        if cell_type_column is not None and 'cell_type' in cells.columns:
            node_attrs['cell_type'] = cells.iloc[i]['cell_type']
        
        # Add attribute if available
        if attribute_column is not None and 'attribute' in cells.columns:
            node_attrs['attribute'] = cells.iloc[i]['attribute']
        
        G.add_node(cell_id, **node_attrs)
    
    # Add edges based on spatial relationships
    if use_polygons:
        # Connect cells based on polygon proximity or overlap
        for i, cell_id1 in enumerate(cell_ids):
            # Skip if no polygon data for this cell
            if cell_id1 not in cell_polygons:
                continue
                
            poly1 = cell_polygons[cell_id1]
            
            for j, cell_id2 in enumerate(cell_ids):
                # Skip self-connections and cells without polygon data
                if cell_id1 == cell_id2 or cell_id2 not in cell_polygons:
                    continue
                    
                poly2 = cell_polygons[cell_id2]
                
                # Calculate distance between polygons
                distance_between = poly1.distance(poly2)
                
                # Check if polygons are in contact or overlap
                if distance_between <= distance_threshold:
                    # Calculate weight based on distance or overlap
                    if weight_by_distance:
                        # Use inverse distance as weight
                        weight = 1 / max(distance_between, 1e-10)
                    else:
                        weight = 1.0
                    
                    # Add edge
                    G.add_edge(cell_id1, cell_id2, weight=weight, distance=distance_between)
                
                # If using edge overlap instead of distance threshold
                elif edge_overlap_threshold > 0:
                    # Get the boundary of each polygon
                    boundary1 = LineString(poly1.exterior.coords)
                    boundary2 = LineString(poly2.exterior.coords)
                    
                    # Check if boundaries overlap or are close
                    if boundary1.distance(boundary2) < distance_threshold:
                        # Calculate the overlap ratio
                        try:
                            # Create a buffer around each boundary to detect "close" edges
                            buffer1 = boundary1.buffer(distance_threshold)
                            buffer2 = boundary2.buffer(distance_threshold)
                            
                            # Calculate overlap
                            overlap = buffer1.intersection(buffer2).length
                            overlap_ratio = overlap / min(boundary1.length, boundary2.length)
                            
                            # Add edge if overlap ratio exceeds threshold
                            if overlap_ratio >= edge_overlap_threshold:
                                # Calculate weight based on overlap ratio
                                weight = overlap_ratio if weight_by_distance else 1.0
                                
                                # Add edge
                                G.add_edge(cell_id1, cell_id2, weight=weight, 
                                          distance=distance_between, overlap=overlap_ratio)
                        except Exception as e:
                            print(f"Warning: Error calculating overlap between cells {cell_id1} and {cell_id2}: {e}")
                
    elif distance_threshold is not None:
        # Connect cells within distance threshold
        for i in range(len(cell_ids)):
            for j in range(len(cell_ids)):
                if i != j and dist_matrix[i, j] <= distance_threshold:
                    # Calculate weight (inverse of distance)
                    if weight_by_distance:
                        # Avoid division by zero
                        weight = 1 / max(dist_matrix[i, j], 1e-10)
                    else:
                        weight = 1.0
                    
                    G.add_edge(cell_ids[i], cell_ids[j], weight=weight, distance=dist_matrix[i, j])
    else:
        # Connect to k nearest neighbors
        for i in range(len(cell_ids)):
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(dist_matrix[i])[1:k_nearest+1]  # Skip self (index 0)
            
            for j in nearest_indices:
                # Calculate weight (inverse of distance)
                if weight_by_distance:
                    # Avoid division by zero
                    weight = 1 / max(dist_matrix[i, j], 1e-10)
                else:
                    weight = 1.0
                
                G.add_edge(cell_ids[i], cell_ids[j], weight=weight, distance=dist_matrix[i, j])
    
    # Calculate basic network metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes > 1:
        density = nx.density(G)
    else:
        density = 0
    
    # Check if the graph is connected
    if not directed and num_nodes > 0:
        if nx.is_connected(G):
            num_components = 1
            largest_component_size = num_nodes
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            components = list(nx.connected_components(G))
            num_components = len(components)
            largest_component_size = max(len(c) for c in components)
            
            # Calculate average path length and diameter for the largest component
            largest_component = G.subgraph(max(components, key=len))
            avg_path_length = nx.average_shortest_path_length(largest_component)
            diameter = nx.diameter(largest_component)
    else:
        # For directed graphs, use weakly connected components
        if directed and num_nodes > 0:
            components = list(nx.weakly_connected_components(G))
            num_components = len(components)
            largest_component_size = max(len(c) for c in components)
            
            # Calculate average path length and diameter for the largest component
            largest_component = G.subgraph(max(components, key=len))
            
            # Check if the largest component has more than one node
            if largest_component.number_of_nodes() > 1:
                avg_path_length = nx.average_shortest_path_length(largest_component)
                diameter = nx.diameter(largest_component)
            else:
                avg_path_length = 0
                diameter = 0
        else:
            num_components = 0
            largest_component_size = 0
            avg_path_length = 0
            diameter = 0
    
    # Calculate clustering coefficient if the graph has nodes
    if num_nodes > 0:
        clustering_coefficient = nx.average_clustering(G)
    else:
        clustering_coefficient = 0
    
    # Calculate degree statistics
    if num_nodes > 0:
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees)
        min_degree = np.min(degrees)
        max_degree = np.max(degrees)
        degree_std = np.std(degrees)
    else:
        avg_degree = 0
        min_degree = 0
        max_degree = 0
        degree_std = 0
    
    # Calculate centrality measures if requested
    node_metrics = None
    
    if centrality_measures and num_nodes > 0:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Initialize dictionary to store node metrics
        node_metrics_dict = {cell_id: {} for cell_id in cell_ids}
        
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(G)
        for node, centrality in degree_centrality.items():
            node_metrics_dict[node]['degree_centrality'] = centrality
        
        # Calculate closeness centrality (only for connected components)
        if not directed and nx.is_connected(G):
            closeness_centrality = nx.closeness_centrality(G)
            for node, centrality in closeness_centrality.items():
                node_metrics_dict[node]['closeness_centrality'] = centrality
        elif directed:
            # For directed graphs, calculate closeness centrality for strongly connected components
            try:
                closeness_centrality = nx.closeness_centrality(G)
                for node, centrality in closeness_centrality.items():
                    node_metrics_dict[node]['closeness_centrality'] = centrality
            except:
                # If there's an error (e.g., not strongly connected), assign zeros
                for node in G.nodes():
                    node_metrics_dict[node]['closeness_centrality'] = 0
        else:
            # For disconnected graphs, calculate closeness centrality for each connected component
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                closeness_centrality = nx.closeness_centrality(subgraph)
                for node, centrality in closeness_centrality.items():
                    node_metrics_dict[node]['closeness_centrality'] = centrality
        
        # Calculate betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(G)
        for node, centrality in betweenness_centrality.items():
            node_metrics_dict[node]['betweenness_centrality'] = centrality
        
        # Calculate eigenvector centrality (may not converge for all graphs)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            for node, centrality in eigenvector_centrality.items():
                node_metrics_dict[node]['eigenvector_centrality'] = centrality
        except:
            # If eigenvector centrality doesn't converge, assign zeros
            for node in G.nodes():
                node_metrics_dict[node]['eigenvector_centrality'] = 0
        
        # Calculate local clustering coefficient
        clustering_coefficients = nx.clustering(G)
        for node, coefficient in clustering_coefficients.items():
            node_metrics_dict[node]['clustering_coefficient'] = coefficient
        
        # Add polygon size metrics if using polygons
        if use_polygons:
            for cell_id in cell_ids:
                if cell_id in cell_polygons:
                    polygon = cell_polygons[cell_id]
                    node_metrics_dict[cell_id]['area'] = polygon.area
                    node_metrics_dict[cell_id]['perimeter'] = polygon.length
                    
                    # Calculate shape features
                    if polygon.area > 0:
                        # Circularity (= 4π * area / perimeter^2)
                        circularity = 4 * np.pi * polygon.area / (polygon.length ** 2)
                        node_metrics_dict[cell_id]['circularity'] = circularity
                        
                        # Solidity (= area / convex hull area)
                        convex_hull = polygon.convex_hull
                        solidity = polygon.area / convex_hull.area if convex_hull.area > 0 else 0
                        node_metrics_dict[cell_id]['solidity'] = solidity
                    else:
                        node_metrics_dict[cell_id]['circularity'] = 0
                        node_metrics_dict[cell_id]['solidity'] = 0
        
        # Convert node metrics to DataFrame
        node_metrics = pd.DataFrame.from_dict(node_metrics_dict, orient='index')
        
        # Add cell metadata
        node_metrics['x'] = [G.nodes[cell_id]['x'] for cell_id in node_metrics.index]
        node_metrics['y'] = [G.nodes[cell_id]['y'] for cell_id in node_metrics.index]
        
        if cell_type_column is not None:
            node_metrics['cell_type'] = [G.nodes[cell_id].get('cell_type', None) for cell_id in node_metrics.index]
        
        if attribute_column is not None:
            node_metrics['attribute'] = [G.nodes[cell_id].get('attribute', None) for cell_id in node_metrics.index]
    
    # Perform community detection if requested
    communities = None
    modularity = None
    
    if community_detection and num_nodes > 0:
        # Use Louvain algorithm for community detection
        partition = community_louvain.best_partition(G)
        
        # Calculate modularity
        modularity = community_louvain.modularity(partition, G)
        
        # Create DataFrame with community assignments
        communities = pd.DataFrame({
            'cell_id': list(partition.keys()),
            'community': list(partition.values())
        })
        
        communities.set_index('cell_id', inplace=True)
        
        # Add cell metadata
        communities['x'] = [G.nodes[cell_id]['x'] for cell_id in communities.index]
        communities['y'] = [G.nodes[cell_id]['y'] for cell_id in communities.index]
        
        if cell_type_column is not None:
            communities['cell_type'] = [G.nodes[cell_id].get('cell_type', None) for cell_id in communities.index]
        
        if attribute_column is not None:
            communities['attribute'] = [G.nodes[cell_id].get('attribute', None) for cell_id in communities.index]
        
        # Calculate number of communities and size statistics
        num_communities = len(set(partition.values()))
        community_sizes = communities.groupby('community').size()
        avg_community_size = community_sizes.mean()
        min_community_size = community_sizes.min()
        max_community_size = community_sizes.max()
        
        # Determine optimal number of communities using silhouette score if graph is large enough
        if num_nodes >= 10:
            # Try different numbers of communities for spectral clustering
            silhouette_scores = []
            n_clusters_range = range(2, min(11, num_nodes))
            
            # Create adjacency matrix for spectral clustering
            adjacency_matrix = nx.to_numpy_array(G)
            
            for n_clusters in n_clusters_range:
                # Perform spectral clustering
                try:
                    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                                n_jobs=n_jobs, assign_labels='kmeans')
                    labels = clustering.fit_predict(adjacency_matrix)
                    
                    # Calculate silhouette score
                    if len(set(labels)) > 1:  # Ensure more than one cluster
                        score = silhouette_score(adjacency_matrix, labels)
                        silhouette_scores.append((n_clusters, score))
                except:
                    continue
            
            # Find optimal number of communities
            if silhouette_scores:
                optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            else:
                optimal_n_clusters = num_communities
        else:
            optimal_n_clusters = num_communities
        
        # Add community metrics
        community_metrics = {
            'num_communities': num_communities,
            'avg_community_size': avg_community_size,
            'min_community_size': min_community_size,
            'max_community_size': max_community_size,
            'modularity': modularity,
            'optimal_n_clusters': optimal_n_clusters
        }
    else:
        community_metrics = None
    
    # Create edge list
    edges = []
    for u, v, data in G.edges(data=True):
        edge_data = {
            'source': u,
            'target': v,
            'weight': data.get('weight', 1.0),
            'distance': data.get('distance', 0.0)
        }
        
        # Add overlap information if available
        if 'overlap' in data:
            edge_data['overlap'] = data['overlap']
        
        edges.append(edge_data)
    
    edge_df = pd.DataFrame(edges)
    
    # Calculate edge-specific statistics
    if num_edges > 0:
        edge_weights = [data.get('weight', 1.0) for _, _, data in G.edges(data=True)]
        edge_distances = [data.get('distance', 0.0) for _, _, data in G.edges(data=True)]
        
        avg_edge_weight = np.mean(edge_weights)
        min_edge_weight = np.min(edge_weights)
        max_edge_weight = np.max(edge_weights)
        
        avg_edge_distance = np.mean(edge_distances)
        min_edge_distance = np.min(edge_distances)
        max_edge_distance = np.max(edge_distances)
        
        # Calculate edge overlap statistics if available
        if use_polygons and edge_overlap_threshold > 0:
            edge_overlaps = [data.get('overlap', 0.0) for _, _, data in G.edges(data=True) if 'overlap' in data]
            
            if edge_overlaps:
                avg_edge_overlap = np.mean(edge_overlaps)
                min_edge_overlap = np.min(edge_overlaps)
                max_edge_overlap = np.max(edge_overlaps)
            else:
                avg_edge_overlap = 0
                min_edge_overlap = 0
                max_edge_overlap = 0
        else:
            avg_edge_overlap = None
            min_edge_overlap = None
            max_edge_overlap = None
    else:
        avg_edge_weight = 0
        min_edge_weight = 0
        max_edge_weight = 0
        
        avg_edge_distance = 0
        min_edge_distance = 0
        max_edge_distance = 0
        
        avg_edge_overlap = None
        min_edge_overlap = None
        max_edge_overlap = None
    
    # Calculate cell type interaction statistics if cell type information is available
    cell_type_interactions = None
    
    if cell_type_column is not None and 'cell_type' in cells.columns:
        # Count interactions between each pair of cell types
        interaction_counts = {}
        
        for u, v, _ in G.edges(data=True):
            u_type = G.nodes[u].get('cell_type', None)
            v_type = G.nodes[v].get('cell_type', None)
            
            if u_type is not None and v_type is not None:
                # Create a sorted tuple to ensure symmetry in undirected graphs
                if not directed:
                    type_pair = tuple(sorted([u_type, v_type]))
                else:
                    type_pair = (u_type, v_type)
                
                interaction_counts[type_pair] = interaction_counts.get(type_pair, 0) + 1
        
        # Convert to DataFrame
        interactions = []
        
        for (type1, type2), count in interaction_counts.items():
            interactions.append({
                'cell_type1': type1,
                'cell_type2': type2,
                'interaction_count': count
            })
        
        cell_type_interactions = pd.DataFrame(interactions)
        
        # Calculate expected interactions under random mixing
        if not directed:
            # Count cells of each type
            type_counts = cells['cell_type'].value_counts()
            total_cells = len(cells)
            
            # Calculate expected number of interactions
            expected_interactions = []
            
            for i, (type1, count1) in enumerate(type_counts.items()):
                for j, (type2, count2) in enumerate(type_counts.items()):
                    if i <= j:  # Only include each type pair once
                        # Expected number of edges between types
                        if type1 == type2:
                            expected = (count1 * (count1 - 1)) / (total_cells * (total_cells - 1)) * num_edges
                        else:
                            expected = (count1 * count2) / (total_cells * (total_cells - 1)) * num_edges * 2
                        
                        # Find actual count
                        type_pair = tuple(sorted([type1, type2]))
                        actual = interaction_counts.get(type_pair, 0)
                        
                        expected_interactions.append({
                            'cell_type1': type1,
                            'cell_type2': type2,
                            'expected_count': expected,
                            'actual_count': actual,
                            'interaction_ratio': actual / expected if expected > 0 else 0
                        })
            
            # Convert to DataFrame
            cell_type_interactions = pd.DataFrame(expected_interactions)
    
    # Create polygon statistics if using polygons
    polygon_stats = None
    
    if use_polygons:
        polygon_data = []
        
        for cell_id in cell_ids:
            if cell_id in cell_polygons:
                polygon = cell_polygons[cell_id]
                
                # Calculate basic polygon metrics
                area = polygon.area
                perimeter = polygon.length
                
                # Calculate shape features
                if area > 0:
                    # Circularity (= 4π * area / perimeter^2)
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Solidity (= area / convex hull area)
                    convex_hull = polygon.convex_hull
                    solidity = area / convex_hull.area if convex_hull.area > 0 else 0
                    
                    # Eccentricity (based on minimum bounding rectangle)
                    try:
                        min_rect = polygon.minimum_rotated_rectangle
                        coords = np.array(min_rect.exterior.coords)
                        edges = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
                        eccentricity = max(edges) / min(edges) if min(edges) > 0 else 0
                    except:
                        eccentricity = 0
                else:
                    circularity = 0
                    solidity = 0
                    eccentricity = 0
                
                # Get centroid
                centroid = polygon.centroid
                
                # Store polygon data
                polygon_data.append({
                    'cell_id': cell_id,
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'solidity': solidity,
                    'eccentricity': eccentricity,
                    'centroid_x': centroid.x,
                    'centroid_y': centroid.y
                })
        
        # Create DataFrame
        if polygon_data:
            polygon_stats = pd.DataFrame(polygon_data)
            polygon_stats.set_index('cell_id', inplace=True)
            
            # Add cell type if available
            if cell_type_column is not None:
                polygon_stats['cell_type'] = [G.nodes[cell_id].get('cell_type', None) 
                                             for cell_id in polygon_stats.index 
                                             if cell_id in G.nodes]
    
    # Create result dictionary
    result = {
        'network_stats': {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'num_components': num_components,
            'largest_component_size': largest_component_size,
            'avg_path_length': avg_path_length,
            'diameter': diameter,
            'clustering_coefficient': clustering_coefficient,
            'avg_degree': avg_degree,
            'min_degree': min_degree,
            'max_degree': max_degree,
            'degree_std': degree_std,
            'avg_edge_weight': avg_edge_weight,
            'min_edge_weight': min_edge_weight,
            'max_edge_weight': max_edge_weight,
            'avg_edge_distance': avg_edge_distance,
            'min_edge_distance': min_edge_distance,
            'max_edge_distance': max_edge_distance,
            'avg_edge_overlap': avg_edge_overlap,
            'min_edge_overlap': min_edge_overlap,
            'max_edge_overlap': max_edge_overlap,
            'directed': directed,
            'use_polygons': use_polygons,
            'connection_method': ('polygon_distance' if use_polygons else 
                                ('distance_threshold' if distance_threshold is not None else 'k_nearest')),
            'threshold_value': (distance_threshold if distance_threshold is not None else 
                             (k_nearest if k_nearest is not None else edge_overlap_threshold))
        },
        'node_metrics': node_metrics,
        'communities': communities,
        'community_metrics': community_metrics,
        'edges': edge_df,
        'cell_type_interactions': cell_type_interactions,
        'polygon_stats': polygon_stats,
        'graph': G  # Include the actual graph for further analysis
    }
    
    # Generate plot configuration if requested
    if plot_result:
        # Define plot configuration based on what's available
        if use_polygons:
            # Network plot with polygons
            plot_config = {
                'type': 'polygon_network',
                'polygons': {cell_id: G.nodes[cell_id].get('polygon', None) 
                           for cell_id in G.nodes() if 'polygon' in G.nodes[cell_id]},
                'node_positions': {cell_id: (G.nodes[cell_id]['x'], G.nodes[cell_id]['y']) 
                                for cell_id in G.nodes()},
                'edges': [(u, v) for u, v in G.edges()],
                'title': 'Cell Interaction Network with Polygons',
                'color_by': 'community' if communities is not None else 
                           ('cell_type' if cell_type_column is not None else None),
                'node_colors': communities['community'].to_dict() if communities is not None else 
                             ({cell_id: G.nodes[cell_id].get('cell_type', None) 
                              for cell_id in G.nodes()} if cell_type_column is not None else None),
                'edge_width_by': 'weight',
                'show_edges': True
            }
        elif community_detection and communities is not None:
            # Community detection plot
            plot_config = {
                'type': 'network',
                'node_positions': {cell_id: (G.nodes[cell_id]['x'], G.nodes[cell_id]['y']) 
                                for cell_id in G.nodes()},
                'node_colors': communities['community'].to_dict(),
                'title': 'Cell Interaction Network with Communities',
                'color_by': 'community',
                'node_size_by': 'degree_centrality' if node_metrics is not None else None,
                'edge_width_by': 'weight',
                'show_labels': False
            }
        elif cell_type_column is not None:
            # Cell type colored network
            plot_config = {
                'type': 'network',
                'node_positions': {cell_id: (G.nodes[cell_id]['x'], G.nodes[cell_id]['y']) 
                                for cell_id in G.nodes()},
                'node_colors': {cell_id: G.nodes[cell_id].get('cell_type', None) 
                              for cell_id in G.nodes()},
                'title': 'Cell Interaction Network by Cell Type',
                'color_by': 'cell_type',
                'node_size_by': 'degree_centrality' if node_metrics is not None else None,
                'edge_width_by': 'weight',
                'show_labels': False
            }
        else:
            # Basic network plot
            plot_config = {
                'type': 'network',
                'node_positions': {cell_id: (G.nodes[cell_id]['x'], G.nodes[cell_id]['y']) 
                                for cell_id in G.nodes()},
                'title': 'Cell Interaction Network',
                'node_size_by': 'degree_centrality' if node_metrics is not None else None,
                'edge_width_by': 'weight',
                'show_labels': False
            }
        
        return result, plot_config
    
    return result

def calculate_gene_spatial_autocorrelation(spatioloji_obj, genes, max_distance=None, 
                                         distance_bins=10, fov_id=None, use_global_coords=True,
                                         cell_type=None, cell_type_column=None,
                                         method='moran', permutations=999, n_jobs=-1, plot_result=False):
    """
    Calculate spatial autocorrelation of gene expression across the tissue.
    
    Args:
        spatioloji_obj: A Spatioloji object
        genes: List of gene names or a single gene name to analyze
        max_distance: Maximum distance (in pixels) to consider for autocorrelation (None for auto-determination)
        distance_bins: Number of distance bins for correlogram analysis
        fov_id: Optional FOV ID to restrict analysis to a specific FOV
        use_global_coords: Whether to use global coordinates (True) or local coordinates (False)
        cell_type: Optional cell type to filter cells (None for all cells)
        cell_type_column: Column name in adata.obs containing cell type information (required if cell_type is provided)
        method: Method for spatial autocorrelation ('moran', 'geary', or 'correlogram')
        permutations: Number of Monte Carlo simulations for significance testing (0 for none)
        n_jobs: Number of parallel jobs for permutations (-1 for all processors)
        plot_result: Whether to return a plot configuration (True) or not (False)
        
    Returns:
        DataFrame with spatial autocorrelation statistics for each gene
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
    import multiprocessing
    from joblib import Parallel, delayed
    from tqdm import tqdm
    from scipy import stats
    import warnings
    
    # Handle single gene input
    if isinstance(genes, str):
        genes = [genes]
    
    # Determine which coordinates to use
    if use_global_coords:
        x_col = 'CenterX_global_px'
        y_col = 'CenterY_global_px'
    else:
        x_col = 'CenterX_local_px'
        y_col = 'CenterY_local_px'
    
    # Get the cell metadata to work with
    if fov_id is not None:
        cells = spatioloji_obj.get_cells_in_fov(fov_id)
    else:
        cells = spatioloji_obj.cell_meta
    
    # Check if we have the necessary columns
    if x_col not in cells.columns or y_col not in cells.columns:
        raise ValueError(f"Required columns {x_col} and {y_col} not found in cell metadata")
    
    # Filter by cell type if specified
    if cell_type is not None:
        if cell_type_column is None:
            raise ValueError("cell_type_column must be provided when cell_type is specified")
            
        if cell_type_column not in spatioloji_obj.adata.obs.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found in adata.obs")
            
        # Create a dictionary mapping cell IDs to cell types
        cell_to_type = spatioloji_obj.adata.obs[cell_type_column].to_dict()
        
        # Add cell type information to the cells DataFrame
        cells['cell_type'] = cells['cell'].map(cell_to_type)
        
        # Filter to only include cells of the specified type
        cells = cells[cells['cell_type'] == cell_type]
        
        if len(cells) == 0:
            raise ValueError(f"No cells of type '{cell_type}' found")
    
    # Extract coordinates and cell IDs
    coords = cells[[x_col, y_col]].values
    cell_ids = cells['cell'].values
    
    # Check if we have enough cells
    if len(cell_ids) < 5:
        raise ValueError("Not enough cells for spatial autocorrelation analysis (need at least 5)")
    
    # Get gene expression values from AnnData object
    # First, validate genes exist in the data
    var_names = set(spatioloji_obj.adata.var_names)
    missing_genes = set(genes) - var_names
    
    if missing_genes:
        warnings.warn(f"The following genes were not found in the data: {missing_genes}")
        
    # Filter to only include genes in the data
    valid_genes = [gene for gene in genes if gene in var_names]
    
    if not valid_genes:
        raise ValueError("None of the specified genes were found in the data")
    
    # Extract gene expression values
    gene_expression = {}
    
    for gene in valid_genes:
        # Get expression values for all cells
        gene_idx = spatioloji_obj.adata.var_names.get_loc(gene)
        expr_values = spatioloji_obj.adata.X[:, gene_idx].toarray().flatten()
        
        # Create mapping from cell ID to expression value
        expr_dict = {cell_id: expr_values[i] for i, cell_id in enumerate(spatioloji_obj.adata.obs.index)}
        
        # Get expression for cells in our filtered set
        expr = np.array([expr_dict.get(cell_id, 0) for cell_id in cell_ids])
        
        # Store gene expression values
        gene_expression[gene] = expr
    
    # Calculate pairwise distances
    dist_matrix = distance.squareform(distance.pdist(coords))
    
    # Determine max distance if not specified
    if max_distance is None:
        # Use half the maximum distance for autocorrelation
        max_distance = np.max(dist_matrix) / 2
    
    # Create distance bins for correlogram
    if method == 'correlogram':
        distance_edges = np.linspace(0, max_distance, distance_bins + 1)
        distance_centers = 0.5 * (distance_edges[1:] + distance_edges[:-1])
    
    # Calculate spatial weights matrix
    # For Moran's I and Geary's C, we use a binary weights matrix with a distance threshold
    weights_matrix = np.zeros_like(dist_matrix)
    weights_matrix[dist_matrix <= max_distance] = 1
    weights_matrix[dist_matrix == 0] = 0  # Exclude self-connections
    
    # Row-normalize the weights matrix
    row_sums = weights_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    weights_matrix = weights_matrix / row_sums[:, np.newaxis]
    
    # Function to calculate Moran's I for a single gene
    def calculate_morans_i(gene, expr, weights_matrix):
        # Standardize expression values
        expr_std = (expr - np.mean(expr)) / np.std(expr) if np.std(expr) > 0 else np.zeros_like(expr)
        
        # Calculate spatial lag (weighted average of neighboring values)
        spatial_lag = weights_matrix.dot(expr_std)
        
        # Calculate Moran's I
        n = len(expr)
        numerator = np.sum(expr_std * spatial_lag)
        denominator = np.sum(expr_std**2)
        
        if denominator == 0:
            return {
                'gene': gene,
                'morans_i': 0,
                'z_score': 0,
                'p_value': 1,
                'pattern': 'No variation'
            }
        
        morans_i = numerator / denominator
        
        # Calculate expected value of Moran's I under the null hypothesis
        expected_i = -1 / (n - 1)
        
        # Calculate variance of Moran's I under the null hypothesis
        # This is a simplified formula; the exact formula is more complex
        s1 = 0.5 * np.sum((weights_matrix + np.transpose(weights_matrix))**2)
        s2 = np.sum((np.sum(weights_matrix, axis=1) + np.sum(weights_matrix, axis=0))**2)
        
        var_i = (n**2 * s1 - n * s2 + 3 * sum(row_sums)**2) / ((n**2 - 1) * sum(row_sums)**2)
        
        # Calculate z-score
        z_score = (morans_i - expected_i) / np.sqrt(var_i)
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        # Determine if spatial pattern is clustered, random, or dispersed
        if p_value < 0.05:
            if morans_i > expected_i:
                pattern = "Clustered"
            else:
                pattern = "Dispersed"
        else:
            pattern = "Random"
        
        return {
            'gene': gene,
            'morans_i': morans_i,
            'expected_i': expected_i,
            'z_score': z_score,
            'p_value': p_value,
            'pattern': pattern
        }
    
    # Function to calculate Geary's C for a single gene
    def calculate_gearys_c(gene, expr, weights_matrix, dist_matrix):
        # Calculate mean expression
        expr_mean = np.mean(expr)
        
        # Calculate Geary's C
        n = len(expr)
        
        numerator = 0
        for i in range(n):
            for j in range(n):
                if i != j and dist_matrix[i, j] <= max_distance:
                    numerator += weights_matrix[i, j] * ((expr[i] - expr[j])**2)
        
        denominator = np.sum((expr - expr_mean)**2)
        
        if denominator == 0:
            return {
                'gene': gene,
                'gearys_c': 1,
                'z_score': 0,
                'p_value': 1,
                'pattern': 'No variation'
            }
        
        # Scale by number of observations and sum of weights
        sum_weights = np.sum(weights_matrix)
        gearys_c = ((n - 1) * numerator) / (2 * sum_weights * denominator)
        
        # Calculate expected value of Geary's C under the null hypothesis
        expected_c = 1.0
        
        # Calculate variance of Geary's C under the null hypothesis
        # This is a simplified formula; the exact formula is more complex
        var_c = (n - 1) / (2 * (n + 1) * sum_weights)
        
        # Calculate z-score
        z_score = (gearys_c - expected_c) / np.sqrt(var_c)
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        # Determine if spatial pattern is clustered, random, or dispersed
        if p_value < 0.05:
            if gearys_c < expected_c:
                pattern = "Clustered"
            else:
                pattern = "Dispersed"
        else:
            pattern = "Random"
        
        return {
            'gene': gene,
            'gearys_c': gearys_c,
            'expected_c': expected_c,
            'z_score': z_score,
            'p_value': p_value,
            'pattern': pattern
        }
    
    # Function to calculate correlogram for a single gene
    def calculate_correlogram(gene, expr, dist_matrix, distance_edges):
        # Calculate Moran's I at different distances
        n = len(expr)
        expr_std = (expr - np.mean(expr)) / np.std(expr) if np.std(expr) > 0 else np.zeros_like(expr)
        
        # Initialize arrays for correlogram
        moran_values = np.zeros(len(distance_edges) - 1)
        n_pairs = np.zeros(len(distance_edges) - 1)
        
        # Calculate Moran's I for each distance bin
        for i in range(len(distance_edges) - 1):
            d_min = distance_edges[i]
            d_max = distance_edges[i+1]
            
            # Create weights matrix for this distance bin
            bin_weights = np.zeros_like(dist_matrix)
            bin_weights[(dist_matrix > d_min) & (dist_matrix <= d_max)] = 1
            bin_weights[dist_matrix == 0] = 0  # Exclude self-connections
            
            # Count pairs in this bin
            n_pairs[i] = np.sum(bin_weights)
            
            if n_pairs[i] > 0:
                # Row-normalize the weights matrix
                row_sums = bin_weights.sum(axis=1)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                bin_weights = bin_weights / row_sums[:, np.newaxis]
                
                # Calculate spatial lag
                spatial_lag = bin_weights.dot(expr_std)
                
                # Calculate Moran's I for this distance bin
                numerator = np.sum(expr_std * spatial_lag)
                denominator = np.sum(expr_std**2)
                
                if denominator > 0:
                    moran_values[i] = numerator / denominator
                else:
                    moran_values[i] = 0
            else:
                moran_values[i] = 0
        
        return {
            'gene': gene,
            'distance_centers': 0.5 * (distance_edges[1:] + distance_edges[:-1]),
            'moran_values': moran_values,
            'n_pairs': n_pairs
        }
    
    # Calculate autocorrelation for all genes
    if method == 'moran':
        # Calculate Moran's I for each gene
        results = []
        
        for gene in valid_genes:
            result = calculate_morans_i(gene, gene_expression[gene], weights_matrix)
            results.append(result)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
    elif method == 'geary':
        # Calculate Geary's C for each gene
        results = []
        
        for gene in valid_genes:
            result = calculate_gearys_c(gene, gene_expression[gene], weights_matrix, dist_matrix)
            results.append(result)
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
    elif method == 'correlogram':
        # Calculate correlogram for each gene
        results = []
        
        for gene in valid_genes:
            result = calculate_correlogram(gene, gene_expression[gene], dist_matrix, distance_edges)
            results.append(result)
        
        # Convert to list of DataFrames, one for each gene
        correlograms = []
        
        for result in results:
            gene = result['gene']
            
            # Create DataFrame for this gene
            df = pd.DataFrame({
                'gene': gene,
                'distance': result['distance_centers'],
                'morans_i': result['moran_values'],
                'n_pairs': result['n_pairs']
            })
            
            correlograms.append(df)
        
        # Combine all DataFrames
        result_df = pd.concat(correlograms)
    
    # Calculate Monte Carlo significance if requested
    if permutations > 0 and method in ['moran', 'geary']:
        # Determine number of cores to use
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Function to calculate autocorrelation for a random permutation
        def calculate_random_autocorrelation(seed, expr, weights_matrix, dist_matrix, method):
            np.random.seed(seed)
            
            # Randomly permute the expression values
            random_expr = np.random.permutation(expr)
            
            if method == 'moran':
                # Calculate Moran's I for random permutation
                result = calculate_morans_i('random', random_expr, weights_matrix)
                return result['morans_i']
            else:
                # Calculate Geary's C for random permutation
                result = calculate_gearys_c('random', random_expr, weights_matrix, dist_matrix)
                return result['gearys_c']
        
        # Calculate autocorrelation for random permutations
        for gene in valid_genes:
            # Get expression values for this gene
            expr = gene_expression[gene]
            
            # Run permutations in parallel
            random_values = Parallel(n_jobs=n_jobs)(
                delayed(calculate_random_autocorrelation)(
                    i, expr, weights_matrix, dist_matrix, method
                ) for i in tqdm(range(permutations), desc=f"Monte Carlo simulations for {gene}")
            )
            
            # Convert to numpy array
            random_values = np.array(random_values)
            
            # Calculate empirical p-value
            gene_idx = result_df[result_df['gene'] == gene].index[0]
            
            if method == 'moran':
                # For Moran's I, count how many random values are >= observed
                observed_i = result_df.loc[gene_idx, 'morans_i']
                p_value = np.mean(np.abs(random_values) >= np.abs(observed_i))
                
                # Update p-value in result_df
                result_df.loc[gene_idx, 'p_value_mc'] = p_value
                
                # Update pattern based on Monte Carlo p-value
                if p_value < 0.05:
                    if observed_i > np.mean(random_values):
                        result_df.loc[gene_idx, 'pattern_mc'] = "Clustered"
                    else:
                        result_df.loc[gene_idx, 'pattern_mc'] = "Dispersed"
                else:
                    result_df.loc[gene_idx, 'pattern_mc'] = "Random"
            else:
                # For Geary's C, count how many random values are <= observed
                observed_c = result_df.loc[gene_idx, 'gearys_c']
                p_value = np.mean(np.abs(random_values - 1) >= np.abs(observed_c - 1))
                
                # Update p-value in result_df
                result_df.loc[gene_idx, 'p_value_mc'] = p_value
                
                # Update pattern based on Monte Carlo p-value
                if p_value < 0.05:
                    if observed_c < 1:
                        result_df.loc[gene_idx, 'pattern_mc'] = "Clustered"
                    else:
                        result_df.loc[gene_idx, 'pattern_mc'] = "Dispersed"
                else:
                    result_df.loc[gene_idx, 'pattern_mc'] = "Random"
    
    # Generate plot configuration if requested
    plot_config = None
    
    if plot_result:
        if method == 'correlogram':
            # Plot correlogram
            plot_config = {
                'type': 'line',
                'x': 'distance',
                'y': 'morans_i',
                'color_by': 'gene',
                'title': 'Spatial Correlogram of Gene Expression',
                'xlabel': 'Distance (pixels)',
                'ylabel': "Moran's I",
                'reference_line': 0  # Add horizontal line at 0
            }
        else:
            # Bar plot of autocorrelation values
            stat_column = 'morans_i' if method == 'moran' else 'gearys_c'
            reference_line = 0 if method == 'moran' else 1
            
            plot_config = {
                'type': 'bar',
                'x': 'gene',
                'y': stat_column,
                'color_by': 'pattern',
                'title': f'Spatial Autocorrelation of Gene Expression ({method.capitalize()})',
                'reference_line': reference_line,
                'ylim': None  # Automatically determine y-axis limits
            }
    
    # Return results
    return {
        'result': result_df,
        'method': method,
        'max_distance': max_distance,
        'cell_type': cell_type,
        'plot_config': plot_config
    }







