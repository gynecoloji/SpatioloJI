import pandas as pd
import numpy as np
import anndata
import pickle
import cv2
import os
from typing import Dict, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
from anndata import AnnData
import scanpy as sc


class spatioloji_qc:
    """
    A class for handling spatial transcriptomics data analysis and quality control.
    """
    
    def __init__(self, expr_matrix=None, cell_metadata=None, output_dir="./output/"):
        """
        Initialize the Spatioloji_qc object.
        
        Parameters:
        -----------
        expr_matrix : pandas.DataFrame
            Expression matrix with genes in columns and cells in rows
        cell_metadata : pandas.DataFrame
            Cell metadata including spatial information
        output_dir : str
            Directory to save output files
        """
        self.expr_matrix = expr_matrix
        self.cell_metadata = cell_metadata
        self.adata = None
        
        # Create output directories
        self.data_dir = os.path.join(output_dir, "data")
        self.analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # FOV IDs will be set during preprocessing
        self.fov_ids = None
        
    def load_data(self, expr_matrix_path, cell_metadata_path):
        """
        Load data from files.
        
        Parameters:
        -----------
        expr_matrix_path : str
            Path to expression matrix file
        cell_metadata_path : str
            Path to cell metadata file
        """
        self.expr_matrix = pd.read_csv(expr_matrix_path, index_col=0)
        self.cell_metadata = pd.read_csv(cell_metadata_path, index_col=0)
        print(f"Loaded expression matrix with shape: {self.expr_matrix.shape}")
        print(f"Loaded cell metadata with shape: {self.cell_metadata.shape}")
    
    def prepare_anndata(self):
        """
        Create AnnData object from expression matrix and prepare it for QC.
        """
        if self.expr_matrix is None:
            raise ValueError("Expression matrix is not loaded.")
        
        # Create a copy of expression matrix
        counts = self.expr_matrix.copy()
        counts['cell'] = counts['fov'].astype(str)+'_'+counts['cell_ID'].astype(str)
        # Set cell ID as index
        counts.index = counts['cell'].tolist()
        # Remove non-gene columns
        counts = counts.iloc[:, ~counts.columns.str.contains("fov|cell_ID|cell")]
        
        # Create AnnData object
        self.adata = AnnData(counts)
        
        # Add gene annotations
        self.adata.var['mt'] = self.adata.var_names.str.startswith("MT-")
        self.adata.var['ribo'] = [name.startswith(("RPS", "RPL")) for name in self.adata.var_names]
        self.adata.var['NegProbe'] = self.adata.var_names.str.startswith("Neg")
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(
            self.adata, qc_vars=["mt", "ribo", "NegProbe"], inplace=True, log1p=True
        )
        
        # Extract FOV IDs from cell names
        self.fov_ids = sorted(list(set([cell.split("_")[0] for cell in self.adata.obs.index])))
        print(f"Identified {len(self.fov_ids)} FOVs: {self.fov_ids}")
        
    def grubbs_test(self, data, alpha=0.05):
        """
        Perform Grubbs test to detect outliers.
        
        Parameters:
        -----------
        data : array-like
            Data to test for outliers
        alpha : float
            Significance level
            
        Returns:
        --------
        int
            Index of outlier or -1 if no outlier detected
        """
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Calculate G statistic
        deviations = np.abs(data - mean)
        max_idx = np.argmax(deviations)
        G = deviations[max_idx] / std

        # Calculate critical G value
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        if G > G_crit:
            return max_idx  # Index of the outlier
        else:
            return -1     # No outlier detected
    
    def qc_negative_probes(self, alpha=0.01):
        """
        Perform QC on negative probes.
        
        Parameters:
        -----------
        alpha : float
            Significance level for outlier detection
        """
        if self.adata is None:
            raise ValueError("AnnData object not created. Run prepare_anndata() first.")
        
        # Get negative probe genes
        neg_probes = self.adata.var[self.adata.var['NegProbe'] == True].index.tolist()
        counts = self.adata.X[:, [self.adata.var_names.get_loc(g) for g in neg_probes]]
        
        # Sum counts per cell
        neg_counts = np.sum(counts, axis=1)
        
        # Plot distribution
        plt.figure(figsize=[6, 4])
        plt.hist(np.log1p(neg_counts), color='skyblue', edgecolor='black')
        plt.title('Distribution of Negative Probe Counts (log1p)')
        plt.xlabel('log1p(Counts)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.analysis_dir, "QC_NegProbe_log1p.png"))
        plt.close()
        
        # Detect outliers
        idx_neg = self.grubbs_test(np.log1p(neg_counts), alpha=alpha)
        if idx_neg != -1:
            outlier_gene = neg_probes[idx_neg]
            self.adata.var['QC_Neg_outlier'] = [gene == outlier_gene for gene in self.adata.var.index]
            print(f"Detected outlier negative probe: {outlier_gene}")
        else:
            print("No outlier negative probes detected.")
    
    def qc_cell_area(self, alpha=0.01):
        """
        Perform QC on cell area.
        
        Parameters:
        -----------
        alpha : float
            Significance level for outlier detection
        """
        if self.adata is None or self.cell_metadata is None:
            raise ValueError("AnnData object or cell metadata not loaded.")
        
        # Get cell areas
        cell_area = self.cell_metadata[['cell', 'Area']]
        cell_area.index = cell_area['cell']
        
        # Plot distribution
        plt.figure(figsize=[6, 4])
        plt.hist(np.log1p(cell_area['Area']), color='skyblue', edgecolor='black')
        plt.title('Distribution of Cell Area (log1p)')
        plt.xlabel('log1p(Area)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.analysis_dir, "QC_cell_area_log1p.png"))
        plt.close()
        
        # Detect outliers
        idx_cell_area = self.grubbs_test(np.log1p(cell_area['Area']), alpha=alpha)
        if idx_cell_area != -1:
            outlier_cell = cell_area.index[idx_cell_area]
            self.adata.obs['QC_Area_outlier'] = [cell == outlier_cell for cell in self.adata.obs.index]
            print(f"Detected outlier cell area: {outlier_cell}")
        else:
            self.adata.obs['QC_Area_outlier'] = False
            print("No cell area outliers detected.")
    
    def qc_cell_metrics(self):
        """
        Perform QC on cell-level metrics.
        """
        if self.adata is None:
            raise ValueError("AnnData object not created. Run prepare_anndata() first.")
        
        # Calculate ratio of counts to genes
        self.adata.obs['ratio_counts_genes'] = self.adata.obs['total_counts'] / self.adata.obs['n_genes_by_counts']
        
        # Get metrics to plot
        metrics = ['ratio_counts_genes', 'total_counts', 'pct_counts_mt', 'pct_counts_NegProbe']
        df = self.adata.obs[metrics]
        
        # Plot distributions
        for metric in metrics:
            plt.figure(figsize=[6, 4])
            plt.hist(df[metric], color='skyblue', edgecolor='black')
            plt.title(f'Distribution of {metric}')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(self.analysis_dir, f"QC_{metric}.png"))
            plt.close()
        
        print("Cell metrics QC plots created.")
    
    def filter_cells(self):
        """
        Filter cells based on QC metrics.
        
        Returns:
        --------
        pandas.DataFrame
            Filtered cell observations
        """
        if self.adata is None:
            raise ValueError("AnnData object not created. Run prepare_anndata() first.")
        
        # Apply filters
        df = self.adata.obs
        df_filtered = df[
            (df['pct_counts_NegProbe'] < 0.1) & 
            (df['pct_counts_mt'] < 0.25) & 
            (df['ratio_counts_genes'] > 1) & 
            (df['total_counts'] > 20) & 
            (df['QC_Area_outlier'] == False)
        ]
        
        print(f"Filtered {len(df)} cells to {len(df_filtered)} cells.")
        return df_filtered
    
    def qc_fov_metrics(self):
        """
        Perform QC on FOV-level metrics.
        """
        if self.adata is None:
            raise ValueError("AnnData object not created. Run prepare_anndata() first.")
        
        # Get counts
        counts = self.adata.to_df()
        
        # Extract FOV IDs
        fov = [cell.split("_")[0] for cell in counts.index]
        
        # QC for FOV average transcripts per cell
        df_tx = counts.copy()
        df_tx['tx_per_cell'] = df_tx.sum(axis=1)
        df_tx['fov'] = fov
        
        plt.figure(figsize=[10, 4])
        sns.boxplot(data=df_tx, x='fov', y='tx_per_cell', hue='fov')
        plt.title('Transcripts per Cell by FOV')
        plt.savefig(os.path.join(self.analysis_dir, "QC_fov_avg_per_cell.png"))
        plt.close()
        
        # Save FOV average per cell
        df_tx[['tx_per_cell', 'fov']].groupby('fov').mean().to_csv(
            os.path.join(self.data_dir, 'fov_avg_per_cell.csv')
        )
        
        # QC for FOV 90th percentile of gene sums vs median of negative probe sums
        df_gene = counts.iloc[:, ~counts.columns.str.contains('Neg')]
        df_gene['genes'] = df_gene.sum(axis=1)
        df_gene['fov'] = fov
        
        df_neg = counts.iloc[:, counts.columns.str.contains('Neg')]
        df_neg['Neg'] = df_neg.sum(axis=1)
        df_neg['fov'] = fov
        
        df_fov = pd.DataFrame(index=df_neg.groupby('fov').Neg.quantile(0.5).index.tolist())
        df_fov['90_percentile_genes'] = df_gene.groupby('fov').genes.quantile(0.9).tolist()
        df_fov['50_percentile_Neg'] = df_neg.groupby('fov').Neg.quantile(0.5).tolist()
        
        # Save FOV percentiles
        df_fov.to_csv(os.path.join(self.data_dir, 'fov_90_gene_50_Neg.csv'))
        
        # Plot FOV percentiles
        df_fov['fov'] = df_fov.index.tolist()
        df_fov['fov'] = df_fov['fov'].astype('category')
        df_fov['fov'] = df_fov['fov'].cat.reorder_categories(self.fov_ids, ordered=True)
        df_fov_melt = df_fov.melt(var_name='category', value_name='counts', id_vars='fov')
        
        plt.figure(figsize=(10, 4))
        sns.barplot(data=df_fov_melt, x='fov', y='counts', hue='category')
        plt.title('90th Percentile Genes vs 50th Percentile Neg Probes by FOV')
        plt.savefig(os.path.join(self.analysis_dir, 'QC_fov_90_gene_50_Neg.png'))
        plt.close()
        
        print("FOV metrics QC completed.")
    
    def filter_genes(self):
        """
        Filter genes based on expression compared to negative probes.
        
        Returns:
        --------
        pandas.DataFrame
            Filtered gene expression matrix
        """
        if self.adata is None:
            raise ValueError("AnnData object not created. Run prepare_anndata() first.")
        
        # Get counts
        counts = self.adata.to_df()
        
        # Separate genes and negative probes
        df_gene = counts.iloc[:, ~counts.columns.str.contains('Neg')]
        df_neg = counts.iloc[:, counts.columns.str.contains('Neg')]
        
        # Filter genes that have higher expression than 50th percentile of negative probes
        neg_threshold = int(np.percentile(df_neg.sum(), 50))
        gene_sums = df_gene.sum()
        mask = (gene_sums > neg_threshold).values
        df_gene_filtered = df_gene.iloc[:, mask]
        print(f"Filtered {df_gene.shape[1]} genes (excluding Negative Probes) to {df_gene_filtered.shape[1]} genes.")
        return df_gene_filtered
    
    def summarize_qc(self, df_filtered_cells):
        """
        Summarize QC results for cells and genes.
        
        Parameters:
        -----------
        df_filtered_cells : pandas.DataFrame
            Filtered cell observations
        """
        if self.adata is None:
            raise ValueError("AnnData object not created. Run prepare_anndata() first.")
        
        # Summary of cell filtering by FOV
        fov_filtered = [cell.split("_")[0] for cell in df_filtered_cells.index]
        df_filtered_cells['fov'] = fov_filtered
        filtered_counts = pd.DataFrame(
            df_filtered_cells.groupby('fov').size(), 
            columns=['cells_after_filtering']
        )
        
        # Get original cell counts by FOV
        fov_all = [cell.split("_")[0] for cell in self.adata.obs.index]
        self.adata.obs['fov'] = fov_all
        all_counts = pd.DataFrame(
            self.adata.obs.groupby('fov').size(), 
            columns=['cells_before_filtering']
        )
        
        # Combine results
        df_summary = pd.concat([filtered_counts, all_counts], axis=1)
        df_summary.to_csv(os.path.join(self.data_dir, 'QC_cells.csv'))
        
        # Plot cell filtering summary
        df_summary['fov'] = df_summary.index.tolist()
        df_summary['fov'] = df_summary['fov'].astype('category')
        df_summary['fov'] = df_summary['fov'].cat.reorder_categories(self.fov_ids, ordered=True)
        df_summary_melt = df_summary.melt(var_name='category', value_name='cells', id_vars='fov')
        df_summary_melt['category'] = df_summary_melt['category'].astype('category')
        df_summary_melt['category'] = df_summary_melt['category'].cat.reorder_categories(
            ['cells_before_filtering', 'cells_after_filtering'], 
            ordered=True
        )
        
        plt.figure(figsize=(10, 4))
        sns.barplot(data=df_summary_melt, x='fov', y='cells', hue='category')
        plt.title('Cells Before vs After Filtering by FOV')
        plt.ylabel('# of cells')
        plt.savefig(os.path.join(self.analysis_dir, 'QC_cells_filtering.png'))
        plt.close()
        
        # Filter genes
        df_gene_filtered = self.filter_genes()
        
        # Summary of gene filtering
        df_genes = pd.DataFrame({
            'genes_before_filtering': [self.adata.var.shape[0]],
            'genes_after_filtering': [df_gene_filtered.shape[1]]
        }, index=['all_fov'])
        df_genes.to_csv(os.path.join(self.data_dir, 'QC_genes.csv'))
        
        # Plot gene filtering summary
        df_genes_melt = df_genes.melt(var_name='category', value_name='genes')
        df_genes_melt['category'] = df_genes_melt['category'].astype('category')
        df_genes_melt['category'] = df_genes_melt['category'].cat.reorder_categories(
            ['genes_before_filtering', 'genes_after_filtering'], 
            ordered=True
        )
        
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df_genes_melt, x='category', y='genes', hue='category')
        plt.title('Genes Before vs After Filtering')
        plt.ylabel('# of genes')
        plt.savefig(os.path.join(self.analysis_dir, 'QC_genes_filtering.png'))
        plt.close()
        
        print("QC summary completed.")
    
    def run_qc_pipeline(self):
        """
        Run the complete QC pipeline.
        
        Returns:
        --------
        tuple
            (filtered_cells, filtered_genes)
        """
        print("Starting QC pipeline...")
        
        # Prepare AnnData
        self.prepare_anndata()
        
        # QC negative probes
        self.qc_negative_probes()
        
        # QC cell area
        self.qc_cell_area()
        
        # QC cell metrics
        self.qc_cell_metrics()
        
        # Filter cells
        filtered_cells = self.filter_cells()
        
        # QC FOV metrics
        self.qc_fov_metrics()
        
        # Filter genes
        filtered_genes = self.filter_genes()
        
        # Summarize QC
        self.summarize_qc(filtered_cells)
        
        print("QC pipeline completed.")
        return filtered_cells, filtered_genes.loc[filtered_cells.index]


class spatioloji:
    """
    A class for managing and analyzing spatial transcriptomics data across multiple fields of view (FOVs).
    
    The spatioloji class provides a comprehensive framework for handling spatial transcriptomics
    data with support for both local and global coordinate systems. It automatically converts
    cell polygon data into GeoDataFrames for spatial analysis and provides methods for
    subsetting data by FOV or cell IDs.
    
    Attributes
    ----------
    polygons : pandas.DataFrame
        DataFrame with polygon data containing coordinates of cell boundaries.
        Required columns: 'cell', 'x_local_px', 'y_local_px', 'x_global_px', 'y_global_px'
    
    gdf_local : geopandas.GeoDataFrame
        GeoDataFrame with polygon geometries using local coordinates (x_local_px, y_local_px)
    
    gdf_global : geopandas.GeoDataFrame
        GeoDataFrame with polygon geometries using global coordinates (x_global_px, y_global_px)
    
    cell_meta : pandas.DataFrame
        DataFrame with cell metadata.
        Required columns: 'cell'
        Common columns: 'fov', 'CenterX_local_px', 'CenterY_local_px', 'CenterX_global_px', 'CenterY_global_px'
    
    adata : anndata.AnnData
        AnnData object containing gene expression data, dimensionality reduction results,
        clustering results, and other cell-level annotations
    
    fov_positions : pandas.DataFrame
        DataFrame with global coordinates of FOVs ('x_global_px', 'y_global_px')
    
    images : Dict[str, numpy.ndarray]
        Dictionary mapping FOV IDs to image arrays
    
    image_shapes : Dict[str, Tuple[int, int]]
        Dictionary mapping FOV IDs to image shapes (width, height)
    
    custom : Dict[str, any]
        Dictionary for any user-defined additional data
    
    Methods
    -------
    subset_by_fovs(fov_ids, fov_column='fov')
        Create a new spatioloji object containing only the specified FOVs
    
    subset_by_cells(cell_ids)
        Create a new spatioloji object containing only the specified cells
    
    get_cells_in_fov(fov_id, fov_column='fov')
        Get all cells within a specific FOV
    
    get_polygon_for_cell(cell_id)
        Get polygon data for a specific cell
    
    get_geometry_for_cell(cell_id, coordinate_type='local')
        Get geometry for a specific cell from the GeoDataFrame
    
    get_image(fov_id)
        Get the image for a specific FOV
    
    summary()
        Get a summary of the object's data
    
    add_custom(key, value)
        Add custom data to the object
    
    get_custom(key)
        Retrieve custom data from the object
    
    to_pickle(filepath)
        Save the object to a pickle file
    
    to_anndata()
        Get the AnnData object
    
    Static Methods
    --------------
    from_pickle(filepath)
        Load a spatioloji object from a pickle file
    
    read_images_from_folder(folder_path, fov_ids, img_format='jpg', prefix_img='CellComposite_F')
        Read images for all FOVs from a folder
    
    from_files(polygons_path, cell_meta_path, adata_path, fov_positions_path, images_folder=None, 
               img_format='jpg', prefix_img='CellComposite_F', fov_id_column='fov')
        Create a spatioloji object from file paths
    """

    def __init__(self,
                 polygons: pd.DataFrame,
                 cell_meta: pd.DataFrame,
                 adata: anndata.AnnData,
                 fov_positions: pd.DataFrame,
                 images: Dict[str, np.ndarray] = None,
                 image_shapes: Dict[str, Tuple[int, int]] = None,
                 images_folder: str = None,
                 img_format: str = 'jpg',
                 prefix_img: str = 'CellComposite_F',
                 fov_id_column: str = 'fov'):
        """
        Initialize a spatioloji object.
        
        Parameters
        ----------
        polygons : pandas.DataFrame
            DataFrame with polygon data, containing at minimum a 'cell' column and
            coordinate columns ('x_local_px', 'y_local_px', 'x_global_px', 'y_global_px')
        
        cell_meta : pandas.DataFrame
            DataFrame with cell metadata, containing at minimum a 'cell' column
        
        adata : anndata.AnnData
            AnnData object containing gene expression data and other cell-level information
        
        fov_positions : pandas.DataFrame
            DataFrame with global coordinates of FOVs
        
        images : Dict[str, numpy.ndarray], optional
            Dictionary mapping FOV IDs to image arrays. If None and images_folder is provided,
            images will be loaded from the folder.
        
        image_shapes : Dict[str, Tuple[int, int]], optional
            Dictionary mapping FOV IDs to image shapes (width, height)
        
        images_folder : str, optional
            Path to folder containing FOV images. If provided and images is None,
            images will be loaded from this folder.
        
        img_format : str, default='jpg'
            Image file format extension (e.g., 'jpg', 'png')
        
        prefix_img : str, default='CellComposite_F'
            Prefix for image filenames
        
        fov_id_column : str, default='fov'
            Column name for FOV IDs in the fov_positions DataFrame
        
        Notes
        -----
        - The constructor automatically converts polygons to GeoDataFrames for both local
        and global coordinate systems.
        - If images_folder is provided, the constructor attempts to load images for all FOVs.
        - The constructor validates that required columns exist and that cell IDs match
        across datasets.
        """
        self.polygons = polygons
        self.cell_meta = cell_meta
        self.adata = adata
        self.fov_positions = fov_positions
        self.custom: Dict[str, any] = {}
        
        # Convert polygons to GeoDataFrames (local and global coordinates)
        self.gdf_local = self._convert_to_geodataframe(polygons, coordinate_type='local')
        self.gdf_global = self._convert_to_geodataframe(polygons, coordinate_type='global')
        
        # Automatically load images if folder is provided
        if images_folder is not None and (images is None or image_shapes is None):
            # Try to determine the FOV ID column if not specified or if it doesn't exist
            if fov_id_column not in fov_positions.columns:
                # First try the default 'fov' column
                if 'fov' in fov_positions.columns:
                    fov_id_column = 'fov'
                # Look for any column with 'fov' in its name
                else:
                    fov_cols = [col for col in fov_positions.columns if 'fov' in col.lower()]
                    if fov_cols:
                        fov_id_column = fov_cols[0]
                    # If we still can't find it, use the first column as a fallback
                    else:
                        fov_id_column = fov_positions.columns[0]
                        print(f"Warning: Could not find a FOV ID column. Using '{fov_id_column}' as FOV ID column.")
            
            # Extract FOV IDs from the determined column
            fov_ids = fov_positions[fov_id_column].astype(str).tolist()
            self.images, self.image_shapes = self.read_images_from_folder(
                folder_path=images_folder,
                fov_ids=fov_ids,
                img_format=img_format,
                prefix_img=prefix_img
            )
        else:
            self.images = images if images is not None else {}
            self.image_shapes = image_shapes if image_shapes is not None else {}

        self._validate(fov_id_column)

    def _convert_to_geodataframe(self, polygons_df: pd.DataFrame, coordinate_type: str = 'local') -> 'geopandas.GeoDataFrame':
        """
        Convert polygon DataFrame to GeoDataFrame.
        
        Args:
            polygons_df: DataFrame containing polygon data
            coordinate_type: Type of coordinates to use ('local' or 'global')
            
        Returns:
            GeoDataFrame with geometry column
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
            
            # Create a copy to avoid modifying the original
            df_copy = polygons_df.copy()
            
            # Determine which coordinate columns to use
            if coordinate_type == 'local':
                x_col, y_col = 'x_local_px', 'y_local_px'
            elif coordinate_type == 'global':
                x_col, y_col = 'x_global_px', 'y_global_px'
            else:
                raise ValueError(f"Invalid coordinate_type: {coordinate_type}. Must be 'local' or 'global'")
                
            # Check if we have the necessary columns for polygon creation
            if all(col in df_copy.columns for col in [x_col, y_col]):
                # Group by cell ID to create polygons
                geometries = []
                for _, group in df_copy.groupby('cell'):
                    # Create polygon from x, y coordinates
                    coords = list(zip(group[x_col], group[y_col]))
                    # Ensure the polygon is closed (first and last points are the same)
                    if coords[0] != coords[-1] and len(coords) > 2:
                        coords.append(coords[0])
                    
                    if len(coords) >= 3:  # Need at least 3 points to form a polygon
                        geometries.append((group['cell'].iloc[0], Polygon(coords)))
                    else:
                        print(f"Warning: Cell {group['cell'].iloc[0]} has fewer than 3 points, cannot create polygon.")
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    data=[g[0] for g in geometries],
                    geometry=[g[1] for g in geometries],
                    columns=['cell']
                )
                
                # Merge with original data to preserve other columns
                # Get one row per cell from original data (for non-coordinate columns)
                cell_info = df_copy.drop([x_col, y_col], axis=1).drop_duplicates('cell')
                gdf = gdf.merge(cell_info, on='cell')
                
                return gdf
            else:
                print(f"Warning: Required columns for polygon creation ({x_col}, {y_col}) not found. Returning empty GeoDataFrame.")
                return gpd.GeoDataFrame()
                
        except ImportError:
            print("Warning: geopandas or shapely not installed. GeoDataFrame conversion skipped.")
            # Return empty DataFrame with geometry column to maintain interface
            df_copy = polygons_df.copy()
            df_copy['geometry'] = None
            return df_copy

    def _validate(self, fov_id_column='fov'):
        # Check for required columns in polygons and cell_meta
        required_columns = {
            'polygons': ['cell'],
            'cell_meta': ['cell']
        }
        
        # Add fov column check only if it exists in both DataFrames
        if fov_id_column in self.polygons.columns and fov_id_column in self.cell_meta.columns:
            required_columns['polygons'].append(fov_id_column)
            required_columns['cell_meta'].append(fov_id_column)
        
        # Validate required columns
        for df_name, columns in required_columns.items():
            df = getattr(self, df_name)
            for col in columns:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found in {df_name}.")
        
        # Validate AnnData object
        assert isinstance(self.adata, anndata.AnnData), "adata must be an AnnData object"
        
        # Validate cell IDs match across datasets
        self._validate_cell_ids()
        
        # Validate images if they're provided and we know the FOV ID column
        if self.images and fov_id_column in self.fov_positions.columns:
            missing_fovs = [fid for fid in self.fov_positions[fov_id_column].astype(str) 
                           if fid not in self.images]
            if missing_fovs:
                print(f"Warning: Missing images for FOVs: {missing_fovs}")

    def _validate_cell_ids(self):
        """Validate that cell IDs match across datasets."""
        # Get sets of cell IDs from each data source
        gdf_local_cells = set(self.gdf_local['cell'].astype(str)) if not self.gdf_local.empty else set()
        meta_cells = set(self.cell_meta['cell'].astype(str))
        
        # Check if adata has cell IDs in obs
        if 'cell' in self.adata.obs.columns:
            adata_cells = set(self.adata.obs['cell'].astype(str))
        else:
            # Try to use index as cell IDs
            adata_cells = set(self.adata.obs.index.astype(str))
        
        # Check for cells in gdf_local but not in cell_meta
        if gdf_local_cells - meta_cells:
            print(f"Warning: {len(gdf_local_cells - meta_cells)} cells in gdf_local are not in cell_meta.")
        
        # Check for cells in cell_meta but not in gdf_local
        if meta_cells - gdf_local_cells:
            print(f"Warning: {len(meta_cells - gdf_local_cells)} cells in cell_meta are not in gdf_local.")
        
        # Check for cells in adata but not in cell_meta
        if adata_cells - meta_cells:
            print(f"Warning: {len(adata_cells - meta_cells)} cells in adata are not in cell_meta.")
        
        # Check for cells in cell_meta but not in adata
        if meta_cells - adata_cells:
            print(f"Warning: {len(meta_cells - adata_cells)} cells in cell_meta are not in adata.")

    def subset_by_fovs(self, fov_ids: List[str], fov_column: str = 'fov') -> 'spatioloji':
        """
        Create a new spatioloji object containing only the specified FOVs.
        
        Parameters
        ----------
        fov_ids : List[str]
            List of FOV IDs to include in the subset
        
        fov_column : str, default='fov'
            Column name for FOV IDs
        
        Returns
        -------
        spatioloji
            A new spatioloji object with the subset of data
        
        Notes
        -----
        This method subsets all components of the spatioloji object:
        - polygons
        - cell_meta
        - adata
        - fov_positions
        - images
        - image_shapes
        
        The GeoDataFrames (gdf_local and gdf_global) are automatically regenerated
        from the subset polygons when the new spatioloji object is created.
        """

        fov_ids = [str(fid) for fid in fov_ids]  # Ensure FOV IDs are strings
        
        # Check if FOV column exists
        if fov_column not in self.cell_meta.columns or fov_column not in self.polygons.columns:
            print(f"Warning: '{fov_column}' column not found in required DataFrames. Cannot subset by FOVs.")
            return self
            
        # Subset polygons and cell_meta
        subset_polygons = self.polygons[self.polygons[fov_column].astype(str).isin(fov_ids)]
        subset_cell_meta = self.cell_meta[self.cell_meta[fov_column].astype(str).isin(fov_ids)]
        
        # Get cell IDs for the FOVs we're keeping
        subset_cells = subset_cell_meta['cell'].astype(str).unique()
        
        # Subset adata based on these cells
        if 'cell' in self.adata.obs.columns:
            # If we have a 'cell' column, use it to subset
            cell_mask = self.adata.obs['cell'].astype(str).isin(subset_cells)
            subset_adata = self.adata[cell_mask].copy()
        else:
            # Otherwise assume cell IDs are the index
            cell_mask = self.adata.obs.index.astype(str).isin(subset_cells)
            subset_adata = self.adata[cell_mask].copy()
        
        # Subset fov_positions
        subset_fov_positions = self.fov_positions[self.fov_positions[fov_column].astype(str).isin(fov_ids)]
        
        # Subset images and image_shapes
        subset_images = {fid: img for fid, img in self.images.items() if fid in fov_ids}
        subset_image_shapes = {fid: shape for fid, shape in self.image_shapes.items() if fid in fov_ids}
        
        # Create new spatioloji instance with subset data
        return spatioloji(
            polygons=subset_polygons,
            cell_meta=subset_cell_meta,
            adata=subset_adata,
            fov_positions=subset_fov_positions,
            images=subset_images,
            image_shapes=subset_image_shapes
        )
        
    def subset_by_cells(self, cell_ids: List[str]) -> 'spatioloji':
        """
        Create a new spatioloji object containing only the specified cells.
        
        Parameters
        ----------
        cell_ids : List[str]
            List of cell IDs to include in the subset
        
        Returns
        -------
        spatioloji
            A new spatioloji object with the subset of data
        
        Notes
        -----
        This method subsets all components of the spatioloji object:
        - polygons
        - cell_meta
        - adata
        - fov_positions (only FOVs containing the specified cells are kept)
        - images (only images for FOVs containing the specified cells are kept)
        - image_shapes (only shapes for FOVs containing the specified cells are kept)
        
        The GeoDataFrames (gdf_local and gdf_global) are automatically regenerated
        from the subset polygons when the new spatioloji object is created.
        """
        cell_ids = [str(cid) for cid in cell_ids]  # Ensure cell IDs are strings
        
        # Subset polygons and cell_meta
        subset_polygons = self.polygons[self.polygons['cell'].astype(str).isin(cell_ids)]
        subset_cell_meta = self.cell_meta[self.cell_meta['cell'].astype(str).isin(cell_ids)]
        
        # Get FOVs for the cells we're keeping
        if 'fov' in subset_cell_meta.columns:
            fov_column = 'fov'
        else:
            # Try to find FOV column
            fov_cols = [col for col in subset_cell_meta.columns if 'fov' in col.lower()]
            if fov_cols:
                fov_column = fov_cols[0]
            else:
                print("Warning: No FOV column found. Using all FOVs.")
                fov_column = None
        
        if fov_column:
            subset_fovs = subset_cell_meta[fov_column].astype(str).unique()
            subset_fov_positions = self.fov_positions[self.fov_positions[fov_column].astype(str).isin(subset_fovs)]
            
            # Subset images and image_shapes
            subset_images = {fid: img for fid, img in self.images.items() if fid in subset_fovs}
            subset_image_shapes = {fid: shape for fid, shape in self.image_shapes.items() if fid in subset_fovs}
        else:
            # Keep all FOVs, images, and shapes
            subset_fov_positions = self.fov_positions.copy()
            subset_images = self.images.copy()
            subset_image_shapes = self.image_shapes.copy()
            
        # Subset adata
        if 'cell' in self.adata.obs.columns:
            # If we have a 'cell' column, use it to subset
            cell_mask = self.adata.obs['cell'].astype(str).isin(cell_ids)
            subset_adata = self.adata[cell_mask].copy()
        else:
            # Otherwise assume cell IDs are the index
            cell_mask = self.adata.obs.index.astype(str).isin(cell_ids)
            subset_adata = self.adata[cell_mask].copy()
            
        # Create new spatioloji instance with subset data
        return spatioloji(
            polygons=subset_polygons,
            cell_meta=subset_cell_meta,
            adata=subset_adata,
            fov_positions=subset_fov_positions,
            images=subset_images,
            image_shapes=subset_image_shapes
        )

    def get_cells_in_fov(self, fov_id: str, fov_column: str = 'fov') -> pd.DataFrame:
        """Get all cells within a specific FOV."""
        if fov_column not in self.cell_meta.columns:
            print(f"Warning: '{fov_column}' column not found in cell_meta. Cannot get cells by FOV.")
            return pd.DataFrame()
        return self.cell_meta[self.cell_meta[fov_column] == fov_id]

    def get_polygon_for_cell(self, cell_id: str) -> pd.DataFrame:
        """Get polygon data for a specific cell."""
        return self.polygons[self.polygons['cell'] == cell_id]
        
    def get_geometry_for_cell(self, cell_id: str, coordinate_type: str = 'local') -> Optional['geopandas.GeoSeries']:
        """
        Get geometry for a specific cell from the GeoDataFrame.
        
        Args:
            cell_id: Cell ID to get geometry for
            coordinate_type: Type of coordinates to use ('local' or 'global')
            
        Returns:
            Geometry for the specified cell or None if not found
        """
        try:
            # Choose the appropriate GeoDataFrame
            gdf = self.gdf_local if coordinate_type == 'local' else self.gdf_global
            
            mask = gdf['cell'] == cell_id
            if not mask.any():
                print(f"Warning: Cell ID '{cell_id}' not found in {coordinate_type} GeoDataFrame.")
                return None
            return gdf[mask].geometry.iloc[0]
        except (AttributeError, KeyError):
            print(f"Warning: {coordinate_type} GeoDataFrame not properly initialized or missing 'geometry' column.")
            return None

    def get_image(self, fov_id: str) -> Optional[np.ndarray]:
        """
        Get the image for a specific FOV.
        
        Parameters
        ----------
        fov_id : str
            FOV ID to get image for
        
        Returns
        -------
        numpy.ndarray or None
            Image array for the specified FOV, or None if the FOV is not found
        """
        return self.images.get(fov_id)

    def summary(self) -> Dict[str, any]:
        """
        Get a summary of the object's data.
        
        Returns
        -------
        Dict[str, any]
            Dictionary containing summary information:
            - n_cells: Number of cells in cell_meta
            - n_fovs: Number of FOVs in fov_positions
            - n_polygons: Number of polygon vertices in polygons
            - n_local_geometries: Number of cell geometries in gdf_local
            - n_global_geometries: Number of cell geometries in gdf_global
            - image_fovs: List of FOV IDs with images
            - adata_shape: Shape of the AnnData object (n_cells, n_genes)
        """
        return {
            "n_cells": len(self.cell_meta),
            "n_fovs": len(self.fov_positions),
            "n_polygons": len(self.polygons),
            "n_local_geometries": len(self.gdf_local) if hasattr(self.gdf_local, '__len__') else 0,
            "n_global_geometries": len(self.gdf_global) if hasattr(self.gdf_global, '__len__') else 0,
            "image_fovs": list(self.images.keys()),
            "adata_shape": self.adata.shape
        }

    def add_custom(self, key: str, value: any):
        """
        Add custom data to the object.
        
        Parameters
        ----------
        key : str
            Key to store the data under
        
        value : any
            Data to store
        """
        self.custom[key] = value

    def get_custom(self, key: str) -> any:
        """
        Retrieve custom data from the object.
        
        Parameters
        ----------
        key : str
            Key to retrieve data for
        
        Returns
        -------
        any
            Data stored under the specified key, or None if the key is not found
        """
        return self.custom.get(key)

    def to_pickle(self, filepath: str):
        """Save the object to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(filepath: str) -> 'spatioloji':
        """Load a spatioloji object from a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def to_anndata(self) -> anndata.AnnData:
        """
        Get the AnnData object.
        
        Returns
        -------
        anndata.AnnData
            AnnData object containing gene expression data and other cell-level information
        """
        return self.adata
        
    @staticmethod
    def read_images_from_folder(folder_path: str, 
                               fov_ids: List[str], 
                               img_format: str = 'jpg',
                               prefix_img: str = 'CellComposite_F') -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
        """
        Read images for all FOVs from a folder.
        
        Args:
            folder_path: Path to the folder containing images
            fov_ids: List of FOV IDs to load images for
            img_format: Image file format extension (e.g., 'jpg', 'png')
            prefix_img: Prefix for image filenames (e.g., 'CellComposite_F')
            
        Returns:
            Tuple containing:
                - Dictionary mapping FOV IDs to image arrays
                - Dictionary mapping FOV IDs to image shapes (width, height)
        """
        images = {}
        image_shapes = {}
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Warning: Image folder '{folder_path}' does not exist.")
            return images, image_shapes
            
        for fov_id in fov_ids:
            # Try with zero-padded format (e.g., '001' for fov_id '1')
            padded_id = fov_id.zfill(3) if fov_id.isdigit() else fov_id
            filename = os.path.join(folder_path, f"{prefix_img}{padded_id}.{img_format}")
            
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            if img is not None:
                images[fov_id] = img
                image_shapes[fov_id] = (img.shape[1], img.shape[0])
            else:
                # If padded version fails, try with original fov_id
                original_filename = os.path.join(folder_path, f"{prefix_img}{fov_id}.{img_format}")
                if filename != original_filename:  # Only try if different from first attempt
                    img = cv2.imread(original_filename, cv2.IMREAD_COLOR)
                    if img is not None:
                        images[fov_id] = img
                        image_shapes[fov_id] = (img.shape[1], img.shape[0])
                    else:
                        print(f"Warning: Image for FOV '{fov_id}' not found at {filename} or {original_filename}")
                else:
                    print(f"Warning: Image for FOV '{fov_id}' not found at {filename}")
        
        if not images:
            print(f"Warning: No images found in folder '{folder_path}' for the provided FOV IDs.")
        
        return images, image_shapes

    @staticmethod
    def from_files(polygons_path: str,
                   cell_meta_path: str,
                   adata_path: str,
                   fov_positions_path: str,
                   images_folder: str = None,
                   img_format: str = 'jpg',
                   prefix_img: str = 'CellComposite_F',
                   fov_id_column: str = 'fov') -> 'spatioloji':
        """
        Create a spatioloji object from file paths.
        
        Args:
            polygons_path: Path to CSV file with polygon data
            cell_meta_path: Path to CSV file with cell metadata
            adata_path: Path to h5ad file with gene expression data
            fov_positions_path: Path to CSV file with FOV positions
            images_folder: Optional path to folder containing FOV images
            img_format: Image file format extension (default: 'jpg')
            prefix_img: Prefix for image filenames (default: 'CellComposite_F')
            fov_id_column: Column name for FOV IDs in fov_positions (default: 'fov')
            
        Returns:
            A new spatioloji object
        """
        polygons = pd.read_csv(polygons_path)
        cell_meta = pd.read_csv(cell_meta_path)
        adata = anndata.read_h5ad(adata_path)
        fov_positions = pd.read_csv(fov_positions_path)

        # Create spatioloji instance with automatic image loading
        return spatioloji(
            polygons=polygons,
            cell_meta=cell_meta,
            adata=adata,
            fov_positions=fov_positions,
            images_folder=images_folder,
            img_format=img_format,
            prefix_img=prefix_img,
            fov_id_column=fov_id_column
        )
        
        
        
        
