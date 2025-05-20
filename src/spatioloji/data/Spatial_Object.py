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

class spatioloji:
    """
    A class for managing and analyzing spatial transcriptomics data across multiple fields of view (FOVs).
    
    Attributes:
        polygons: DataFrame with polygon data ('fov', 'x_local_px', 'y_local_px', 'x_global_px', 'y_global_px', 'cell')
        cell_meta: DataFrame with cell metadata ('fov', 'CenterX_local_px', 'CenterY_local_px', 'CenterX_global_px', 'CenterY_global_px', 'cell', others)
        adata: AnnData object containing gene expression data, UMAP, cell types, etc.
        fov_positions: DataFrame with global coordinates of FOVs ('x_global_px','y_global_px')
        images: Dict mapping FOV IDs to image arrays
        image_shapes: Dict mapping FOV IDs to (width, height)
        custom: Dict for any user-defined additional data
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

        self.polygons = polygons
        self.cell_meta = cell_meta
        self.adata = adata
        self.fov_positions = fov_positions
        self.custom: Dict[str, any] = {}
        
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
        
        # Validate images if they're provided and we know the FOV ID column
        if self.images and fov_id_column in self.fov_positions.columns:
            missing_fovs = [fid for fid in self.fov_positions[fov_id_column].astype(str) 
                           if fid not in self.images]
            if missing_fovs:
                print(f"Warning: Missing images for FOVs: {missing_fovs}")

    def get_cells_in_fov(self, fov_id: str, fov_column: str = 'fov') -> pd.DataFrame:
        """Get all cells within a specific FOV."""
        if fov_column not in self.cell_meta.columns:
            print(f"Warning: '{fov_column}' column not found in cell_meta. Cannot get cells by FOV.")
            return pd.DataFrame()
        return self.cell_meta[self.cell_meta[fov_column] == fov_id]

    def get_polygon_for_cell(self, cell_id: str) -> pd.DataFrame:
        """Get polygon data for a specific cell."""
        return self.polygons[self.polygons['cell'] == cell_id]

    def get_image(self, fov_id: str) -> Optional[np.ndarray]:
        """Get the image for a specific FOV."""
        return self.images.get(fov_id)

    def summary(self) -> Dict[str, any]:
        """Get a summary of the object's data."""
        return {
            "n_cells": len(self.cell_meta),
            "n_fovs": len(self.fov_positions),
            "n_polygons": len(self.polygons),
            "image_fovs": list(self.images.keys()),
            "adata_shape": self.adata.shape
        }

    def add_custom(self, key: str, value: any):
        """Add custom data to the object."""
        self.custom[key] = value

    def get_custom(self, key: str) -> any:
        """Retrieve custom data from the object."""
        return self.custom.get(key)

    def to_pickle(self, filepath: str):
        """Save the object to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(filepath: str) -> 'Spatioloji':
        """Load a Spatioloji object from a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def to_anndata(self) -> anndata.AnnData:
        """Get the AnnData object."""
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
                   fov_id_column: str = 'fov') -> 'Spatioloji':
        """
        Create a Spatioloji object from file paths.
        
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
            A new Spatioloji object
        """
        polygons = pd.read_csv(polygons_path)
        cell_meta = pd.read_csv(cell_meta_path)
        adata = anndata.read_h5ad(adata_path)
        fov_positions = pd.read_csv(fov_positions_path)

        # Create Spatioloji instance with automatic image loading
        return Spatioloji(
            polygons=polygons,
            cell_meta=cell_meta,
            adata=adata,
            fov_positions=fov_positions,
            images_folder=images_folder,
            img_format=img_format,
            prefix_img=prefix_img,
            fov_id_column=fov_id_column
        )
        


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



