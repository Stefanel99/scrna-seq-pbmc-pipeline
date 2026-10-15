import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

def parse_arguments():
    parser = argparse.ArgumentParser(description="Comprehensive Single-Cell RNA-seq Exploratory Pipeline")
    parser.add_argument("--input_dir", type=str, default="data/filtered_feature_bc_matrix/", help="Path to 10x mtx directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs and figures")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=100, facecolor='white', figsize=(8, 6), vector_friendly=True)

    print(">>> Loading data...")
    adata = sc.read_10x_mtx(args.input_dir, var_names='gene_symbols', cache=True)
    adata.var_names_make_unique()
    adata.raw = adata

    print(">>> Calculating QC metrics...")
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    adata.var['hb'] = adata.var_names.str.startswith('HB')

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo', 'hb'], percent_top=None, log1p=False, inplace=True)

    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False)
    plt.savefig(os.path.join(fig_dir, "qc_violin.png"), bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axes[0], show=False)
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=axes[1], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "qc_metrics.png"))
    plt.close()

    print(">>> Filtering cells and genes...")
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[
        (adata.obs['n_genes_by_counts'] > 500) &
        (adata.obs['n_genes_by_counts'] < 5000) &
        (adata.obs['pct_counts_mt'] < 15) &
        (adata.obs['total_counts'] > 1000) &
        (adata.obs['total_counts'] < 30000)
    ].copy()

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=500)

    print(">>> Normalizing and selecting highly variable genes...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    plt.figure(figsize=(8, 4))
    sc.pl.highly_variable_genes(adata, show=False)
    plt.title("Highly Variable Genes")
    plt.savefig(os.path.join(fig_dir, "hvg.png"))
    plt.close()

    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)

    print(">>> Running PCA...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

    pca_coords = adata.obsm['X_pca']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    variance_ratio = adata.uns['pca']['variance_ratio']
    axes[0,0].plot(range(1, len(variance_ratio)+1), variance_ratio, 'bo-', markersize=3)
    axes[0,0].set_xlabel('Principal component')
    axes[0,0].set_ylabel('Variance ratio')
    axes[0,0].set_title('PCA variance ratio')
    axes[0,0].set_yscale('log')

    s1 = axes[0,1].scatter(pca_coords[:, 0], pca_coords[:, 1], c=adata.obs['total_counts'], s=1, alpha=0.7, cmap='viridis')
    axes[0,1].set_title('PCA - Total counts')
    plt.colorbar(s1, ax=axes[0,1])

    s2 = axes[1,0].scatter(pca_coords[:, 0], pca_coords[:, 1], c=adata.obs['n_genes_by_counts'], s=1, alpha=0.7, cmap='viridis')
    axes[1,0].set_title('PCA - Number of genes')
    plt.colorbar(s2, ax=axes[1,0])

    s3 = axes[1,1].scatter(pca_coords[:, 0], pca_coords[:, 1], c=adata.obs['pct_counts_mt'], s=1, alpha=0.7, cmap='viridis')
    axes[1,1].set_title('PCA - Mitochondrial %')
    plt.colorbar(s3, ax=axes[1,1])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pca.png"))
    plt.close()

    print(">>> Computing neighbors, UMAP, and multi-resolution clustering...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata, random_state=42)


    sc.tl.leiden(adata, resolution=0.15, random_state=42, flavor="igraph", directed=False, n_iterations=2, key_added='leiden_0.15')
    sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5')
    sc.tl.leiden(adata, resolution=0.8, key_added='leiden_0.8')
    sc.tl.leiden(adata, resolution=1.0, key_added='leiden_1.0')
    sc.tl.louvain(adata, flavor="igraph", directed=False, random_state=42)

    adata.obs['clusters'] = adata.obs['leiden_0.15']

    print(">>> Calculating Differential Expression (Marker Genes)...")
    sc.tl.rank_genes_groups(adata, groupby='leiden_0.15', method='wilcoxon', key_added='rank_genes_groups', pts=True)
    
    sc.pl.rank_genes_groups(adata, n_genes=5, sharey=False, show=False, save="_rank_genes.png")

    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    marker_df = pd.DataFrame({
        f"{group}_{key}": result[key][group]
        for group in groups for key in ['names', 'pvals_adj', 'logfoldchanges']
    })
    marker_df.to_csv(os.path.join(args.output_dir, "marker_genes_per_cluster.csv"))


    sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=5, groupby='leiden_0.15', use_raw=True, show=False, save="marker_genes.png")

    print(">>> Mapping canonical lineage markers and cell types...")
    pbmc_markers = {
        'CD4+ T cells': ['CD4', 'IL7R', 'CCR7'],
        'CD8+ T cells': ['CD8A', 'CD8B', 'GZMK'],
        'Cytotoxic T/NK Cells': ['GZMA', 'GZMB', 'NKG7', 'KLRB1'],
        'NK cells': ['GNLY', 'NKG7', 'FGFBP2'],
        'B cells': ['MS4A1', 'CD79A', 'CD79B'],
        'CD14+ Monocytes': ['CD14', 'LYZ', 'CD68'],
        'FCGR3A+ Monocytes': ['FCGR3A', 'MS4A7'],
        'Dendritic Cells': ['FCER1A', 'CST3'],
        'Megakaryocytes': ['PPBP', 'PF4'],
        'Progenitor-like': ['SOX4', 'HMGB1', 'RPL15']
    }

    sc.pl.dotplot(adata, var_names=pbmc_markers, groupby='leiden_0.15', use_raw=True, show=False, save="marker_gene_sign.png")
    sc.pl.matrixplot(adata, var_names=pbmc_markers, groupby='leiden_0.15', use_raw=True, standard_scale='var', cmap='Reds', show=False, save="marker_gene_sign.png")

    cluster_annotation = {
        '0': 'CD4+ T cells',
        '1': 'Cytotoxic T/NK Cells',
        '2': 'Dendritic Cells',
        '3': 'Monocytes',
        '4': 'FCGR3A+ Monocytes',
        '5': 'NK Cells',
        '6': 'B Cells',
        '7': 'CD8+ T cells',
        '8': 'Megakaryocytes',
        '9': 'Progenitor-like',
        '10': 'Other'
    }

    adata.obs['cell_type'] = adata.obs['leiden_0.15'].astype(str).map(cluster_annotation).fillna('Unknown')

    sc.pl.umap(adata, color='cell_type', legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon=False, show=False, save='_clusters_assigned.png')

    sc.pl.umap(adata, color=['LYZ', 'RPL15', 'PPBP'], use_raw=True, vmin=0, vmax=100, cmap='Reds', size=20, show=False, save='_val_ann.png')

    print(">>> Computing PAGA Trajectory Topology...")
    sc.tl.paga(adata, groups='leiden_0.15')
    sc.pl.paga(adata, show=False, save=".png")
    
    sc.tl.paga(adata, groups='cell_type')
    sc.pl.paga(adata, show=False, save='_cell_type.png')

    print(">>> Full exploratory pipeline run completed successfully.")

if __name__ == "__main__":
    main()