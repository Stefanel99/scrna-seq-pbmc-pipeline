# Single-Cell RNA-Seq Exploratory Analysis Pipeline

## Overview
This repository contains an exploratory single-cell RNA sequencing (scRNA-seq) analysis workflow utilizing peripheral blood mononuclear cell (PBMC) data. Developed as a hands-on learning exercise, this project implements a complete bioinformatics pipeline using Python and Scanpy, moving from raw count matrices to biologically annotated cell populations and trajectory inference.

## Workflow Architecture
The analysis pipeline executes the following sequential steps:
1. **Quality Control:** Filtering cells based on total UMI counts, detected gene counts, and mitochondrial gene percentage thresholds to eliminate low-quality droplets and dying cells.
2. **Normalization and Scaling:** Total count normalization, log1p transformation, highly variable gene (HVG) selection, and regression of technical covariates.
3. **Dimensionality Reduction:** Principal Component Analysis (PCA) followed by Uniform Manifold Approximation and Projection (UMAP) for non-linear manifold visualization.
4. **Graph-Based Clustering:** Construction of a K-nearest neighbor graph and partitioning using the Leiden algorithm across multiple resolution scales (e.g., $0.15$ to $1.0$) to resolve macro and micro-clusters.
5. **Differential Expression and Annotation:** Identification of cluster marker genes via the Wilcoxon rank-sum test, visualized through dot plots, matrix plots, and spatial UMAP overlays mapped to canonical lineage markers.
6. **Trajectory Inference:** Partition-based Graph Abstraction (PAGA) to map global topological connectivity and developmental relationships between cell clusters.

## Repository Structure
* `analysis.ipynb`: The main exploratory notebook containing the complete workflow execution.
* `environment.yml`: Conda environment configuration specifying required software dependencies and versions.
* `figures/`: Directory containing exported high-resolution plots, including quality control metrics, PCA variance ratios, UMAP cluster projections, and PAGA topology graphs.

## Requirements and Installation
To set up the environment and run the analysis, clone the repository and build the conda environment:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
conda env create -f environment.yml
conda activate scRNA-seq-analysis