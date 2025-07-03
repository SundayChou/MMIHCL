"""
Functions for preprocessing input data.
"""
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad


def readDataset(dataset_name):
    """
    Read original single-cell AnnDatas from dataset.

    Parameters:
    dataset_name (str): The name of the dataset to be read.

    Returns:
    shared_adata_1 (anndata.AnnData): The AnnData about the shared features of first modality.
    shared_adata_2 (anndata.AnnData): The AnnData about the shared features of second modality.
    all_adata_1 (anndata.AnnData): The AnnData about the all features of first modality.
    all_adata_2 (anndata.AnnData): The AnnData about the all features of second modality.
    """
    # The case where the two modal feature codes are inconsistent.
    if dataset_name in ['CITEseq_PBMC', 'ABseq_BMC', 'CODEX_tonsil', 'TEAseq_PBMC', 'CITEseq_BMC']:
        all_adata_1 = ad.read_h5ad('../data/' + dataset_name + '/RNA.h5ad')
        all_adata_2 = ad.read_h5ad('../data/' + dataset_name + '/protein.h5ad')

        correspondence = pd.read_csv('../data/protein_gene_conversion.csv')
        rna_protein_correspondence = []

        for i in range(correspondence.shape[0]):
            curr_protein_name, curr_rna_names = correspondence.iloc[i]
            if curr_protein_name not in all_adata_2.var_names:
                continue
            if curr_rna_names.find('Ignore') != -1:
                continue
            curr_rna_names = curr_rna_names.split('/')
            for r in curr_rna_names:
                if r in all_adata_1.var_names:
                    rna_protein_correspondence.append([r, curr_protein_name])

        rna_protein_correspondence = np.array(rna_protein_correspondence)
        shared_adata_1 = all_adata_1[:, rna_protein_correspondence[:, 0]].copy()
        shared_adata_2 = all_adata_2[:, rna_protein_correspondence[:, 1]].copy()

    elif dataset_name == '10X_PBMC':
        all_adata_1 = ad.read_h5ad('../data/' + dataset_name + '/RNA.h5ad')
        all_adata_2 = ad.read_h5ad('../data/' + dataset_name + '/ATAC.h5ad')
        
        correspondence = pd.read_csv('../data/atac_gene_conversion.csv')
        rna_atac_correspondence = []

        for i in range(correspondence.shape[0]):
            curr_atac_name, curr_rna_names = correspondence.iloc[i]
            if curr_atac_name not in all_adata_2.var_names:
                continue
            if curr_rna_names.find('Ignore') != -1:
                continue
            curr_rna_names = curr_rna_names.split('/')
            for r in curr_rna_names:
                if r in all_adata_1.var_names:
                    rna_atac_correspondence.append([r, curr_atac_name])
                    
        rna_atac_correspondence = np.array(rna_atac_correspondence)
        shared_adata_1 = all_adata_1[:, rna_atac_correspondence[:, 0]].copy()
        shared_adata_2 = all_adata_2[:, rna_atac_correspondence[:, 1]].copy()   

    # The case where the two modal feature codes are consistent.
    elif dataset_name in ['CITEseq_CyTOF_PBMC', 'CyTOF_human']:
        if dataset_name == 'CITEseq_CyTOF_PBMC':
            all_adata_1 = ad.read_h5ad('../data/' + dataset_name + '/CITEseq.h5ad')
            all_adata_2 = ad.read_h5ad('../data/' + dataset_name + '/CyTOF.h5ad')

        elif dataset_name == 'CyTOF_human':
            all_adata_1 = ad.read_h5ad('../data/' + dataset_name + '/H1N1.h5ad')
            all_adata_2 = ad.read_h5ad('../data/' + dataset_name + '/IFNG.h5ad')

        shared_var = np.intersect1d(all_adata_1.var_names.to_numpy(), all_adata_2.var_names.to_numpy())
        shared_adata_1 = all_adata_1[:, shared_var].copy()
        shared_adata_2 = all_adata_2[:, shared_var].copy()

    return shared_adata_1, shared_adata_2, all_adata_1, all_adata_2


def preprocessAnnData(dataset_name, shared_adata_1, shared_adata_2, all_adata_1, all_adata_2, n_sample_1=None, n_sample_2=None):
    """
    Apply standard scanpy preprocessing steps on AnnDatas.
    Considering the difference in data, the way of preprocessing is different.

    Parameters:
    dataset_name (str): The name of the dataset.
    shared_adata_1 (anndata.AnnData): The AnnData about the shared features of first modality.
    shared_adata_2 (anndata.AnnData): The AnnData about the shared features of second modality.
    all_adata_1 (anndata.AnnData): The AnnData about the all features of first modality.
    all_adata_2 (anndata.AnnData): The AnnData about the all features of second modality.
    n_sample_1 (int): The number of downsamples of first modality.
        None means take the full dataset.
    n_sample_2 (int): The number of downsamples of second modality.
        None means take the full dataset.

    Returns:
    new_shared_adata_1 (anndata.AnnData): New AnnData object of shared_adata_1.
    new_shared_adata_2 (anndata.AnnData): New AnnData object of shared_adata_2.
    new_all_adata_1 (anndata.AnnData): New AnnData object of all_adata_1.
    new_all_adata_2 (anndata.AnnData): New AnnData object of all_adata_2.
    """
    # Procrocess shared_adata_1. 
    if dataset_name in ['CITEseq_PBMC', 'ABseq_BMC', 'CODEX_tonsil', 'TEAseq_PBMC', 'CITEseq_BMC', '10X_PBMC']:
        sc.pp.normalize_total(shared_adata_1)
        sc.pp.log1p(shared_adata_1)
    sc.pp.scale(shared_adata_1)

    # Procrocess shared_adata_2. 
    if dataset_name in ['CITEseq_PBMC', 'ABseq_BMC', 'TEAseq_PBMC', 'CITEseq_BMC', '10X_PBMC']:
        sc.pp.normalize_total(shared_adata_2)
        sc.pp.log1p(shared_adata_2)
    sc.pp.scale(shared_adata_2)

    # Procrocess all_adata_1.
    if dataset_name in ['CITEseq_PBMC', 'ABseq_BMC', 'CODEX_tonsil', 'TEAseq_PBMC', 'CITEseq_BMC', '10X_PBMC']:
        sc.pp.normalize_total(all_adata_1)
        sc.pp.log1p(all_adata_1)
        sc.pp.highly_variable_genes(all_adata_1)
        all_adata_1 = all_adata_1[:, all_adata_1.var['highly_variable']].copy()
    sc.pp.scale(all_adata_1)

    # Procrocess all_adata_2. 
    if dataset_name in ['CITEseq_PBMC', 'ABseq_BMC', 'TEAseq_PBMC', 'CITEseq_BMC', '10X_PBMC']:
        sc.pp.normalize_total(all_adata_2)
        sc.pp.log1p(all_adata_2)
        if dataset_name == '10X_PBMC':
            sc.pp.highly_variable_genes(all_adata_2)
            all_adata_2 = all_adata_2[:, all_adata_2.var['highly_variable']].copy()
    sc.pp.scale(all_adata_2)

    # Downsample datasets.
    if n_sample_1 is None:
        id_1 = range(shared_adata_1.shape[0])
    else:
        id_1 = np.random.choice(shared_adata_1.shape[0], size=n_sample_1, replace=False)
    new_shared_adata_1, new_all_adata_1 = shared_adata_1[id_1], all_adata_1[id_1].copy()

    if n_sample_2 is None:
        id_2 = range(shared_adata_2.shape[0])
    else:
        id_2 = np.random.choice(shared_adata_2.shape[0], size=n_sample_2, replace=False)
    new_shared_adata_2, new_all_adata_2 = shared_adata_2[id_2], all_adata_2[id_2].copy()

    # Make sure no column is static.
    mask = (
        (new_shared_adata_1.X.std(axis=0) > 1e-5) 
        & (new_shared_adata_2.X.std(axis=0) > 1e-5)
    )
    new_shared_adata_1 = new_shared_adata_1[:, mask].copy()
    new_shared_adata_2 = new_shared_adata_2[:, mask].copy()

    return new_shared_adata_1, new_shared_adata_2, new_all_adata_1, new_all_adata_2


def getLabels(dataset_name, adata_1, adata_2):
    """
    Get cell annotation labels from AnnDatas.

    Parameters:
    dataset_name (str): The name of the dataset.
    adata_1 (anndata.AnnData): The AnnData for first modality.
    adata_2 (anndata.AnnData): The AnnData for second modality.

    Returns:
    labels_1 (list): The list of cell labels for first modality.
    labels_2 (list): The list of cell labels for second modality.
    """
    if dataset_name == 'CITEseq_PBMC':
        labels_1 = adata_1.obs['celltype.l1'].tolist()
        labels_2 = adata_2.obs['celltype.l1'].tolist()

    elif dataset_name in ['ABseq_BMC', '10X_PBMC', 'CITEseq_BMC']:
        labels_1 = adata_1.obs['celltype.l2'].tolist()
        labels_2 = adata_2.obs['celltype.l2'].tolist()

    else:
        labels_1 = adata_1.obs['celltype'].tolist()
        labels_2 = adata_2.obs['celltype'].tolist()

    return labels_1, labels_2