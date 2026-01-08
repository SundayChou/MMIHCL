"""
The main object MMIHCL for running the pipeline.
"""
import scanpy as sc
import anndata as ad

from Model import *
from Utils import *
from scipy.optimize import linear_sum_assignment


# The Multimodal Integration through Hypergraph Contrastive Learning (MMIHCL) class.
class MMIHCL:
    def __init__(self, shared_array_1, shared_array_2, all_array_1, all_array_2):
        """
        Initialization for MMIHCL object.

        Parameters:
        shared_array_1 (numpy.ndarray): The first dataset with shared features.
        shared_array_2 (numpy.ndarray): The second dataset with shared features.
        all_array_1 (numpy.ndarray): The first dataset with all active features.
        all_array_2 (numpy.ndarray): The second dataset with all active features.

        Returns:
        None.
        """
        # Initialize shared and all activate arrays from datasets.
        self.shared_array_1 = shared_array_1
        self.shared_array_2 = shared_array_2
        self.all_array_1 = all_array_1
        self.all_array_2 = all_array_2
        self.all_array_cca_1 = None
        self.all_array_cca_2 = None

        # Initialize dataset sizes.
        self.n1 = self.shared_array_1.shape[0]
        self.n2 = self.shared_array_2.shape[0]

        # Initialize matching scale parameters.
        self.n_intermediate_match = None
        self.n_final_match = None

        # Initialize graph adjacent matrices over two datasets.
        self.k_neighbor_adj_1 = None
        self.k_neighbor_adj_2 = None
        self.agg_mat_1 = None
        self.agg_mat_2 = None

        # Initialize models and optimizers.
        self.model = None
        self.opt = None

        # Initialize cell embeddings learned from models.
        self.shared_embeds_1 = None
        self.shared_embeds_2 = None
        self.all_embeds_1 = None
        self.all_embeds_2 = None

        # Initialize the distances between cell embeddings.
        self.init_dist = None
        self.refined_dist = None
        self.final_dist = None

        # Initialize the cell matchings between two modalities.
        self.init_matching = None
        self.refined_matching = None
        self.final_matching = None

    def setMatchingParameters(self, n_intermediate_match, n_final_match, verbose=False):
        """
        Set how many cells in the second dataset will be matched with one in the first dataset.

        Parameters:
        n_match (int): Each row in the first dataset is matched to at most n_match rows in the second dataset.
        verbose (bool): Whether to print the progress.

        Returns:
        None.
        """
        # Avoid insufficient number of cells in the second dataset.
        if self.n1 * n_final_match > self.n2:
            raise ValueError("Not enough cells in the second dataset!")

        # Set parameters about matching scale.
        self.n_intermediate_match = n_intermediate_match
        self.n_final_match = n_final_match

        if verbose:
            if n_final_match == 1:
                print('We will perform one-to-one matching between two datasets.')
            else:
                print(str(n_final_match) + ' cells in the second dataset will be matched with one in the first dataset.')

    def constructGraph(self, n_neighbors_1=15, n_neighbors_2=15, n_components_1=30, n_components_2=30, verbose=False):
        """
        Construct graph matrices over two all active feature datasets.

        Parameters:
        n_neighbors_1 (int): The number of neighbors for adjacent graph construction over self.all_array_1.
        n_neighbors_2 (int): The number of neighbors for adjacent graph construction over self.all_array_2.
        n_components_1 (int): Number of components of self.all_array_1 to keep.
        n_components_2 (int): Number of components of self.all_array_2 to keep.
        verbose (bool): Whether to print the progress.

        Returns:
        None.
        """
        # Calculate adaptively k-neighbor adjacent matrices over two all active feature datasets.
        if verbose:
            print('Calculating adaptively k-neighbor adjacent matrices...')

        adata_1 = ad.AnnData(svdDenoise(self.all_array_1, n_components_1))
        adata_2 = ad.AnnData(svdDenoise(self.all_array_2, n_components_2))

        sc.pp.neighbors(adata_1, n_neighbors=n_neighbors_1, n_pcs=None, use_rep='X', metric='correlation')
        sc.pp.neighbors(adata_2, n_neighbors=n_neighbors_2, n_pcs=None, use_rep='X', metric='correlation')

        rows_1, cols_1 = adata_1.obsp['connectivities'].nonzero()
        rows_2, cols_2 = adata_2.obsp['connectivities'].nonzero()

        vals_1 = adata_1.obsp['connectivities'][(rows_1, cols_1)].A1
        vals_2 = adata_2.obsp['connectivities'][(rows_2, cols_2)].A1

        self.k_neighbor_adj_1 = np.zeros(shape=(adata_1.shape[0],adata_1.shape[0]))
        self.k_neighbor_adj_2 = np.zeros(shape=(adata_2.shape[0],adata_2.shape[0]))

        for i in range(len(rows_1)):
            self.k_neighbor_adj_1[rows_1[i], cols_1[i]] = vals_1[i]
        for i in range(len(rows_2)):
            self.k_neighbor_adj_2[rows_2[i], cols_2[i]] = vals_2[i]

        if verbose:
            print('Two adaptively k-neighbor adjacent matrices have been calculated.')
            print('Calculating GCN aggregate matrices...')

        # Calculate GCN aggregate matrices over two adjacent matrices.
        self.agg_mat_1 = aggregateAdjacentMatrix(self.k_neighbor_adj_1)
        self.agg_mat_2 = aggregateAdjacentMatrix(self.k_neighbor_adj_2)

        if verbose:
            print('Two GCN aggregate matrices have been calculation calculated.')

    def learnEmbeddings(self, init_embeds_1, init_embeds_2, hyperedge_dim=32, n_epochs=1000, 
                        n_layers=2, learn_rate=2e-2, weight_decay=1e-2, temp=0.1, verbose=False):
        """
        Learn hypergraph-based cell embeddings.

        Parameters:
        init_embeds_1 (numpy.ndarray): The first list of embeddings to generate their learned embeddings.
        init_embeds_2 (numpy.ndarray): The second list of embeddings to generate their learned embeddings.
        hyperedge_dim (int): The dimension of cell-hyperedge embeddings.
        n_epochs (int): The model iteration epoch.
        n_layers (int): The number of layers of GCN and HGNN.
        learn_rate (float): The learning rate of model.
        weight_decay (float): The factor controlling weight decay in the optimizer.
        temp (float): The temperature parameter.
        verbose (bool): Whether to print the progress.

        Returns:
        learned_embeds_1 (numpy.ndarray): The learned embeddings of first modality.
        learned_embeds_2 (numpy.ndarray): The learned embeddings of second modality.
        """
        # Initialize the model and optimizer.
        if verbose:
            print('Initializing the model and optimizer...')

        self.model = HypergraphModel(
            init_embeds_1=init_embeds_1,
            init_embeds_2=init_embeds_2,
            hyperedge_dim=hyperedge_dim
        ).cuda()

        self.opt = torch.optim.Adam(
            params=self.model.parameters(),
            lr=learn_rate,
            weight_decay=weight_decay
        )

        # Fit the model to get cell embeddings of cell features.
        if verbose:
            print('Fitting the model...')

        adj_1 = torch.from_numpy(self.agg_mat_1.astype('float32')).cuda()
        adj_2 = torch.from_numpy(self.agg_mat_2.astype('float32')).cuda()

        for i in range(n_epochs):
            self.model.train()
            _, _, g1, g2, h1, h2 = self.model.forward(adj_1, adj_2, n_layers)
            loss = self.model.calculateLoss(g1, g2, h1, h2, n_layers, temp)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        # Get learned embeddings for two modalities.
        learned_embeds_1, learned_embeds_2, _, _, _, _ = self.model.forward(adj_1, adj_2, n_layers)
        learned_embeds_1 = learned_embeds_1.detach().cpu().numpy()
        learned_embeds_2 = learned_embeds_2.detach().cpu().numpy()

        if verbose:
            print('The hypergraph-based embedding learning is complete.')

        return learned_embeds_1, learned_embeds_2

    def findMatching(self, dist_mat, mode):
        """
        Find matching based on distance matrix and mode.

        Parameters:
        dist_mat (numpy.ndarray): The distance matrix of cross-modal embeddings.
        mode (str): The matching mode, 'intermediate' or 'final'.

        Returns:
        matching (list): A list of length three.
            The i-th matched pair is (matching[0][i], matching[1][i]).
            The score of the i-th matching pair is matching[2][i].
        """
        # generate an intermediate many-to-many matching
        if mode == 'intermediate':
            idx_1, idx_2 = linear_sum_assignment(dist_mat)
            dist = np.array([dist_mat[i, j] for i, j in zip(idx_1, idx_2)])
            idx_1_list = list(idx_1)
            idx_2_list = list(idx_2)
            dist_list = list(dist)

            for k in range(1, self.n_intermediate_match):
                for l in range(len(idx_1_list)):
                    dist_mat[idx_1_list[l]][idx_2_list[l]] = 1
                idx_1, idx_2 = linear_sum_assignment(dist_mat)
                dist = np.array([dist_mat[i, j] for i, j in zip(idx_1, idx_2)])
                idx_1_list += list(idx_1)
                idx_2_list += list(idx_2)
                dist_list += list(dist)

            matching = [idx_1_list, idx_2_list, dist_list]

        # generate the final one-to-N matching
        elif mode == 'final':
            if self.n_final_match > 1:
                dist_mat = np.tile(dist_mat, (self.n_final_match, 1))

            argsort_k = np.argsort(dist_mat, axis=1)
            argsort_k = argsort_k[:, -(self.n2 - self.n2 // 2):]
            dist_mat_cp = np.copy(dist_mat)
            dist_mat_cp[np.arange(dist_mat.shape[0])[:, None], argsort_k] = np.Inf

            idx_1, idx_2 = linear_sum_assignment(dist_mat_cp)
            idx_1 = idx_1 % self.n1
            dist = np.array([dist_mat[i, j] for i, j in zip(idx_1, idx_2)])
            matching = [list(idx_1), list(idx_2), list(dist)]            

        return matching

    def filterMatching(self, matching, threshold=0.7):
        """
        Filter out low-quality pairs from matching pairs.

        Parameters:
        matching (list): A list of length three.
            The i-th matched pair is (matching[0][i], matching[1][i]).
            The score of the i-th matching pair is matching[2][i].
        threshold (float): Filtering threshold.

        Returns:
        filtered_matching (list): A list of length three.
            The i-th filtered matched pair is (matching[0][i], matching[1][i]).
            The score of the i-th filtered matching pair is matching[2][i].
        """
        filtered_matching = [[], [], []]

        for i in range(self.n1):
            idx_1_list, idx_2_list, dist_list = [], [], []

            for j in range(self.n_intermediate_match):
                idx_1_list.append(matching[0][j * self.n1 + i])
                idx_2_list.append(matching[1][j * self.n1 + i])
                dist_list.append(matching[2][j * self.n1 + i])

            # Every cell in first modality is guaranteed to have at least one match in second modality.
            cur_min_dist = min(dist_list)

            for j in range(self.n_intermediate_match):
                if dist_list[j] == cur_min_dist or dist_list[j] < threshold:
                    filtered_matching[0].append(idx_1_list[j])
                    filtered_matching[1].append(idx_2_list[j])
                    filtered_matching[2].append(dist_list[j])

        return filtered_matching

    def findInitialMatching(self, hyperegde_dim=32, n_epochs=1000, n_layers=2, learn_rate=2e-2, 
                            weight_decay=1e-3, temp=0.1, min_dist=1e-7, threshold=0.7, verbose=False):
        """
        Find initial matching between two modalities.

        Parameters:
        hyperedge_dim (int): The dimension of cell-hyperedge embeddings.
        n_epochs (int): The model iteration epoch.
        n_layers (int): The number of layers of GCN and HGNN.
        learn_rate (float): The learning rate of model.
        weight_decay (float): The factor controlling weight decay in the optimizer.
        temp (float): The temperature parameter.
        min_dist (float): The minimum distance used to correct.
        threshold (float): Filtering threshold.
        verbose (bool): Whether to print the progress.

        Returns:
        None.
        """ 
        if verbose:
            print('Learning shared feature embeddings...')

        # Learn hypergraph-based shared feature embeddings for two modalities.
        self.shared_embeds_1, self.shared_embeds_2 = self.learnEmbeddings(
            init_embeds_1=self.shared_array_1,
            init_embeds_2=self.shared_array_2,
            hyperedge_dim=hyperegde_dim,
            n_epochs=n_epochs,
            n_layers=n_layers,
            learn_rate=learn_rate,
            weight_decay=weight_decay,
            temp=temp,
            verbose=verbose
        )

        # Calculate and correct the distance of shared feature embeddings.
        self.init_dist = correlationDistance(self.shared_embeds_1, self.shared_embeds_2)
        self.init_dist = correctDistance(self.init_dist, min_dist)

        # Match cells based on self.init_dist and filter out bad matching pairs.
        dist_mat = np.copy(self.init_dist)

        if verbose:
            print('Finding the initial matching...')
        self.init_matching = self.findMatching(dist_mat, 'intermediate')

        if verbose:
            print('Filtering the initial matching...')
        self.init_matching = self.filterMatching(self.init_matching.copy(), threshold)

        if verbose:
            print('The initial matching has been found and filtered.')

    def CCAProjection(self, matching, n_components=20, max_iter=1000):
        """
        Build a CCA projection of all active arrays for two modalities based on matching.

        Parameters:
        matching (list): A list of length three.
            The i-th matched pair is (matching[0][i], matching[1][i]).
            The score of the i-th matching pair is matching[2][i].
        n_components (int): Number of components of CCA projection results to keep.
        max_iter (int): Maximum number of iterations.

        Returns:
        None.
        """ 
        X, Y = [], []
        Y_idx_list = [[] for _ in range(self.n1)]

        for i in range(len(matching[0])):
            Y_idx_list[matching[0][i]].append(matching[1][i])

        for i in range(self.n1):
            X.append(self.all_array_1[i])
            Y_temp = np.zeros(self.all_array_2.shape[1])
            for j in Y_idx_list[i]:
                Y_temp += self.all_array_2[j]
            Y.append(Y_temp / len(Y_idx_list[i]))

        _, cca = getCancor(X, Y, n_components, max_iter)
        self.all_array_cca_1, self.all_array_cca_2 = cca.transform(self.all_array_1, self.all_array_2)
        self.all_array_cca_1, self.all_array_cca_2 = self.all_array_cca_1.astype(np.float32), self.all_array_cca_2.astype(np.float32)

    def findRefinedMatching(self, hyperegde_dim=32, n_epochs=1000, n_layers=2, learn_rate=2e-2, weight_decay=1e-3, 
                            temp=0.1, min_dist=1e-7, threshold=0.7, n_iter=2, n_components=20, max_iter=1000, verbose=False):
        """
        Find refined matching between two modalities.

        Parameters:
        hyperedge_dim (int): The dimension of cell-hyperedge embeddings.
        n_epochs (int): The model iteration epoch.
        n_layers (int): The number of layers of GCN and HGNN.
        learn_rate (float): The learning rate of model.
        weight_decay (float): The factor controlling weight decay in the optimizer.
        temp (float): The temperature parameter.
        min_dist (float): The minimum distance used to correct.
        threshold (float): Filtering threshold.
        n_iter (int): The number of rounds of iterative optimization.
        n_components (int): Number of components of CCA projection results to keep.
        max_iter (int): Maximum number of iterations in CCA projection.
        verbose (bool): Whether to print the progress.

        Returns:
        None.
        """
        self.refined_matching = self.init_matching

        for t in range(n_iter):
            if verbose:
                print('The current number of iterations: ' + str(t + 1) + '.')

            # Use CCA projection to get joint embeddings for two modalities.
            if verbose:
                print('Using CCA projection to get joint embeddings...')

            self.CCAProjection(
                matching=self.refined_matching,
                n_components=n_components,
                max_iter=max_iter
            )

            # Learn hypergraph-based all feature embeddings for two modalities.
            if verbose:
                print('Learning all feature embeddings......')           

            self.all_embeds_1, self.all_embeds_2 = self.learnEmbeddings(
                init_embeds_1=self.all_array_cca_1,
                init_embeds_2=self.all_array_cca_2,
                hyperedge_dim=hyperegde_dim,
                n_epochs=n_epochs,
                n_layers=n_layers,
                learn_rate=learn_rate,
                weight_decay=weight_decay,
                temp=temp,
                verbose=False
            )

            if t + 1 == n_iter:
                break

            # Find refined matching between two modalities.
            if verbose:
                print('Finding refind matching based on all feature embeddings.......')

            self.refined_dist = correlationDistance(self.all_embeds_1, self.all_embeds_2)
            self.refined_dist = correctDistance(self.refined_dist, min_dist)

            dist_mat = np.copy(self.refined_dist)

            self.refined_matching = self.findMatching(dist_mat, 'intermediate')
            self.refined_matching = self.filterMatching(self.refined_matching.copy(), threshold)

    def findFinalMatching(self, min_dist=1e-7, threshold=0.7, n_components=20, max_iter=1000, verbose=False):
        """
        Find final matching between two modalities.

        Parameters:
        min_dist (float): The minimum distance used to correct.
        threshold (float): Filtering threshold.
        n_components (int): Number of components of CCA projection results to keep.
        max_iter (int): Maximum number of iterations in CCA projection.
        verbose (bool): Whether to print the progress.

        Returns:
        None.
        """ 
        # Find final matching between two modalities.
        self.final_dist = correlationDistance(self.all_embeds_1, self.all_embeds_2)
        self.final_dist = correctDistance(self.final_dist, min_dist)

        dist_mat = np.copy(self.final_dist)

        if verbose:
            print('Finding the final matching...')        
        self.final_matching = self.findMatching(dist_mat, 'final')

        if verbose:
            print('Filtering the final matching...')
        self.final_matching = self.filterMatching(self.final_matching.copy(), threshold)

        if verbose:
            print('The final matching has been found and filtered.')

        # Use CCA projection to get final joint embeddings.
        if verbose:
            print('Using CCA projection to get final joint embeddings...')

        self.CCAProjection(
            matching=self.final_matching,
            n_components=n_components,
            max_iter=max_iter
        )

        if verbose:
            print('The final joint embeddings have been geted.')
