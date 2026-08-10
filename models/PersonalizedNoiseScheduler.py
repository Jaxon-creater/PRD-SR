import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import networkx as nx


def ensure_csr_matrix(x):
    """Convert sparse inputs to a stable scipy csr_matrix across scipy versions."""
    if isinstance(x, csr_matrix):
        return x

    if sp.issparse(x):
        # COO-like object with explicit row/col
        if hasattr(x, 'row') and hasattr(x, 'col'):
            return csr_matrix((np.asarray(x.data), (np.asarray(x.row), np.asarray(x.col))), shape=x.shape)

        # sparse array variants may expose coords
        if hasattr(x, 'coords'):
            row, col = x.coords
            return csr_matrix((np.asarray(x.data), (np.asarray(row), np.asarray(col))), shape=x.shape)

        # generic scipy sparse object
        if hasattr(x, 'tocsr'):
            return x.tocsr()

        if hasattr(x, 'tocoo'):
            y = x.tocoo()
            if hasattr(y, 'row') and hasattr(y, 'col'):
                return csr_matrix((np.asarray(y.data), (np.asarray(y.row), np.asarray(y.col))), shape=y.shape)
            if hasattr(y, 'coords'):
                row, col = y.coords
                return csr_matrix((np.asarray(y.data), (np.asarray(row), np.asarray(col))), shape=y.shape)

        # fallback via nonzero
        row, col = x.nonzero()
        data = np.asarray(x[row, col]).reshape(-1)
        return csr_matrix((data, (row, col)), shape=x.shape)

    return csr_matrix(x)


class PersonalizedNoiseScheduler:
    """
    Adjust each user's diffusion noise level from activity and influence.
    """
    
    def __init__(self, n_users, train_mat, trust_mat, 
                 activity_weight=0.5, influence_weight=0.5,
                 min_noise_scale=0.3, max_noise_scale=1.0):
        """
        Args:
            n_users: Number of users.
            train_mat: Sparse user-item interaction matrix.
            trust_mat: Sparse user-user social matrix.
            activity_weight: Contribution of normalized user activity.
            influence_weight: Contribution of normalized social influence.
            min_noise_scale: Noise scale for highly reliable users.
            max_noise_scale: Noise scale for less reliable users.
        """
        self.n_users = n_users
        self.activity_weight = activity_weight
        self.influence_weight = influence_weight
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale

        # Normalize sparse formats to avoid SciPy matrix/array API differences.
        train_mat = ensure_csr_matrix(train_mat)
        trust_mat = ensure_csr_matrix(trust_mat)
        
        # Estimate interaction activity and social influence.
        self.user_activity = self._calculate_activity(train_mat)
        self.user_influence = self._calculate_influence(trust_mat)

        # Convert reliability estimates into personalized noise scales.
        self.noise_scales = self._generate_noise_scales()
        
        print(f"[PersonalizedNoiseScheduler] Initialized")
        print(f"  Activity range: [{self.user_activity.min():.4f}, {self.user_activity.max():.4f}]")
        print(f"  Influence range: [{self.user_influence.min():.4f}, {self.user_influence.max():.4f}]")
        print(f"  Noise scale range: [{self.noise_scales.min():.4f}, {self.noise_scales.max():.4f}]")
    
    def _calculate_activity(self, train_mat):
        """Compute min-max-normalized interaction counts per user."""
        activity = np.array(train_mat.sum(axis=1)).flatten()

        # Normalize activity to [0, 1].
        min_val, max_val = activity.min(), activity.max()
        if max_val > min_val:
            activity = (activity - min_val) / (max_val - min_val)
        else:
            activity = np.ones_like(activity) * 0.5
        
        return activity
    
    def _calculate_influence(self, trust_mat):
        """Estimate social influence from degree centrality and PageRank."""
        n_users = trust_mat.shape[0]

        # Column and row sums capture incoming and outgoing social degree.
        out_degree = np.array(trust_mat.sum(axis=0)).flatten()
        in_degree = np.array(trust_mat.sum(axis=1)).flatten()

        degree_centrality = out_degree + in_degree * 0.5

        if degree_centrality.max() > 0:
            degree_centrality = degree_centrality / degree_centrality.max()

        try:
            # Support both current and legacy NetworkX/SciPy APIs.
            if hasattr(nx, 'from_scipy_sparse_array'):
                G = nx.from_scipy_sparse_array(trust_mat, create_using=nx.DiGraph)
            else:
                G = nx.from_scipy_sparse_matrix(trust_mat, create_using=nx.DiGraph)
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
            pagerank_scores = np.array([pagerank.get(i, 0) for i in range(n_users)])
            
            if pagerank_scores.max() > 0:
                pagerank_scores = pagerank_scores / pagerank_scores.max()

            influence = 0.6 * degree_centrality + 0.4 * pagerank_scores
        except Exception:
            # Fall back to degree centrality when PageRank does not converge.
            influence = degree_centrality
            print("[Warning] PageRank calculation failed, using degree centrality only")
        
        return influence
    
    def _generate_noise_scales(self):
        """
        Map reliability to noise scales.

        Highly active and influential users receive lower noise, while less
        reliable users receive higher noise.
        """
        # Combine the two normalized reliability signals.
        combined_score = (self.activity_weight * self.user_activity + 
                         self.influence_weight * self.user_influence)
        
        # Invert reliability so that higher scores produce lower noise.
        noise_scales = self.max_noise_scale - (combined_score * 
                                               (self.max_noise_scale - self.min_noise_scale))
        
        return noise_scales
    
    def get_noise_scale(self, user_indices):
        """
        Return noise scales for selected users as a NumPy array.
        
        Args:
            user_indices: User indices as a NumPy array or torch tensor.
        
        Returns:
            Noise scales as a NumPy array.
        """
        if isinstance(user_indices, torch.Tensor):
            user_indices = user_indices.cpu().numpy()
        
        return self.noise_scales[user_indices]
    
    def get_noise_scale_tensor(self, user_indices, device):
        """
        Return noise scales for selected users as a torch tensor.
        
        Args:
            user_indices: User-index tensor.
            device: Target torch device.
        
        Returns:
            Noise-scale tensor with shape ``[batch_size, 1]``.
        """
        scales = self.get_noise_scale(user_indices)
        return torch.tensor(scales, dtype=torch.float32, device=device).view(-1, 1)
    
    def get_statistics(self):
        """Return summary statistics for the reliability signals and noise."""
        stats = {
            'activity': {
                'mean': float(self.user_activity.mean()),
                'std': float(self.user_activity.std()),
                'min': float(self.user_activity.min()),
                'max': float(self.user_activity.max())
            },
            'influence': {
                'mean': float(self.user_influence.mean()),
                'std': float(self.user_influence.std()),
                'min': float(self.user_influence.min()),
                'max': float(self.user_influence.max())
            },
            'noise_scale': {
                'mean': float(self.noise_scales.mean()),
                'std': float(self.noise_scales.std()),
                'min': float(self.noise_scales.min()),
                'max': float(self.noise_scales.max())
            }
        }
        return stats
    
    def visualize_distribution(self, save_path=None):
        """Visualize activity, influence, noise, and their relationship."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Activity distribution.
            axes[0, 0].hist(self.user_activity, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('User Activity Distribution')
            axes[0, 0].set_xlabel('Activity Score')
            axes[0, 0].set_ylabel('Count')
            
            # Influence distribution.
            axes[0, 1].hist(self.user_influence, bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('User Influence Distribution')
            axes[0, 1].set_xlabel('Influence Score')
            axes[0, 1].set_ylabel('Count')
            
            # Noise-scale distribution.
            axes[1, 0].hist(self.noise_scales, bins=50, alpha=0.7, color='red')
            axes[1, 0].set_title('Noise Scale Distribution')
            axes[1, 0].set_xlabel('Noise Scale')
            axes[1, 0].set_ylabel('Count')
            
            # Relationship between activity and assigned noise.
            axes[1, 1].scatter(self.user_activity, self.noise_scales, alpha=0.3, s=10)
            axes[1, 1].set_title('Activity vs Noise Scale')
            axes[1, 1].set_xlabel('Activity Score')
            axes[1, 1].set_ylabel('Noise Scale')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[PersonalizedNoiseScheduler] Visualization saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("[Warning] Matplotlib not available, skipping visualization")
