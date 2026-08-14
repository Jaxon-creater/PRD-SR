import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


class SemanticFeatureGenerator:
    """
    Semantic feature generator using a local all-MiniLM-L6-v2 model.
    Generates 384-dimensional semantic vectors from text data.
    """

    def __init__(self, model_path='models/all-MiniLM-L6-v2', device=None):
        """
        Args:
            model_path: Local all-MiniLM-L6-v2 directory. Relative paths are
                resolved from the code_github project root.
            device: torch device (cuda or cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.expanduser(model_path)
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        self.model_path = os.path.abspath(model_path)

        required_files = ('modules.json', 'config.json')
        missing_files = [
            filename for filename in required_files
            if not os.path.isfile(os.path.join(self.model_path, filename))
        ]
        if not os.path.isdir(self.model_path) or missing_files:
            missing_text = ', '.join(missing_files) if missing_files else 'model directory'
            raise FileNotFoundError(
                f"Local MiniLM model is incomplete at {self.model_path}; "
                f"missing: {missing_text}. Download all-MiniLM-L6-v2 before training."
            )

        # Enforce local-only loading. This process must never contact the Hub.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; install requirements.txt before training"
            ) from exc

        self.model = SentenceTransformer(self.model_path)
        self.model = self.model.to(self.device)
        print(f"[SemanticFeatureGenerator] Loaded local model: {self.model_path}")
        print(f"[SemanticFeatureGenerator] Embedding dimension: {self.embedding_dim}")

    def generate_embeddings(self, texts, batch_size=64, show_progress=True):
        """
        Generate semantic embeddings for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_texts, 384)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def load_and_process_reviews(self, csv_path, n_users, n_items):
        """
        Load review texts from CSV and generate user/item semantic features.

        The CSV should have columns: split, user_idx, item_idx, review_text

        Args:
            csv_path: Path to the review_texts.csv file
            n_users: Total number of users
            n_items: Total number of items

        Returns:
            user_semantic_features: torch.Tensor of shape (n_users, 384)
            item_semantic_features: torch.Tensor of shape (n_items, 384)
        """
        print(f"[SemanticFeatureGenerator] Loading reviews from {csv_path}")

        # Fit semantic features on training reviews only. Using validation/test
        # text here would leak held-out information into model inputs.
        df = pd.read_csv(csv_path)
        if 'split' not in df.columns:
            raise ValueError("review CSV must contain an explicit 'split' column")
        df = df[df['split'] == 'train'].reset_index(drop=True)
        print(f"[SemanticFeatureGenerator] Loaded {len(df)} training reviews")

        # Initialize aggregation containers
        user_embeddings_list = {i: [] for i in range(n_users)}
        item_embeddings_list = {i: [] for i in range(n_items)}

        # Generate embeddings for all review texts
        print("[SemanticFeatureGenerator] Generating embeddings for all reviews...")
        all_texts = df['review_text'].tolist()
        all_embeddings = self.generate_embeddings(all_texts)

        # Aggregate embeddings by user and item
        print("[SemanticFeatureGenerator] Aggregating embeddings by user and item...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Aggregating"):
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            embedding = all_embeddings[idx]

            if user_idx < n_users:
                user_embeddings_list[user_idx].append(embedding)
            if item_idx < n_items:
                item_embeddings_list[item_idx].append(embedding)

        # Compute mean embeddings for users and items
        print("[SemanticFeatureGenerator] Computing mean embeddings...")
        user_semantic_features = np.zeros((n_users, self.embedding_dim))
        item_semantic_features = np.zeros((n_items, self.embedding_dim))

        for user_idx in range(n_users):
            if len(user_embeddings_list[user_idx]) > 0:
                user_semantic_features[user_idx] = np.mean(user_embeddings_list[user_idx], axis=0)

        for item_idx in range(n_items):
            if len(item_embeddings_list[item_idx]) > 0:
                item_semantic_features[item_idx] = np.mean(item_embeddings_list[item_idx], axis=0)

        # Convert to torch tensors
        user_semantic_features = torch.tensor(user_semantic_features, dtype=torch.float32)
        item_semantic_features = torch.tensor(item_semantic_features, dtype=torch.float32)

        print(f"[SemanticFeatureGenerator] User semantic features shape: {user_semantic_features.shape}")
        print(f"[SemanticFeatureGenerator] Item semantic features shape: {item_semantic_features.shape}")

        return user_semantic_features, item_semantic_features

    def save_features(self, user_features, item_features, save_dir):
        """
        Save generated semantic features to disk.

        Args:
            user_features: User semantic feature tensor
            item_features: Item semantic feature tensor
            save_dir: Directory to save the features
        """
        os.makedirs(save_dir, exist_ok=True)

        user_path = os.path.join(save_dir, 'user_semantic_features.pt')
        item_path = os.path.join(save_dir, 'item_semantic_features.pt')

        torch.save(user_features, user_path)
        torch.save(item_features, item_path)

        print(f"[SemanticFeatureGenerator] Saved user features to {user_path}")
        print(f"[SemanticFeatureGenerator] Saved item features to {item_path}")

    def load_features(self, save_dir):
        """
        Load pre-computed semantic features from disk.

        Args:
            save_dir: Directory containing the saved features

        Returns:
            user_features: User semantic feature tensor
            item_features: Item semantic feature tensor
        """
        user_path = os.path.join(save_dir, 'user_semantic_features.pt')
        item_path = os.path.join(save_dir, 'item_semantic_features.pt')

        if os.path.exists(user_path) and os.path.exists(item_path):
            user_features = torch.load(user_path)
            item_features = torch.load(item_path)
            print(f"[SemanticFeatureGenerator] Loaded user features from {user_path}")
            print(f"[SemanticFeatureGenerator] Loaded item features from {item_path}")
            return user_features, item_features
        else:
            return None, None


class SemanticProjection(nn.Module):
    """
    Projects semantic features (384-dim) to the same dimension as
    denoised interaction vectors and social vectors.
    """

    def __init__(self, input_dim=384, output_dim=64, dropout=0.1):
        """
        Args:
            input_dim: Dimension of semantic features (384 for all-MiniLM-L6-v2)
            output_dim: Target dimension (should match n_hid in GCN)
            dropout: Dropout rate
        """
        super(SemanticProjection, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, semantic_features):
        """
        Project semantic features to target dimension.

        Args:
            semantic_features: Tensor of shape (batch_size, 384)

        Returns:
            projected_features: Tensor of shape (batch_size, output_dim)
        """
        return self.projection(semantic_features)
