import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DualGraphCollaboration(nn.Module):
    """
    Reliability-aware Dual-Graph Collaboration (RDGC).

    RDGC estimates how much each user should trust the interaction view and the
    denoised social view, then fuses both views with reliability-aware gates.
    """

    def __init__(self, n_users, n_items, hidden_dim, temperature=0.1,
                 interaction_reliability=None, social_struct_reliability=None,
                 social_shift_penalty=1.0, use_cross_view=True,
                 reliability_temperature=1.0, residual_weight=0.2,
                 cross_weight=0.1):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            hidden_dim: Dimension of hidden embeddings
            temperature: Kept for backwards compatibility; RDGC uses a stable
                reliability softmax temperature internally
            interaction_reliability: User-level interaction reliability prior
            social_struct_reliability: User-level social-structure reliability prior
            social_shift_penalty: Penalty strength for diffusion-induced shifts
            use_cross_view: Whether to add lightweight cross-view collaboration
            reliability_temperature: Temperature for reliability softmax gates
            residual_weight: Weight for preserving original graph representations
            cross_weight: Strength of cross-view representation
        """
        super(DualGraphCollaboration, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.reliability_temperature = max(reliability_temperature, 1e-6)
        self.social_shift_penalty = social_shift_penalty
        self.use_cross_view = use_cross_view
        self.residual_weight = min(max(residual_weight, 0.0), 1.0)
        self.cross_weight = cross_weight

        interaction_prior = self._make_reliability_prior(interaction_reliability, n_users)
        social_prior = self._make_reliability_prior(social_struct_reliability, n_users)
        self.register_buffer("interaction_prior", interaction_prior)
        self.register_buffer("social_prior", social_prior)

        self.cross_view_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Optional distribution alignment layers (for KL divergence alignment)
        self.social_mean = nn.Linear(hidden_dim, hidden_dim)
        self.social_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.interaction_mean = nn.Linear(hidden_dim, hidden_dim)
        self.interaction_logvar = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _make_reliability_prior(self, values, n_users):
        if values is None:
            return torch.full((n_users,), 0.5, dtype=torch.float32)

        if isinstance(values, torch.Tensor):
            prior = values.detach().float().view(-1).cpu()
        else:
            prior = torch.tensor(values, dtype=torch.float32).view(-1)

        if prior.numel() != n_users:
            raise ValueError(f"Reliability prior length {prior.numel()} does not match n_users={n_users}")

        min_val = prior.min()
        max_val = prior.max()
        if torch.isfinite(min_val) and torch.isfinite(max_val) and max_val > min_val:
            prior = (prior - min_val) / (max_val - min_val)
        else:
            prior = torch.full_like(prior, 0.5)

        return prior.clamp(0.0, 1.0)

    def _normalize_shift(self, shift):
        shift = shift.detach()
        min_val = shift.min()
        max_val = shift.max()
        if max_val > min_val:
            return (shift - min_val) / (max_val - min_val + 1e-8)
        return torch.zeros_like(shift)

    def compute_reliability_weights(self, user_social_emb, denoised_social_emb, user_indices):
        """
        Compute RDGC reliability weights [w_i, w_s] for a batch of users.
        """
        if user_indices is None:
            if user_social_emb.size(0) != self.n_users:
                raise ValueError("user_indices is required when passing a mini-batch to RDGC")
            user_indices = torch.arange(self.n_users, device=user_social_emb.device)

        user_indices = user_indices.long().to(user_social_emb.device)
        interaction_reliability = self.interaction_prior[user_indices].view(-1, 1)
        social_struct_reliability = self.social_prior[user_indices].view(-1, 1)

        social_shift = torch.norm(user_social_emb - denoised_social_emb, p=2, dim=-1, keepdim=True)
        social_shift = self._normalize_shift(social_shift)
        social_reliability = social_struct_reliability * torch.exp(
            -self.social_shift_penalty * social_shift
        )

        logits = torch.cat([interaction_reliability, social_reliability], dim=-1)
        weights = F.softmax(logits / self.reliability_temperature, dim=-1)

        stats = {
            "interaction_reliability": interaction_reliability,
            "social_reliability": social_reliability,
            "social_shift": social_shift,
            "gate_weights": weights
        }
        return weights, stats

    def compute_distribution_params(self, social_emb, interaction_emb):
        """
        Compute mean and log-variance for distribution alignment.

        Args:
            social_emb: Social embeddings
            interaction_emb: Interaction embeddings

        Returns:
            social_mean, social_logvar, interaction_mean, interaction_logvar
        """
        social_mean = self.social_mean(social_emb)
        social_logvar = self.social_logvar(social_emb)

        interaction_mean = self.interaction_mean(interaction_emb)
        interaction_logvar = self.interaction_logvar(interaction_emb)

        return social_mean, social_logvar, interaction_mean, interaction_logvar

    def compute_kl_divergence_loss(self, social_emb, interaction_emb):
        """
        Compute KL divergence loss to align user embedding distributions
        across the two domains (social and interaction).

        Args:
            social_emb: Social embeddings (batch_size, hidden_dim)
            interaction_emb: Interaction embeddings (batch_size, hidden_dim)

        Returns:
            kl_loss: KL divergence loss
        """
        social_mean, social_logvar, interaction_mean, interaction_logvar = \
            self.compute_distribution_params(social_emb, interaction_emb)

        # KL divergence: KL(social || interaction)
        kl_loss = 0.5 * torch.mean(
            interaction_logvar - social_logvar
            - 1.0
            + torch.exp(social_logvar - interaction_logvar)
            + (social_mean - interaction_mean) ** 2 * torch.exp(-interaction_logvar)
        )

        return kl_loss

    def forward(self, user_social_emb, user_interaction_emb, item_emb,
                denoised_social_emb=None, user_indices=None):
        """
        Forward pass for reliability-aware dual-graph collaboration.

        Args:
            user_social_emb: Original user social embeddings before denoising
            user_interaction_emb: User interaction embeddings
            item_emb: Item embeddings, kept for backwards-compatible signature
            denoised_social_emb: Denoised user social embeddings
            user_indices: Batch user IDs for reliability-prior lookup

        Returns:
            guided_interaction_emb: Reliability-weighted interaction contribution
            guided_social_emb: Reliability-weighted social contribution
            kl_loss: KL divergence alignment loss
        """
        if denoised_social_emb is None:
            denoised_social_emb = user_social_emb

        gate_weights, _ = self.compute_reliability_weights(
            user_social_emb, denoised_social_emb, user_indices
        )
        interaction_weight = gate_weights[:, 0:1]
        social_weight = gate_weights[:, 1:2]

        rdgc_interaction = interaction_weight * user_interaction_emb
        rdgc_social = social_weight * denoised_social_emb
        rdgc_cross = torch.zeros_like(user_interaction_emb)

        if self.use_cross_view:
            cross_input = torch.cat([
                user_interaction_emb,
                denoised_social_emb,
                user_interaction_emb * denoised_social_emb
            ], dim=-1)
            cross_emb = self.cross_view_mlp(cross_input)
            cross_weight = interaction_weight * social_weight
            rdgc_cross = self.cross_weight * cross_weight * cross_emb

        residual_scale = self.residual_weight
        rdgc_scale = 1.0 - residual_scale
        guided_interaction_emb = residual_scale * user_interaction_emb + \
            rdgc_scale * (rdgc_interaction + 0.5 * rdgc_cross)
        guided_social_emb = residual_scale * denoised_social_emb + \
            rdgc_scale * (rdgc_social + 0.5 * rdgc_cross)

        kl_loss = self.compute_kl_divergence_loss(denoised_social_emb, user_interaction_emb)

        return guided_interaction_emb, guided_social_emb, kl_loss


class MultimodalFeatureFusion(nn.Module):
    """
    Multimodal Feature Fusion using attention mechanisms to fuse
    semantic, social, and interaction features.

    The network dynamically learns to assign weights to each modality:
    - For highly social users, the social vector gets higher weight
    - For niche projects with highly descriptive text, the semantic vector takes precedence
    """

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        """
        Args:
            hidden_dim: Dimension of all feature vectors
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultimodalFeatureFusion, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        # Cross-modal attention layers
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Modality-specific importance scoring
        self.semantic_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.social_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.interaction_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Residual gate
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_modality_weights(self, semantic_feat, social_feat, interaction_feat):
        """
        Compute dynamic weights for each modality based on their content.

        Args:
            semantic_feat: Semantic features (batch_size, hidden_dim)
            social_feat: Social features (batch_size, hidden_dim)
            interaction_feat: Interaction features (batch_size, hidden_dim)

        Returns:
            weights: Tensor of shape (batch_size, 3) containing modality weights
        """
        semantic_score = self.semantic_importance(semantic_feat)  # (batch_size, 1)
        social_score = self.social_importance(social_feat)  # (batch_size, 1)
        interaction_score = self.interaction_importance(interaction_feat)  # (batch_size, 1)

        # Concatenate and apply softmax
        scores = torch.cat([semantic_score, social_score, interaction_score], dim=-1)  # (batch_size, 3)
        weights = F.softmax(scores, dim=-1)

        return weights

    def cross_modal_attention(self, query_feat, key_feat, value_feat):
        """
        Apply cross-modal multi-head attention.

        Args:
            query_feat: Query features (batch_size, hidden_dim)
            key_feat: Key features (batch_size, hidden_dim)
            value_feat: Value features (batch_size, hidden_dim)

        Returns:
            attended_feat: Attended features (batch_size, hidden_dim)
        """
        batch_size = query_feat.size(0)

        # Project to multi-head space
        Q = self.query_proj(query_feat).view(batch_size, self.n_heads, self.head_dim)
        K = self.key_proj(key_feat).view(batch_size, self.n_heads, self.head_dim)
        V = self.value_proj(value_feat).view(batch_size, self.n_heads, self.head_dim)

        # Compute attention scores
        attention_scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended = torch.bmm(attention_weights, V)

        # Reshape back
        attended_feat = attended.view(batch_size, self.hidden_dim)

        return attended_feat

    def forward(self, semantic_feat, social_feat, interaction_feat, base_feat=None):
        """
        Fuse semantic, social, and interaction features.

        Args:
            semantic_feat: Projected semantic features (batch_size, hidden_dim)
            social_feat: Social features from graph (batch_size, hidden_dim)
            interaction_feat: Interaction features from graph (batch_size, hidden_dim)
            base_feat: Optional base feature for residual connection

        Returns:
            fused_feat: Fused feature vector (batch_size, hidden_dim)
            modality_weights: Dynamic modality weights for analysis
        """
        # Compute dynamic modality weights
        modality_weights = self.compute_modality_weights(semantic_feat, social_feat, interaction_feat)

        # Weighted combination
        weighted_semantic = semantic_feat * modality_weights[:, 0:1]
        weighted_social = social_feat * modality_weights[:, 1:2]
        weighted_interaction = interaction_feat * modality_weights[:, 2:3]

        # Concatenate for fusion
        concat_feat = torch.cat([weighted_semantic, weighted_social, weighted_interaction], dim=-1)

        # Project to fused representation
        fused_feat = self.output_proj(concat_feat)

        # Optional residual connection
        if base_feat is not None:
            gate_input = torch.cat([fused_feat, base_feat], dim=-1)
            gate = self.residual_gate(gate_input)
            fused_feat = gate * fused_feat + (1 - gate) * base_feat

        return fused_feat, modality_weights


class ContrastiveLearning(nn.Module):
    """
    Contrastive Learning module for two tasks:

    Task A: Diffusion-Aware Contrastive Learning (Intra-modal)
        - Pre-diffused noisy structural vector and post-diffused clean structural
          vector are treated as positive pairs.

    Task B: Cross-modal alignment (Inter-modal)
        - Direct: Item A's structural vector and Item A's text vector (positive pairs)
        - Adversarial: Item A's structural vector and Item B's text vector (negative pairs)
        - Uses InfoNCE loss
    """

    def __init__(self, hidden_dim, temperature=0.07):
        """
        Args:
            hidden_dim: Dimension of feature vectors
            temperature: Temperature parameter for InfoNCE loss
        """
        super(ContrastiveLearning, self).__init__()

        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Projection heads for contrastive learning
        self.structural_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.semantic_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_info_nce_loss(self, anchor, positive, negatives=None):
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings (batch_size, hidden_dim)
            positive: Positive embeddings (batch_size, hidden_dim)
            negatives: Optional negative embeddings (batch_size, n_neg, hidden_dim)
                      If None, uses in-batch negatives

        Returns:
            loss: InfoNCE loss
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / self.temperature

        if negatives is not None:
            # Use provided negatives
            negatives = F.normalize(negatives, dim=-1)
            neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(-2, -1)).squeeze(1) / self.temperature
            logits = torch.cat([pos_sim, neg_sim], dim=-1)
        else:
            # Use in-batch negatives
            all_sim = torch.mm(anchor, positive.t()) / self.temperature
            # Positive pairs are on the diagonal
            logits = all_sim

        # Labels: positive is at index 0 (for explicit negatives) or diagonal (for in-batch)
        if negatives is not None:
            labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # In-batch contrastive loss
            labels = torch.arange(anchor.size(0), device=anchor.device)
            loss = F.cross_entropy(logits, labels)

        return loss

    def diffusion_aware_contrastive_loss(self, noisy_struct, clean_struct):
        """
        Task A: Diffusion-Aware Contrastive Learning (Intra-modal)

        Treats pre-diffused noisy structural vector and post-diffused clean
        structural vector as positive pairs.

        Args:
            noisy_struct: Noisy structural embeddings before denoising (batch_size, hidden_dim)
            clean_struct: Clean structural embeddings after denoising (batch_size, hidden_dim)

        Returns:
            loss: Diffusion-aware contrastive loss
        """
        # Project both embeddings
        noisy_proj = self.structural_projector(noisy_struct)
        clean_proj = self.structural_projector(clean_struct)

        # Compute InfoNCE loss with in-batch negatives
        loss = self.compute_info_nce_loss(noisy_proj, clean_proj)

        return loss

    def cross_modal_contrastive_loss(self, structural_emb, semantic_emb, neg_semantic_emb=None):
        """
        Task B: Cross-modal alignment (Inter-modal)

        Treats a node's structural representation and its semantic representation
        as two different "views" of the same entity.

        Args:
            structural_emb: Structural embeddings (batch_size, hidden_dim)
            semantic_emb: Semantic embeddings for same entities (batch_size, hidden_dim)
            neg_semantic_emb: Optional negative semantic embeddings (batch_size, n_neg, hidden_dim)

        Returns:
            loss: Cross-modal contrastive loss
        """
        # Project embeddings
        struct_proj = self.structural_projector(structural_emb)
        semantic_proj = self.semantic_projector(semantic_emb)

        if neg_semantic_emb is not None:
            neg_semantic_proj = self.semantic_projector(neg_semantic_emb.view(-1, self.hidden_dim))
            neg_semantic_proj = neg_semantic_proj.view(structural_emb.size(0), -1, self.hidden_dim)
            loss = self.compute_info_nce_loss(struct_proj, semantic_proj, neg_semantic_proj)
        else:
            # Use in-batch negatives
            loss = self.compute_info_nce_loss(struct_proj, semantic_proj)

        return loss

    def forward(self, noisy_struct, clean_struct, structural_emb, semantic_emb,
                neg_semantic_emb=None, alpha=0.5, beta=0.5):
        """
        Compute combined contrastive learning loss.

        Args:
            noisy_struct: Noisy structural embeddings
            clean_struct: Clean structural embeddings
            structural_emb: Structural embeddings for cross-modal
            semantic_emb: Semantic embeddings
            neg_semantic_emb: Optional negative semantic embeddings
            alpha: Weight for diffusion-aware loss
            beta: Weight for cross-modal loss

        Returns:
            total_loss: Combined contrastive loss
            diffusion_loss: Diffusion-aware contrastive loss
            cross_modal_loss: Cross-modal contrastive loss
        """
        # Task A: Diffusion-Aware Contrastive Learning
        diffusion_loss = self.diffusion_aware_contrastive_loss(noisy_struct, clean_struct)

        # Task B: Cross-modal alignment
        cross_modal_loss = self.cross_modal_contrastive_loss(
            structural_emb, semantic_emb, neg_semantic_emb
        )

        # Combined loss
        total_loss = alpha * diffusion_loss + beta * cross_modal_loss

        return total_loss, diffusion_loss, cross_modal_loss
