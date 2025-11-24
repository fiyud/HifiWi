import torch
import torch.nn as nn
from einops import rearrange
import time
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.mamba = Mamba(d_model=d_model, **kwargs)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + identity

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * x / (rms + self.eps)


class BFMBlock(nn.Module):
    def __init__(self, d_model, **mamba_kwargs):
        super().__init__()
        self.norm_in = RMSNorm(d_model)
        self.norm_out = RMSNorm(d_model)

        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)
        self.mamba_bwd = Mamba(d_model=d_model, **mamba_kwargs)

        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        h = self.norm_in(x)

        fwd = self.mamba_fwd(h)

        bwd = torch.flip(h, dims=[1])
        bwd = self.mamba_bwd(bwd)
        bwd = torch.flip(bwd, dims=[1])

        fused_cat = torch.cat([fwd, bwd], dim=-1)
        g = torch.sigmoid(self.gate(fused_cat))
        fused = g * fwd + (1 - g) * bwd

        return self.norm_out(fused + x)


class GSFM(nn.Module):
    def __init__(self, d_model, expansion_ratio=2):
        super().__init__()
        D_inner = int(d_model * expansion_ratio)
        self.proj_up = nn.Linear(d_model, D_inner * 2)
        self.split = D_inner
        self.proj_down = nn.Linear(D_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_up = self.proj_up(x)
        x_v, x_g = x_up.split(self.split, dim=-1)
        x_gated = x_v * torch.sigmoid(x_g)
        x_out = self.proj_down(x_gated)
        return self.norm(x_out + x)


class SkeletonPriorEmbedding(nn.Module):
    """
    Novel Module: Embeds anatomical skeleton structure as prior knowledge.
    Uses graph structure to guide feature learning with joint-level embeddings.
    
    STABILIZED VERSION with better initialization and normalization.
    """
    def __init__(self, num_joints, embed_dim, dataset='mmfi-csi'):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        
        # Better initialization - smaller scale to prevent instability
        self.joint_embeddings = nn.Parameter(torch.randn(num_joints, embed_dim) * 0.02)
        
        if dataset == 'mmfi-csi':
            bone_pairs = [
                [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                [12, 13], [8, 14], [14, 15], [15, 16]
            ]
        elif dataset == 'person-in-wifi-3d':
            bone_pairs = [
                [0, 1], [0, 2], [2, 5], [3, 0], [4, 2], [5, 7],
                [6, 3], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7],
                [12, 9], [13, 11]
            ]
        elif dataset == 'wipose':
            bone_pairs = [
                [17, 15], [0, 15], [0, 14], [14, 16], [1, 2], [2, 3],
                [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
                [9, 10], [1, 11], [11, 12], [12, 13]
            ]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        self.register_buffer('bone_pairs', torch.tensor(bone_pairs))
        
        # Add normalization layers to stabilize training
        self.bone_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),  # ADD NORMALIZATION
            nn.ReLU(),
            nn.Dropout(0.1),          # ADD DROPOUT for regularization
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)   # ADD NORMALIZATION
        )
        
        # Initialize MLP weights with Xavier initialization (smaller scale)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with smaller scale for stability"""
        for m in self.bone_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self):
        # Normalize embeddings to prevent explosion
        # L2 normalize then rescale by sqrt(embed_dim)
        joint_emb = nn.functional.normalize(self.joint_embeddings, p=2, dim=-1)
        joint_emb = joint_emb * (self.embed_dim ** 0.5)
        
        # Concatenate bone endpoint embeddings
        bone_emb_pairs = torch.cat([
            joint_emb[self.bone_pairs[:, 0]],
            joint_emb[self.bone_pairs[:, 1]]
        ], dim=-1)
        
        # Process through MLP with normalization
        bone_features = self.bone_mlp(bone_emb_pairs)
        
        return joint_emb, bone_features


class SkeletonGuidedFusion(nn.Module):
    """
    Fuses temporal features with skeleton prior knowledge.
    Uses cross-attention mechanism between temporal features and skeleton embeddings.
    
    STABILIZED VERSION with gradient clipping and better numerical stability.
    """
    def __init__(self, d_model, num_joints):
        super().__init__()
        self.d_model = d_model
        self.num_joints = num_joints
        
        # Cross-attention between temporal features and skeleton prior
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Initialize with smaller weights for stability
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.1)
        
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(0.1)
        
        # Bone feature integration
        self.bone_gate = nn.Linear(d_model * 2, d_model)
        nn.init.xavier_uniform_(self.bone_gate.weight, gain=0.1)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, temporal_feat, joint_emb, bone_feat):
        """
        Args:
            temporal_feat: (B, d_model) - fused temporal features
            joint_emb: (num_joints, d_model) - joint embeddings
            bone_feat: (num_bones, d_model) - bone features
        Returns:
            enhanced_feat: (B, num_joints, d_model) - skeleton-aware features
        """
        B = temporal_feat.shape[0]
        
        # Add small epsilon to prevent numerical issues
        temporal_feat = temporal_feat + 1e-8
        
        # Expand temporal features for each joint
        temporal_expanded = temporal_feat.unsqueeze(1).expand(B, self.num_joints, self.d_model)
        
        # Cross-attention: temporal queries attend to joint embeddings
        Q = self.query_proj(temporal_expanded)  # (B, num_joints, d_model)
        K = self.key_proj(joint_emb).unsqueeze(0).expand(B, -1, -1)  # (B, num_joints, d_model)
        V = self.value_proj(joint_emb).unsqueeze(0).expand(B, -1, -1)  # (B, num_joints, d_model)
        
        # Stabilized attention computation
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_joints, num_joints)
        
        # Clamp to prevent overflow in softmax
        attn = torch.clamp(attn, min=-10, max=10)
        attn = torch.softmax(attn, dim=-1)
        
        # Add small value to prevent all-zero attention
        attn = attn + 1e-8
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        attn = self.dropout(attn)
        
        # Apply attention
        attended = torch.matmul(attn, V)  # (B, num_joints, d_model)
        
        # Integrate bone features (average bone features for connected joints)
        bone_feat_mean = bone_feat.mean(dim=0, keepdim=True)  # (1, d_model)
        bone_feat_expanded = bone_feat_mean.unsqueeze(0).expand(B, self.num_joints, -1)
        
        # Gated fusion with bone features
        combined = torch.cat([attended, bone_feat_expanded], dim=-1)
        gate = torch.sigmoid(self.bone_gate(combined))
        
        # Clamp gate to reasonable range (prevent extreme values)
        gate = torch.clamp(gate, min=0.01, max=0.99)
        
        enhanced = gate * attended + (1 - gate) * temporal_expanded
        
        return self.norm(enhanced + temporal_expanded)


class HCMamba_CSI_HPE(nn.Module):
    def __init__(self, Tx, Rx_Subc, d_model, N_person, N_kp, D_coord,
                 num_stm_layers=4, num_ltm_layers=2, downsample_rate=4, 
                 dataset='mmfi-csi', use_skeleton_prior=True, **mamba_kwargs):
        super().__init__()

        feature_dim = Tx * Rx_Subc
        self.d_model = d_model
        self.downsample_rate = downsample_rate

        self.N_person = N_person
        self.N_kp = N_kp
        self.D_coord = D_coord
        self.use_skeleton_prior = use_skeleton_prior
        target_dim = N_person * N_kp * D_coord

        # --- Embedding ---
        self.proj_in = nn.Linear(feature_dim, d_model)
        self.sf_mixer = GSFM(d_model=d_model)

        # --- Skeleton Prior Module (STABILIZED) ---
        if self.use_skeleton_prior:
            self.skeleton_prior = SkeletonPriorEmbedding(
                num_joints=N_kp,
                embed_dim=d_model,
                dataset=dataset
            )
            self.skeleton_fusion = SkeletonGuidedFusion(
                d_model=d_model,
                num_joints=N_kp
            )

        # --- Short-Term Blocks ---
        self.stm_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, **mamba_kwargs) for _ in range(num_stm_layers)
        ])

        # --- Downsample: depthwise ---
        self.downsample = nn.Conv1d(
            d_model, d_model,
            kernel_size=downsample_rate,
            stride=downsample_rate,
            groups=d_model
        )

        # --- Long-Term Blocks ---
        self.ltm_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, **mamba_kwargs) for _ in range(num_ltm_layers)
        ])

        # --- UPSAMPLE: Interpolate + Conv1d
        self.upsample_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        # --- Fusion + Output ---
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # Modified output head for skeleton-guided prediction
        if self.use_skeleton_prior:
            self.fc_out = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),  # ADD NORMALIZATION
                nn.GELU(),
                nn.Dropout(0.1),              # ADD DROPOUT
                nn.Linear(d_model // 2, D_coord)  # Output per joint
            )
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, target_dim)
            )

    def upsample(self, x, target_L):
        """
        Interpolate + Conv1d
        x: (B, d_model, L_down)
        """
        x = torch.nn.functional.interpolate(
            x, size=target_L, mode='linear', align_corners=False
        )
        return self.upsample_conv(x)

    def forward(self, x):
        B, Tx, F, L = x.shape

        # ---- Flatten ----
        x_flat = rearrange(x, 'b tx f l -> b l (tx f)')
        x_emb = self.proj_in(x_flat.reshape(B * L, -1))
        x_emb = x_emb.reshape(B, L, self.d_model)
        x_emb = self.sf_mixer(x_emb)

        x_residual = x_emb

        # ---- STM ----
        x_stm = x_emb
        for block in self.stm_blocks:
            x_stm = block(x_stm)

        # ---- Downsample ----
        x_down = self.downsample(x_stm.permute(0, 2, 1))
        x_ltm = x_down.permute(0, 2, 1)

        # ---- LTM ----
        for block in self.ltm_blocks:
            x_ltm = block(x_ltm)

        # ---- UPSAMPLE ----
        x_up = self.upsample(x_ltm.permute(0, 2, 1), L)
        x_up = x_up.permute(0, 2, 1)

        # ---- Fusion ----
        x_fused = x_stm + x_up + x_residual

        # ---- Only last frame ----
        last = x_fused[:, -1, :]
        last = self.fusion_norm(last)

        # ---- Skeleton-Guided Prediction ----
        if self.use_skeleton_prior:
            # Get skeleton embeddings (normalized and stabilized)
            joint_emb, bone_feat = self.skeleton_prior()
            
            # Fuse temporal features with skeleton prior
            skeleton_guided_feat = self.skeleton_fusion(last, joint_emb, bone_feat)
            # skeleton_guided_feat: (B, N_kp, d_model)
            
            # Predict coordinates for each joint
            out = self.fc_out(skeleton_guided_feat)  # (B, N_kp, D_coord)
            
        else:
            # Original prediction
            out = self.fc_out(last)  # (B, target_dim)
            out = out.reshape(B, self.N_person, self.N_kp, self.D_coord)
            
            # Squeeze if single person to match expected output shape
            if self.N_person == 1:
                out = out.squeeze(1)  # (B, N_kp, D_coord)

        return out


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total:,}")
    return total


if __name__ == "__main__":
    B = 16
    L = 5
    Tx = 3
    Rx_Subc = 90 * 3
    d_model = 128

    N_person = 1
    N_kp = 18  # Changed to 18 for wipose dataset
    D_coord = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = torch.randn(1, 3, 90*3, 5).to(device)

    # Test with skeleton prior (STABILIZED)
    print("=" * 70)
    print("Testing WITH Skeleton Prior (STABILIZED)")
    print("=" * 70)
    model_hpe_with_prior = HCMamba_CSI_HPE(
        Tx=Tx,
        Rx_Subc=Rx_Subc,
        d_model=d_model,
        N_person=N_person,
        N_kp=N_kp,
        D_coord=D_coord,
        num_stm_layers=16,
        num_ltm_layers=8,
        downsample_rate=4,
        dataset='wipose',
        use_skeleton_prior=True
    ).to(device)

    start_time = time.time()
    output_hpe_prior = model_hpe_with_prior(test_input)
    end_time = time.time()

    print(f"Output Shape: {output_hpe_prior.shape}")
    print(f"Expected Shape: (1, 18, 3)")
    print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Check for NaN or Inf
    if torch.isnan(output_hpe_prior).any():
        print("WARNING: NaN detected in output!")
    if torch.isinf(output_hpe_prior).any():
        print("WARNING: Inf detected in output!")
    
    print(f"Output stats - min: {output_hpe_prior.min().item():.4f}, "
          f"max: {output_hpe_prior.max().item():.4f}, "
          f"mean: {output_hpe_prior.mean().item():.4f}")
    
    count_parameters(model_hpe_with_prior)

    print("\n" + "=" * 70)
    print("Testing WITHOUT Skeleton Prior (Original)")
    print("=" * 70)
    
    # Test without skeleton prior
    model_hpe_no_prior = HCMamba_CSI_HPE(
        Tx=Tx,
        Rx_Subc=Rx_Subc,
        d_model=d_model,
        N_person=N_person,
        N_kp=N_kp,
        D_coord=D_coord,
        num_stm_layers=16,
        num_ltm_layers=8,
        downsample_rate=4,
        use_skeleton_prior=False
    ).to(device)

    start_time = time.time()
    output_hpe_no_prior = model_hpe_no_prior(test_input)
    end_time = time.time()

    print(f"Output Shape: {output_hpe_no_prior.shape}")
    print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
    count_parameters(model_hpe_no_prior)