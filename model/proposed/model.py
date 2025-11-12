import torch
import torch.nn as nn
import torch.nn.functional as F
from model.proposed.module import GhostConv, FiLMBlock, Gaussian_Position, simam_module

class GhostPose(nn.Module):
    """
    Frequency-Spatial decoupling for WiFi-specific processing
    """
    def __init__(self, num_keypoints, num_coor, subcarrier_num, time_packets=10, 
                 num_person=1, dataset='mmfi-csi', base_channels=64):
        super(GhostPose, self).__init__()
        
        self.num_keypoints = num_keypoints
        self.num_coor = num_coor
        self.subcarrier_num = subcarrier_num
        self.time_packets = time_packets
        self.num_person = num_person
        self.dataset = dataset
        self.base_channels = base_channels
        
        # ==================== Stage 1: Ghost Feature Extraction ====================
        self.stem = nn.Sequential(
            GhostConv(3, base_channels, k=3, s=1),  # 3 antennas -> base_channels
            simam_module(),  # Parameter-free spatial attention
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # Reduce subcarrier dim
        )
        
        # ==================== Stage 2: Frequency-Spatial Decoupling ====================
        # Process frequency (subcarrier) and spatial (time) separately
        self.freq_branch = FrequencyBranch(base_channels, base_channels * 2, 
                                           subcarrier_num // 2)
        self.spatial_branch = SpatialBranch(base_channels, base_channels * 2, 
                                            time_packets)
        
        # ==================== Stage 3: FiLM-conditioned Fusion ====================
        # Dynamic feature modulation based on signal characteristics
        self.film_modulator = FiLMModulator(base_channels * 4, base_channels * 2)
        self.film_block = FiLMBlock()
        
        # ==================== Stage 4: Temporal Modeling ====================
        # Gaussian position encoding for WiFi packet sequences
        self.temporal_encoder = TemporalEncoder(
            base_channels * 2, 
            time_packets,
            num_heads=4
        )
        
        # ==================== Stage 5: Keypoint-aware Refinement ====================
        # Ghost convolution blocks with attention
        self.refinement = nn.ModuleList([
            GhostBottleneck(base_channels * 2, base_channels * 2, use_attention=True)
            for _ in range(2)
        ])
        
        # ==================== Stage 6: Pose Regression ====================
        # Adaptive regression head based on dataset
        self.regression_head = self._build_regression_head(base_channels * 2)
        
        # ==================== Additional Components ====================
        # Signal quality estimation (for FiLM conditioning)
        self.quality_estimator = SignalQualityEstimator(self.base_channels, self.base_channels * 4)
        
        self.fusion_conv = nn.Conv2d(self.base_channels * 4, self.base_channels * 2, 
                                     kernel_size=1, bias=False)
        
    def _build_regression_head(self, in_channels):
        if self.dataset == 'person-in-wifi-3d':
            out_channels = in_channels // 2
            pool_h = self.num_keypoints * self.num_person
            pool_w = 4
            return nn.Sequential(
                GhostConv(in_channels, out_channels, k=3),
                nn.AdaptiveAvgPool2d((pool_h, pool_w)),
                nn.Flatten(),
                nn.Linear(out_channels * pool_h * pool_w, self.num_keypoints * self.num_coor * self.num_person)
            )
        elif self.dataset == 'wipose':
            out_channels = in_channels // 2
            pool_h = self.num_keypoints
            pool_w = 2
            return nn.Sequential(
                GhostConv(in_channels, out_channels, k=3),
                nn.AdaptiveAvgPool2d((pool_h, pool_w)),
                nn.Flatten(),
                nn.Linear(out_channels * pool_h * pool_w, self.num_keypoints * self.num_coor)
            )
        else:  # mmfi-csi
            # Standard 3D pose
            out_channels = in_channels // 2
            pool_h = self.num_keypoints
            pool_w = 3
            return nn.Sequential(
                GhostConv(in_channels, out_channels, k=3),
                nn.AdaptiveAvgPool2d((pool_h, pool_w)),
                nn.Flatten(),
                nn.Linear(out_channels * pool_h * pool_w, self.num_keypoints * self.num_coor)
            )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, subcarrier_num, time_packets) - WiFi CSI tensor
        Returns:
            pose: (B, num_keypoints, num_coor) - Estimated pose
            features: (B, channels) - Extracted features for downstream tasks
        """
        batch_size = x.size(0)
        
        # Stage 1: Ghost Feature Extraction
        x = self.stem(x)  # (B, base_channels, subcarrier_num//2, time_packets)
        
        # Estimate signal quality for FiLM conditioning
        quality_features = self.quality_estimator(x)
        gamma, beta = self.film_modulator(quality_features)
        
        # Stage 2: Frequency-Spatial Decoupling
        freq_features = self.freq_branch(x)  # (B, base_channels*2, reduced_freq, time)
        spatial_features = self.spatial_branch(x)  # (B, base_channels*2, freq, reduced_time)
        
        # Fuse branches
        fused = torch.cat([freq_features, spatial_features], dim=1)  # (B, base_channels*4, ...)
        fused = F.adaptive_avg_pool2d(fused, (x.size(2), x.size(3)))  # Align dimensions
        
        # Reduce channels after fusion
        fused = self.fusion_conv(fused)
        
        # Stage 3: FiLM-conditioned Fusion
        fused = self.film_block(fused, gamma, beta)
        
        # Stage 4: Temporal Modeling
        fused = self.temporal_encoder(fused)
        
        # Stage 5: Keypoint-aware Refinement
        for refine_block in self.refinement:
            fused = refine_block(fused)
        
        # Extract global features
        features = F.adaptive_avg_pool2d(fused, (1, 1)).squeeze(-1).squeeze(-1)
        
        # Stage 6: Pose Regression
        pose = self.regression_head(fused)
        
        # Reshape output based on dataset
        if self.dataset == 'person-in-wifi-3d':
            pose = pose.view(batch_size, self.num_person, self.num_keypoints, self.num_coor)
        else:
            pose = pose.view(batch_size, self.num_keypoints, self.num_coor)
        
        return pose, features


# ==================== Supporting Modules ====================

class FrequencyBranch(nn.Module):
    """Process subcarrier dimension with frequency-aware convolutions"""
    
    def __init__(self, in_channels, out_channels, freq_dim):
        super().__init__()
        self.branch = nn.Sequential(
            GhostConv(in_channels, out_channels, k=3, s=1),
            simam_module(),
            # Frequency-specific: convolve along subcarrier dimension
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), 
                     padding=(2, 0), groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.branch(x)


class SpatialBranch(nn.Module):
    """Process temporal dimension with spatial convolutions"""
    
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.branch = nn.Sequential(
            GhostConv(in_channels, out_channels, k=3, s=1),
            simam_module(),
            # Temporal-specific: convolve along time dimension
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), 
                     padding=(0, 2), groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.branch(x)


class FiLMModulator(nn.Module):
    def __init__(self, quality_dim, target_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(quality_dim, quality_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(quality_dim // 2, target_channels * 2)  # gamma and beta
        )
        self.target_channels = target_channels
    
    def forward(self, quality_features):
        params = self.fc(quality_features)
        gamma = params[:, :self.target_channels]
        beta = params[:, self.target_channels:]
        return gamma, beta


class SignalQualityEstimator(nn.Module):
    """Estimate WiFi signal quality for adaptive processing"""
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Take original CSI input and estimate quality metrics"""
        return self.estimator(x)


class TemporalEncoder(nn.Module):
    """Encode temporal dependencies with Gaussian positional encoding"""
    def __init__(self, channels, time_dim, num_heads=4):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        
        self.pos_encoding = Gaussian_Position(channels, time_dim, K=min(10, time_dim))
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape to (B*H, W, C) for temporal processing
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B * H, W, C)
        
        # Apply Gaussian positional encoding
        x = self.pos_encoding(x)  # (B*H, W, C)
        
        # Temporal self-attention
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ffn(self.norm(x))
        
        # Reshape back to (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        hidden_channels = in_channels // 2
        
        self.conv1 = GhostConv(in_channels, hidden_channels, k=1)
        self.conv2 = GhostConv(hidden_channels, hidden_channels, k=3)
        self.conv3 = GhostConv(hidden_channels, out_channels, k=1)
        
        if use_attention:
            self.attention = simam_module()
        
        self.shortcut = nn.Identity() if in_channels == out_channels else \
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        out = self.conv3(out)
        out = out + identity
        
        return F.relu(out, inplace=True)

# ==================== Weight Initialization ====================

def ghostpose_weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def create_ghostpose(dataset='mmfi-csi', base_channels=64):
    if dataset == 'mmfi-csi':
        model = GhostPose(
            num_keypoints=17,
            num_coor=3,
            subcarrier_num=114,
            time_packets=10,
            num_person=1,
            dataset=dataset,
            base_channels=base_channels
        )
    elif dataset == 'person-in-wifi-3d':
        model = GhostPose(
            num_keypoints=14,
            num_coor=3,
            subcarrier_num=180,
            time_packets=20,
            num_person=1,  # Will be adjusted dynamically
            dataset=dataset,
            base_channels=base_channels
        )
    elif dataset == 'wipose':
        model = GhostPose(
            num_keypoints=18,
            num_coor=2,
            subcarrier_num=90,
            time_packets=5,
            num_person=1,
            dataset=dataset,
            base_channels=base_channels
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    model.apply(ghostpose_weights_init)
    
    return model

if __name__ == '__main__':
    model = create_ghostpose('mmfi-csi', base_channels=64)
    x = torch.randn(4, 3, 114, 10)  # Batch of 4 CSI samples
    pose, features = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Pose output shape: {pose.shape}")  # (4, 17, 3)
    print(f"Feature shape: {features.shape}")  # (4, 128)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")