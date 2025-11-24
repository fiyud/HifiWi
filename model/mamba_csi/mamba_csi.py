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

class HCMamba_CSI_HPE(nn.Module):
    def __init__(self, Tx, Rx_Subc, d_model, N_person, N_kp, D_coord,
                 num_stm_layers=4, num_ltm_layers=2, downsample_rate=4, **mamba_kwargs):
        super().__init__()

        feature_dim = Tx * Rx_Subc
        self.d_model = d_model
        self.downsample_rate = downsample_rate

        self.N_person = N_person
        self.N_kp = N_kp
        self.D_coord = D_coord
        target_dim = N_person * N_kp * D_coord

        self.proj_in = nn.Linear(feature_dim, d_model)
        self.sf_mixer = GSFM(d_model=d_model)

        self.stm_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, **mamba_kwargs) for _ in range(num_stm_layers)
        ])

        self.downsample = nn.Conv1d(d_model, d_model, downsample_rate, downsample_rate, groups=d_model)

        self.ltm_blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, **mamba_kwargs) for _ in range(num_ltm_layers)
        ])

        self.upsample = nn.ConvTranspose1d(d_model, d_model, downsample_rate, downsample_rate, groups=d_model)

        self.fusion_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, target_dim)
        )

    def forward(self, x):
        B, Tx, F, L = x.shape

        x_flat = rearrange(x, 'b tx f l -> b l (tx f)')
        x_emb = self.proj_in(x_flat.reshape(B * L, -1))
        x_sf_mixed = self.sf_mixer(x_emb)
        x_proj = x_sf_mixed.reshape(B, L, self.d_model)
        x_residual = x_proj

        x_stm = x_proj
        for block in self.stm_blocks:
            x_stm = block(x_stm)

        x_down_in = x_stm.permute(0, 2, 1)
        x_down = self.downsample(x_down_in)
        x_ltm_in = x_down.permute(0, 2, 1)
        x_ltm = x_ltm_in
        for block in self.ltm_blocks:
            x_ltm = block(x_ltm)

        x_up_in = x_ltm.permute(0, 2, 1)
        x_ltm_up = self.upsample(x_up_in, output_size=(x_up_in.shape[0], x_up_in.shape[1], L))

        x_ltm_up = x_ltm_up.permute(0, 2, 1)
        x_fused = x_stm + x_ltm_up + x_residual

        last_packet_output = x_fused[:, -1, :]
        x_norm = self.fusion_norm(last_packet_output)

        flat_output = self.fc_out(x_norm)

        output = flat_output.reshape(B, self.N_person, self.N_kp, self.D_coord)

        return output

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
    N_kp = 18
    D_coord = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_input = torch.randn(1, 3, 90*3, 5).to(device)

    model_hpe = HCMamba_CSI_HPE(
        Tx=Tx,
        Rx_Subc=Rx_Subc,
        d_model=d_model,
        N_person=N_person,
        N_kp=N_kp,
        D_coord=D_coord,
        num_stm_layers=16,
        num_ltm_layers=8,
        downsample_rate=4
    ).to(device)

    start_time = time.time()
    output_hpe = model_hpe(test_input)
    end_time = time.time()


    print("--- HCMamba-CSI cho Multi-Person HPE (Sử dụng Mamba thực) ---")
    print(f"Output Shape (Dự đoán Keypoints): {output_hpe.shape}")
    print(f"Thời gian chạy (xấp xỉ): {(end_time - start_time) * 1000:.2f} ms")

    OUT = count_parameters(model_hpe)
    print(OUT)
