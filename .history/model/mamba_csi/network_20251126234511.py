import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PairwiseRefine(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        """
        x: B x K x D
        returns: B x K x D (refined)
        """
        B, K, D = x.shape
        out = []
        for i in range(K):
            # concatenate 
            xi = x[:, i:i+1, :].expand(-1, K, -1)  # B x K x D
            pair = torch.cat([xi, x], dim=-1)       # B x K x 2D
            out.append(self.mlp(pair).mean(1, keepdim=True))  # B x 1 x D
        out = torch.cat(out, dim=1)  # B x K x D
        return out + x  # residual

class MultiPersonDecoderPerPerson(nn.Module):
    def __init__(self, emb_dim=256, num_keypoints=17, num_person=2, num_layers=3, nhead=8, coor_num=3, hidden_pairwise=128):
        super().__init__()
        self.num_person = num_person
        self.num_keypoints = num_keypoints
        self.emb_dim = emb_dim

        # queries
        self.person_queries = nn.Parameter(torch.randn(num_person, emb_dim))
        self.keypoint_queries = nn.Parameter(torch.randn(num_keypoints, emb_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # replace HGNN with PairwiseRefine
        self.refine1 = PairwiseRefine(dim=emb_dim, hidden=hidden_pairwise)
        self.refine2 = PairwiseRefine(dim=emb_dim, hidden=hidden_pairwise)

        # final head: embedding -> coord
        self.fc_out = nn.Linear(emb_dim, coor_num)

    def forward(self, memory, H=None):
        """
        memory: (B, T, D)
        H: ignored
        returns: coords (B, P, K, 3)
        """
        B, T, D = memory.shape

        person_q = self.person_queries.unsqueeze(0).repeat(B, 1, 1)  # B x P x D
        key_q = self.keypoint_queries.unsqueeze(0).repeat(B, 1, 1)   # B x K x D

        all_person_outputs = []
        for p in range(self.num_person):
            per_person_query = person_q[:, p:p+1, :]
            tgt = per_person_query + key_q  # B x K x D

            out_emb = self.transformer_decoder(tgt=tgt, memory=memory)  # B x K x D

            # PairwiseRefine
            out_ref1 = self.refine1(out_emb)
            out_ref2 = self.refine2(out_ref1)

            coords = self.fc_out(out_ref2)  # B x K x 3
            all_person_outputs.append(coords.unsqueeze(1))  # add person dim

        all_person_outputs = torch.cat(all_person_outputs, dim=1)  # B x P x K x 3
        return all_person_outputs

# CSI Encoder
class CSI_Encoder(nn.Module):
    def __init__(self, in_channels=3, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, emb_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))  # compress frequency dimension
        self.norm = nn.LayerNorm(emb_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # B x emb_dim x H x 1
        x = x.squeeze(-1) # B x emb_dim x H
        x = rearrange(x, 'b d t -> b t d')  # B x T x D
        x = self.norm(x)
        return x  # memory


class CSI_HPE_withPairwiseRefine(nn.Module):
    def __init__(self, emb_dim=128, num_keypoints=17, num_person=1, coor_num=3):
        super().__init__()
        self.encoder = CSI_Encoder(in_channels=3, emb_dim=emb_dim)
        self.decoder = MultiPersonDecoderPerPerson(
            emb_dim=emb_dim,
            num_keypoints=num_keypoints,
            num_person=num_person,
            coor_num=coor_num
        )

    def forward(self, x, H=None):
        memory = self.encoder(x)
        coords = self.decoder(memory, H)
        return coords

    
def init_csi_hpe_weights_pairwise(m):
    """
    Weight initialization for CSI_HPE_withPairwiseRefine model.
    """
    # Conv2d
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    # Linear
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    # LayerNorm / BatchNorm
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
    # PairwiseRefine
    elif isinstance(m, PairwiseRefine):
        for layer in m.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    # TransformerDecoderLayer (Linear + LayerNorm)
    elif isinstance(m, nn.TransformerDecoderLayer):
        for name, param in m.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
