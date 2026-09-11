"""AASIST and AASIST-L Graph Attention Network architecture for Audio Anti-Spoofing.

Ported from the official Clova AI implementation (Jung et al., ICASSP 2022)
and adapted for self-supervised speech front-ends (WavLM / wav2vec2) following
Tak et al. (Odyssey 2022) and Pan et al. (Interspeech 2024).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphAttentionLayer(nn.Module):
    """Spectro-Temporal Graph Attention Layer."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (#batch, #node, #dim)
        """
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_bn(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x: torch.Tensor) -> torch.Tensor:
        nb_nodes = x.size(1)
        x_4d = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x_4d.transpose(1, 2)
        return x_4d * x_mirror

    def _derive_att_map(self, x: torch.Tensor) -> torch.Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_bn(self, x: torch.Tensor) -> torch.Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x


class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous Stacking Graph Attention Layer (HS-GAL)."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weight22 = Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weight12 = Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weightM = Parameter(torch.FloatTensor(out_dim, 1))

        for w in [self.att_weight11, self.att_weight22, self.att_weight12, self.att_weightM]:
            nn.init.xavier_normal_(w)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, master: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)

        x = self._project(x, att_map)
        x = self._apply_bn(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _derive_att_map_master(self, x: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project_master(self, x: torch.Tensor, master: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _derive_att_map(self, x: torch.Tensor, num_type1: int, num_type2: int) -> torch.Tensor:
        nb_nodes = x.size(1)
        x_4d = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x_4d.transpose(1, 2)
        att_map = torch.tanh(self.att_proj(x_4d * x_mirror))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )
        att_board = att_board / self.temp
        return F.softmax(att_board, dim=-2)

    def _project(self, x: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_bn(self, x: torch.Tensor) -> torch.Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x


class GraphPool(nn.Module):
    """Top-k Graph Pooling module."""

    def __init__(self, k: float, in_dim: int, p: float = 0.3) -> None:
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.drop(h)
        scores = torch.sigmoid(self.proj(z))
        _, n_nodes, n_feat = h.size()
        k_nodes = max(int(n_nodes * self.k), 1)
        _, idx = torch.topk(scores, k_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


class AttentiveMerging(nn.Module):
    """Attentive merging of hierarchical hidden layer embeddings (Pan et al., 2024)."""

    def __init__(self, num_layers: int = 12) -> None:
        super().__init__()
        self.weights = Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Args:
            hidden_states: List of (B, T, D) tensors from selected transformer layers.

        Returns:
            Merged tensor of shape (B, T, D).
        """
        norm_weights = F.softmax(self.weights[: len(hidden_states)], dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # (L, B, T, D)
        weights_exp = norm_weights.view(-1, 1, 1, 1)  # (L, 1, 1, 1)
        return (stacked * weights_exp).sum(dim=0)


class AASISTLHead(nn.Module):
    """AASIST-L Graph Attention Head operating on SSL (WavLM) representations.

    Parameter budget: ~85k parameters.
    """

    def __init__(
        self,
        in_dim: int = 768,
        gat_dims: Tuple[int, int] = (24, 32),
        pool_ratios: Tuple[float, float, float, float] = (0.4, 0.5, 0.7, 0.5),
        temperatures: Tuple[float, float, float, float] = (2.0, 2.0, 100.0, 100.0),
        merge_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.gat_dims = gat_dims

        # Optional attentive merging over several WavLM hidden layers
        # (Pan et al. 2024, [R10] in docs/08). When None the head keeps its
        # original single-layer contract, so every checkpoint trained before
        # this existed loads and scores identically -- `merge` is simply
        # absent from its state_dict and `forward` never looks for it.
        #
        # Motivated by this repo's own measurement, not only the paper:
        # docs/08 §12.6 found layer 5 decisively better on A17/A21/A24/A29
        # and layer 12 better on A19/A20, i.e. no single layer wins across
        # attack families. A learned merge is the obvious response to that.
        self.merge_layers = merge_layers
        self.merge = AttentiveMerging(num_layers=merge_layers) if merge_layers else None

        # SSL feature projection into initial node dimension
        self.proj = nn.Linear(in_dim, 64)
        self.bn_in = nn.BatchNorm1d(64)
        self.selu = nn.SELU(inplace=True)

        # Spectral and temporal graph layers
        self.GAT_layer_S = GraphAttentionLayer(64, gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(64, gat_dims[0], temperature=temperatures[1])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], p=0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], p=0.3)

        # Heterogeneous stacking layers
        self.master1 = Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = Parameter(torch.randn(1, 1, gat_dims[0]))

        self.HtrgGAT_11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], p=0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], p=0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], p=0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], p=0.3)

        self.drop_way = nn.Dropout(0.2)
        self.drop_out = nn.Dropout(0.5)

        # 5 readout components: [T_max, T_avg, S_max, S_avg, master] * gat_dims[1]
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(
        self, feat: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            feat: (B, T, in_dim) SSL feature sequence, or -- when this head
                was built with ``merge_layers`` -- a list of such tensors,
                one per selected WavLM hidden layer, merged attentively
                before the graph stack sees them.

        Returns:
            last_hidden: (B, 5 * gat_dims[1])
            logits: (B, 2) where index 0=spoof, 1=bonafide.
        """
        if isinstance(feat, (list, tuple)):
            if self.merge is None:
                raise ValueError(
                    "AASISTLHead received a list of hidden states but was built "
                    "without merge_layers; construct it with "
                    "AASISTLHead(merge_layers=N) to enable attentive merging"
                )
            if len(feat) != self.merge_layers:
                raise ValueError(
                    f"expected {self.merge_layers} hidden states to merge, got {len(feat)} "
                    "-- train and inference must select the same layer range"
                )
            feat = self.merge(list(feat))
        elif self.merge is not None:
            raise ValueError(
                f"this head was built with merge_layers={self.merge_layers} and needs a "
                "list of that many hidden states, not a single tensor"
            )

        bs, seq_len, _ = feat.size()

        # Project and normalize
        h = self.proj(feat)  # (B, T, 64)
        h = h.transpose(1, 2)  # (B, 64, T)
        h = self.bn_in(h)
        h = self.selu(h).transpose(1, 2)  # (B, T, 64)

        # Temporal chunks for spectral nodes
        chunk_size = max(seq_len // 4, 1)
        s_nodes = h[:, : chunk_size * 4, :].view(bs, 4, -1, 64).mean(dim=2)  # (B, 4, 64)

        # GAT pass
        gat_t = self.GAT_layer_T(h)
        out_t = self.pool_T(gat_t)

        gat_s = self.GAT_layer_S(s_nodes)
        out_s = self.pool_S(gat_s)

        master1 = self.master1.expand(bs, -1, -1)
        master2 = self.master2.expand(bs, -1, -1)

        # Heterogeneous Stack Branch 1
        out_t1, out_s1, master1 = self.HtrgGAT_11(out_t, out_s, master=master1)
        out_s1 = self.pool_hS1(out_s1)
        out_t1 = self.pool_hT1(out_t1)
        out_t_aug, out_s_aug, master_aug = self.HtrgGAT_12(out_t1, out_s1, master=master1)
        out_t1 = out_t1 + out_t_aug
        out_s1 = out_s1 + out_s_aug
        master1 = master1 + master_aug

        # Heterogeneous Stack Branch 2
        out_t2, out_s2, master2 = self.HtrgGAT_21(out_t, out_s, master=master2)
        out_s2 = self.pool_hS2(out_s2)
        out_t2 = self.pool_hT2(out_t2)
        out_t_aug2, out_s_aug2, master_aug2 = self.HtrgGAT_22(out_t2, out_s2, master=master2)
        out_t2 = out_t2 + out_t_aug2
        out_s2 = out_s2 + out_s_aug2
        master2 = master2 + master_aug2

        out_t = torch.max(self.drop_way(out_t1), self.drop_way(out_t2))
        out_s = torch.max(self.drop_way(out_s1), self.drop_way(out_s2))
        master = torch.max(self.drop_way(master1), self.drop_way(master2))

        t_max, _ = torch.max(torch.abs(out_t), dim=1)
        t_avg = torch.mean(out_t, dim=1)
        s_max, _ = torch.max(torch.abs(out_s), dim=1)
        s_avg = torch.mean(out_s, dim=1)

        last_hidden = torch.cat([t_max, t_avg, s_max, s_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop_out(last_hidden)
        logits = self.out_layer(last_hidden)

        return last_hidden, logits
