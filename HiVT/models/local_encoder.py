# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from datasets.av2_constants import NUM_LANE_TYPES
from datasets.av2_constants import NUM_MARK_TYPES
from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import DistanceDropEdge
from utils import TemporalData
from utils import init_weights
from utils import rotate_2d
from utils import rotate_2d_pair


class LocalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False,
                 num_actor_types: int = 0,
                 map_mode: str = 'av1') -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel
        self.map_mode = map_mode

        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel)
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        self.actor_type_embed = nn.Embedding(num_actor_types, embed_dim) if num_actor_types > 0 else None
        self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    map_mode=map_mode,
                                    num_lane_types=NUM_LANE_TYPES,
                                    num_mark_types=NUM_MARK_TYPES)

    def _build_parallel_aa_graph(self, data: TemporalData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized temporal AA graph for parallel=True (no per-t subgraph / Batch.from_data_list).

        Equivalent to: for each t, subgraph(valid_t) → distance drop → Batch offset by t*N.
        Returns (batch_x [T*N, F], batch_edge_index [2, K], batch_edge_attr [K, 2]).
        """
        t_hist = self.historical_steps
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        row, col = edge_index[0], edge_index[1]
        valid = ~data['padding_mask'][:, :t_hist]  # [N, T]
        edge_valid = valid[row] & valid[col]  # [E, T]
        pos_diff = data['positions'][row, :t_hist] - data['positions'][col, :t_hist]  # [E, T, 2]
        keep = edge_valid
        max_distance_sq = self.drop_edge.max_distance_sq
        if max_distance_sq is not None:
            keep = keep & ((pos_diff * pos_diff).sum(dim=-1) < max_distance_sq)
        e_idx, t_idx = keep.nonzero(as_tuple=True)
        offset = t_idx * num_nodes
        batch_edge_index = torch.stack([row[e_idx] + offset, col[e_idx] + offset], dim=0)
        batch_edge_attr = pos_diff[e_idx, t_idx]
        batch_x = data.x[:, :t_hist].transpose(0, 1).reshape(t_hist * num_nodes, -1)
        return batch_x, batch_edge_index, batch_edge_attr

    def forward(self, data: TemporalData) -> torch.Tensor:
        if self.parallel:
            batch_x, batch_edge_index, batch_edge_attr = self._build_parallel_aa_graph(data)
            out = self.aa_encoder(x=batch_x, t=None, edge_index=batch_edge_index, edge_attr=batch_edge_attr,
                                  bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
            out = out.reshape(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        else:
            for t in range(self.historical_steps):
                data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
                data[f'edge_attr_{t}'] = \
                    data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
                out[t] = self.aa_encoder(x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                         bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])
            out = torch.stack(out)  # [T, N, D]
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
        if self.actor_type_embed is not None:
            out = out + self.actor_type_embed(data['actor_type'])
        edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        if self.map_mode == 'av2':
            out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                                  is_intersections=data['is_intersections'], lane_types=data['lane_types'],
                                  left_mark_types=data['left_mark_types'],
                                  right_mark_types=data['right_mark_types'], rotate_mat=data['rotate_mat'])
        else:
            out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                                  is_intersections=data['is_intersections'],
                                  turn_directions=data['turn_directions'],
                                  traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat'])
        return out


class AAEncoder(MessagePassing):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 **kwargs) -> None:
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                t: Optional[int],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        if self.parallel:
            x_steps = x.reshape(self.historical_steps, x.shape[0] // self.historical_steps, -1)
            if rotate_mat is None:
                center_embed = self.center_embed(x_steps)
            else:
                center_embed = self.center_embed(
                    rotate_2d(x_steps, rotate_mat.expand(self.historical_steps, *rotate_mat.shape)))
            center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                       self.bos_token.unsqueeze(-2),
                                       center_embed).reshape(x.shape[0], -1)
        else:
            if rotate_mat is None:
                center_embed = self.center_embed(x)
            else:
                center_embed = self.center_embed(rotate_2d(x, rotate_mat))
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat,
                                                      size)
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                # Nodes are laid out [t0..tT); rotate_mat is per-actor and shared across time.
                center_rotate_mat = rotate_mat[edge_index[1] % rotate_mat.size(0)]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            x_j_r, edge_attr_r = rotate_2d_pair(x_j, edge_attr, center_rotate_mat)
            nbr_embed = self.nbr_embed([x_j_r, edge_attr_r])
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size) -> torch.Tensor:
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, size=size))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


class ALEncoder(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 map_mode: str = 'av1',
                 num_lane_types: int = 4,
                 num_mark_types: int = 15,
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.map_mode = map_mode

        self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        if map_mode == 'av2':
            self.lane_type_embed = nn.Parameter(torch.Tensor(num_lane_types, embed_dim))
            self.left_mark_embed = nn.Parameter(torch.Tensor(num_mark_types, embed_dim))
            self.right_mark_embed = nn.Parameter(torch.Tensor(num_mark_types, embed_dim))
            nn.init.normal_(self.lane_type_embed, mean=0., std=.02)
            nn.init.normal_(self.left_mark_embed, mean=0., std=.02)
            nn.init.normal_(self.right_mark_embed, mean=0., std=.02)
        else:
            self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
            self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
            nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
            nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                is_intersections: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None,
                turn_directions: Optional[torch.Tensor] = None,
                traffic_controls: Optional[torch.Tensor] = None,
                lane_types: Optional[torch.Tensor] = None,
                left_mark_types: Optional[torch.Tensor] = None,
                right_mark_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        if self.map_mode == 'av2':
            lane_types = lane_types.long()
            left_mark_types = left_mark_types.long()
            right_mark_types = right_mark_types.long()
            x_actor = x_actor + self._mha_block(
                self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections, rotate_mat, size,
                lane_types=lane_types, left_mark_types=left_mark_types, right_mark_types=right_mark_types)
        else:
            turn_directions = turn_directions.long()
            traffic_controls = traffic_controls.long()
            x_actor = x_actor + self._mha_block(
                self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections, rotate_mat, size,
                turn_directions=turn_directions, traffic_controls=traffic_controls)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: Adj,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                is_intersections_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int],
                turn_directions_j=None,
                traffic_controls_j=None,
                lane_types_j=None,
                left_mark_types_j=None,
                right_mark_types_j=None) -> torch.Tensor:
        if self.map_mode == 'av2':
            categorical = [self.is_intersection_embed[is_intersections_j],
                           self.lane_type_embed[lane_types_j],
                           self.left_mark_embed[left_mark_types_j],
                           self.right_mark_embed[right_mark_types_j]]
        else:
            categorical = [self.is_intersection_embed[is_intersections_j],
                           self.turn_direction_embed[turn_directions_j],
                           self.traffic_control_embed[traffic_controls_j]]
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr], categorical)
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j_r, edge_attr_r = rotate_2d_pair(x_j, edge_attr, rotate_mat)
            x_j = self.lane_embed([x_j_r, edge_attr_r], categorical)
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   is_intersections: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   size: Size,
                   turn_directions: Optional[torch.Tensor] = None,
                   traffic_controls: Optional[torch.Tensor] = None,
                   lane_types: Optional[torch.Tensor] = None,
                   left_mark_types: Optional[torch.Tensor] = None,
                   right_mark_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        kwargs = dict(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                      is_intersections=is_intersections, rotate_mat=rotate_mat, size=size)
        if self.map_mode == 'av2':
            kwargs.update(lane_types=lane_types, left_mark_types=left_mark_types,
                          right_mark_types=right_mark_types)
        else:
            kwargs.update(turn_directions=turn_directions, traffic_controls=traffic_controls)
        x_actor = self.out_proj(self.propagate(**kwargs))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)
