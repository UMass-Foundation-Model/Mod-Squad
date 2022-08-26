import torch
import torch.nn as nn

from parallel_experts import MoE

from timm.models.layers import DropPath, to_2tuple
from functools import partial

class MoEAttention(nn.Module):
    def __init__(self, dim, num_experts=24, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
        sample_topk=2, cvloss=0.0, switchloss=0.01, zloss=0.001):
        super().__init__()
        self.num_experts = num_experts
        self.sample_topk = sample_topk

        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = MoE(dim, head_dim, num_experts, num_heads, cvloss=cvloss, switchloss=switchloss, zloss=zloss)

        self.kv_proj = nn.Sequential(
            nn.Linear(dim, head_dim * 2),
        )

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):

        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, aux_loss = self.q_proj.map(x, sample_topk=self.sample_topk)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.head_dim)
        v = v.reshape(B, N, self.head_dim)

        attn = torch.einsum('bihd,bjd->bhij', q, k) * self.scale
        # attn = attn.premute(0,3,1,2) # b, h, i, j

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        attn = torch.einsum('bhij,bjd->bihd', attn, v)

        x = self.q_proj.reduce(attn)
        x = self.proj_drop(x)
        return x, aux_loss


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MoEBlock(nn.Module):

    def __init__(self, dim, num_heads, num_attn_experts=24, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None, z_weight=0.000,
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, sample_topk=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MoEAttention(
            dim, num_heads=num_heads, num_experts=num_attn_experts, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            cvloss=cvloss, switchloss=switchloss, zloss=zloss, sample_topk=sample_topk)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        assert z_weight == 0

    def forward(self, x, mask=None):
        y, z_loss = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, z_loss
