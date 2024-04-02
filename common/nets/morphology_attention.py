import torch.nn as nn
from einops import repeat
from main.draw_utils import *

class HMA(nn.Module):
    def __init__(self, dim=256, depth=1, num_heads=4, mlp_ratio=4.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))
        self.up_sample = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.mlp = nn.Conv2d(64, 1, 1, 1, 0)
        self.expand = nn.Conv2d(64, 256, 1, 1, 0)
        self.down_sample = nn.MaxPool2d(2, 2)
        self.norm_f = nn.BatchNorm2d(256)
        self.norm_out = nn.BatchNorm2d(64)

    def forward(self, structure_f, profile_f):
        Q_K = self.down_sample(self.expand(profile_f))
        V = structure_f
        for i, layer in enumerate(self.layers):
            output = layer(query_key=Q_K, value=V)
        output_f = self.up_sample(self.norm_f(output))
        whole_seg_f = self.mlp(self.norm_out(output_f) + profile_f)
        return whole_seg_f


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, key, value):
        B, N, C = query.shape
        query = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, N, C)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()



        self.channels = dim

        self.encode_value = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_query = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.qk_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))
        self.v_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query_key, value):
        b, c, h, w = query_key.shape
        query_key_embed = repeat(self.qk_embedding, '() n c d -> b n c d', b=b)
        value_embed = repeat(self.v_embedding, '() n c d -> b n c d', b=b)

        qk_embed = self.with_pos_embed(query_key, query_key_embed)
        v_embed = self.with_pos_embed(value, value_embed)

        v = self.encode_value(v_embed).view(b, self.channels, -1)
        v = v.permute(0, 2, 1)

        q = self.encode_query(qk_embed).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)

        k = self.encode_key(qk_embed).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)

        query_key = query_key.view(b, self.channels, -1).permute(0, 2, 1)

        query_key = query_key + self.attn(query=q, key=k, value=v)
        query_key = query_key + self.mlp(self.norm2(query_key))
        query_key = query_key.permute(0, 2, 1).contiguous().view(b, self.channels, h, w)

        return query_key
