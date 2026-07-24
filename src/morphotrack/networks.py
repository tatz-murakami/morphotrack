import torch.nn as nn
import torch
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation='ReLU', use_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()
        self.activation = getattr(nn, activation)() if activation else nn.Identity()
        self.shortcut = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        return out + self.shortcut(x)
    

def build_mlp_layer(sizes, activation='ReLU', use_norm=False, use_residual=False):
    layers = []
    for i in range(len(sizes) - 1):
        in_size, out_size = sizes[i], sizes[i + 1]
        if use_residual:
            layers.append(
                ResidualMLPBlock(
                    in_dim=in_size,
                    out_dim=out_size,
                    activation=activation,
                    use_norm=use_norm,
                )
            )
        else:
            layers.append(nn.Linear(in_size, out_size))
            if use_norm:
                layers.append(nn.LayerNorm(out_size))
            if activation:
                layers.append(getattr(nn, activation)())
    return nn.Sequential(*layers)

    
    

class SimpleMLP(nn.Module):
    def __init__(self, 
                 output_dim=3,
                 hidden_sizes=[256, 128, 64],
                 use_norm=True,
                 use_residual=True,
                 activation_func='ReLU'):
        super(SimpleMLP, self).__init__()
        self.net = build_mlp_layer(
            sizes=[3] + hidden_sizes + [output_dim],
            activation=activation_func,
            use_norm=use_norm,
            use_residual=use_residual
        )

    def forward(self, x):
        return self.net(x)
    


class SpatialEncoder(nn.Module):
    def __init__(self, z_dim=16, hidden_dim=64, num_freqs=6, use_pos_enc=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.use_pos_enc = use_pos_enc
        input_dim = 2 * (2 * num_freqs) if use_pos_enc else 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def _encode(self, x):
        freq_bands = 2.0 ** torch.arange(self.num_freqs, dtype=x.dtype, device=x.device) * torch.pi
        x = x.unsqueeze(-1)
        enc = torch.cat([torch.sin(freq_bands * x), torch.cos(freq_bands * x)], dim=-1)
        return enc.squeeze(-2)

    def forward(self, uv):
        if self.use_pos_enc:
            feat = torch.cat([self._encode(uv[:, 0:1]), self._encode(uv[:, 1:2])], dim=-1)
        else:
            feat = uv
        return self.net(feat)


class LinearSpeedNet(nn.Module):
    """
    Predicts r(u,v) = tanh(...) ∈ [−1, 1].
    Physical scaling: a(u,v) = 2·L·r;   c(u,v,t) = a·(t − 1/2) + L.
    Warp: t_warped(t) = t + r·t·(t − 1).
    """
    def __init__(self, z_dim=16, hidden_dim=64, num_freqs=6, use_pos_enc=True):
        super().__init__()
        self.encoder = SpatialEncoder(z_dim, hidden_dim, num_freqs, use_pos_enc)
        self.head = nn.Linear(z_dim, 1)

    def forward(self, uv):                       # uv: (M, 2) → r: (M,)
        z = self.encoder(uv)
        return torch.tanh(self.head(z).squeeze(-1))


class NormalizedField:
    def __init__(self, model, pos_min, pos_max):
        self.model = model
        self.pos_min = pos_min
        self.pos_max = pos_max
    def __call__(self, x):
        x_norm = (x - self.pos_min) / (self.pos_max - self.pos_min)
        return F.normalize(self.model(x_norm), p=2, dim=1)
    def eval(self):
        self.model.eval()
        return self
