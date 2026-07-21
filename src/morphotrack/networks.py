import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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
                 activation_func='ReLU',
                 use_output_norm=True,
                 unit_factor=1.0,
                 gamma1=1.0,
                 gamma2=1.0):
        super(SimpleMLP, self).__init__()

        self.use_output_norm = use_output_norm
        self.unit_factor = unit_factor
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.net = build_mlp_layer(
            sizes=[3] + hidden_sizes + [output_dim],
            activation=activation_func,
            use_norm=use_norm,
            use_residual=use_residual
        )

    def forward(self, x):
        raw_v = self.net(x)  # (N, output_dim)

        if self.use_output_norm:
            direction = F.normalize(raw_v, p=2, dim=1)
            raw_magnitude = raw_v.norm(p=2, dim=1, keepdim=True)

            min_mag = self.gamma1
            max_mag = self.gamma2
            magnitude = min_mag + (max_mag - min_mag) * torch.sigmoid(raw_magnitude)

            v = direction * magnitude * self.unit_factor
        else:
            v = raw_v

        return v
    

class SimpleMLP2(nn.Module):
    def __init__(self, 
                 output_dim=3,
                 hidden_sizes=[256, 128, 64],
                 use_norm=True,
                 use_residual=True,
                 activation_func='ReLU'):
        super(SimpleMLP2, self).__init__()
        self.net = build_mlp_layer(
            sizes=[3] + hidden_sizes + [output_dim],
            activation=activation_func,
            use_norm=use_norm,
            use_residual=use_residual
        )

    def forward(self, x):
        return self.net(x)
    

class SimpleMLP0(nn.Module):
    def __init__(self, 
                 hidden_sizes=[256, 128, 64],
                 use_norm=True,
                 use_residual=True,
                 activation_func='ReLU',
                 epsilon=1e-3):
        super(SimpleMLP2, self).__init__()
        self.epsilon = epsilon

        self.net = build_mlp_layer(
            sizes=[3] + hidden_sizes,
            activation=activation_func,
            use_norm=use_norm,
            use_residual=use_residual
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)


    def forward(self, x):
        raw_v = self.net(x)  # (N, output_dim)
        alpha = F.softplus(self.output_layer(raw_v)).squeeze(-1) + self.epsilon  # shape: (N,)

        return alpha
    

# class MagnitudePredictor(nn.Module):
#     def __init__(self, input_dim=3, hidden_sizes=[64, 32], use_norm=False, epsilon=1e-3):
#         super().__init__()
#         self.epsilon = epsilon

#         layers = []
#         in_dim = input_dim

#         for h in hidden_sizes:
#             layers.append(nn.Linear(in_dim, h))
#             if use_norm:
#                 layers.append(nn.LayerNorm(h))
#             layers.append(nn.SiLU())
#             in_dim = h

#         self.net = nn.Sequential(*layers)
#         self.output_layer = nn.Linear(in_dim, 1)

#     def forward(self, x):
#         h = self.net(x)
#         alpha = F.softplus(self.output_layer(h)).squeeze(-1) + self.epsilon  # shape: (N,)
#         return alpha


def positional_encoding(x, num_freqs=6):
    """
    x: (..., 1)
    Returns: (..., 2*num_freqs)
    """
    freq_bands = 2.0 ** torch.arange(num_freqs, dtype=x.dtype, device=x.device) * torch.pi
    x = x.unsqueeze(-1)  # (..., 1, 1)
    enc = torch.cat([torch.sin(freq_bands * x), torch.cos(freq_bands * x)], dim=-1)  # (..., 1, 2*num_freqs)
    return enc.squeeze(1)  # (..., 2*num_freqs)


class DeltaMotionNet(nn.Module):
    def __init__(self, num_freqs=6, hidden_dim=64, dims=3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dims = dims
        input_dim = dims * 2 * num_freqs  # 2 for sin and cos

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X):  # X: (..., input_dims) → (...,) raw delta scalars
        if X.shape[-1] != self.input_dims:
            raise ValueError(f"Expected input with last dim = {self.input_dims}, got {X.shape[-1]}")

        encoded = [
            positional_encoding(X[..., i:i+1], self.num_freqs)
            for i in range(self.input_dims)
        ]
        feat = torch.cat(encoded, dim=-1)  # (..., input_dim)
        return self.net(feat).squeeze(-1)


class SimpleMLP3(nn.Module):
    """SimpleMLP2 with optional NeRF-style positional encoding on the input.

    With use_positional_encoding=False, behaves identically to SimpleMLP2:
    input (N,3) is forwarded through build_mlp_layer with no encoding.

    With use_positional_encoding=True, applies γ(x) per-dim
    (same encoding as DeltaMotionNet, via positional_encoding helper)
    and optionally concatenates raw x for low-freq channels (NeRF default).
    """

    def __init__(self,
                 output_dim=3,
                 hidden_sizes=[256, 128, 64],
                 use_norm=True,
                 use_residual=True,
                 activation_func='ReLU',
                 use_positional_encoding=True,
                 num_freqs=6,
                 include_raw=True):
        super().__init__()
        self.use_pe      = use_positional_encoding
        self.num_freqs   = num_freqs
        self.include_raw = include_raw

        if use_positional_encoding:
            pe_dim = 3 * 2 * num_freqs              # xyz × sin/cos × num_freqs
            in_dim = pe_dim + (3 if include_raw else 0)
        else:
            in_dim = 3

        self.net = build_mlp_layer(
            sizes=[in_dim] + hidden_sizes + [output_dim],
            activation=activation_func,
            use_norm=use_norm,
            use_residual=use_residual,
        )

    def forward(self, x):                            # x: (N, 3)
        if not self.use_pe:
            return self.net(x)
        encoded = [
            positional_encoding(x[..., i:i + 1], self.num_freqs)
            for i in range(3)
        ]
        feat = torch.cat(encoded, dim=-1)            # (N, 6·num_freqs)
        if self.include_raw:
            feat = torch.cat([x, feat], dim=-1)      # (N, 3 + 6·num_freqs)
        return self.net(feat)


class HarmonicGradField(nn.Module):
    """Adapter: scalar phi_net → v_field via ∇φ + per-axis scale correction.

    Returns (N, 3) ∇_real φ on normalized input (N, 3).

    create_graph (init flag, default False):
        - False : default for pure inference; no retained graph.
        - True  : graph always retained.
    Regardless of this flag, if the input has `requires_grad=True`
    (e.g., caller is in a grad-enabled context like `divergence_vhat`),
    the inner autograd is run with `create_graph=True` so the output
    is differentiable w.r.t. the input.
    """

    def __init__(self, phi_net, scale, create_graph=False):
        super().__init__()
        self.phi_net = phi_net
        if not torch.is_tensor(scale):
            scale = torch.as_tensor(scale, dtype=torch.float32)
        self.register_buffer('inv_scale', 1.0 / scale)
        self.create_graph = create_graph

    def forward(self, x_norm):
        # Auto-detect outer grad need: if x_norm requires grad, the caller
        # will want to backprop through our output → retain inner graph.
        need_graph = self.create_graph or x_norm.requires_grad

        with torch.enable_grad():
            if x_norm.requires_grad:
                x = x_norm
            else:
                x = x_norm.detach().requires_grad_(True)
            phi = self.phi_net(x).squeeze(-1)
            grad_norm = torch.autograd.grad(
                phi.sum(), x, create_graph=need_graph
            )[0]
        return grad_norm * self.inv_scale


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
