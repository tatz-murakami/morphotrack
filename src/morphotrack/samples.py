import torch


# -----------------------
# Data Generation
# -----------------------
def generate_surface_data(u, v, t, noise_std=0.02):
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32, device=u.device)
    f_uv = torch.sin(torch.pi * (3/4.) * u) + torch.cos(torch.pi * (3/4.) * v) + 2.0
    z = f_uv * (t + 1)**2 + t
    x = u * 10
    y = v * 10 + 2 * torch.sin(torch.pi * (1/2.) * t)
    xyz = torch.stack([x, y, z], dim=1)
    if noise_std > 0:
        xyz += torch.randn_like(xyz) * noise_std
    return xyz


def partial_G_t(u, v, t):
    f_uv = torch.sin(torch.pi * (3/4.) * u) + torch.cos(torch.pi * (3/4.) * v) + 2.0
    dx_dt = torch.zeros_like(t)
    dy_dt = torch.pi * torch.cos(torch.pi * 0.5 * t)
    dz_dt = 2 * (t + 1) * f_uv + 1
    return torch.stack([dx_dt, dy_dt, dz_dt], dim=1)


# -----------------------
# Sample Data for Alignment
# -----------------------
def generate_alignment_data(n_samples=1000, n_t=10, start=0, end=1, device='cuda'):
    u = (torch.rand(n_samples, device=device) * 2) - 1
    v = (torch.rand(n_samples, device=device) * 2) - 1
    t_steps = torch.linspace(start, end, n_t, device=device)
    vec_pos = []
    dGdt_val = []

    for t_val in t_steps:
        t_batch = torch.full((n_samples,), t_val, device=device)
        xyz = generate_surface_data(u, v, t_batch, noise_std=0.01)
        dGdt = partial_G_t(u, v, t_batch)
        vec_pos.append(xyz)
        dGdt_val.append(dGdt)

    vec_pos_tensor = torch.cat(vec_pos, dim=0)
    dGdt_val_tensor = torch.cat(dGdt_val, dim=0)
    norms = torch.norm(dGdt_val_tensor, dim=1, keepdim=True) + 1e-8
    dGdt_unit_tensor = dGdt_val_tensor / norms

    return vec_pos_tensor, dGdt_unit_tensor


# -----------------------
# Point cloud data generation
# -----------------------
# def generate_gaussian_sample(mu, sigma, num_samples, device='cuda'):
#     M = len(mu)
#     N = num_samples
#     output = torch.zeros((M, N), device=device)

#     for i in range(M):
#         # Generate N samples from a Gaussian distribution for column i
#         samples = torch.normal(mu[i], sigma[i], size=(N,), device=device)
        
#         # Clip the values to be in the range [0, 1]
#         samples = torch.clamp(samples, 0, 1)
        
#         # Sort the values in ascending order
#         output[i] = torch.sort(samples).values

#     return output

def generate_gaussian_sample(mu, sigma, num_samples, device='cuda'):
    if not torch.is_tensor(mu):
        mu = torch.tensor(mu, device=device)
    else:
        mu = mu.to(device)
    mu = mu.view(-1, 1)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=device)
    else:
        sigma = sigma.to(device)
    sigma = sigma.view(-1, 1)

    eps = torch.randn((mu.size(0), num_samples), device=device)  # Standard normal samples
    samples = mu + sigma * eps
    samples = torch.clamp(samples, 0, 1)
    samples, _ = torch.sort(samples, dim=1)
    return samples


def compute_sample_val(u, v, mean=0.2, coeff=0.5, device='cuda'):
    u = u.to(device)
    v = v.to(device)
    
    # Example of a smooth function: sinusoidal variation
    val = mean + coeff * mean * torch.sin(torch.pi * u) * torch.cos(torch.pi * v)
    
    return val