import torch
import torch.nn.functional as F
    

def alignment_loss(vectors, references, norm_vector=True, norm_ref=False, mixed_orientation=False, return_vector=False, eps=1e-8):
    if norm_vector:
        vectors_unit = vectors / (vectors.norm(dim=1, keepdim=True) + eps)
    else:
        vectors_unit = vectors
        
    if norm_ref:
        references_unit = references / (references.norm(dim=1, keepdim=True) + eps)
    else:
        references_unit = references

    if mixed_orientation:
        # Calculate cosine similarity and its negative
        cos_sim = torch.sum(vectors_unit * references_unit, dim=1)
        # Adjust for mixed orientation by flipping signs based on the sign of the cosine similarity
        cos_sim = torch.where(cos_sim < 0, -cos_sim, cos_sim)
    else:
        cos_sim = torch.sum(vectors_unit * references_unit, dim=1)

    if return_vector:
        return 1 - cos_sim
    else:
        return 1 - cos_sim.mean()


def approx_const_curvature_loss(v_field, points, epsilon=1e-3):
    v0 = F.normalize(v_field(points), p=2, dim=1)
    x1 = points + epsilon * v0
    v1 = F.normalize(v_field(x1), p=2, dim=1)
    x2 = x1 + epsilon * v1
    v2 = F.normalize(v_field(x2), p=2, dim=1)
    return ((v2 - 2 * v1 + v0) ** 2).sum(dim=1).mean()  # ≈ ε⁴ · ‖d²v̂/ds²‖²


def approx_zero_torsion_loss(v_field, points, epsilon=1e-3):
    """Penalize (T · (T' × T''))² ≈ κ⁴ τ² along streamlines.
    Drives streamlines to be locally planar (zero torsion)
    or straight (zero curvature). Weighted by curvature⁴ — natural
    handling of straight regions (no division)."""
    v0 = F.normalize(v_field(points), p=2, dim=1)
    x1 = points + epsilon * v0
    v1 = F.normalize(v_field(x1), p=2, dim=1)
    x2 = x1 + epsilon * v1
    v2 = F.normalize(v_field(x2), p=2, dim=1)

    T        = v1
    T_prime  = (v2 - v0) / (2 * epsilon)
    T_dprime = (v2 - 2 * v1 + v0) / (epsilon ** 2)

    triple = (T * torch.cross(T_prime, T_dprime, dim=1)).sum(dim=1)   # scalar per pt
    return triple.pow(2).mean()


def dirichlet_energy(net, X):
    """Monte-Carlo estimate of ∫|∇φ|² over interior samples X. (Deep Ritz.)"""
    Xg  = X.detach().requires_grad_(True)
    phi = net(Xg).squeeze(-1)
    grad_phi = torch.autograd.grad(
        phi.sum(), Xg, create_graph=True, retain_graph=True
    )[0]                                                   # (N, 3)
    return grad_phi.pow(2).sum(dim=-1).mean()
    

