from geomloss import SamplesLoss
import torch
import torch.nn as nn

# -----------------------
# Point cloud Loss
# -----------------------
def chamfer_one_way():
    def chamfer(X, Y):
        x_exp = X.unsqueeze(1)  # (N, 1, 3)
        y_exp = Y.unsqueeze(0)  # (1, M, 3)
        dist = torch.sum((x_exp - y_exp) ** 2, dim=2)  # (N, M)
        x2y = torch.min(dist, dim=1)[0]
        return x2y.mean()
    return chamfer

def chamfer_distance():
    def chamfer(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
        """
        Chamfer Distance between two point clouds of shape [N, 3] and [M, 3].

        Args:
            pc1: Tensor of shape [N, 3]
            pc2: Tensor of shape [M, 3]

        Returns:
            Scalar Chamfer distance (mean of nearest neighbor distances in both directions)
        """
        # Compute pairwise distances: [N, M]
        dist = torch.cdist(pc1.unsqueeze(0), pc2.unsqueeze(0), p=2).squeeze(0)

        # For each point in pc1, find nearest in pc2
        min_dist_pc1, _ = torch.min(dist, dim=1)  # [N]

        # For each point in pc2, find nearest in pc1
        min_dist_pc2, _ = torch.min(dist, dim=0)  # [M]

        # Mean over both directions
        chamfer_loss = min_dist_pc1.mean() + min_dist_pc2.mean()
        return chamfer_loss
    return chamfer

    
def sinkhorn(**kwargs):
    # Default parameters
    default_params = {
        "loss": "sinkhorn",
        "p": 2,
        "blur": 0.01,
    }

    # Override defaults with kwargs
    default_params.update(kwargs)

    # Create and return the loss function with updated parameters
    sinkhorn_loss_fn = SamplesLoss(
        default_params["loss"],
        p=default_params["p"],
        blur=default_params["blur"]
    )

    return sinkhorn_loss_fn


def match_loss(X, Y, loss_func=sinkhorn()):
    loss = loss_func(X, Y)
    return loss


# -----------------------
# Vector Alignment Loss
# -----------------------
def alignment_angle_loss(vectors, references, norm_vector=True, norm_ref=False, mixed_orientation=False, return_vector=False, eps=1e-8):
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

    cos_sim_clamped = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.arccos(cos_sim_clamped)
    
    if return_vector:
        return angle*2# 1 - cos_sim
    else:
        return angle.mean()*2# 1 - cos_sim.mean()
    

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
    

def alignment_loss_mse(vectors, references, norm_vector=True, norm_ref=False, return_vector=False, eps=1e-8):
    if norm_vector:
        vectors_unit = vectors / (vectors.norm(dim=1, keepdim=True) + eps)
    else:
        vectors_unit = vectors
        
    if norm_ref:
        references_unit = references / (references.norm(dim=1, keepdim=True) + eps)
    else:
        references_unit = references

    # Calculate cosine similarity and its negative
    mse = torch.sum((vectors_unit - references_unit) ** 2, dim=1)
    
    if return_vector:
        return mse
    else:
        return mse.mean()