import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from typing import Tuple, Callable, List
from tqdm import tqdm
import math


def D_transform_model(raw_deltaD):
    """
    Monotonic transformation output using learned positive deltas.

    Inputs:
        raw_deltaD: learnable parameters of shape (u_num, v_num - 1)
            These are transformed via softplus to ensure positivity.

    Returns:
        D_adjusted: transformed values, shape (u_num, v_num)
    """
    dim = raw_deltaD.dim()
    if dim==2:
        u_num, _ = raw_deltaD.shape
        # Ensure positive deltas using softplus
        soft_deltaD = F.softplus(raw_deltaD)  # shape: (u_num, v_num - 1)
        # Cumulative sum (integral) to get strictly increasing mapping
        D_no_bound = torch.cumsum(soft_deltaD, dim=1)  # shape: (u_num, v_num - 1)
        # Add a leading zero to make D start at 0
        D_adjusted = torch.cat([torch.zeros(u_num, 1, device=raw_deltaD.device), D_no_bound], dim=1)  # (u_num, v_num)
        # Normalize so D ends at 1
        D_adjusted = D_adjusted / (D_adjusted[:, -1:].clamp(min=1e-6))  # shape: (u_num, v_num)
    elif dim==3:
        u_num, v_num,_ = raw_deltaD.shape
        # Ensure positive deltas using softplus
        soft_deltaD = F.softplus(raw_deltaD)  # shape: (u_num, v_num, t_num-1)
        # Cumulative sum (integral) to get strictly increasing mapping
        D_no_bound = torch.cumsum(soft_deltaD, dim=2)  # shape: (u_num, v_num, t_num-1)
        # Add a leading zero to make D start at 0
        D_adjusted = torch.cat([torch.zeros(u_num, v_num, 1, device=raw_deltaD.device), D_no_bound], dim=2)  # (u_num, v_num, t_num)
        # Normalize so D ends at 1
        D_adjusted = D_adjusted / (D_adjusted[:, :, -1:].clamp(min=1e-6))  # shape: (u_num, v_num, t_num)
    else:
        raise ValueError("Only 2 or 3 dim input is supported.")
        
    return D_adjusted



def displacement_transform(positions, displacement, k=3, eps=1e-8):
    """
    Create a displacement-based forward transform function using weighted KNN interpolation.
    
    positions:    (N, 2) tensor — sample points where displacement is known
    displacement: (N, 2) tensor — displacement vectors at those points
    k: Number of nearest neighbors to use (1 = nearest neighbor, >1 = weighted average)
    """
    N = positions.shape[0]

    def transform_fn(uv_points):
        """
        uv_points: (M, 2) tensor — query points
        Returns:   (M, 2) tensor — displaced points
        """
        # Compute pairwise distances between query and known points
        diff = uv_points.unsqueeze(1) - positions.unsqueeze(0)  # (M, N, 2)
        dist2 = (diff ** 2).sum(dim=2)  # (M, N)

        # Get indices of k nearest neighbors
        knn_dists, knn_indices = torch.topk(dist2, k=k, dim=1, largest=False)

        # Inverse distance weights (with small epsilon for stability)
        weights = 1.0 / (knn_dists + eps)  # (M, k)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Gather corresponding displacements
        knn_displacements = displacement[knn_indices]  # (M, k, 2)

        # Weighted sum
        interpolated_disp = (weights.unsqueeze(2) * knn_displacements).sum(dim=1)  # (M, 2)

        # Apply displacement
        return uv_points + interpolated_disp

    return transform_fn


def displacement_transform_with_kdTree(positions, displacements, k=3, eps=1e-8):
    """
    Fast KNN-based displacement interpolator using KDTree.

    Parameters:
        positions: (N, 2) numpy array of source positions
        displacements: (N, 2) numpy array of displacements
        k: number of nearest neighbors
        eps: small number to avoid division by zero

    Returns:
        transform_fn(uv_points): returns uv_points + interpolated displacement
    """
    tree = KDTree(positions)

    def transform_fn(uv_points):
        # Query k nearest neighbors
        dists, indices = tree.query(uv_points, k=k)  # dists: (M, k), indices: (M, k)

        # Gather displacements for neighbors
        knn_displacements = displacements[indices]  # shape (M, k, 2)

        # Inverse distance weighting
        weights = 1.0 / (dists + eps)  # (M, k)
        weights /= np.sum(weights, axis=1, keepdims=True)  # normalize

        # Weighted average of displacements
        interpolated_disp = np.sum(weights[..., None] * knn_displacements, axis=1)  # (M, 2)

        return uv_points + interpolated_disp  # (M, 2)

    return transform_fn


# Soft assignment of each point to bins
def soft_bin_assign_gauss(values, centers, sigma):
    # values: [N], centers: [B] -> output: [N, B]
    diff = values[:, None] - centers[None, :]  # [N, B]
    weights = torch.exp(-0.5 * (diff / sigma)**2)  # Gaussian soft binning
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    return weights


def soft_bin_assign_triangular(values, centers, width):
    """
    Soft bin assignment using a triangular kernel (differentiable).
    - values: [N]
    - centers: [B]
    - width: kernel support (half-width of triangle base)
    Returns:
    - weights: [N, B]
    """
    diff = torch.abs(values[:, None] - centers[None, :])  # [N, B]
    weights = torch.clamp(1 - diff / width, min=0.0)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    return weights


def soft_bin_assign_softmax(values, centers, temperature):
    """
    Soft bin assignment using softmax over negative distances.
    - temperature controls the sharpness (lower = sharper)
    """
    diff = values[:, None] - centers[None, :]
    weights = torch.softmax(-torch.abs(diff) / temperature, dim=1)
    return weights


def soft_bin_function(bin_param=1.0, func_choice='triangular'):
    if func_choice == 'triangular':
        def soft_bin_assign(values, centers):
            return soft_bin_assign_triangular(values, centers, width=bin_param)
    elif func_choice == 'gaussian':
        def soft_bin_assign(values, centers):
            return soft_bin_assign_gauss(values, centers, sigma=bin_param)
    elif func_choice == 'softmax':
        def soft_bin_assign(values, centers):
            return soft_bin_assign_softmax(values, centers, temperature=bin_param)
    else:
        raise ValueError("func_choice must be either triangular/gaussian/softmax") 
    return soft_bin_assign


def generate_bin_center(x, bins, lower=0, upper=100):
    device = x.device
    # x_min, x_max = torch.quantile(x, lower), torch.quantile(x, upper)
    x_min = np.percentile(x.detach().cpu().numpy(), lower)
    x_max = np.percentile(x.detach().cpu().numpy(), upper)
    bin_centers = torch.linspace(x_min, x_max, bins, device=device)
    return bin_centers


def soft_histogram_nd(
    coords: torch.Tensor,
    bin_centers: Tuple[torch.Tensor, ...],  # tuple of D tensors
    params: Tuple[float],
    cdf: bool = False,
    chunk_size: int = 10000,
):
    """
    Memory-efficient soft histgram generator using chunked accumulation.
    Args:
        coords: [N, D] input coordinates
        bin_centers: tuple of 1D tensors with bin centers for each dimension
        params: tuple of floats, softness parameters per dimension
        chunk_size: number of samples per chunk
        cdf: if True, compute cumulative sum along last axis

    Returns:
        hist: [bins[0], bins[1], ..., bins[D-1]] soft histogram
    """
    device = coords.device
    N, D = coords.shape
    assert len(bin_centers) == D and len(params) == D

    # Compute global bin centers once (assumes min/max over all data)
    bins = [i.shape[0] for i in bin_centers]

    # Initialize histogram
    hist_shape = tuple(bins)
    hist = torch.zeros(hist_shape, device=device)

    for chunk_start in tqdm(range(0, N, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk = coords[chunk_start:chunk_end]

        weights_per_dim = []
        for d in range(D):
            weights = soft_bin_assign_triangular(
                chunk[:, d], bin_centers[d], params[d]
            )
            weights_per_dim.append(weights)

        # Construct einsum string dynamically
        dims = [chr(105 + i) for i in range(D)]  # ['i', 'j', ...]
        einsum_inputs = ','.join([f'n{d}' for d in dims])
        einsum_output = ''.join(dims)
        einsum_str = f'{einsum_inputs}->{einsum_output}'

        partial_hist = torch.einsum(einsum_str, *weights_per_dim)
        hist += partial_hist

    if cdf:
        hist = torch.cumsum(hist, dim=-1)

    return hist


def soft_bin_assign_triangular_sparse(values, centers, width, skip=1):
    """
    Vectorized soft bin assignment using triangular kernel with stride-based bin skipping.
    
    Args:
        values: [N]
        centers: [B] (sorted)
        width: float
        skip: int, spacing between neighboring bins (1 = no skip, 2 = every other bin, etc.)
    
    Returns:
        weights: [N, K], indices: [N, K]
    """
    device = values.device
    N = values.shape[0]
    B = centers.shape[0]

    if B < 2:
        raise ValueError("Need at least two bin centers to compute spacing.")
    
    delta = centers[1] - centers[0]  # assumes uniform spacing
    full_support = int(torch.ceil(2 * width / delta).item()) + 1  # like before
    all_offsets = torch.arange(-(full_support // 2), full_support // 2 + 1, device=device)
    
    # Apply skipping
    sampled_offsets = all_offsets[::skip]  # e.g. [-50, -48, ..., 0, ..., 48, 50]
    K = sampled_offsets.numel()

    # Find nearest center index per value
    indices = torch.bucketize(values, centers)
    indices = torch.clamp(indices, 1, B - 1)  # avoid edges

    center_ids = indices.unsqueeze(1) + sampled_offsets.view(1, -1)  # [N, K]
    center_ids = torch.clamp(center_ids, 0, B - 1)

    selected_centers = centers[center_ids]  # [N, K]
    dists = torch.abs(values.unsqueeze(1) - selected_centers)
    weights = torch.clamp(1 - dists / width, min=0.0)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    return weights, center_ids


def estimate_values_from_histogram(
    positions: torch.Tensor,             # [M, D] (D=2 or 3)
    histogram: torch.Tensor,             # [B1, B2, (B3)]
    bin_centers: Tuple[torch.Tensor, ...],  # tuple of D tensors
    params: Tuple[float, ...],              # tuple of D widths
    normalize: bool = True,
    skips: Tuple[int, ...] = (1, 1),
    chunk_size: int = 100000
):
    """
    Memory-efficient estimation from soft histogram using per-dimension skip-based binning and chunking.

    Args:
        positions: [M, D] (D=2 or 3)
        histogram: histogram tensor of shape [B1, B2, (B3)]
        bin_centers: tuple of bin center tensors per dimension
        params: tuple of triangular widths per dimension
        normalize: if True, normalize result by histogram.sum()
        skips: tuple of ints, stride between neighboring bins per dimension
        chunk_size: chunk size for processing positions

    Returns:
        values: [M] estimated values from the histogram
    """
    D = positions.shape[1]
    M = positions.shape[0]
    assert D in (2, 3), "Only 2D or 3D histograms are supported."
    assert len(skips) == D, f"`skips` must be a tuple of length {D}."

    result_list = []

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = positions[start:end]  # [m, D]
        weights = []
        indices = []

        for d in range(D):
            w_d, i_d = soft_bin_assign_triangular_sparse(
                chunk[:, d], bin_centers[d], params[d], skip=skips[d]
            )
            weights.append(w_d)
            indices.append(i_d)

        if D == 2:
            wu, wv = weights
            iu, iv = indices

            wu = wu.unsqueeze(2)        # [m, K1, 1]
            wv = wv.unsqueeze(1)        # [m, 1, K2]
            W = wu * wv                 # [m, K1, K2]

            K1, K2 = wu.shape[1], wv.shape[2]
            iu_exp = iu.unsqueeze(2).expand(-1, -1, K2)
            iv_exp = iv.unsqueeze(1).expand(-1, K1, -1)

            hist_vals = histogram[iu_exp, iv_exp]  # [m, K1, K2]

        else:  # D == 3
            wu, wv, ww = weights
            iu, iv, iw = indices

            wu = wu.unsqueeze(2).unsqueeze(3)  # [m, K1, 1, 1]
            wv = wv.unsqueeze(1).unsqueeze(3)  # [m, 1, K2, 1]
            ww = ww.unsqueeze(1).unsqueeze(2)  # [m, 1, 1, K3]
            W = wu * wv * ww                   # [m, K1, K2, K3]

            K1, K2, K3 = wu.shape[1], wv.shape[2], ww.shape[3]
            iu_exp = iu.unsqueeze(2).unsqueeze(3).expand(-1, -1, K2, K3)
            iv_exp = iv.unsqueeze(1).unsqueeze(3).expand(-1, K1, -1, K3)
            iw_exp = iw.unsqueeze(1).unsqueeze(2).expand(-1, K1, K2, -1)

            hist_vals = histogram[iu_exp, iv_exp, iw_exp]  # [m, K1, K2, K3]

        values = (W * hist_vals).sum(dim=tuple(range(1, W.ndim)))  # [m]
        result_list.append(values)

    values = torch.cat(result_list, dim=0)  # [M]

    if normalize:
        values = values / (histogram.sum() + 1e-8)

    return values


def compute_effective_support(width: float, centers: torch.Tensor, skip: int = 1) -> int:
    """
    Estimate number of bin neighbors per point when using 'skip'-based triangular soft binning.
    
    Args:
        width: half-width of the triangle kernel
        centers: tensor of bin centers (must be >= 2)
        skip: integer skip value (1 = no skipping)

    Returns:
        int: number of neighbors per point after skipping
    """
    if centers.shape[0] < 2:
        raise ValueError("At least two bin centers are needed.")

    delta = centers[1] - centers[0]
    full_support = int(torch.ceil(2 * width / delta).item()) + 1
    effective_support = int(math.ceil(full_support / skip))
    return effective_support


def build_warp_field(delta_flat: torch.Tensor, u_num: int, v_num: int, t_num: int) -> torch.Tensor:
    """
    Converts flat raw delta predictions from DeltaMotionNet into a monotonic
    warp field Tmoved in [0, 1].

    Parameters
    ----------
    delta_flat : torch.Tensor, shape (u_num * v_num * (t_num - 1),)
        Raw scalar outputs from the network for all grid points except t_min.
    u_num, v_num, t_num : int
        Grid dimensions of the output warp field.

    Returns
    -------
    Tmoved : torch.Tensor, shape (u_num, v_num, t_num)
        Monotonically increasing warp field along the t-axis, in [0, 1].
        Tmoved[:, :, 0] == 0 and Tmoved[:, :, -1] == 1 for all (u, v).
    """
    delta = delta_flat.view(u_num, v_num, t_num - 1)
    soft_delta = F.softplus(delta)
    cumulative = torch.cumsum(soft_delta, dim=2)
    cumulative = torch.cat([
        torch.zeros(u_num, v_num, 1, device=delta_flat.device),
        cumulative
    ], dim=2)
    return cumulative / cumulative[:, :, -1:].clamp(min=1e-6)


###############
# Garbage
###############

# def estimate_values_from_histogram(
#     positions: torch.Tensor,          # [M, 2]
#     histogram: torch.Tensor,          # [B1, B2]
#     bin_centers: Tuple[torch.Tensor, torch.Tensor],
#     params: Tuple[float, float],
#     normalize=True,
#     skip: int = 1,
#     chunk_size: int = 100000
# ):
#     """
#     Memory-efficient estimation from 2D histogram using skip-based soft binning and chunking.

#     Args:
#         positions: [M, 2]
#         histogram: [B1, B2]
#         bin_centers: (centers_u, centers_v)
#         params: (width_u, width_v)
#         normalize: if True, normalize result by histogram.sum()
#         skip: stride for bin skipping (1 = full neighbors, 2 = every other bin, etc.)
#         chunk_size: number of positions per chunk

#     Returns:
#         values: [M] estimated values from the histogram
#     """
#     M = positions.shape[0]
#     device = positions.device
#     centers_u, centers_v = bin_centers
#     width_u, width_v = params

#     result_list = []

#     for start in range(0, M, chunk_size):
#         end = min(start + chunk_size, M)
#         chunk = positions[start:end]

#         # Use skip-aware soft binning
#         wu, iu = soft_bin_assign_triangular_sparse(chunk[:, 0], centers_u, width_u, skip)
#         wv, iv = soft_bin_assign_triangular_sparse(chunk[:, 1], centers_v, width_v, skip)

#         wu = wu.unsqueeze(2)  # [m, K1, 1]
#         wv = wv.unsqueeze(1)  # [m, 1, K2]
#         weights = wu * wv     # [m, K1, K2]

#         K1 = wu.shape[1]
#         K2 = wv.shape[2]

#         iu_exp = iu.unsqueeze(2).expand(-1, -1, K2)  # [m, K1, K2]
#         iv_exp = iv.unsqueeze(1).expand(-1, K1, -1)  # [m, K1, K2]

#         hist_vals = histogram[iu_exp, iv_exp]        # [m, K1, K2]
#         values = (weights * hist_vals).sum(dim=(1, 2))  # [m]

#         result_list.append(values)

#     values = torch.cat(result_list, dim=0)  # [M]

#     if normalize:
#         values = values / (histogram.sum() + 1e-8)

#     return values



# def estimate_values_from_histogram(
#     positions: torch.Tensor,     # [M, D]
#     histogram: torch.Tensor,   # [N, D]
#     bin_centers: Tuple[int],
#     params: Tuple[float],
#     normalize=True
# ):
#     """
#     Estimate values using histogram and interpolation.

#     Args:
#         positions: [M, D] query points
#         histogram: input histogram
#         bin_centers: center of bins
#         params: tuple of floats, softness params per dimension
#         normalize: True if normalization necessary

#     Returns:
#         Tensor of shape [M] with values at query positions.
#     """
#     D = histogram.dim()
#     M = positions.shape[0]
#     assert len(params) == D

#     # 2) Compute soft bin weights for query positions
#     query_weights_per_dim = []
#     for d in range(D):
#         weights = soft_bin_assign_triangular(positions[:, d], bin_centers[d], params[d])
#         query_weights_per_dim.append(weights)

#     # 3) Build einsum string dynamically
#     dims = [chr(105 + i) for i in range(D)]  # ['i', 'j', 'k', ...]
#     weights_subscripts = [f'm{d}' for d in dims]  # e.g. ['mi','mj','mk']
#     hist_subscript = ''.join(dims)                 # e.g. 'ijk'
#     output_subscript = 'm'                          # output shape: [M]

#     einsum_str = ','.join(weights_subscripts) + ',' + hist_subscript + '->' + output_subscript
#     values = torch.einsum(einsum_str, *query_weights_per_dim, histogram)  # shape: [M]

#     # 4) Normalize if requested
#     if normalize:
#         values = values / histogram.sum()

#     return values  # shape: [M]


# def soft_histogram_nd(coords: torch.Tensor, bins: Tuple[int], params: Tuple[float], cdf=False):
#     """
#     Compute an N-dimensional soft histogram.

#     Args:
#         coords: [N, D] input coordinates
#         bins: tuple of ints, number of bins per dimension
#         params: tuple of floats, softness parameters per dimension
#         cdf: if True, compute cumulative sum along last axis

#     Returns:
#         hist: [bins[0], bins[1], ..., bins[D-1]] soft histogram
#     """
#     N, D = coords.shape
#     assert len(bins) == D and len(params) == D

#     bin_centers = []
#     weights_per_dim = []
#     for d in range(D):
#         centers = generate_bin_center(coords[:, d], bins[d])
#         weights = soft_bin_assign_triangular(coords[:, d], centers, params[d])
#         bin_centers.append(centers)
#         weights_per_dim.append(weights)
        

#     # Generate einsum subscripts dynamically
#     dims = [chr(105 + i) for i in range(D)]  # ['i', 'j', 'k', ...]
#     einsum_inputs = ','.join([f'n{d}' for d in dims])  # e.g. 'ni,nj,nk'
#     einsum_output = ''.join(dims)                       # e.g. 'ijk'
#     einsum_str = f'{einsum_inputs}->{einsum_output}'

#     hist = torch.einsum(einsum_str, *weights_per_dim)  # shape: bins[...]

#     if cdf:
#         hist = torch.cumsum(hist, dim=-1)

#     return hist, bin_centers
