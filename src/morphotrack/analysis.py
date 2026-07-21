import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import map_coordinates
import trimesh


def rk4_step(X, dt, v_field):
    # RK4 Integration
    k1 = v_field(X)
    k2 = v_field(X + 0.5 * dt * k1)
    k3 = v_field(X + 0.5 * dt * k2)
    k4 = v_field(X + dt * k3)
    return X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4(X0, v_field, steps=10, dt=0.1):
    X = X0.clone()
    traj = [X]
    for _ in range(steps):
        X = rk4_step(X, dt, v_field)
        traj.append(X)
    traj = torch.stack(traj, dim=0)
    return traj


def rk4_step_batched(X, dt, v_field):
    # X: (N, D)
    # dt: (N,) - step sizes for each point in the batch
    k1 = v_field(X)                         # (N, D)
    k2 = v_field(X + 0.5 * dt.unsqueeze(1) * k1)
    k3 = v_field(X + 0.5 * dt.unsqueeze(1) * k2)
    k4 = v_field(X + dt.unsqueeze(1) * k3)
    return X + (dt.unsqueeze(1) / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4_adaptive_dt(X0, v_field, dt_model, steps=10, return_dt_history=False):
    """
    Returns:
        - traj: (steps+1, N, D)
        - dt_history (optional): (steps, N)
    """
    X = X0.clone()
    traj = [X]
    dt_history = []

    for _ in range(steps):
        dt = dt_model(X).squeeze()  # shape: (N,)
        dt_history.append(dt)
        X = rk4_step_batched(X, dt, v_field)
        traj.append(X)

    traj = torch.stack(traj, dim=0)  # (steps+1, N, D)
    if return_dt_history:
        dt_history = torch.stack(dt_history, dim=0)  # (steps, N)
        return traj, dt_history
    
    return traj

# def rk4_step_batched2(X, dt, v_field):
#     # X: (B, N, D)
#     # dt: (B, N) - individual time steps for each point in each batch

#     # Flatten X to pass into v_field: (B*N, D)
#     B, N, D = X.shape
#     X_flat = X.view(B * N, D)

#     k1 = v_field(X_flat).view(B, N, D)
#     k2 = v_field((X + 0.5 * dt.unsqueeze(-1) * k1).view(B * N, D)).view(B, N, D)
#     k3 = v_field((X + 0.5 * dt.unsqueeze(-1) * k2).view(B * N, D)).view(B, N, D)
#     k4 = v_field((X + dt.unsqueeze(-1) * k3).view(B * N, D)).view(B, N, D)

#     delta = (dt.unsqueeze(-1) / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
#     return X + delta


# def integrate_rk4_adaptive_dt2(X0, v_field, dt_model, steps=10, return_dt_history=False):
#     """
#     Batched RK4 integration with adaptive dt per sample.

#     Args:
#         X0: (B, N, D)
#         v_field: velocity model
#         dt_model: model that returns dt, should output (B, N)
#         steps: number of integration steps
#         return_dt_history: whether to return history of dt values

#     Returns:
#         traj: (B, steps+1, N, D)
#         dt_history: (B, steps, N) if return_dt_history is True
#     """
#     B, N, D = X0.shape
#     X = X0.clone()
#     traj = [X]
#     dt_history = []

#     for _ in range(steps):
#         dt = dt_model(X).view(B, N)  # dt: (B, N)
#         dt_history.append(dt)
#         X = rk4_step_batched2(X, dt, v_field)
#         traj.append(X)

#     traj = torch.stack(traj, dim=1)  # (B, steps+1, N, D)

#     if return_dt_history:
#         dt_history = torch.stack(dt_history, dim=1)  # (B, steps, N)
#         return traj, dt_history

#     return traj


def integrate_rk4_with_scalar(X0, v_field, alpha_model, steps=10, dt=0.1):
    """
    Integrate the scaled velocity field: alpha(X) * v_field(X)

    Args:
        X0: (N, D) initial positions
        v_field: function mapping X -> v(X), shape (N, D)
        alpha_model: function mapping X -> alpha(X), shape (N,)
        steps: number of RK4 steps
        dt: base time step

    Returns:
        traj: list of (N, D) positions over time (len = steps + 1)
    """
    
    
    def scaled_v_field(X):
        alpha = alpha_model(X) # (N, 1)
        v = v_field(X)          # (N, D)
        return alpha * v  # (N, D)
    
    X = X0.clone()
    traj = [X]
    for _ in range(steps):
        X = rk4_step(X, dt, scaled_v_field)
        traj.append(X)

    traj = torch.stack(traj, dim=0)

    return traj


def integrate_rk4_until_time_with_fractional_steps(
    X0, v_field, dt_model, target_times, max_steps=100, return_dt_history=False
):
    """
    Integrate using RK4 until each point reaches its corresponding target_time.

    Args:
        X0: (N, D) starting positions
        v_field: callable (X) -> velocity (N, D)
        dt_model: callable (X) -> suggested time step (N,)
        target_times: (N,) target integration time per point
        max_steps: max number of RK4 steps
        return_dt_history: if True, return (T, N) dt history

    Returns:
        traj: (T+1, N, D) trajectory
        dt_history: (T, N) if return_dt_history=True
        step_counts: (N,) float, effective number of steps per point
    """
    X = X0.clone()
    N, D = X.shape
    device = X.device

    traj = [X]
    dt_history = []
    accum_time = torch.zeros(N, device=device)
    step_counts = torch.zeros(N, device=device)  # float step count
    active_mask = torch.ones(N, dtype=torch.bool, device=device)

    for _ in range(max_steps):
        if not active_mask.any():
            break

        # Raw model-suggested dt
        dt_raw = dt_model(X)  # (N,)
        dt_raw = torch.clamp(dt_raw, min=1e-8)  # avoid divide-by-zero later

        # Clip to avoid overshooting
        remaining_time = target_times - accum_time
        dt = torch.minimum(dt_raw, remaining_time.clamp(min=0.0))
        dt = torch.where(active_mask, dt, torch.zeros_like(dt))

        # RK4 step
        X_new = rk4_step_batched(X, dt, v_field)
        X = torch.where(active_mask.unsqueeze(1), X_new, X)

        # Update time
        accum_time += dt

        # Fractional step count
        step_counts += (dt / dt_raw) * active_mask.float()

        # Update mask
        active_mask = accum_time < target_times

        traj.append(X)
        if return_dt_history:
            dt_history.append(dt)

    traj = torch.stack(traj, dim=0)
    if return_dt_history:
        dt_history = torch.stack(dt_history, dim=0)
        return traj, dt_history, step_counts

    return traj, step_counts


def calculate_intersection(trajectory, mesh):
    # Build segments: start and end of each segment
    segment_starts = trajectory[:-1]
    segment_ends = trajectory[1:]

    # Compute directions and lengths
    directions = segment_ends - segment_starts
    lengths = np.linalg.norm(directions, axis=1)
    valid_mask = lengths > 1e-8  # avoid zero-length segments

    segment_starts = segment_starts[valid_mask]
    segment_ends = segment_ends[valid_mask]
    directions = directions[valid_mask]
    lengths = lengths[valid_mask]
    norm_directions = directions / lengths[:, None]

    # Store the original time index for each segment
    time_indices = np.nonzero(valid_mask)[0]  # This gives t for segment (t, t+1)

    # Use Trimesh ray intersection per segment
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=segment_starts,
        ray_directions=norm_directions,
        multiple_hits=False
    )

    # Find the first valid intersection (within segment bounds)
    first_hit = np.asarray([np.nan]*trajectory.shape[-1],dtype=trajectory.dtype)
    intersection_time = np.nan

    if len(locations) > 0:
        # Measure distance from segment start to intersection point
        hit_distances = np.linalg.norm(locations - segment_starts[index_ray], axis=1)
        within_segment = hit_distances <= lengths[index_ray]

        if np.any(within_segment):
            valid_locs = locations[within_segment]
            valid_indices = index_ray[within_segment]
            valid_dists = hit_distances[within_segment]
            valid_lengths = lengths[valid_indices]
            valid_times = time_indices[valid_indices]

            # Choose the first hit in trajectory order (lowest t)
            sort_order = valid_times.argsort()
            chosen = sort_order[0]

            first_hit = valid_locs[chosen]

            # Interpolate fractional time
            t0 = valid_times[chosen]
            alpha = valid_dists[chosen] / valid_lengths[chosen]  # 0 <= alpha <= 1
            intersection_time = t0 + alpha  # Fractional time step

    return first_hit, intersection_time


def analyze_trajectory_mesh_interaction(trajectory, mesh,
                                         distance_threshold=0.5,
                                         min_duration=3,
                                         gap_tolerance=2):
    """
    Independently detect both intersection and near-surface lingering events.
    Handles oscillating trajectories by merging nearby near-surface episodes.

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, 3)
    mesh : trimesh.Trimesh
    distance_threshold : float
        Distance below which a point is considered "near" the surface.
    min_duration : int
        Minimum total time steps (after merging) to qualify as lingering.
    gap_tolerance : int
        Maximum number of consecutive steps outside the threshold
        that are allowed before splitting into separate events.

    Returns
    -------
    result : dict
    """
    nan_point = np.full(trajectory.shape[-1], np.nan, dtype=trajectory.dtype)

    # --- 1. Intersection detection ---
    hit_point, hit_time = calculate_intersection(trajectory, mesh)
    has_intersection = not np.isnan(hit_time)

    # --- 2. Proximity analysis ---
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(mesh, trajectory)
    near_mask = distances < distance_threshold

    # --- 3. Find raw runs, merge, then filter by duration ---
    raw_runs = _find_consecutive_runs(near_mask, min_length=1)
    merged_runs = _merge_nearby_runs(raw_runs, gap_tolerance=gap_tolerance)
    # Apply min_duration after merging, using the full span
    merged_runs = [(s, e) for s, e in merged_runs if (e - s + 1) >= min_duration]

    # --- 4. Build event info ---
    near_events = []
    for start, end in merged_runs:
        # Use ALL points in the merged window, including brief excursions
        window_distances = distances[start:end + 1]
        window_near_mask = near_mask[start:end + 1]

        local_idx = np.argmin(window_distances)
        global_idx = start + local_idx

        # Count how many steps are actually within threshold
        steps_near = int(np.sum(window_near_mask))
        # Fraction of time spent near surface
        nearness_ratio = steps_near / len(window_distances)

        contains_intersection = False
        if has_intersection:
            contains_intersection = (start <= hit_time <= end + 1)

        near_events.append({
            "start": start,
            "end": end,
            "total_duration": end - start + 1,
            "steps_near_surface": steps_near,
            "nearness_ratio": nearness_ratio,
            "closest_approach": {
                "point": closest_points[global_idx],
                "trajectory_point": trajectory[global_idx],
                "distance": float(distances[global_idx]),
                "time": global_idx
            },
            "mean_distance": float(np.mean(window_distances)),
            "mean_distance_when_near": float(np.mean(window_distances[window_near_mask]))
                                       if steps_near > 0 else np.nan,
            "contains_intersection": contains_intersection
        })

    return {
        "intersection": {
            "point": hit_point if has_intersection else nan_point,
            "time": hit_time,
            "found": has_intersection
        },
        "near_surface_events": near_events
    }


def _find_consecutive_runs(mask, min_length=1):
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= min_length:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _merge_nearby_runs(runs, gap_tolerance=2):
    if len(runs) == 0:
        return runs
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_tolerance + 1:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(r) for r in merged]



def inverse_transform(points_in_space2, disp_field, num_iters=10, tol=1e-3):
    """
    Estimate positions in space 1 corresponding to given points in space 2,
    by inverting the forward displacement field using iterative refinement.
    
    Parameters:
    - points_in_space2: (N, 3) numpy array of points in space 2
    - disp_field: (Z, Y, X, 3) numpy array of displacement field from space 1 to space 2
    - num_iters: number of iterations to perform
    - tol: convergence threshold (optional)
    
    Returns:
    - points_in_space1: (N, 3) numpy array of estimated positions in space 1
    """
    # Initialize with space2 points as guess
    points_est = points_in_space2.copy()
    
    for _ in range(num_iters):
        disp_interp = np.vstack([
            map_coordinates(disp_field[..., i], points_est.T, order=1, mode='nearest')
            for i in range(3)
        ]).T
        
        new_points_est = points_in_space2 - disp_interp
        
        # Check convergence
        if np.linalg.norm(new_points_est - points_est) < tol:
            break
            
        points_est = new_points_est
    
    return points_est


def second_partial_derivative(X: torch.Tensor, dim1: int, dim2: int, dx1: float, dx2: float) -> torch.Tensor:
    """
    Numerically approximates the second partial derivative ∂²X / ∂(dim1)∂(dim2)
    using central differences.

    Args:
        X: Input tensor of at least 3D (e.g., shape (U, V, T))
        dim1: First dimension to differentiate with respect to
        dim2: Second dimension to differentiate with respect to
        dx1: Grid spacing along dim1
        dx2: Grid spacing along dim2

    Returns:
        A tensor of the same dtype as X, but smaller along the specified dims
    """

    if dim1 > dim2:
        # Ensure dim1 <= dim2 for consistent slicing
        dim1, dim2 = dim2, dim1
        dx1, dx2 = dx2, dx1

    if dim1 == dim2:
        # Pure second derivative: ∂²X / ∂(dim)^2
        slice_before = [slice(None)] * X.ndim
        slice_center = [slice(None)] * X.ndim
        slice_after = [slice(None)] * X.ndim

        slice_before[dim1] = slice(0, -2)
        slice_center[dim1] = slice(1, -1)
        slice_after[dim1] = slice(2, None)

        X_before = X[slice_before]
        X_center = X[slice_center]
        X_after = X[slice_after]

        return (X_before - 2 * X_center + X_after) / (dx1 ** 2)

    else:
        # Mixed second derivative: ∂²X / ∂(dim1)∂(dim2)
        s11 = [slice(None)] * X.ndim
        s12 = [slice(None)] * X.ndim
        s21 = [slice(None)] * X.ndim
        s22 = [slice(None)] * X.ndim

        s11[dim1] = slice(2, None)     # +1 along dim1
        s11[dim2] = slice(2, None)     # +1 along dim2

        s12[dim1] = slice(2, None)     # +1 along dim1
        s12[dim2] = slice(0, -2)       # -1 along dim2

        s21[dim1] = slice(0, -2)       # -1 along dim1
        s21[dim2] = slice(2, None)     # +1 along dim2

        s22[dim1] = slice(0, -2)       # -1 along dim1
        s22[dim2] = slice(0, -2)       # -1 along dim2

        return (
            X[s11] - X[s12] - X[s21] + X[s22]
        ) / (4 * dx1 * dx2)
    
    
def first_partial_derivative(X: torch.Tensor, dim: int, dx: float) -> torch.Tensor:
    """
    Numerically approximates the first partial derivative ∂X / ∂(dim)
    using central differences.

    Args:
        X: Input tensor of at least 1D
        dim: Dimension to differentiate along
        dx: Grid spacing along that dimension

    Returns:
        A tensor of the same dtype as X, with size reduced by 2 along `dim`
    """

    slice_before = [slice(None)] * X.ndim
    slice_after = [slice(None)] * X.ndim

    slice_before[dim] = slice(0, -2)
    slice_after[dim] = slice(2, None)

    X_before = X[slice_before]
    X_after = X[slice_after]

    return (X_after - X_before) / (2 * dx)


def compute_spatial_gradient(pos_field):
    # pos_field: (B, 3, D, H, W) — absolute position map
    dx = pos_field[..., 1:, :, :] - pos_field[..., :-1, :, :]
    dy = pos_field[..., :, 1:, :] - pos_field[..., :, :-1, :]
    dz = pos_field[..., :, :, 1:] - pos_field[..., :, :, :-1]
    
    # Pad to keep original size
    dx = F.pad(dx, (0, 0, 0, 0, 0, 1))  # pad D
    dy = F.pad(dy, (0, 0, 0, 1))        # pad H
    dz = F.pad(dz, (0, 1))              # pad W
    
    # Form gradient tensor: (B, 3, 3, D, H, W)
    # For each point, we get a 3×3 Jacobian matrix
    gradient = torch.stack([dx, dy, dz], dim=2)  # shape: (B, 3, 3, D, H, W)
    return gradient


def warp_vector_field(vector_field, deformation_gradient):
    # vector_field: (B, 3, D, H, W)
    # deformation_gradient: (B, 3, 3, D, H, W)
    
    # Do matrix multiplication per voxel
    # Resulting vector = F(x) @ v(x)
    B, _, D, H, W = vector_field.shape
    v = vector_field.view(B, 3, 1, D, H, W)
    
    v_warped = torch.matmul(deformation_gradient, v)  # shape: (B, 3, 1, D, H, W)
    return v_warped.view(B, 3, D, H, W)

def denormalize_grid(grid, shape):
    """
    Convert grid from [-1, 1] to voxel coordinates.
    grid: (B, 3, D, H, W)
    shape: (D, H, W)
    """
    D, H, W = shape
    scale = torch.tensor([(D - 1) / 2, (H - 1) / 2, (W - 1) / 2], device=grid.device).view(1, 3, 1, 1, 1)
    shift = torch.tensor([1.0, 1.0, 1.0], device=grid.device).view(1, 3, 1, 1, 1)
    
    return (grid + shift) * scale

def normalize_coords(points, D, H, W):
    """
    Convert voxel indices (z,y,x) to normalized coordinates in [-1,1]
    """
    z = points[:, 0] / (D - 1) * 2 - 1
    y = points[:, 1] / (H - 1) * 2 - 1
    x = points[:, 2] / (W - 1) * 2 - 1
    return np.stack([z, y, x], axis=-1)

def denormalize_coords(points_norm, D, H, W):
    """
    Convert normalized coordinates [-1,1] to voxel indices (z,y,x)
    """
    z = ((points_norm[:, 0] + 1) / 2) * (D - 1)
    y = ((points_norm[:, 1] + 1) / 2) * (H - 1)
    x = ((points_norm[:, 2] + 1) / 2) * (W - 1)
    return np.stack([z, y, x], axis=-1)



# def integrate_rk4_batched(X0, v_field, dt_list):
#     """
#     X0: (N, D) tensor of initial positions
#     v_field: function taking (N, D) and returning (N, D)
#     dt_list: (N, T) tensor of step sizes for each point (sum along dim=1 == 1.0)
    
#     Returns: (T+1, N, D) trajectory for each point over T steps
#     """
#     N, D = X0.shape
#     T = dt_list.shape[1]

#     X = X0.clone()
#     traj = [X]

#     for t in range(T):
#         dt = dt_list[:, t]          # shape: (N,)
#         X = rk4_step_batched(X, dt, v_field)
#         traj.append(X)

#     return torch.stack(traj, dim=0)  # shape: (T+1, N, D)