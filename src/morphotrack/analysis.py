import torch
import numpy as np


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