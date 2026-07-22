import torch
import numpy as np


def tracks_to_vectors(points, track_ids, dense=True):
    """
    Per-track local displacement vectors from ordered track points.

    points:    (N, D) array of ordered points, concatenated over tracks
    track_ids: (N,) grouping label per point
    dense:     True -> every segment; False -> first and last segment only

    Returns (positions, vectors), each (M, D).
    """
    positions, vectors = [], []
    for i in np.unique(track_ids):
        p = points[track_ids == i]
        v = np.diff(p, axis=0)
        if dense:
            positions.append(p[1:])
            vectors.append(v)
        else:
            positions.append(p[[0, -2]])
            vectors.append(v[[0, -1]])
    return np.vstack(positions), np.vstack(vectors)


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
