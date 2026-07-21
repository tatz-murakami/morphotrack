import torch


def divergence_vhat(vhat_model, X, create_graph=False):
    """
    κ(x) = div( v̂ )(x) for batch X (N,3).
    Builds just enough graph to take ∂v̂/∂x; returns a detached tensor.
    """
    with torch.enable_grad():
        X = X.detach().requires_grad_(True)
        V = vhat_model(X)          # (N,3)
        # Take trace of Jacobian ∂V/∂X
        kappa_terms = []
        for i in range(3):
            g = torch.autograd.grad(
                V[:, i].sum(), X,
                retain_graph=(i < 2),            # False on last component
                create_graph=create_graph,
                allow_unused=False
            )[0]                                  # (N,3)
            kappa_terms.append(g[:, i])          # ∂V_i/∂x_i
        kappa = (kappa_terms[0] + kappa_terms[1] + kappa_terms[2]).detach()
    return kappa

def divergence_vhat_chunked(vhat_model, X, chunk_size=10000):
    out = torch.empty(X.shape[0], device=X.device)
    for i in range(0, X.shape[0], chunk_size):
        out[i:i+chunk_size] = divergence_vhat(vhat_model, X[i:i+chunk_size])
    return out


def rk4_step_vec(X, dt_vec, v_field):  # dt_vec: (N,1)
    k1 = v_field(X)
    k2 = v_field(X + 0.5 * dt_vec * k1)
    k3 = v_field(X + 0.5 * dt_vec * k2)
    k4 = v_field(X + dt_vec * k3)
    return X + (dt_vec / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@torch.no_grad()
def compute_X_from_c(X0, vhat_model, c_t, t_step=0.001):
    """
    Given c(t) on a uniform t grid, integrate  dx/dt = c(t) * v̂(x)
    to get X(t).

    Parameters:
        X0:         (N, 3) seed positions at t=0
        vhat_model: unit vector field, callable X -> v̂(X)
        c_t:        (T+1, N) speed at each t step
        t_step:     t spacing

    Returns:
        X_t:        (T+1, N, 3) positions at each t step
    """
    T1, N = c_t.shape
    device = X0.device
    X_t = torch.empty(T1, N, 3, device=device)
    X = X0.clone()
    X_t[0] = X
    dt = t_step
    for k in range(T1 - 1):
        c = c_t[k].unsqueeze(-1)  # (N, 1)
        # RK4 in t-space: dx/dt = c(t) * v̂(x)
        # Using c at current t for all RK4 substeps (no midpoint interpolation)
        k1 = c * vhat_model(X)
        k2 = c * vhat_model(X + 0.5 * dt * k1)
        k3 = c * vhat_model(X + 0.5 * dt * k2)
        k4 = c * vhat_model(X + dt * k3)
        X = X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        X_t[k + 1] = X
    return X_t



"""
Solve   div(c * v̂) = 0   along streamlines of a unit vector field v̂,
with the unit-transit-time normalization  ∫_0^L ds/c(s) = 1.
 
Math (b=0 case):
    dc/ds = -(∇·v̂) c        ⇒   c(s) = c(0) / J(s),   J(s) = exp(∫_0^s ∇·v̂ ds')
    1 = ∫_0^L ds/c           ⇒   c(0) = ∫_0^L J(s) ds  =: c0
    t(s) = ∫_0^s J(s')ds' / c0       (transit time, ∈ [0,1])
 
The `b` knob softens the exponent: J = exp(logJ / (1+b)).  b=0 is the
"true" case; b is retained in the code for now and will be removed later.
"""
 
 
@torch.no_grad()
def integrate_s(X0, vhat_model, L, M_s=2000, b=0.0):
    """
    Integrate along v̂ in arc-length s, storing full history.
 
    Args:
        X0:         (N, 3) entry points, one per streamline
        vhat_model: callable, X -> unit vector field v̂(X)
        L:          (N,) or scalar, total arc length per streamline
        M_s:        number of arc-length steps
        b:          exponent softener (b=0 is the math case)
 
    Returns:
        X_hist:    (M_s+1, N, 3)  positions along each streamline
        logJ_hist: (M_s+1, N)     ∫_0^s ∇·v̂ ds'  (= log of flow Jacobian)
        cumJ_hist: (M_s+1, N)     ∫_0^s J ds'    (cumulative volume factor)
        c0:        (N,)           entry speed = ∫_0^L J ds = cumJ_hist[-1]
    """
    device = X0.device
    N = X0.shape[0]
    L = L.to(device) if torch.is_tensor(L) else torch.tensor(L, device=device).expand(N)
    ds = (L / M_s).view(N, 1)
 
    X = X0.clone()
    logJ_hist = torch.empty(M_s + 1, N, device=device)
    cumJ_hist = torch.empty(M_s + 1, N, device=device)
    X_hist    = torch.empty(M_s + 1, N, 3, device=device)
    logJ_hist[0] = 0.0
    cumJ_hist[0] = 0.0
    X_hist[0]    = X
 
    logJ_prev = torch.zeros(N, device=device)
    cumJ_prev = torch.zeros(N, device=device)
    div_prev  = divergence_vhat(vhat_model, X)
    J_prev    = torch.ones(N, device=device)
 
    for m in range(M_s):
        X_next    = rk4_step_vec(X, ds, lambda Y: vhat_model(Y))
        div_next  = divergence_vhat(vhat_model, X_next)
 
        # logJ <- ∫ ∇·v̂ ds  (trapezoidal)
        dlogJ     = 0.5 * (div_prev + div_next) * ds.view(-1)
        logJ_next = logJ_prev + dlogJ
        J_next    = torch.exp(logJ_next / (b + 1))
 
        # cumJ <- ∫ J ds  (trapezoidal)
        dcumJ     = 0.5 * (J_prev + J_next) * ds.view(-1)
        cumJ_next = cumJ_prev + dcumJ
 
        logJ_hist[m + 1] = logJ_next
        cumJ_hist[m + 1] = cumJ_next
        X_hist[m + 1]    = X_next
 
        X = X_next
        logJ_prev, cumJ_prev = logJ_next, cumJ_next
        div_prev,  J_prev    = div_next,  J_next
 
    c0 = cumJ_prev  # entry speed: c(0) = ∫_0^L J ds
    return X_hist, logJ_hist, cumJ_hist, c0
 
 
def resample_to_uniform_t(X_hist, logJ_hist, cumJ_hist, c0, L,
                          t_step=0.01, to_cpu=False, b=0.0):
    """
    Resample arc-length history to a uniform transit-time grid t ∈ [0, 1].
 
    Inverts t(s) = cumJ(s) / c0  to find s for each target t, then evaluates
    c(s) = c0 / J(s) = c0 * exp(-logJ(s) / (1+b)).
 
    Returns:
        X_t: (T+1, N, 3)  positions at uniform t
        c_t: (T+1, N)     speed c(t)
        s_t: (T+1, N)     arc length corresponding to each t
    """
    device = c0.device
    M_s = logJ_hist.shape[0] - 1
    N = c0.shape[0]
    L = L.to(device) if torch.is_tensor(L) else torch.tensor(L, device=device).expand(N)
    ds = (L / M_s).view(-1)
 
    T = int(round(1.0 / t_step))
    t_grid = torch.linspace(0.0, 1.0, T + 1, device=device)
 
    X_t = torch.empty(T + 1, N, 3, device=device)
    c_t = torch.empty(T + 1, N, device=device)
    s_t = torch.empty(T + 1, N, device=device)
 
    # t = 0 (entry)
    X_t[0] = X_hist[0]
    c_t[0] = c0
    s_t[0] = 0.0
 
    # t = 1 (exit, exact)
    X_t[T] = X_hist[-1]
    c_t[T] = torch.exp(-logJ_hist[-1] / (b + 1)) * c0
    s_t[T] = L
 
    n_idx = torch.arange(N, device=device)
    for k in range(1, T):
        cumJ_target = t_grid[k] * c0  # invert t = cumJ / c0
        idx = torch.searchsorted(cumJ_hist.T.contiguous(),
                                 cumJ_target.unsqueeze(-1)).squeeze(-1)
        idx = idx.clamp(1, M_s)
        lo  = idx - 1
 
        cumJ_lo = cumJ_hist[lo,  n_idx]
        cumJ_hi = cumJ_hist[idx, n_idx]
        w = (cumJ_target - cumJ_lo) / (cumJ_hi - cumJ_lo).clamp_min(1e-12)
 
        logJ_interp = (1 - w) * logJ_hist[lo, n_idx] + w * logJ_hist[idx, n_idx]
        X_interp    = (1 - w.unsqueeze(-1)) * X_hist[lo, n_idx] \
                    +      w.unsqueeze(-1)  * X_hist[idx, n_idx]
        s_interp    = (lo.float() + w) * ds
 
        X_t[k] = X_interp
        c_t[k] = torch.exp(-logJ_interp / (b + 1)) * c0
        s_t[k] = s_interp
 
    if to_cpu:
        X_t = X_t.cpu()
        c_t = c_t.cpu()
        s_t = s_t.cpu()
    return X_t, c_t, s_t

# @torch.no_grad()
# def compute_Q_end(X0, vhat_model, tau_end, M_tau=2000, b=0.0):
#     """
#     Returns Q_end (N,), where Q(τ) = ∫_0^τ exp(K/(b+1)) dτ, K(τ)=∫_0^τ κ dτ.

#     b is the exponent in the DRG density assumption σ = C·α^b
#     (σ = 3D fiber segment density).
#     b=0 (default) gives equivolume reparameterization (∇·v = 0, σ = const).
#     """
#     device = X0.device
#     X = X0.clone()
#     N = X.shape[0]


#     tau_end = tau_end.to(device) if torch.is_tensor(tau_end) else torch.tensor(tau_end, device=device).expand(N)
#     dtaus = (tau_end / M_tau).view(N,1)                      # (N,1)

#     # Initialize K(0)=0, Q(0)=0
#     K_prev = torch.zeros(N, device=device)
#     Q_prev = torch.zeros(N, device=device)

#     # κ at τ=0
#     kappa_prev = divergence_vhat(vhat_model, X)              # (N,)
#     expK_prev = torch.ones(N, device=device)

#     for _ in range(M_tau):
#         # Step along unit field
#         X_next = rk4_step_vec(X, dtaus, lambda Y: vhat_model(Y))

#         # κ at new locations
#         kappa_next = divergence_vhat(vhat_model, X_next)

#         # Trapezoid updates
#         dK = 0.5 * (kappa_prev + kappa_next) * dtaus.view(-1)    # (N,)
#         K_next = K_prev + dK

#         expK_next = torch.exp(K_next / (b + 1))
#         dQ = 0.5 * (expK_prev + expK_next) * dtaus.view(-1)
#         Q_next = Q_prev + dQ

#         # Roll
#         X = X_next
#         K_prev, Q_prev = K_next, Q_next
#         kappa_prev = kappa_next
#         expK_prev = expK_next

#     return Q_prev   # (N,)


# @torch.no_grad()
# def integrate_and_resample_uniform_t(
#     X0, vhat_model, tau_end, Q_end,
#     t_step=0.01, M_tau=2000, to_cpu=False, b=0.0
# ):
#     """
#     Returns:
#       X_t:     (T+1, N, 3) positions at t = 0, t_step, ..., 1
#       alpha_t: (T+1, N) magnitudes α at the same times
#       tau_t:   (T+1, N) arc length τ at the same times
#     """
#     device = X0.device
#     X = X0.clone()
#     N = X.shape[0]

#     tau_end = tau_end.to(device) if torch.is_tensor(tau_end) else torch.tensor(tau_end, device=device).expand(N)
#     dtaus = (tau_end / M_tau).view(N, 1)

#     # Uniform t-grid
#     T = int(round(1.0 / t_step))
#     t_grid = torch.linspace(0.0, 1.0, T + 1, device=device)

#     # Outputs
#     X_t = torch.empty(T + 1, N, 3, device=device)
#     alpha_t = torch.empty(T + 1, N, device=device)
#     tau_t = torch.empty(T + 1, N, device=device)

#     # Initialize
#     K_prev = torch.zeros(N, device=device)
#     Q_prev = torch.zeros(N, device=device)
#     tau_prev = torch.zeros(N, device=device)
#     alpha_prev = Q_end.clone()

#     # At t=0:
#     X_t[0] = X if not to_cpu else X.cpu()
#     alpha_t[0] = alpha_prev if not to_cpu else alpha_prev.cpu()
#     tau_t[0] = 0.0

#     # Next sample index per fiber
#     k_idx = torch.ones(N, dtype=torch.long, device=device)
#     Q_thresh = t_grid[k_idx] * Q_end

#     # κ at τ=0
#     kappa_prev = divergence_vhat(vhat_model, X)
#     expK_prev = torch.ones(N, device=device)

#     for _ in range(M_tau):
#         X_next = rk4_step_vec(X, dtaus, lambda Y: vhat_model(Y))
#         kappa_next = divergence_vhat(vhat_model, X_next)

#         dK = 0.5 * (kappa_prev + kappa_next) * dtaus.view(-1)
#         K_next = K_prev + dK

#         expK_next = torch.exp(K_next / (b + 1))
#         dQ = 0.5 * (expK_prev + expK_next) * dtaus.view(-1)
#         Q_next = Q_prev + dQ

#         alpha_next = torch.exp(-K_next / (b + 1)) * Q_end
#         tau_next = tau_prev + dtaus.view(-1)

#         # Emit samples whose Q-threshold is crossed
#         while True:
#             mask = (k_idx <= T) & (Q_prev < Q_thresh) & (Q_next >= Q_thresh)
#             if not mask.any():
#                 break

#             w = ((Q_thresh[mask] - Q_prev[mask]) / (Q_next[mask] - Q_prev[mask]).clamp_min(1e-12)).unsqueeze(-1)

#             X_interp = (1 - w) * X[mask] + w * X_next[mask]
#             K_interp = (1 - w.squeeze(-1)) * K_prev[mask] + w.squeeze(-1) * K_next[mask]
#             alpha_interp = torch.exp(-K_interp / (b + 1)) * Q_end[mask]
#             tau_interp = (1 - w.squeeze(-1)) * tau_prev[mask] + w.squeeze(-1) * tau_next[mask]

#             kk = k_idx[mask]
#             row_idx = torch.arange(N, device=device)[mask]
#             if to_cpu:
#                 X_t[kk, row_idx.cpu(), :] = X_interp.cpu()
#                 alpha_t[kk, row_idx.cpu()] = alpha_interp.cpu()
#                 tau_t[kk, row_idx.cpu()] = tau_interp.cpu()
#             else:
#                 X_t[kk, row_idx, :] = X_interp
#                 alpha_t[kk, row_idx] = alpha_interp
#                 tau_t[kk, row_idx] = tau_interp

#             k_idx[mask] += 1
#             active = (k_idx <= T)
#             Q_thresh[active] = t_grid[k_idx[active]] * Q_end[active]

#         # Roll
#         X = X_next
#         K_prev, Q_prev = K_next, Q_next
#         tau_prev = tau_next
#         alpha_prev = alpha_next
#         kappa_prev = kappa_next
#         expK_prev = expK_next

#     # Ensure last sample t=1 is filled
#     if (k_idx <= T).any():
#         m = (k_idx <= T)
#         if to_cpu:
#             X_t[T, m.cpu(), :] = X[m].cpu()
#             alpha_t[T, m.cpu()] = alpha_prev[m].cpu()
#             tau_t[T, m.cpu()] = tau_prev[m].cpu()
#         else:
#             X_t[T, m, :] = X[m]
#             alpha_t[T, m] = alpha_prev[m]
#             tau_t[T, m] = tau_prev[m]

#     if to_cpu:
#         X_t = X_t.cpu()
#         alpha_t = alpha_t.cpu()
#         tau_t = tau_t.cpu()
#     return X_t, alpha_t, tau_t


# @torch.no_grad()
# def integrate_tau(X0, vhat_model, tau_end, M_tau=2000, b=0.0):
#     """
#     Integrate along v̂ in τ-space, storing full history.

#     Returns:
#       X_hist: (M_tau+1, N, 3)
#       K_hist: (M_tau+1, N)
#       Q_hist: (M_tau+1, N)
#       Q_end:  (N,)
#     """
#     device = X0.device
#     N = X0.shape[0]

#     tau_end = tau_end.to(device) if torch.is_tensor(tau_end)@torch.no_grad()
# def compute_Q_end(X0, vhat_model, tau_end, M_tau=2000, b=0.0):
#     """
#     Returns Q_end (N,), where Q(τ) = ∫_0^τ exp(K/(b+1)) dτ, K(τ)=∫_0^τ κ dτ.

#     b is the exponent in the DRG density assumption σ = C·α^b
#     (σ = 3D fiber segment density).
#     b=0 (default) gives equivolume reparameterization (∇·v = 0, σ = const).
#     """
#     device = X0.device
#     X = X0.clone()
#     N = X.shape[0]


#     tau_end = tau_end.to(device) if torch.is_tensor(tau_end) else torch.tensor(tau_end, device=device).expand(N)
#     dtaus = (tau_end / M_tau).view(N,1)                      # (N,1)

#     # Initialize K(0)=0, Q(0)=0
#     K_prev = torch.zeros(N, device=device)
#     Q_prev = torch.zeros(N, device=device)

#     # κ at τ=0
#     kappa_prev = divergence_vhat(vhat_model, X)              # (N,)
#     expK_prev = torch.ones(N, device=device)

#     for _ in range(M_tau):
#         # Step along unit field
#         X_next = rk4_step_vec(X, dtaus, lambda Y: vhat_model(Y))

#         # κ at new locations
#         kappa_next = divergence_vhat(vhat_model, X_next)

#         # Trapezoid updates
#         dK = 0.5 * (kappa_prev + kappa_next) * dtaus.view(-1)    # (N,)
#         K_next = K_prev + dK

#         expK_next = torch.exp(K_next / (b + 1))
#         dQ = 0.5 * (expK_prev + expK_next) * dtaus.view(-1)
#         Q_next = Q_prev + dQ

#         # Roll
#         X = X_next
#         K_prev, Q_prev = K_next, Q_next
#         kappa_prev = kappa_next
#         expK_prev = expK_next

#     return Q_prev   # (N,)


# @torch.no_grad()
# def integrate_and_resample_uniform_t(
#     X0, vhat_model, tau_end, Q_end,
#     t_step=0.01, M_tau=2000, to_cpu=False, b=0.0
# ):
#     """
#     Returns:
#       X_t:     (T+1, N, 3) positions at t = 0, t_step, ..., 1
#       alpha_t: (T+1, N) magnitudes α at the same times
#       tau_t:   (T+1, N) arc length τ at the same times
#     """
#     device = X0.device
#     X = X0.clone()
#     N = X.shape[0]

#     tau_end = tau_end.to(device) if torch.is_tensor(tau_end) else torch.tensor(tau_end, device=device).expand(N)
#     dtaus = (tau_end / M_tau).view(N, 1)

#     # Uniform t-grid
#     T = int(round(1.0 / t_step))
#     t_grid = torch.linspace(0.0, 1.0, T + 1, device=device)

#     # Outputs
#     X_t = torch.empty(T + 1, N, 3, device=device)
#     alpha_t = torch.empty(T + 1, N, device=device)
#     tau_t = torch.empty(T + 1, N, device=device)

#     # Initialize
#     K_prev = torch.zeros(N, device=device)
#     Q_prev = torch.zeros(N, device=device)
#     tau_prev = torch.zeros(N, device=device)
#     alpha_prev = Q_end.clone()

#     # At t=0:
#     X_t[0] = X if not to_cpu else X.cpu()
#     alpha_t[0] = alpha_prev if not to_cpu else alpha_prev.cpu()
#     tau_t[0] = 0.0

#     # Next sample index per fiber
#     k_idx = torch.ones(N, dtype=torch.long, device=device)
#     Q_thresh = t_grid[k_idx] * Q_end

#     # κ at τ=0
#     kappa_prev = divergence_vhat(vhat_model, X)
#     expK_prev = torch.ones(N, device=device)

#     for _ in range(M_tau):
#         X_next = rk4_step_vec(X, dtaus, lambda Y: vhat_model(Y))
#         kappa_next = divergence_vhat(vhat_model, X_next)

#         dK = 0.5 * (kappa_prev + kappa_next) * dtaus.view(-1)
#         K_next = K_prev + dK

#         expK_next = torch.exp(K_next / (b + 1))
#         dQ = 0.5 * (expK_prev + expK_next) * dtaus.view(-1)
#         Q_next = Q_prev + dQ

#         alpha_next = torch.exp(-K_next / (b + 1)) * Q_end
#         tau_next = tau_prev + dtaus.view(-1)

#         # Emit samples whose Q-threshold is crossed
#         while True:
#             mask = (k_idx <= T) & (Q_prev < Q_thresh) & (Q_next >= Q_thresh)
#             if not mask.any():
#                 break

#             w = ((Q_thresh[mask] - Q_prev[mask]) / (Q_next[mask] - Q_prev[mask]).clamp_min(1e-12)).unsqueeze(-1)

#             X_interp = (1 - w) * X[mask] + w * X_next[mask]
#             K_interp = (1 - w.squeeze(-1)) * K_prev[mask] + w.squeeze(-1) * K_next[mask]
#             alpha_interp = torch.exp(-K_interp / (b + 1)) * Q_end[mask]
#             tau_interp = (1 - w.squeeze(-1)) * tau_prev[mask] + w.squeeze(-1) * tau_next[mask]

#             kk = k_idx[mask]
#             row_idx = torch.arange(N, device=device)[mask]
#             if to_cpu:
#                 X_t[kk, row_idx.cpu(), :] = X_interp.cpu()
#                 alpha_t[kk, row_idx.cpu()] = alpha_interp.cpu()
#                 tau_t[kk, row_idx.cpu()] = tau_interp.cpu()
#             else:
#                 X_t[kk, row_idx, :] = X_interp
#                 alpha_t[kk, row_idx] = alpha_interp
#                 tau_t[kk, row_idx] = tau_interp

#             k_idx[mask] += 1
#             active = (k_idx <= T)
#             Q_thresh[active] = t_grid[k_idx[active]] * Q_end[active]

#         # Roll
#         X = X_next
#         K_prev, Q_prev = K_next, Q_next
#         tau_prev = tau_next
#         alpha_prev = alpha_next
#         kappa_prev = kappa_next
#         expK_prev = expK_next

#     # Ensure last sample t=1 is filled
#     if (k_idx <= T).any():
#         m = (k_idx <= T)
#         if to_cpu:
#             X_t[T, m.cpu(), :] = X[m].cpu()
#             alpha_t[T, m.cpu()] = alpha_prev[m].cpu()
#             tau_t[T, m.cpu()] = tau_prev[m].cpu()
#         else:
#             X_t[T, m, :] = X[m]
#             alpha_t[T, m] = alpha_prev[m]
#             tau_t[T, m] = tau_prev[m]

#     if to_cpu:
#         X_t = X_t.cpu()
#         alpha_t = alpha_t.cpu()
#         tau_t = tau_t.cpu()
#     return X_t, alpha_t, tau_t