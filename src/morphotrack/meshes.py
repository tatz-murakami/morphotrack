import trimesh
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS


def build_weighted_mesh_graph(vertices, faces, values, exponent=0.5):
    """
    Constructs a sparse graph from mesh connectivity with
    value-weighted edge lengths for equivolume embedding.

    Parameters:
        vertices: (n, 3) array of vertex coordinates
        faces: (m, 3) array of triangle indices
        values: (n,) array of per-vertex values (e.g., alpha)
        exponent: power applied to values (0.5 for equivolume)

    Returns:
        graph: sparse (n x n) matrix with weighted edge distances
    """
    n_vertices = vertices.shape[0]
    edge_set = set()
    for tri in faces:
        i, j, k = tri
        edge_set.update({tuple(sorted((i, j))), tuple(sorted((j, k))), tuple(sorted((k, i)))})

    rows, cols, weights = [], [], []
    for i, j in edge_set:
        eu_dist = np.linalg.norm(vertices[i] - vertices[j])
        avg_value = 0.5 * (values[i] + values[j])
        weight = eu_dist * (avg_value ** exponent)
        rows.append(i)
        cols.append(j)
        weights.append(weight)

    graph = sp.csr_matrix((weights + weights, (rows + cols, cols + rows)), shape=(n_vertices, n_vertices))
    return graph


def embed_mesh_isomap(vertices, faces, values, exponent=0.5, n_components=2,
                      align_vertices=None, align_direction=None, flip_v=False, max_iter=300, eps=1e-06):
    """
    Embeds a mesh into 2D using value-weighted Isomap.

    Parameters:
        vertices: (n, 3) vertex positions
        faces: (m, 3) triangle indices
        values: (n,) per-vertex values (e.g., alpha)
        exponent: power applied to values (0.5 for equivolume)
        n_components: embedding dimension
        align_vertices: optional tuple of two vertex indices (i, j)
        align_direction: optional (2,) array, desired UV direction from i to j
        flip_v: if True, negate the v axis

    Returns:
        embedding: (n, 2) 2D embedding
    """
    graph = build_weighted_mesh_graph(vertices, faces, values, exponent=exponent)
    dist_matrix = shortest_path(csgraph=graph, directed=False)
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, max_iter=max_iter, eps=eps)
    embedding = mds.fit_transform(dist_matrix)

    if flip_v:
        embedding[:, 1] = -embedding[:, 1]

    # --- Rotate to align with desired direction ---
    if align_vertices is not None and align_direction is not None:
        i, j = align_vertices
        current = embedding[j] - embedding[i]
        target = np.array(align_direction, dtype=float)

        angle_current = np.arctan2(current[1], current[0])
        angle_target = np.arctan2(target[1], target[0])
        theta = angle_target - angle_current

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])

        centroid = embedding.mean(axis=0)
        embedding = (embedding - centroid) @ R.T + centroid

    return embedding


def filter_vertices_by_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh:
    """
    Return a new mesh with only the vertices where vertex_mask is True.
    Faces with any removed vertex are also removed.

    Parameters:
    - mesh: trimesh.Trimesh
        The original mesh.
    - vertex_mask: np.ndarray of shape (n_vertices,)
        Boolean array where True means keep the vertex.
W
    Returns:
    - trimesh.Trimesh
        New mesh with filtered vertices and faces.
    """
    if vertex_mask.shape[0] != mesh.vertices.shape[0]:
        raise ValueError("vertex_mask must have the same length as the number of vertices in the mesh.")
    
    # Map from old vertex indices to new ones
    old_to_new_index = -np.ones(len(vertex_mask), dtype=int)
    old_to_new_index[vertex_mask] = np.arange(np.sum(vertex_mask))

    # Keep only the vertices where mask is True
    new_vertices = mesh.vertices[vertex_mask]

    # Keep only the faces where all 3 vertices are in the mask
    face_mask = vertex_mask[mesh.faces].all(axis=1)
    new_faces = mesh.faces[face_mask]

    # Remap face indices to the new vertex array
    new_faces = old_to_new_index[new_faces]

    # Create and return new mesh
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)


def interpolate_mesh_values(vertices, faces, values, known_mask):
    """
    Interpolate/extrapolate values on a mesh via harmonic (Laplace) extension.

    Parameters
    ----------
    vertices : (N, 3) array — vertex positions
    faces    : (M, 3) array — triangle indices
    values   : (N,) or (N, K) array — values (only entries where known_mask=True matter)
    known_mask : (N,) bool array — True where value is known

    Returns
    -------
    result : (N,) or (N, K) array — values with unknowns filled in
    """
    n = len(vertices)

    # --- 1. Build the cotangent-weight Laplacian (or use uniform weights) ---
    # Uniform weights are simpler and work well for most cases.
    row, col, data = [], [], []

    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    row.append(f[i])
                    col.append(f[j])
                    data.append(1.0)

    # Adjacency matrix (duplicates are summed by coo_matrix)
    A = sparse.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    # Degree (number of neighbors per vertex)
    degree = np.array(A.sum(axis=1)).flatten()
    degree[degree == 0] = 1  # avoid division by zero for isolated vertices

    # Laplacian: L = D^{-1} A - I  (row-normalized)
    D_inv = sparse.diags(1.0 / degree)
    L = D_inv @ A - sparse.eye(n)

    # --- 2. Set up the linear system ---
    # For known vertices:   x_i = known_value_i
    # For unknown vertices:  L @ x = 0  (harmonic condition)

    # Build system matrix: for unknowns use L rows, for knowns use identity rows
    # Rewrite as: M @ x = b  (b may be 1D or 2D; spsolve handles both)
    M = L.tolil()
    b = np.zeros((n,) + values.shape[1:], dtype=values.dtype)

    for i in range(n):
        if known_mask[i]:
            M[i, :] = 0
            M[i, i] = 1.0
            b[i] = values[i]
        # else: row stays as L row, b[i] = 0 (Laplace equation)

    M = M.tocsr()
    result = spsolve(M, b)
    return result

# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────
def remove_na_and_fill(mesh_or_verts, keep_mask, faces=None):
    """
    Parameters
    ----------
    mesh_or_verts : trimesh.Trimesh  **or**  (N, 3) ndarray of vertices
        If a Trimesh, `faces` is taken from the mesh.
    keep_mask : (N,) bool array
        True  = vertex has a valid value (keep it).
        False = vertex should be removed.
    faces : (M, 3) int array, optional
        Required only when mesh_or_verts is a raw vertex array.
 
    Returns
    -------
    If input was Trimesh  → trimesh.Trimesh
    If input was ndarray  → (new_vertices, new_faces)
    """
    try:
        import trimesh as _tm
        is_trimesh = isinstance(mesh_or_verts, _tm.Trimesh)
    except ImportError:
        is_trimesh = False
 
    if is_trimesh:
        vertices = np.asarray(mesh_or_verts.vertices)
        faces_arr = np.asarray(mesh_or_verts.faces)
    else:
        vertices = np.asarray(mesh_or_verts)
        faces_arr = np.asarray(faces)
 
    keep_mask = np.asarray(keep_mask, dtype=bool)
    new_v, new_f = _remove_and_fill(vertices, faces_arr, keep_mask)
 
    if is_trimesh:
        import trimesh as _tm
        return _tm.Trimesh(vertices=new_v, faces=new_f, process=False)
    return new_v, new_f
 


def clean_mesh(mesh_or_verts, faces=None, min_faces=None):
    """
    Clean a triangle mesh:
      1. Remove degenerate faces (zero-area / repeated vertices)
      2. Remove duplicate faces
      3. Remove non-manifold edges (keep at most 2 faces per edge)
      4. Remove non-manifold vertices (bowtie vertices)
      5. Keep only the largest connected component(s)
      6. Remove orphan vertices

    Parameters
    ----------
    mesh_or_verts : trimesh.Trimesh  **or**  (N, 3) ndarray of vertices
    faces : (M, 3) int array, optional
        Required only when mesh_or_verts is a raw vertex array.
    min_faces : int, optional
        If given, keep every component with at least this many faces
        instead of only the single largest one.

    Returns
    -------
    If input was Trimesh  → (trimesh.Trimesh, kept_mask)
    If input was ndarray  → (new_vertices, new_faces, kept_mask)

    kept_mask : (N,) bool array
        True = vertex was kept, False = vertex was rejected.
        N is the number of *original* input vertices.
    """
    try:
        import trimesh as _tm
        is_trimesh = isinstance(mesh_or_verts, _tm.Trimesh)
    except ImportError:
        is_trimesh = False

    if is_trimesh:
        vertices = np.asarray(mesh_or_verts.vertices)
        faces_arr = np.asarray(mesh_or_verts.faces)
    else:
        vertices = np.asarray(mesh_or_verts)
        faces_arr = np.asarray(faces)

    n_orig = len(vertices)
    new_v, new_f, kept_mask = _clean(vertices, faces_arr, min_faces)

    # If bowtie splitting added extra vertices, truncate kept_mask
    # to original length. The extra (split) copies are always kept,
    # so the caller only needs to know about originals.
    if len(kept_mask) > n_orig:
        kept_mask = kept_mask[:n_orig]

    if is_trimesh:
        import trimesh as _tm
        return _tm.Trimesh(vertices=new_v, faces=new_f, process=False), kept_mask
    return new_v, new_f, kept_mask


# ─────────────────────────────────────────────
#  Core clean algorithm
# ─────────────────────────────────────────────
def _clean(vertices, faces, min_faces):
    if len(faces) == 0:
        return (vertices[:0].copy(), faces.copy(),
                np.zeros(len(vertices), dtype=bool))

    # ── Step 1: Remove degenerate faces (repeated vertex indices) ───
    non_degen = np.array([len(set(f)) == 3 for f in faces])
    faces = faces[non_degen]

    # ── Step 2: Remove duplicate faces (same 3 verts, any order) ────
    sorted_faces = np.sort(faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    faces = faces[np.sort(unique_idx)]

    # ── Step 3: Remove non-manifold edges (>2 faces per edge) ───────
    faces = _fix_non_manifold_edges(faces)

    # ── Step 4: Split non-manifold (bowtie) vertices ────────────────
    faces, vertices = _fix_non_manifold_vertices(faces, vertices)

    # ── Step 5: Connected-component filtering ───────────────────────
    if len(faces) == 0:
        return (vertices[:0].copy(), faces,
                np.zeros(len(vertices), dtype=bool))

    edge_to_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for i in range(3):
            edge_to_faces[_edge(f[i], f[(i + 1) % 3])].append(fi)

    n_faces = len(faces)
    comp_id = np.full(n_faces, -1, dtype=np.intp)
    current = 0
    for seed in range(n_faces):
        if comp_id[seed] >= 0:
            continue
        queue = [seed]
        comp_id[seed] = current
        while queue:
            fi = queue.pop()
            f = faces[fi]
            for i in range(3):
                for nb in edge_to_faces[_edge(f[i], f[(i + 1) % 3])]:
                    if comp_id[nb] < 0:
                        comp_id[nb] = current
                        queue.append(nb)
        current += 1

    comp_sizes = np.bincount(comp_id, minlength=current)
    if min_faces is not None:
        keep_comps = set(np.where(comp_sizes >= min_faces)[0])
    else:
        keep_comps = {np.argmax(comp_sizes)}

    keep_face_mask = np.isin(comp_id, list(keep_comps))
    kept_faces = faces[keep_face_mask]

    # ── Step 6: Remove orphan vertices ──────────────────────────────
    used = np.zeros(len(vertices), dtype=bool)
    used[kept_faces.ravel()] = True

    old2new = np.full(len(vertices), -1, dtype=np.intp)
    old2new[used] = np.arange(used.sum())

    new_verts = vertices[used]
    new_faces = old2new[kept_faces]

    return new_verts, new_faces, used


# ─────────────────────────────────────────────
#  Non-manifold fixers
# ─────────────────────────────────────────────
def _fix_non_manifold_edges(faces):
    """Keep at most 2 faces per edge; drop extras."""
    edge_to_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for i in range(3):
            edge_to_faces[_edge(f[i], f[(i + 1) % 3])].append(fi)

    bad = set()
    for flist in edge_to_faces.values():
        if len(flist) > 2:
            bad.update(flist[2:])

    if bad:
        keep = np.ones(len(faces), dtype=bool)
        keep[list(bad)] = False
        faces = faces[keep]
    return faces


def _fix_non_manifold_vertices(faces, vertices):
    """
    Detect bowtie (non-manifold) vertices.  Keep only the largest
    fan for each; remove faces from smaller fans.
    This avoids creating new vertices so the output vertex set is
    always a subset of the input.
    """
    max_iters = 10
    for _ in range(max_iters):
        bad_faces = _find_bowtie_bad_faces(faces)
        if not bad_faces:
            break
        keep = np.ones(len(faces), dtype=bool)
        keep[list(bad_faces)] = False
        faces = faces[keep]
    return faces, vertices


def _find_bowtie_bad_faces(faces):
    """
    For every bowtie vertex, find the face indices belonging to
    the smaller fan(s).  These will be removed.
    """
    vert_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[v].append(fi)

    bad_faces = set()
    for vi, fi_list in vert_faces.items():
        if len(fi_list) < 2:
            continue

        # Local adjacency: two faces are neighbours if they share
        # an edge that passes through vi
        edge_to_local = defaultdict(set)
        for fi in fi_list:
            f = faces[fi]
            for i in range(3):
                if f[i] == vi or f[(i + 1) % 3] == vi:
                    e = _edge(f[i], f[(i + 1) % 3])
                    edge_to_local[e].add(fi)

        face_neighbors = defaultdict(set)
        for fis in edge_to_local.values():
            fis = list(fis)
            for a in fis:
                for b in fis:
                    if a != b:
                        face_neighbors[a].add(b)

        # BFS to find connected fans
        visited = set()
        fans = []
        for seed in fi_list:
            if seed in visited:
                continue
            fan = []
            queue = [seed]
            visited.add(seed)
            while queue:
                cur = queue.pop()
                fan.append(cur)
                for nb in face_neighbors[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            fans.append(fan)

        if len(fans) > 1:
            # Keep the largest fan, remove the rest
            fans.sort(key=len, reverse=True)
            for fan in fans[1:]:
                bad_faces.update(fan)

    return bad_faces


# ─────────────────────────────────────────────
#  Core remove-and-fill algorithm
# ─────────────────────────────────────────────
def _remove_and_fill(vertices, faces, keep_mask):
    if keep_mask.all():
        return vertices.copy(), faces.copy()

    remove_mask = ~keep_mask

    # 1. Original boundary edges (in OLD indices)
    orig_bnd = _boundary_edges(faces)

    # 2. Remove every face that touches a removed vertex
    face_has_removed = remove_mask[faces].any(axis=1)
    kept_faces = faces[~face_has_removed]

    # 3. Reindex: drop removed vertices
    old2new = np.full(len(vertices), -1, dtype=np.intp)
    old2new[keep_mask] = np.arange(keep_mask.sum())

    new_verts = vertices[keep_mask]
    new_faces = old2new[kept_faces]

    # 4. Map old boundary into new index space
    orig_bnd_new = set()
    for a, b in orig_bnd:
        na, nb = old2new[a], old2new[b]
        if na >= 0 and nb >= 0:
            orig_bnd_new.add(_edge(na, nb))

    # 5. New boundary = boundary of reduced mesh minus old boundary
    reduced_bnd = _boundary_edges(new_faces)
    hole_edges  = reduced_bnd - orig_bnd_new

    # 6. Extract closed loops from hole edges
    loops = _edge_loops(hole_edges)

    # 7. Ear-clip each closed loop to triangulate the hole
    fill = []
    for loop in loops:
        tris = _triangulate_3d_polygon(new_verts, loop, new_faces)
        fill.extend(tris)

    if fill:
        new_faces = np.vstack([new_faces, np.array(fill, dtype=np.intp)])

    return new_verts, new_faces


# ─────────────────────────────────────────────
#  Boundary / edge helpers
# ─────────────────────────────────────────────
def _edge(a, b):
    return (min(a, b), max(a, b))


def _boundary_edges(faces):
    """Return the set of boundary (non-manifold) edges as sorted tuples."""
    cnt = defaultdict(int)
    for f in faces:
        for i in range(3):
            cnt[_edge(f[i], f[(i + 1) % 3])] += 1
    return {e for e, c in cnt.items() if c == 1}


def _edge_loops(edges):
    """
    Chain undirected edges into closed loops.
    Skips open chains (which correspond to boundary extensions, not holes).
    """
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited = set()
    loops = []

    for start in list(adj):
        if start in visited:
            continue
        path = [start]
        visited.add(start)
        cur = start
        prev = None
        stuck = False
        while True:
            nbrs = adj[cur] - ({prev} if prev is not None else set())
            nxt = None
            for n in nbrs:
                if n == start and len(path) >= 3:
                    nxt = start
                    break
                if n not in visited:
                    nxt = n
                    break
            if nxt is None:
                stuck = True
                break
            if nxt == start:
                break
            path.append(nxt)
            visited.add(nxt)
            prev = cur
            cur = nxt

        if not stuck and len(path) >= 3:
            loops.append(path)
        else:
            queue = [cur]
            while queue:
                node = queue.pop()
                for n in adj[node]:
                    if n not in visited:
                        visited.add(n)
                        queue.append(n)
    return loops


# ─────────────────────────────────────────────
#  3-D polygon triangulation  (ear clipping)
# ─────────────────────────────────────────────
def _triangulate_3d_polygon(vertices, loop_indices, existing_faces):
    """
    Triangulate a planar(-ish) 3-D polygon defined by `loop_indices`
    using ear clipping.  Returns list of (i, j, k) index triples
    (indices into `vertices`).
    """
    pts3d = vertices[loop_indices]
    n = len(loop_indices)
    if n < 3:
        return []
    if n == 3:
        tri = _orient_triangle(loop_indices, vertices, existing_faces)
        return [tri]

    normal = _polygon_normal(pts3d)
    u_ax, v_ax = _orthonormal_basis(normal)
    centroid = pts3d.mean(axis=0)
    rel = pts3d - centroid
    pts2d = np.column_stack([rel @ u_ax, rel @ v_ax])

    if _signed_area_2d(pts2d) < 0:
        pts2d = pts2d[::-1]
        loop_indices = loop_indices[::-1]

    idx = list(range(n))
    tris = []

    max_iter = n * n
    it = 0
    while len(idx) > 3 and it < max_iter:
        it += 1
        clipped = False
        for i in range(len(idx)):
            p  = idx[(i - 1) % len(idx)]
            c  = idx[i]
            nx = idx[(i + 1) % len(idx)]

            if _cross_2d(pts2d[p], pts2d[c], pts2d[nx]) <= 0:
                continue

            ear_ok = True
            for j in idx:
                if j in (p, c, nx):
                    continue
                if _point_in_triangle_2d(pts2d[j], pts2d[p],
                                         pts2d[c], pts2d[nx]):
                    ear_ok = False
                    break
            if ear_ok:
                tri = _orient_triangle(
                    [loop_indices[p], loop_indices[c], loop_indices[nx]],
                    vertices, existing_faces,
                )
                tris.append(tri)
                idx.remove(c)
                clipped = True
                break
        if not clipped:
            p  = idx[0]
            c  = idx[1]
            nx = idx[2]
            tri = _orient_triangle(
                [loop_indices[p], loop_indices[c], loop_indices[nx]],
                vertices, existing_faces,
            )
            tris.append(tri)
            idx.remove(c)

    if len(idx) == 3:
        tri = _orient_triangle(
            [loop_indices[idx[0]], loop_indices[idx[1]],
             loop_indices[idx[2]]],
            vertices, existing_faces,
        )
        tris.append(tri)

    return tris


# ─────────────────────────────────────────────
#  Orientation helper
# ─────────────────────────────────────────────
def _orient_triangle(tri_indices, vertices, existing_faces):
    """
    Return `tri_indices` with winding that best matches
    the orientation of neighbouring existing faces.
    """
    tri = list(tri_indices)
    tri_set = set(tri)

    directed = {}
    for f in existing_faces:
        fset = set(f)
        if len(fset & tri_set) >= 2:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                if a in tri_set and b in tri_set:
                    directed[_edge(a, b)] = (a, b)

    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        key = _edge(a, b)
        if key in directed:
            da, db = directed[key]
            if (a, b) == (da, db):
                tri = tri[::-1]
            return tuple(tri)

    return tuple(tri)


# ─────────────────────────────────────────────
#  Geometry utilities
# ─────────────────────────────────────────────
def _polygon_normal(pts):
    """Newell's method for the normal of a 3-D polygon."""
    n = np.zeros(3)
    m = len(pts)
    for i in range(m):
        cur = pts[i]
        nxt = pts[(i + 1) % m]
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else np.array([0, 0, 1.0])


def _orthonormal_basis(normal):
    """Return two unit vectors perpendicular to `normal`."""
    n = normal / np.linalg.norm(normal)
    seed = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(n, seed)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _signed_area_2d(pts):
    """Signed area of a 2-D polygon (positive = CCW)."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]
    return area * 0.5


def _cross_2d(o, a, b):
    """2-D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle_2d(p, a, b, c):
    """Check if point p lies strictly inside triangle abc (CCW)."""
    d1 = _cross_2d(a, b, p)
    d2 = _cross_2d(b, c, p)
    d3 = _cross_2d(c, a, p)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

class MeshMapper:
    """
    Bijective mapping between UV and XYZ spaces using barycentric interpolation.

    Parameters:
        uv: (n, 2) array of UV coordinates
        vertices: (n, 3) array of XYZ coordinates
        faces: (m, 3) array of triangle indices
        uv_offset: (2,) translation (du, dv) applied to the UV coordinates.
                   xyz_to_uv returns offset UV; uv_to_xyz (and other methods
                   taking UV queries) expect offset UV.
    """

    def __init__(self, uv, vertices, faces, uv_offset=(0.0, 0.0)):
        self._uv_base = np.asarray(uv, dtype=np.float64)
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=int)

        self._tree_xyz = KDTree(self.vertices)

        self._vert_faces = [[] for _ in range(len(self.vertices))]
        for fi, tri in enumerate(self.faces):
            for v in tri:
                self._vert_faces[v].append(fi)

        self.set_uv_offset(uv_offset)

        # Precompute per-vertex Jacobian (translation-invariant in UV,
        # so independent of uv_offset)
        self._compute_vertex_jacobian()

    def set_uv_offset(self, uv_offset):
        """
        Set the (du, dv) translation applied to the mesh's UV coordinates.

        Replaces any previous offset (always relative to the original UVs,
        not cumulative).

        Parameters:
            uv_offset: (2,) array-like, e.g. (-100, 100)
        """
        self.uv_offset = np.asarray(uv_offset, dtype=np.float64).reshape(2)
        self.uv = self._uv_base + self.uv_offset
        self._tree_uv = KDTree(self.uv)

    def _compute_vertex_jacobian(self):
        p0 = self.vertices[self.faces[:, 0]]
        p1 = self.vertices[self.faces[:, 1]]
        p2 = self.vertices[self.faces[:, 2]]
        a_3d = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

        q0 = self.uv[self.faces[:, 0]]
        q1 = self.uv[self.faces[:, 1]]
        q2 = self.uv[self.faces[:, 2]]
        a_uv = 0.5 * np.abs((q1[:, 0] - q0[:, 0]) * (q2[:, 1] - q0[:, 1]) -
                              (q2[:, 0] - q0[:, 0]) * (q1[:, 1] - q0[:, 1]))

        tri_jac = a_3d / np.maximum(a_uv, 1e-16)

        vert_jac = np.zeros(len(self.vertices))
        vert_weight = np.zeros(len(self.vertices))
        np.add.at(vert_jac, self.faces[:, 0], a_uv * tri_jac)
        np.add.at(vert_jac, self.faces[:, 1], a_uv * tri_jac)
        np.add.at(vert_jac, self.faces[:, 2], a_uv * tri_jac)
        np.add.at(vert_weight, self.faces[:, 0], a_uv)
        np.add.at(vert_weight, self.faces[:, 1], a_uv)
        np.add.at(vert_weight, self.faces[:, 2], a_uv)

        self._vertex_jacobian = vert_jac / np.maximum(vert_weight, 1e-16)

    def _barycentric(self, p, t0, t1, t2):
        e1 = t1 - t0
        e2 = t2 - t0
        ep = p - t0

        d11 = np.dot(e1, e1)
        d12 = np.dot(e1, e2)
        d22 = np.dot(e2, e2)
        d1p = np.dot(e1, ep)
        d2p = np.dot(e2, ep)

        denom = d11 * d22 - d12 * d12
        if abs(denom) < 1e-16:
            return None

        b1 = (d22 * d1p - d12 * d2p) / denom
        b2 = (d11 * d2p - d12 * d1p) / denom
        b0 = 1.0 - b1 - b2
        return b0, b1, b2

    def _find_triangle(self, query, source_coords, tree, k_neighbors=5):
        """
        Find the containing triangle and barycentric coordinates.

        Returns:
            (face_index, b0, b1, b2) or None
        """
        _, near_verts = tree.query(query, k=k_neighbors)
        if k_neighbors == 1:
            near_verts = [near_verts]

        candidates = set()
        for v in near_verts:
            candidates.update(self._vert_faces[v])

        best_dist = np.inf
        best = None

        for fi in candidates:
            tri = self.faces[fi]
            bary = self._barycentric(query,
                                     source_coords[tri[0]],
                                     source_coords[tri[1]],
                                     source_coords[tri[2]])
            if bary is None:
                continue

            b0, b1, b2 = bary
            if b0 >= -1e-4 and b1 >= -1e-4 and b2 >= -1e-4:
                projected = (b0 * source_coords[tri[0]] +
                             b1 * source_coords[tri[1]] +
                             b2 * source_coords[tri[2]])
                dist = np.linalg.norm(query - projected)
                if dist < best_dist:
                    best_dist = dist
                    best = (fi, b0, b1, b2)

        return best

    def uv_to_xyz(self, query_uv, k_neighbors=5):
        """
        Map UV point(s) to XYZ.

        Parameters:
            query_uv: (2,) or (k, 2) array
            k_neighbors: number of nearest vertices to check

        Returns:
            xyz: (3,) or (k, 3) array, NaN if outside mesh
        """
        query_uv = np.asarray(query_uv, dtype=np.float64)
        single = query_uv.ndim == 1
        if single:
            query_uv = query_uv[np.newaxis]

        result = np.full((len(query_uv), 3), np.nan)
        for i, q in enumerate(query_uv):
            found = self._find_triangle(q, self.uv, self._tree_uv, k_neighbors)
            if found is not None:
                fi, b0, b1, b2 = found
                tri = self.faces[fi]
                result[i] = (b0 * self.vertices[tri[0]] +
                             b1 * self.vertices[tri[1]] +
                             b2 * self.vertices[tri[2]])

        return result[0] if single else result

    def xyz_to_uv(self, query_xyz, k_neighbors=5):
        """
        Map XYZ point(s) on the surface to UV.

        Parameters:
            query_xyz: (3,) or (k, 3) array
            k_neighbors: number of nearest vertices to check

        Returns:
            uv: (2,) or (k, 2) array, NaN if not found
        """
        query_xyz = np.asarray(query_xyz, dtype=np.float64)
        single = query_xyz.ndim == 1
        if single:
            query_xyz = query_xyz[np.newaxis]

        result = np.full((len(query_xyz), 2), np.nan)
        for i, q in enumerate(query_xyz):
            found = self._find_triangle(q, self.vertices, self._tree_xyz, k_neighbors)
            if found is not None:
                fi, b0, b1, b2 = found
                tri = self.faces[fi]
                result[i] = (b0 * self.uv[tri[0]] +
                             b1 * self.uv[tri[1]] +
                             b2 * self.uv[tri[2]])

        return result[0] if single else result

    def jacobian(self, query, space='uv', k_neighbors=5):
        """
        Compute the Jacobian determinant (area ratio A_3D / A_UV)
        at the given point(s), using barycentric interpolation
        of per-vertex values.

        Parameters:
            query: (2,) or (k, 2) if space='uv',
                   (3,) or (k, 3) if space='xyz'
            space: 'uv' or 'xyz', which space the query is in
            k_neighbors: number of nearest vertices to check

        Returns:
            jac: float or (k,) array, NaN if outside mesh
        """
        query = np.asarray(query, dtype=np.float64)
        single = query.ndim == 1
        if single:
            query = query[np.newaxis]

        if space == 'uv':
            source_coords = self.uv
            tree = self._tree_uv
        else:
            source_coords = self.vertices
            tree = self._tree_xyz

        result = np.full(len(query), np.nan)
        for i, q in enumerate(query):
            found = self._find_triangle(q, source_coords, tree, k_neighbors)
            if found is not None:
                fi, b0, b1, b2 = found
                tri = self.faces[fi]
                result[i] = (b0 * self._vertex_jacobian[tri[0]] +
                             b1 * self._vertex_jacobian[tri[1]] +
                             b2 * self._vertex_jacobian[tri[2]])

        return result[0] if single else result
    
    
    def is_valid(self, query, space='uv', k_neighbors=5,
             distance_threshold=None, relative_threshold=0.1):
        """
        Check if query point(s) lie within the mesh.

        For UV: valid if inside a UV triangle.
        For XYZ: valid if projection distance to nearest triangle
                is below threshold.

        Parameters:
            query: (2,) or (k, 2) if space='uv',
                (3,) or (k, 3) if space='xyz'
            space: 'uv' or 'xyz'
            k_neighbors: number of nearest vertices to check
            distance_threshold: absolute distance. If set, overrides
                                relative_threshold.
            relative_threshold: fraction of mean edge length (default 0.1 = 10%).
                                Only used when distance_threshold is None.

        Returns:
            valid: bool or (k,) bool array
        """
        query = np.asarray(query, dtype=np.float64)
        single = query.ndim == 1
        if single:
            query = query[np.newaxis]

        if space == 'uv':
            source_coords = self.uv
            tree = self._tree_uv
        else:
            source_coords = self.vertices
            tree = self._tree_xyz
            if distance_threshold is None:
                edges = self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]]
                distance_threshold = relative_threshold * np.linalg.norm(edges, axis=1).mean()

        valid = np.zeros(len(query), dtype=bool)

        for i, q in enumerate(query):
            found = self._find_triangle(q, source_coords, tree, k_neighbors)
            if found is None:
                continue

            if space == 'uv':
                valid[i] = True
            else:
                fi, b0, b1, b2 = found
                tri = self.faces[fi]
                projected = (b0 * self.vertices[tri[0]] +
                            b1 * self.vertices[tri[1]] +
                            b2 * self.vertices[tri[2]])
                valid[i] = np.linalg.norm(q - projected) < distance_threshold

        return valid[0] if single else valid