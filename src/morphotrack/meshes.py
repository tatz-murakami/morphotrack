import numpy as np
from scipy.spatial import KDTree, cKDTree
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS
import trimesh
import pyvista as pv
import pyacvd


def remesh_uniform(mesh, target_area, subdivide=3):
    """
    Uniformly remesh a triangular mesh so each face covers roughly
    target_area (in squared mesh units), using ACVD clustering.

    Parameters:
        mesh: trimesh.Trimesh to remesh
        target_area: desired area per face; sets the target vertex count
        subdivide: subdivisions applied before clustering, so the
                   clusterer has enough points to work with

    Returns:
        trimesh.Trimesh with approximately uniform face areas
    """
    # faces ~ area / target_area; verts ~ faces / 2
    n_target = int((mesh.area // target_area) // 2)

    # trimesh -> pyvista
    faces_pv = np.hstack(
        [np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]
    ).ravel()
    pmesh = pv.PolyData(mesh.vertices, faces_pv)

    # uniform remeshing
    clus = pyacvd.Clustering(pmesh)
    clus.subdivide(subdivide)
    clus.cluster(n_target)
    remesh = clus.create_mesh()

    # pyvista -> trimesh (drop the leading "3" per face)
    faces_tm = remesh.faces.reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(remesh.points, faces_tm, process=False)


def sample_points_between_surfaces(mesh_a, mesh_b, n_candidates, use_bbox=False):
    tree_a = cKDTree(mesh_a.vertices)
    tree_b = cKDTree(mesh_b.vertices)

    all_verts = np.vstack([mesh_a.vertices, mesh_b.vertices])
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)

    candidates = np.random.uniform(bbox_min, bbox_max, size=(n_candidates, 3))

    if use_bbox:
        return candidates

    _, idx_a = tree_a.query(candidates)
    _, idx_b = tree_b.query(candidates)

    closest_a = tree_a.data[idx_a]
    closest_b = tree_b.data[idx_b]

    vec_ab = closest_b - closest_a
    vec_ac = candidates - closest_a

    dot_ab = np.einsum('ij,ij->i', vec_ab, vec_ab)
    dot_ac = np.einsum('ij,ij->i', vec_ab, vec_ac)

    t = np.where(dot_ab > 1e-12, dot_ac / np.maximum(dot_ab, 1e-12), -1.0)
    mask = (t > 0.0) & (t < 1.0)

    return candidates[mask]


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