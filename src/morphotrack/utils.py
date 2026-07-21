import numpy as np
import napari
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader



def omezarr2dask(path):
    store = parse_url(path)
    reader = Reader(store)
    nodes = list(reader())
    image_node = nodes[0]
    data = image_node.data # Dask array
    return data

def random_idx_with_max(N,M):
    if M == -1 or M >= N:
        idx = np.arange(N)
    else:
        idx = np.random.choice(N, size=M, replace=False)
    return idx


def track_visualizer(X_all, target=None, indices=None, size=(4,30), edge_color='yellow'):
    """Visualize the tracking results in napari."""
    viewer = napari.Viewer(ndisplay=3)

    if indices is None:
        shapes = [X_all[:, i, :] for i in range(X_all.shape[1])]
    else:
        shapes = [X_all[:, i, :] for i in indices]

    # Trajectory lines
    viewer.add_shapes(
        shapes,
        shape_type='path',
        edge_color=edge_color,
        edge_width=size[0],
        name='Trajectories',
        # blending='additive'
    )

    # Sampled initial points (magenta)
    viewer.add_points(
        X_all[0],
        name='X (t=0)',
        size=size[1],
        face_color='magenta',
        blending='additive'
    )

    # Show all warped points, not just subset
    viewer.add_points(
        X_all[-1],
        name='X (t=1) (all)',
        size=size[1],
        face_color='red',
        blending='additive'
    )

    if target is not None:
        viewer.add_points(
            target,
            name='Target',
            size=size[1],
            face_color='lime',
            blending='additive'
        )

    viewer.show()

    return viewer
