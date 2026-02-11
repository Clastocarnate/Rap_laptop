import os
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Optional PyVista
try:
    import pyvista as pv
    PV_AVAILABLE = True
except Exception:
    PV_AVAILABLE = False


# ============================================================
# Mesh utilities (version-safe)
# ============================================================
def load_and_repair(path):
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    try:
        trimesh.repair.fix_inversion(mesh)
    except Exception:
        pass

    if hasattr(mesh, "remove_unreferenced_vertices"):
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass

    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass

    # Center mesh
    mesh.apply_translation(-mesh.centroid)
    return mesh


# ============================================================
# Geometry helpers
# ============================================================
def get_briefcase_interior_bounds(briefcase_mesh, margin_ratio=0.1):
    bc_min, bc_max = briefcase_mesh.bounds
    margin = margin_ratio * (bc_max - bc_min)
    interior_min = bc_min + margin
    interior_max = bc_max - margin
    return interior_min, interior_max


def random_rotation_matrix():
    return trimesh.transformations.random_rotation_matrix()


def fits_inside_bounds(mesh, interior_min, interior_max):
    m_min, m_max = mesh.bounds
    return np.all(m_min >= interior_min) and np.all(m_max <= interior_max)


def generate_random_laptop_poses(
    laptop_mesh,
    interior_min,
    interior_max,
    num_poses=5,
    max_trials=500
):
    poses = []

    for _ in range(num_poses):
        placed = False

        for _ in range(max_trials):
            test_mesh = laptop_mesh.copy()

            # Random rotation
            R = random_rotation_matrix()
            test_mesh.apply_transform(R)

            # Random translation
            available_space = interior_max - interior_min
            t = interior_min + np.random.rand(3) * available_space
            test_mesh.apply_translation(t - test_mesh.centroid)

            # Containment check
            if fits_inside_bounds(test_mesh, interior_min, interior_max):
                poses.append(test_mesh)
                placed = True
                break

        if not placed:
            raise RuntimeError("Failed to place laptop inside briefcase")

    return poses


# ============================================================
# Voxelization
# ============================================================
def mesh_to_voxels(mesh, pitch=None, target_dim=256):
    bbox = mesh.bounds
    dims = bbox[1] - bbox[0]

    if pitch is None:
        pitch = dims.max() / float(target_dim)

    v = mesh.voxelized(pitch)
    mat = v.matrix.astype(bool)
    origin = v.bounds[0]
    return mat, pitch, origin


def voxelize_to_reference(mesh, pitch, ref_origin, ref_shape):
    v = mesh.voxelized(pitch)
    out = np.zeros(ref_shape, dtype=bool)

    offset = np.round((v.bounds[0] - ref_origin) / pitch).astype(int)

    sz, sy, sx = v.matrix.shape
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                if v.matrix[z, y, x]:
                    zi = z + offset[0]
                    yi = y + offset[1]
                    xi = x + offset[2]
                    if (
                        0 <= zi < ref_shape[0]
                        and 0 <= yi < ref_shape[1]
                        and 0 <= xi < ref_shape[2]
                    ):
                        out[zi, yi, xi] = True
    return out


# ============================================================
# Attenuation model
# ============================================================
def attenuation_from_labels(labels):
    mu = np.zeros(labels.shape, dtype=np.float32)
    mu[labels == 0] = 0.01   # air
    mu[labels == 1] = 0.6   # briefcase
    mu[labels == 2] = 1.00   # laptop
    mu = gaussian_filter(mu, sigma=1.0)
    return mu


# ============================================================
# Visualization
# ============================================================
def show_slices(volume, title=""):
    zc, yc, xc = np.array(volume.shape) // 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(volume[zc], cmap="gray")
    ax[1].imshow(volume[:, yc, :], cmap="gray")
    ax[2].imshow(volume[:, :, xc], cmap="gray")
    fig.suptitle(title)
    plt.show()


def view_3d(volume, spacing):
    if not PV_AVAILABLE:
        return
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = spacing
    grid.cell_data["values"] = volume.flatten(order="F")

    p = pv.Plotter()
    p.add_volume(grid, scalars="values", opacity="sigmoid", cmap="gray")
    p.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    os.makedirs("scenes", exist_ok=True)

    briefcase = load_and_repair("briefcase.stl")
    laptop = load_and_repair("laptop.stl")

    interior_min, interior_max = get_briefcase_interior_bounds(briefcase)

    laptop_poses = generate_random_laptop_poses(
        laptop,
        interior_min,
        interior_max,
        num_poses=5
    )

    # Voxelize briefcase once (reference grid)
    bc_occ, pitch, origin = mesh_to_voxels(briefcase, target_dim=256)
    ref_shape = bc_occ.shape

    for idx, lap_mesh in enumerate(laptop_poses):
        lap_occ = voxelize_to_reference(
            lap_mesh, pitch, origin, ref_shape
        )

        labels = np.zeros(ref_shape, dtype=np.uint8)
        labels[bc_occ] = 1
        labels[lap_occ] = 2

        volume = attenuation_from_labels(labels)
        mask = (labels == 2).astype(np.uint8)

        np.save(f"scenes/scene_{idx:02d}_volume.npy", volume)
        np.save(f"scenes/scene_{idx:02d}_mask.npy", mask)

        show_slices(volume, title=f"Scene {idx}")
        view_3d(volume, spacing=(pitch, pitch, pitch))
