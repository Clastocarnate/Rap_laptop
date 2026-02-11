import os
import numpy as np
import trimesh
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Optional 3D Visualization (PyVista)
# ------------------------------------------------------------
try:
    import pyvista as pv
    PV_AVAILABLE = True
except Exception:
    PV_AVAILABLE = False


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OBJECT_DIR = BASE_DIR / "objects"
OUTPUT_DIR = BASE_DIR / "scene_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Mesh Utilities
# ------------------------------------------------------------
def load_and_repair(path):
    mesh = trimesh.load(str(path), force="mesh")

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    try:
        trimesh.repair.fix_inversion(mesh)
    except Exception:
        pass

    mesh.apply_translation(-mesh.centroid)
    return mesh


# ------------------------------------------------------------
# Geometry Helpers
# ------------------------------------------------------------
def limited_rotation(max_deg=30):
    """
    Random rotation within ±max_deg on each axis.
    """
    angles = np.deg2rad(np.random.uniform(-max_deg, max_deg, size=3))
    Rx = trimesh.transformations.rotation_matrix(angles[0], [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(angles[1], [0, 1, 0])
    Rz = trimesh.transformations.rotation_matrix(angles[2], [0, 0, 1])
    return trimesh.transformations.concatenate_matrices(Rx, Ry, Rz)



def get_briefcase_interior(briefcase, margin_ratio=0.15):
    bc_min, bc_max = briefcase.bounds
    margin = margin_ratio * (bc_max - bc_min)
    return bc_min + margin, bc_max - margin

def shrink_bounds(bounds, shrink_factor=0.8):
    """
    Shrink AABB to given percentage (e.g., 0.8 = 80%).
    Keeps center fixed.
    """
    min_corner, max_corner = bounds
    center = (min_corner + max_corner) / 2.0
    size = (max_corner - min_corner) * shrink_factor
    new_min = center - size / 2.0
    new_max = center + size / 2.0
    return np.array([new_min, new_max])



def aabb_collision(bounds1, bounds2, shrink_factor=0.85):
    b1 = shrink_bounds(bounds1, shrink_factor)
    b2 = shrink_bounds(bounds2, shrink_factor)

    min1, max1 = b1
    min2, max2 = b2

    return np.all(max1 > min2) and np.all(max2 > min1)



def clamp_mesh_inside(mesh, interior_min, interior_max):
    """
    Minimal translation to bring mesh fully inside interior.
    """
    m_min, m_max = mesh.bounds
    shift = np.zeros(3)

    for i in range(3):
        if m_min[i] < interior_min[i]:
            shift[i] = interior_min[i] - m_min[i]
        elif m_max[i] > interior_max[i]:
            shift[i] = interior_max[i] - m_max[i]

    mesh.apply_translation(shift)
    return mesh


def resolve_collision(mesh, other_bounds_list, step_size=2.0, max_push=10):
    """
    Push object away minimally if colliding.
    """
    for _ in range(max_push):
        collision = False
        for bounds in other_bounds_list:
            if aabb_collision(mesh.bounds, bounds, shrink_factor=0.8):
                collision = True
                direction = mesh.centroid - (bounds[0] + bounds[1]) / 2
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    direction = np.random.randn(3)
                    norm = np.linalg.norm(direction)
                direction /= norm
                mesh.apply_translation(direction * step_size)
                break
        if not collision:
            return True
    return False


# ------------------------------------------------------------
# Placement Logic (NEW BEHAVIOR)
# ------------------------------------------------------------
def place_objects(
    object_meshes,
    interior_min,
    interior_max,
):
    placed = []
    placed_bounds = []

    for name, base_mesh in object_meshes:

        mesh = base_mesh.copy()

        # 1️⃣ Random rotation
        if name == "laptop":
            R = limited_rotation(max_deg=30)
        else:
            R = trimesh.transformations.random_rotation_matrix()

        mesh.apply_transform(R)

        # 2️⃣ Random center spawn
        center_spawn = interior_min + np.random.rand(3) * (interior_max - interior_min)
        mesh.apply_translation(center_spawn - mesh.centroid)

        # 3️⃣ Clamp fully inside
        mesh = clamp_mesh_inside(mesh, interior_min, interior_max)

        # 4️⃣ Resolve collisions minimally
        success = resolve_collision(mesh, placed_bounds)

        # 5️⃣ Final containment check
        mesh = clamp_mesh_inside(mesh, interior_min, interior_max)

        if not success:
            print(f"Skipping {name} (collision unresolved)")
            continue

        # Final overlap check
        collision = False
        for bounds in placed_bounds:
            if aabb_collision(mesh.bounds, bounds):
                collision = True
                break

        if collision:
            print(f"Skipping {name} (still colliding)")
            continue

        placed.append((name, mesh))
        placed_bounds.append(mesh.bounds)

    return placed


# ------------------------------------------------------------
# Voxelization
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Attenuation
# ------------------------------------------------------------
def attenuation_from_labels(labels):
    mu = np.zeros(labels.shape, dtype=np.float32)

    mu[labels == 0] = 0.01 + 0.
    mu[labels == 1] = 0.15 + 0.5
    mu[labels == 2] = 1.0
    mu[labels == 3] = 0.8
    mu[labels == 4] = 0.2 +0.3

    return gaussian_filter(mu, sigma=1.0)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def show_slices(volume):
    zc, yc, xc = np.array(volume.shape) // 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(volume[zc], cmap="gray")
    ax[1].imshow(volume[:, yc, :], cmap="gray")
    ax[2].imshow(volume[:, :, xc], cmap="gray")
    plt.show()

def view_3d(volume, spacing):
    if not PV_AVAILABLE:
        print("PyVista not installed.")
        return

    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = spacing
    grid.origin = (0.0, 0.0, 0.0)

    grid.cell_data["values"] = volume.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        scalars="values",
        cmap="gray",
        opacity="sigmoid",
        shade=True
    )

    plotter.show_axes()
    plotter.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    briefcase = load_and_repair(OBJECT_DIR / "2xBriefcase.stl")

    object_files = [
        ("laptop", OBJECT_DIR / "laptop.stl"),
        ("charger", OBJECT_DIR / "Charger.stl"),
        ("airpods", OBJECT_DIR / "AirPods.stl"),
        ("comb", OBJECT_DIR / "comb.stl"),
        ("diary", OBJECT_DIR / "Diary.stl"),
        ("pen", OBJECT_DIR / "Pen.stl"),
        ("brush", OBJECT_DIR / "buerste.STL"),
        ("hanger", OBJECT_DIR / "hanger.stl"),
    ]

    object_meshes = [(name, load_and_repair(path)) for name, path in object_files]

    interior_min, interior_max = get_briefcase_interior(briefcase)

    placed = place_objects(
        object_meshes,
        interior_min,
        interior_max,
    )

    bc_occ, pitch, origin = mesh_to_voxels(briefcase, target_dim=256)
    ref_shape = bc_occ.shape

    labels = np.zeros(ref_shape, dtype=np.uint8)
    labels[bc_occ] = 1

    for name, mesh in placed:
        occ = voxelize_to_reference(mesh, pitch, origin, ref_shape)

        if name == "laptop":
            labels[occ] = 2
        elif name in ["charger", "airpods"]:
            labels[occ] = 3
        else:
            labels[occ] = 4

    volume = attenuation_from_labels(labels)

    np.save(OUTPUT_DIR / "volume.npy", volume)
    np.save(OUTPUT_DIR / "labels.npy", labels)

    show_slices(volume)
    view_3d(volume, spacing=(pitch, pitch, pitch))

