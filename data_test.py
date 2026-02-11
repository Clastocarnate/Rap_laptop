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
# Rotation
# ------------------------------------------------------------
def limited_rotation(max_deg=30):
    angles = np.deg2rad(np.random.uniform(-max_deg, max_deg, size=3))
    Rx = trimesh.transformations.rotation_matrix(angles[0], [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(angles[1], [0, 1, 0])
    Rz = trimesh.transformations.rotation_matrix(angles[2], [0, 0, 1])
    return trimesh.transformations.concatenate_matrices(Rx, Ry, Rz)

# ------------------------------------------------------------
# Geometry Helpers
# ------------------------------------------------------------
def get_briefcase_interior(briefcase, margin_ratio=0.20):
    bc_min, bc_max = briefcase.bounds
    margin = margin_ratio * (bc_max - bc_min)
    return bc_min + margin, bc_max - margin

def shrink_bounds(bounds, shrink_factor=0.8):
    min_corner, max_corner = bounds
    center = (min_corner + max_corner) / 2.0
    size = (max_corner - min_corner) * shrink_factor
    new_min = center - size / 2.0
    new_max = center + size / 2.0
    return np.array([new_min, new_max])

def aabb_collision(bounds1, bounds2, shrink_factor=0.8):
    b1 = shrink_bounds(bounds1, shrink_factor)
    b2 = shrink_bounds(bounds2, shrink_factor)
    min1, max1 = b1
    min2, max2 = b2
    return np.all(max1 > min2) and np.all(max2 > min1)

def clamp_mesh_inside(mesh, interior_min, interior_max):
    m_min, m_max = mesh.bounds
    shift = np.zeros(3)

    for i in range(3):
        if m_min[i] < interior_min[i]:
            shift[i] = interior_min[i] - m_min[i]
        elif m_max[i] > interior_max[i]:
            shift[i] = interior_max[i] - m_max[i]

    mesh.apply_translation(shift)
    return mesh

def resolve_collision(mesh, other_bounds_list, interior_min, interior_max,
                      step_size=3.0, max_push=15):
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
                mesh = clamp_mesh_inside(mesh, interior_min, interior_max)
                break
        if not collision:
            return True
    return False

# ------------------------------------------------------------
# Placement Logic
# ------------------------------------------------------------
def place_objects(object_meshes, interior_min, interior_max):
    placed = []
    placed_bounds = []

    for name, base_mesh in object_meshes:

        mesh = base_mesh.copy()

        # Rotation
        if name == "laptop":
            R = limited_rotation(max_deg=30)
        else:
            R = trimesh.transformations.random_rotation_matrix()
        mesh.apply_transform(R)

        # Random center spawn
        center_spawn = interior_min + np.random.rand(3) * (interior_max - interior_min)
        mesh.apply_translation(center_spawn - mesh.centroid)

        mesh = clamp_mesh_inside(mesh, interior_min, interior_max)

        success = resolve_collision(mesh, placed_bounds,
                                    interior_min, interior_max)

        mesh = clamp_mesh_inside(mesh, interior_min, interior_max)

        if not success:
            print(f"Skipping {name} (collision unresolved)")
            continue

        # Final collision verification
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
    return v.matrix.astype(bool), pitch, v.bounds[0]

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
                    if (0 <= zi < ref_shape[0] and
                        0 <= yi < ref_shape[1] and
                        0 <= xi < ref_shape[2]):
                        out[zi, yi, xi] = True
    return out

# ------------------------------------------------------------
# Attenuation
# ------------------------------------------------------------
def attenuation_from_labels(labels):
    mu = np.zeros(labels.shape, dtype=np.float32)
    mu[labels == 0] = 0.01
    mu[labels == 1] = 0.4      # briefcase
    mu[labels == 2] = 1.0      # laptop
    mu[labels == 3] = 0.8      # electronics
    mu[labels == 4] = 0.5      # misc
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
        return
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.cell_data["values"] = volume.flatten(order="F")

    p = pv.Plotter()
    p.add_volume(grid, scalars="values",
                 opacity="sigmoid", cmap="gray")
    p.show_axes()
    p.show()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    briefcase = load_and_repair(OBJECT_DIR / "2xBriefcase.stl")

    object_meshes = []

    # Laptop (always 1)
    object_meshes.append(("laptop",
                          load_and_repair(OBJECT_DIR / "laptop.stl")))

    # Random small objects
    for _ in range(np.random.randint(1, 6)):
        object_meshes.append(("pen",
                              load_and_repair(OBJECT_DIR / "Pen.stl")))

    for _ in range(np.random.randint(1, 6)):
        object_meshes.append(("airpods",
                              load_and_repair(OBJECT_DIR / "AirPods.stl")))

    for _ in range(np.random.randint(2, 5)):
        object_meshes.append(("comb",
                              load_and_repair(OBJECT_DIR / "comb.stl")))

    # Single medium objects
    object_meshes.extend([
        ("charger", load_and_repair(OBJECT_DIR / "Charger.stl")),
        ("diary", load_and_repair(OBJECT_DIR / "Diary.stl")),
        ("brush", load_and_repair(OBJECT_DIR / "buerste.STL")),
        ("hanger", load_and_repair(OBJECT_DIR / "hanger.stl")),
    ])

    interior_min, interior_max = get_briefcase_interior(briefcase)

    placed = place_objects(object_meshes,
                           interior_min, interior_max)

    bc_occ, pitch, origin = mesh_to_voxels(briefcase, target_dim=256)
    ref_shape = bc_occ.shape

    labels = np.zeros(ref_shape, dtype=np.uint8)
    labels[bc_occ] = 1

    for name, mesh in placed:
        occ = voxelize_to_reference(mesh, pitch,
                                    origin, ref_shape)
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
