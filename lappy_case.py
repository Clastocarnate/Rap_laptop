"""
briefcase_laptop_voxel_scene.py

Creates a synthetic luggage scene with:
- a briefcase container
- a laptop placed inside it
- multi-material attenuation
- interactive 3D visualization

Requirements:
trimesh, numpy, scipy, matplotlib, nibabel, pyvista (optional)
"""

import os
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib

# ----------------------------
# Optional PyVista
# ----------------------------
try:
    import pyvista as pv
    PV_AVAILABLE = True
    print("PyVista available")
except Exception:
    PV_AVAILABLE = False
    print("PyVista not available")

# ----------------------------
# Mesh utilities
# ----------------------------
def load_and_repair(path):
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    # Fix face orientation if possible
    try:
        trimesh.repair.fix_inversion(mesh)
    except Exception:
        pass

    # --- Version-safe cleanup ---
    # Some trimesh versions expose these as methods, some don't
    if hasattr(mesh, "remove_duplicate_faces"):
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass

    if hasattr(mesh, "remove_degenerate_faces"):
        try:
            mesh.remove_degenerate_faces()
        except Exception:
            pass

    if hasattr(mesh, "remove_unreferenced_vertices"):
        try:
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass

    # Try to make watertight (best effort)
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass

    # Absolute must for stable voxelization
    mesh.apply_translation(-mesh.centroid)

    return mesh


# ----------------------------
# Voxelization
# ----------------------------
def mesh_to_voxels(mesh, pitch=None, target_dim=256):
    bbox = mesh.bounds
    dims = bbox[1] - bbox[0]

    if pitch is None:
        pitch = dims.max() / float(target_dim)

    v = mesh.voxelized(pitch)
    mat = v.matrix.copy().astype(bool)
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
                        0 <= zi < ref_shape[0] and
                        0 <= yi < ref_shape[1] and
                        0 <= xi < ref_shape[2]
                    ):
                        out[zi, yi, xi] = True
    return out

# ----------------------------
# Attenuation model
# ----------------------------
def multi_material_attenuation(label_vol):
    """
    label_vol:
    0 = air
    1 = briefcase
    2 = laptop
    """
    mu = np.zeros(label_vol.shape, dtype=np.float32)

    mu[label_vol == 0] = 0.01   # air
    mu[label_vol == 1] = 0.6   # briefcase shell
    mu[label_vol == 2] = 1.00   # laptop electronics

    mu = gaussian_filter(mu, sigma=0.8)
    return mu

# ----------------------------
# Visualization
# ----------------------------
def show_slices(volume):
    zc = volume.shape[0] // 2
    yc = volume.shape[1] // 2
    xc = volume.shape[2] // 2

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(volume[zc], cmap="gray")
    ax[0].set_title("Z slice")
    ax[1].imshow(volume[:, yc, :], cmap="gray")
    ax[1].set_title("Y slice")
    ax[2].imshow(volume[:, :, xc], cmap="gray")
    ax[2].set_title("X slice")
    plt.tight_layout()
    plt.show()

def view_3d_pyvista(volume, spacing):
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = spacing
    grid.cell_data["values"] = volume.flatten(order="F")

    p = pv.Plotter()
    p.add_volume(
        grid,
        scalars="values",
        cmap="gray",
        opacity="sigmoid",
        shade=True
    )
    p.show_axes()
    p.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    out_dir = "out_scene"
    os.makedirs(out_dir, exist_ok=True)

    # Load meshes
    briefcase = load_and_repair("briefcase.stl")
    laptop = load_and_repair("laptop.stl")

    print("Briefcase extents:", briefcase.extents)
    print("Laptop extents:", laptop.extents)

    # -------------------------------------------------
    # Place laptop inside briefcase
    # -------------------------------------------------
    bc_min, bc_max = briefcase.bounds
    margin = 0.1 * (bc_max - bc_min)

    interior_min = bc_min + margin
    interior_max = bc_max - margin
    target_center = 0.5 * (interior_min + interior_max)

    laptop.apply_translation(target_center - laptop.centroid)

    # -------------------------------------------------
    # Voxelization (briefcase defines reference grid)
    # -------------------------------------------------
    bc_occ, pitch, origin = mesh_to_voxels(
        briefcase,
        pitch=None,
        target_dim=256
    )

    ref_shape = bc_occ.shape

    lap_occ = voxelize_to_reference(
        laptop,
        pitch=pitch,
        ref_origin=origin,
        ref_shape=ref_shape
    )

    print("Voxel grid shape:", ref_shape)
    print("Pitch:", pitch)

    # -------------------------------------------------
    # Merge with priority
    # -------------------------------------------------
    labels = np.zeros(ref_shape, dtype=np.uint8)
    labels[bc_occ] = 1
    labels[lap_occ] = 2

    # -------------------------------------------------
    # Attenuation + mask
    # -------------------------------------------------
    volume = multi_material_attenuation(labels)
    laptop_mask = (labels == 2).astype(np.uint8)

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    np.save(os.path.join(out_dir, "attenuation.npy"), volume)
    np.save(os.path.join(out_dir, "laptop_mask.npy"), laptop_mask)

    nii = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nii, os.path.join(out_dir, "volume.nii.gz"))

    mask_nii = nib.Nifti1Image(laptop_mask, affine=np.eye(4))
    nib.save(mask_nii, os.path.join(out_dir, "mask.nii.gz"))

    # -------------------------------------------------
    # Visualize
    # -------------------------------------------------
    show_slices(volume)

    if PV_AVAILABLE:
        view_3d_pyvista(volume, spacing=(pitch, pitch, pitch))
