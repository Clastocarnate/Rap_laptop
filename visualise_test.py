"""
stl_to_voxel.py
Convert an STL to a voxel volume and visualize slices (pure Python).
Requires: trimesh, numpy, scipy, matplotlib, nibabel (optional), pyvista (optional)
"""
import trimesh
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib
import os



# Optional: for 3D interactive viewing (pip install pyvista)
try:
    import pyvista as pv
    PV_AVAILABLE = True
    print("Available")
except Exception:
    print("Fuck off")
    PV_AVAILABLE = False

def load_and_repair(path):
    mesh = trimesh.load(path, force='mesh')
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty")

    # Fix normals / winding
    try:
        trimesh.repair.fix_inversion(mesh)
    except Exception:
        pass

    # These are METHODS on the mesh, not in trimesh.repair
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # Try hole filling if not watertight
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass

    return mesh


def mesh_to_voxels(mesh, pitch=None, target_dim=None):
    bbox = mesh.bounds
    dims = bbox[1] - bbox[0]

    if pitch is None:
        assert target_dim is not None
        pitch = dims.max() / float(target_dim)

    v = mesh.voxelized(pitch)
    mat = v.matrix.copy()

    origin = v.bounds[0]   # ✅ correct
    return mat, pitch, origin


def occupancy_to_attenuation(occ, base_mu=0.02, laptop_mu=0.8, smoothing_sigma=1.0):
    """
    Map occupancy to attenuation map (float). This is a simple mapping:
    - background (False) -> air-like small value (base_mu)
    - object (True) -> larger value (laptop_mu)
    Then smooth for partial-volume effects.
    """
    vol = np.where(occ, laptop_mu, base_mu).astype(np.float32)
    if smoothing_sigma > 0:
        vol = gaussian_filter(vol, sigma=smoothing_sigma)
    return vol

def save_as_nifti(volume, outpath, spacing=(1.0,1.0,1.0)):
    # volume is numpy array with shape (Z,Y,X) or (nx,ny,nz)
    affine = np.diag([spacing[2], spacing[1], spacing[0], 1.0])  # crude affine
    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nii, outpath)

def show_slices(volume):
    # show three orthogonal central slices
    zc = volume.shape[0] // 2
    yc = volume.shape[1] // 2
    xc = volume.shape[2] // 2
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(volume[zc,:,:], cmap='gray'); axes[0].set_title(f'Z slice {zc}')
    axes[1].imshow(volume[:,yc,:], cmap='gray'); axes[1].set_title(f'Y slice {yc}')
    axes[2].imshow(volume[:,:,xc], cmap='gray'); axes[2].set_title(f'X slice {xc}')
    plt.show()

def view_3d_pyvista(volume, spacing=(1.0, 1.0, 1.0)):
    """
    Interactive 3D volume rendering using PyVista.
    volume shape: (Z, Y, X)
    """
    volume = np.asarray(volume)
    nz, ny, nx = volume.shape

    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = spacing

    # 🔑 correct API for new PyVista
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



if __name__ == "__main__":
    stl_path = "Closed_position.stl"       # replace with your file
    out_dir = "out_vox"
    os.makedirs(out_dir, exist_ok=True)

    mesh = load_and_repair(stl_path)
    print("Bounds:", mesh.bounds, "extents:", mesh.extents)

    # Option A: specify pitch directly (voxel size in same units as STL)
    # mat, pitch, origin = mesh_to_voxels(mesh, pitch=1.0)

    # Option B: specify a target dimension for largest axis (e.g. 256)
    mat, pitch, origin = mesh_to_voxels(mesh, pitch=None, target_dim=256)
    print("voxel grid shape (z,y,x) ~", mat.shape, "pitch:", pitch, "origin:", origin)

    # Note/trick: trimesh returns matrix in order (filled axes); treat mat as (Z,Y,X)
    occ = mat.astype(bool)

    # Optionally convert occupancy -> attenuation map
    vol = occupancy_to_attenuation(occ, base_mu=0.01, laptop_mu=1.0, smoothing_sigma=1.2)

    # Save occupancy and attenuation
    np.save(os.path.join(out_dir, "occupancy.npy"), occ)
    np.save(os.path.join(out_dir, "attenuation.npy"), vol)
    # Also NIfTI (useful for ML pipelines)
    save_as_nifti(vol, os.path.join(out_dir, "vol.nii.gz"), spacing=(pitch,pitch,pitch))
    save_as_nifti(occ.astype(np.uint8), os.path.join(out_dir, "mask.nii.gz"), spacing=(pitch,pitch,pitch))

    # Quick visualization
    show_slices(vol)
    if PV_AVAILABLE:
        view_3d_pyvista(vol, spacing=(pitch,pitch,pitch))
        view_volume_pyvista(vol, pitch)

