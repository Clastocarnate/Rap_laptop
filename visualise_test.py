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
except Exception:
    PV_AVAILABLE = False

def load_and_repair(path):
    mesh = trimesh.load(path, force='mesh')
    if mesh.is_empty:
        raise ValueError("Loaded mesh empty")
    # Basic cleanups
    trimesh.repair.fix_inversion(mesh)        # fix flipped faces
    trimesh.repair.remove_duplicate_faces(mesh)
    trimesh.repair.remove_degenerate_faces(mesh)
    mesh.remove_unreferenced_vertices()
    # Fill holes if not watertight (best-effort)
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
    return mesh

def mesh_to_voxels(mesh, pitch=None, target_dim=None):
    # If user provides pitch (size of voxel in mesh units) use it.
    # Otherwise compute pitch to fit mesh bounding box into target_dim
    bbox = mesh.bounds  # array [[minx,miny,minz],[maxx,maxy,maxz]]
    dims = bbox[1] - bbox[0]
    if pitch is None:
        assert target_dim is not None, "Either pitch or target_dim must be set"
        max_dim = dims.max()
        pitch = max_dim / float(target_dim)
    # Voxelize
    v = mesh.voxelized(pitch)
    mat = v.matrix.copy()  # boolean 3D array (shape roughly (nx,ny,nz))
    # Compute origins so we know mapping from voxel indices to mesh coordinates:
    origin = v.origin  # world coords of voxel [0,0,0] corner
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

def view_3d_pyvista(volume, spacing=(1,1,1)):
    if not PV_AVAILABLE:
        print("PyVista not available. pip install pyvista for 3D viewing")
        return
    # volume should be a boolean or float grid; convert to pyvista UniformGrid
    nz, ny, nx = volume.shape
    grid = pv.UniformGrid()
    grid.dimensions = np.array(volume.shape) + 1  # dims are n+1 for cells
    grid.origin = (0,0,0)
    grid.spacing = spacing  # voxel spacing
    # set cell data
    grid.cell_arrays["values"] = volume.flatten(order="F")  # Fortran order
    p = pv.Plotter()
    p.add_volume(grid, cmap="gray", opacity="sigmoid")
    p.show()

if __name__ == "__main__":
    stl_path = "example.stl"       # replace with your file
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
