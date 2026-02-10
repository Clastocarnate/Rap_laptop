# Dependencies
pip install trimesh numpy scipy matplotlib nibabel pyvista


# Visualiser Testing
- visualiser.py = .stl -> voxels -> visualise
### Theory
- Input: .stl file which are basically meshes
- Visualiser.py -> converts meshes to voxels + assign attenuation + visualise in 3D space
## The Key Conceptual Gap

### What a CT Scanner Measures

A computed tomography (CT) scanner does **not** directly measure a 3D image of an object. Instead, it measures how X‑rays are attenuated as they pass through the object from many different directions.

Each CT measurement corresponds to a **line integral** of the object’s attenuation field and can be written as:

[
p(\theta, s) = \int \mu(x, y, z) , dl
]

where:

* ( \mu(x, y, z) ) is the linear attenuation coefficient at a spatial location,
* ( dl ) denotes integration along the X‑ray path,
* ( \theta ) is the projection angle, and
* ( s ) is the detector position.

By acquiring these measurements over many angles, the scanner builds a set of projections (often called a *sinogram*). Reconstruction algorithms such as **Filtered Backprojection (FBP)** or **iterative reconstruction methods** are then used to recover the volumetric attenuation field ( \mu(x, y, z) ).

---

### What the Current Pipeline Produces

The current pipeline directly constructs the volumetric attenuation field:

[
\mu(x, y, z)
]

This is done by voxelizing CAD geometry and assigning attenuation values to the resulting voxels. While this volume represents the *ideal* reconstruction target of a CT scanner, it is **not** what a scanner measures directly.

---

### The Missing Link

Because the pipeline starts directly from ( \mu(x, y, z) ), it bypasses the CT forward model entirely:

* X‑ray projections are not simulated
* Multi‑angle line integrals are not computed
* Reconstruction artifacts and acquisition effects are not introduced

Forward projection followed by reconstruction is the critical missing step that bridges the gap between a geometrically correct synthetic volume and a physically realistic CT scan.
