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
# Configuration
# ------------------------------------------------------------
class PackingConfig:
    # Collision margins (as fraction of object size)
    LAPTOP_MIN_SEPARATION = 0.15  # Laptop needs more space to be distinguishable
    OBJECT_MARGIN = 0.05          # General object margin
    
    # Placement attempts
    MAX_PLACEMENT_ATTEMPTS = 50
    MAX_COLLISION_RESOLVE_STEPS = 20
    
    # Physics-like settling
    GRAVITY_STEPS = 10
    GRAVITY_STEP_SIZE = 2.0
    
    # Interior margins
    BRIEFCASE_MARGIN_RATIO = 0.15

# ------------------------------------------------------------
# Mesh Utilities
# ------------------------------------------------------------
def load_and_repair(path):
    mesh = trimesh.load(str(path), force="mesh")

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    try:
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    mesh.apply_translation(-mesh.centroid)
    return mesh

def get_mesh_size(mesh):
    """Get characteristic size of mesh"""
    bounds = mesh.bounds
    return np.linalg.norm(bounds[1] - bounds[0])

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
def get_briefcase_interior(briefcase, margin_ratio=PackingConfig.BRIEFCASE_MARGIN_RATIO):
    bc_min, bc_max = briefcase.bounds
    margin = margin_ratio * (bc_max - bc_min)
    return bc_min + margin, bc_max - margin

def get_expanded_bounds(bounds, margin):
    """Expand bounding box by margin"""
    min_corner, max_corner = bounds
    size = max_corner - min_corner
    expansion = margin * size
    return min_corner - expansion, max_corner + expansion

def aabb_collision(bounds1, bounds2, margin1=0.0, margin2=0.0):
    """Check AABB collision with configurable margins"""
    exp_bounds1 = get_expanded_bounds(bounds1, margin1)
    exp_bounds2 = get_expanded_bounds(bounds2, margin2)
    
    min1, max1 = exp_bounds1
    min2, max2 = exp_bounds2
    
    return np.all(max1 > min2) and np.all(max2 > min1)

def is_inside_bounds(mesh_bounds, interior_min, interior_max, tolerance=0.1):
    """Check if mesh is fully inside interior with tolerance"""
    m_min, m_max = mesh_bounds
    return np.all(m_min >= interior_min - tolerance) and \
           np.all(m_max <= interior_max + tolerance)

def clamp_mesh_inside(mesh, interior_min, interior_max):
    """Clamp mesh to stay inside bounds"""
    m_min, m_max = mesh.bounds
    shift = np.zeros(3)

    for i in range(3):
        if m_min[i] < interior_min[i]:
            shift[i] = interior_min[i] - m_min[i]
        elif m_max[i] > interior_max[i]:
            shift[i] = interior_max[i] - m_max[i]

    if np.any(shift != 0):
        mesh.apply_translation(shift)
    return mesh

def get_separation_distance(bounds1, bounds2):
    """Calculate minimum distance between two bounding boxes"""
    center1 = (bounds1[0] + bounds1[1]) / 2
    center2 = (bounds2[0] + bounds2[1]) / 2
    return np.linalg.norm(center1 - center2)

# ------------------------------------------------------------
# Improved Collision Resolution
# ------------------------------------------------------------
def find_valid_position(mesh, placed_objects, interior_min, interior_max, 
                       is_laptop=False, max_attempts=PackingConfig.MAX_PLACEMENT_ATTEMPTS):
    """
    Try multiple random positions to find valid placement
    """
    base_mesh = mesh.copy()
    best_mesh = None
    best_collision_count = float('inf')
    
    for attempt in range(max_attempts):
        test_mesh = base_mesh.copy()
        
        # Random position within interior
        pos = interior_min + np.random.rand(3) * (interior_max - interior_min)
        test_mesh.apply_translation(pos - test_mesh.centroid)
        
        # Apply gravity-like settling (move down until collision or floor)
        test_mesh = simulate_gravity_settle(test_mesh, placed_objects, 
                                            interior_min, interior_max)
        
        # Check collisions
        collision_count = 0
        valid = True
        
        for obj_name, obj_mesh in placed_objects:
            obj_margin = PackingConfig.OBJECT_MARGIN
            test_margin = PackingConfig.LAPTOP_MIN_SEPARATION if is_laptop else PackingConfig.OBJECT_MARGIN
            
            # Special handling for laptop - needs more separation
            if is_laptop or obj_name == "laptop":
                test_margin = max(test_margin, PackingConfig.LAPTOP_MIN_SEPARATION)
            
            if aabb_collision(test_mesh.bounds, obj_mesh.bounds, 
                            test_margin, obj_margin):
                collision_count += 1
                valid = False
        
        # Check if inside bounds
        if not is_inside_bounds(test_mesh.bounds, interior_min, interior_max):
            valid = False
            collision_count += 100  # Penalize heavily
        
        if valid:
            return test_mesh, True
        
        # Track best attempt
        if collision_count < best_collision_count:
            best_collision_count = collision_count
            best_mesh = test_mesh.copy()
    
    # Return best attempt even if not perfect
    return best_mesh, False

def simulate_gravity_settle(mesh, placed_objects, interior_min, interior_max):
    """
    Simulate gravity by moving object downward until collision or floor
    """
    gravity_direction = np.array([0, 0, -1])  # Assuming Z is up
    
    for _ in range(PackingConfig.GRAVITY_STEPS):
        test_mesh = mesh.copy()
        test_mesh.apply_translation(gravity_direction * PackingConfig.GRAVITY_STEP_SIZE)
        
        # Check if hit floor
        if test_mesh.bounds[0][2] < interior_min[2]:
            break
        
        # Check if hit another object
        collision = False
        for _, obj_mesh in placed_objects:
            if aabb_collision(test_mesh.bounds, obj_mesh.bounds, 
                            PackingConfig.OBJECT_MARGIN, PackingConfig.OBJECT_MARGIN):
                collision = True
                break
        
        if collision:
            break
        
        mesh = test_mesh
    
    return mesh

def resolve_collision_iterative(mesh, placed_objects, interior_min, interior_max,
                                is_laptop=False):
    """
    Iteratively push object away from collisions
    """
    for step in range(PackingConfig.MAX_COLLISION_RESOLVE_STEPS):
        collision_vectors = []
        
        for obj_name, obj_mesh in placed_objects:
            obj_margin = PackingConfig.OBJECT_MARGIN
            test_margin = PackingConfig.LAPTOP_MIN_SEPARATION if is_laptop else PackingConfig.OBJECT_MARGIN
            
            if is_laptop or obj_name == "laptop":
                test_margin = max(test_margin, PackingConfig.LAPTOP_MIN_SEPARATION)
            
            if aabb_collision(mesh.bounds, obj_mesh.bounds, test_margin, obj_margin):
                # Calculate push direction
                my_center = (mesh.bounds[0] + mesh.bounds[1]) / 2
                obj_center = (obj_mesh.bounds[0] + obj_mesh.bounds[1]) / 2
                direction = my_center - obj_center
                
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    direction = np.random.randn(3)
                    norm = np.linalg.norm(direction)
                
                collision_vectors.append(direction / norm)
        
        if not collision_vectors:
            return True  # No collisions
        
        # Average push direction
        avg_direction = np.mean(collision_vectors, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)
        
        # Push with adaptive step size
        step_size = 5.0 * (1.0 + step * 0.2)  # Increase push over iterations
        mesh.apply_translation(avg_direction * step_size)
        mesh = clamp_mesh_inside(mesh, interior_min, interior_max)
    
    return False

# ------------------------------------------------------------
# Improved Placement Logic
# ------------------------------------------------------------
def place_objects(object_meshes, interior_min, interior_max, verbose=True):
    """
    Place objects with improved collision handling and packing
    """
    # Sort objects by size (largest first) except laptop (always first)
    laptop_obj = None
    other_objects = []
    
    for name, mesh in object_meshes:
        if name == "laptop":
            laptop_obj = (name, mesh)
        else:
            size = get_mesh_size(mesh)
            other_objects.append((name, mesh, size))
    
    # Sort by size descending
    other_objects.sort(key=lambda x: x[2], reverse=True)
    
    # Reorder: laptop first, then by size
    ordered_objects = []
    if laptop_obj:
        ordered_objects.append(laptop_obj)
    ordered_objects.extend([(name, mesh) for name, mesh, _ in other_objects])
    
    placed = []
    skipped = []
    
    for idx, (name, base_mesh) in enumerate(ordered_objects):
        is_laptop = (name == "laptop")
        
        # Apply rotation
        mesh = base_mesh.copy()
        if is_laptop:
            R = limited_rotation(max_deg=30)
        else:
            R = trimesh.transformations.random_rotation_matrix()
        mesh.apply_transform(R)
        
        # Try to find valid position
        placed_mesh, success = find_valid_position(
            mesh, placed, interior_min, interior_max, is_laptop
        )
        
        if not success:
            # Try collision resolution as fallback
            success = resolve_collision_iterative(
                placed_mesh, placed, interior_min, interior_max, is_laptop
            )
        
        # Final validation
        final_valid = True
        if success:
            for obj_name, obj_mesh in placed:
                margin1 = PackingConfig.LAPTOP_MIN_SEPARATION if is_laptop else PackingConfig.OBJECT_MARGIN
                margin2 = PackingConfig.LAPTOP_MIN_SEPARATION if obj_name == "laptop" else PackingConfig.OBJECT_MARGIN
                
                if aabb_collision(placed_mesh.bounds, obj_mesh.bounds, margin1, margin2):
                    final_valid = False
                    break
            
            if not is_inside_bounds(placed_mesh.bounds, interior_min, interior_max, tolerance=1.0):
                final_valid = False
        else:
            final_valid = False
        
        if final_valid:
            placed.append((name, placed_mesh))
            if verbose:
                print(f"✓ Placed {name} (object {idx+1}/{len(ordered_objects)})")
        else:
            skipped.append(name)
            if verbose:
                print(f"✗ Skipped {name} - could not resolve collisions")
    
    if verbose:
        print(f"\nPlacement summary: {len(placed)}/{len(ordered_objects)} objects placed")
        if skipped:
            print(f"Skipped objects: {', '.join(skipped)}")
    
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
# Scene Generation
# ------------------------------------------------------------
def generate_random_scene(briefcase, interior_min, interior_max, scene_id, verbose=False):
    """Generate a single random scene"""
    object_meshes = []

    # Laptop (always 1)
    object_meshes.append(("laptop",
                          load_and_repair(OBJECT_DIR / "laptop.stl")))

    # Random small objects with varying quantities
    num_pens = np.random.randint(3, 8)
    for i in range(num_pens):
        object_meshes.append((f"pen_{i}",
                              load_and_repair(OBJECT_DIR / "Pen.stl")))

    num_airpods = np.random.randint(2, 6)
    for i in range(num_airpods):
        object_meshes.append((f"airpods_{i}",
                              load_and_repair(OBJECT_DIR / "AirPods.stl")))

    num_combs = np.random.randint(3, 7)
    for i in range(num_combs):
        object_meshes.append((f"comb_{i}",
                              load_and_repair(OBJECT_DIR / "comb.stl")))

    # Medium/large objects
    object_meshes.extend([
        ("charger", load_and_repair(OBJECT_DIR / "Charger.stl")),
        ("diary", load_and_repair(OBJECT_DIR / "Diary.stl")),
        ("brush", load_and_repair(OBJECT_DIR / "buerste.STL")),
        ("hanger", load_and_repair(OBJECT_DIR / "hanger.stl")),
    ])

    # Place objects
    if verbose:
        print(f"\n{'='*60}")
        print(f"Scene {scene_id}: Placing {len(object_meshes)} objects...")
        print(f"{'='*60}")
    
    placed = place_objects(object_meshes, interior_min, interior_max, verbose=verbose)

    # Voxelization
    bc_occ, pitch, origin = mesh_to_voxels(briefcase, target_dim=256)
    ref_shape = bc_occ.shape

    labels = np.zeros(ref_shape, dtype=np.uint8)
    labels[bc_occ] = 1

    for name, mesh in placed:
        occ = voxelize_to_reference(mesh, pitch, origin, ref_shape)
        if name == "laptop":
            labels[occ] = 2
        elif "airpods" in name or name == "charger":
            labels[occ] = 3
        else:
            labels[occ] = 4

    volume = attenuation_from_labels(labels)
    
    # Save scene
    scene_dir = OUTPUT_DIR / f"scene_{scene_id:03d}"
    scene_dir.mkdir(exist_ok=True)
    np.save(scene_dir / "volume.npy", volume)
    np.save(scene_dir / "labels.npy", labels)
    
    return volume, labels, pitch, len(placed)

# ------------------------------------------------------------
# Visualization - Multiple Scenes
# ------------------------------------------------------------
def show_all_scenes_matplotlib(scenes_data):
    """Show all 10 scenes in a single matplotlib window"""
    n_scenes = len(scenes_data)
    
    # Create figure with subplots (2 rows x 5 columns for 10 scenes)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('10 Random Briefcase Scenes (Z-slice view)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (volume, labels, scene_id) in enumerate(scenes_data):
        zc = volume.shape[0] // 2
        
        # Show volume
        axes[idx].imshow(volume[zc], cmap="gray")
        axes[idx].set_title(f'Scene {scene_id}', fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_all_scenes_pyvista(scenes_data):
    """Show all scenes sequentially in PyVista with user control"""
    if not PV_AVAILABLE:
        print("PyVista not available for 3D visualization")
        return
    
    print("\n" + "="*60)
    print("3D Visualization - Press 'q' to move to next scene")
    print("="*60)
    
    for idx, (volume, labels, scene_id, pitch) in enumerate(scenes_data):
        print(f"\nShowing Scene {scene_id} ({idx+1}/{len(scenes_data)})")
        
        # Create PyVista grid
        grid = pv.ImageData()
        grid.dimensions = np.array(volume.shape) + 1
        grid.spacing = (pitch, pitch, pitch)
        grid.origin = (0, 0, 0)
        grid.cell_data["values"] = volume.flatten(order="F")
        
        # Create plotter
        p = pv.Plotter()
        p.add_text(f"Scene {scene_id} - Press 'q' for next scene", 
                   position='upper_edge', font_size=12, color='white')
        
        p.add_volume(grid, scalars="values",
                     opacity="sigmoid", cmap="gray")
        p.show_axes()
        p.camera_position = 'iso'
        
        # Show and wait for user to press 'q'
        p.show()
        p.close()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("3D Briefcase Packing - 10 Random Scenes Generator")
    print("=" * 60)
    
    # Load briefcase once
    print("\nLoading briefcase...")
    briefcase = load_and_repair(OBJECT_DIR / "2xBriefcase.stl")
    interior_min, interior_max = get_briefcase_interior(briefcase)
    print(f"Briefcase interior: {interior_min} to {interior_max}")
    
    # Generate 10 scenes
    n_scenes = 10
    scenes_data_matplotlib = []
    scenes_data_pyvista = []
    
    print(f"\nGenerating {n_scenes} random scenes...")
    print("=" * 60)
    
    for i in range(n_scenes):
        scene_id = i + 1
        print(f"\n{'#'*60}")
        print(f"# GENERATING SCENE {scene_id}/{n_scenes}")
        print(f"{'#'*60}")
        
        volume, labels, pitch, num_placed = generate_random_scene(
            briefcase, interior_min, interior_max, scene_id, verbose=True
        )
        
        # Store for matplotlib (without pitch)
        scenes_data_matplotlib.append((volume, labels, scene_id))
        
        # Store for pyvista (with pitch)
        scenes_data_pyvista.append((volume, labels, scene_id, pitch))
        
        print(f"\n✓ Scene {scene_id} complete: {num_placed} objects placed")
        
        # Statistics
        laptop_voxels = np.sum(labels == 2)
        total_object_voxels = np.sum(labels > 1)
        print(f"  Laptop voxels: {laptop_voxels:,}")
        print(f"  Total object voxels: {total_object_voxels:,}")
        print(f"  Laptop ratio: {laptop_voxels/max(total_object_voxels, 1)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("ALL SCENES GENERATED!")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_DIR}")
    
    # Show all scenes in matplotlib
    print("\nDisplaying all scenes in matplotlib...")
    show_all_scenes_matplotlib(scenes_data_matplotlib)
    
    # Show all scenes in PyVista sequentially
    print("\nPreparing PyVista 3D visualization...")
    show_all_scenes_pyvista(scenes_data_pyvista)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)