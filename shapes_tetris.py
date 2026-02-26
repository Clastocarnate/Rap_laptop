import pyvista as pv
import numpy as np
import random
import os
from itertools import permutations

# ---------------------------------------------------------------------------
# Container dimensions (from 2xBriefcase bounding box)
# ---------------------------------------------------------------------------
CONTAINER = (236.0, 324.0, 149.4)  # W, H, D  (2xBriefcase reduced by 10% per dim)

# ---------------------------------------------------------------------------
# Object cuboid dimensions (computed from STL bounding boxes)
# Excluding Briefcase.stl and 2xBriefcase.stl (container)
# ---------------------------------------------------------------------------
OBJECTS = {
    "laptop":  (68.60,  8.94,  56.13),
    "AirPods": (19.37, 19.48,  39.09),
    "Charger": (41.00, 59.00,  30.00),
    "Diary":   (91.00, 71.40,   6.00),
    "Pen":     (57.00,  4.00,   4.00),
    "buerste": (170.59, 34.14, 22.85),
    "comb":    (177.80,  2.54, 55.88),
    "hanger":  (180.34, 385.05, 4.33),
}

# Map object names to STL filenames
STL_FILES = {
    "laptop":  "laptop.stl",
    "AirPods": "AirPods.stl",
    "Charger": "Charger.stl",
    "Diary":   "Diary.stl",
    "Pen":     "Pen.stl",
    "buerste": "buerste.STL",
    "comb":    "comb.stl",
    "hanger":  "hanger.stl",
}

OBJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objects")


# ---------------------------------------------------------------------------
# STL mesh loading + caching
# ---------------------------------------------------------------------------

_mesh_cache = {}

def load_stl_centered(name):
    """Load an STL, center it at origin, and cache it."""
    if name in _mesh_cache:
        return _mesh_cache[name]
    path = os.path.join(OBJECTS_DIR, STL_FILES[name])
    mesh = pv.read(path)
    # Center at origin
    center = np.array(mesh.center)
    mesh.translate(-center, inplace=True)
    _mesh_cache[name] = mesh
    return mesh


def orient_mesh_to_placement(mesh, orig_dims, placed_dims):
    """
    Given an STL mesh centered at origin with original bounding box dims
    (orig_w, orig_h, orig_d), and the packer chose orientation (pw, ph, pd),
    apply 90-degree rotations so the mesh axes match the placement.

    Strategy: figure out which permutation maps orig_dims → placed_dims,
    then apply the corresponding axis swaps via rotation matrices.
    """
    mesh = mesh.copy()
    ow, oh, od = orig_dims
    pw, ph, pd = placed_dims

    # Find which permutation of original dims matches placed dims
    # We compare with tolerance since floats
    orig = np.array([ow, oh, od])
    target = np.array([pw, ph, pd])

    # Try all 6 permutations of axes
    best_perm = None
    best_err = float('inf')
    for perm in permutations(range(3)):
        err = np.sum(np.abs(orig[list(perm)] - target))
        if err < best_err:
            best_err = err
            best_perm = perm

    # Apply rotation based on permutation
    # Identity = (0,1,2), no rotation needed
    # We use 90-degree rotations around principal axes
    if best_perm == (0, 1, 2):
        pass  # no rotation
    elif best_perm == (0, 2, 1):
        mesh.rotate_x(90, inplace=True)
    elif best_perm == (1, 0, 2):
        mesh.rotate_z(90, inplace=True)
    elif best_perm == (1, 2, 0):
        mesh.rotate_z(90, inplace=True)
        mesh.rotate_x(90, inplace=True)
    elif best_perm == (2, 0, 1):
        mesh.rotate_x(90, inplace=True)
        mesh.rotate_z(90, inplace=True)
    elif best_perm == (2, 1, 0):
        mesh.rotate_y(90, inplace=True)

    # Scale to exactly match the placed box dimensions
    bounds = mesh.bounds
    current_dims = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    # Avoid division by zero
    scale = np.array(target) / np.maximum(current_dims, 1e-6)
    mesh.scale(scale, inplace=True)

    return mesh


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_orientations(w, h, d):
    """Return all unique axis-aligned orientations (permutations of dims)."""
    return list(set(permutations((w, h, d))))


def fits_in_container(x, y, z, w, h, d, container):
    """Check if a box placed at (x,y,z) with size (w,h,d) fits in container."""
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= container[0] + 0.01 and
            y + h <= container[1] + 0.01 and
            z + d <= container[2] + 0.01)


def overlaps(box_a, box_b):
    """Check if two axis-aligned boxes overlap. Each box = (x, y, z, w, h, d)."""
    ax, ay, az, aw, ah, ad = box_a
    bx, by, bz, bw, bh, bd = box_b
    return (ax < bx + bw and ax + aw > bx and
            ay < by + bh and ay + ah > by and
            az < bz + bd and az + ad > bz)


# ---------------------------------------------------------------------------
# Extreme-Point Best-Fit Packing
# ---------------------------------------------------------------------------

def generate_eps_from_placement(px, py, pz, pw, ph, pd):
    """Generate new extreme points from a placed box."""
    return [
        (px + pw, py,      pz),       # right face
        (px,      py + ph, pz),       # top face
        (px,      py,      pz + pd),  # front face
        (px + pw, py + ph, pz),       # right-top edge
        (px + pw, py,      pz + pd),  # right-front edge
        (px,      py + ph, pz + pd),  # top-front edge
    ]


def pack_remaining(container, placed, other_objects):
    """
    Pack remaining objects using Extreme-Point Best-Fit.
    Objects are distributed across the full container height (not just bottom)
    by randomising a target Y for each object and seeding EPs at multiple levels.
    """
    CW, CH, CD = container

    extreme_points = set()

    # Seed extreme points at multiple heights & positions so objects
    # can start packing anywhere inside the container (not just origin).
    for y_frac in [0.0, 0.25, 0.5, 0.75]:
        for x_frac in [0.0, 0.5]:
            for z_frac in [0.0, 0.5]:
                extreme_points.add((x_frac * CW, y_frac * CH, z_frac * CD))

    # Add EPs from already-placed objects (e.g. laptop)
    for p in placed:
        _, px, py, pz, pw, ph, pd = p
        for ep in generate_eps_from_placement(px, py, pz, pw, ph, pd):
            if ep[0] < CW and ep[1] < CH and ep[2] < CD:
                extreme_points.add(ep)

    # Sort objects by volume descending (largest first for better packing)
    sorted_objs = sorted(other_objects.items(),
                         key=lambda item: item[1][0] * item[1][1] * item[1][2],
                         reverse=True)

    for name, dims in sorted_objs:
        best = None
        best_score = float('inf')

        # Random target Y spreads objects across the full height
        target_y = random.uniform(0, CH)

        for ep in list(extreme_points):
            ex, ey, ez = ep
            for orientation in get_orientations(*dims):
                ow, oh, od = orientation

                # Containment check
                if not fits_in_container(ex, ey, ez, ow, oh, od, container):
                    continue

                # Collision check against all placed boxes
                new_box = (ex, ey, ez, ow, oh, od)
                collision = False
                for p in placed:
                    if overlaps(new_box, (p[1], p[2], p[3], p[4], p[5], p[6])):
                        collision = True
                        break
                if collision:
                    continue

                # Score: prefer placement near the random target_y
                # Also lightly prefer tight packing on X and Z
                score = abs(ey - target_y) * 10 + ez * 3 + ex * 1
                if score < best_score:
                    best_score = score
                    best = (name, ex, ey, ez, ow, oh, od)

        if best is not None:
            placed.append(best)
            _, bx, by, bz, bw, bh, bd = best
            for ep in generate_eps_from_placement(bx, by, bz, bw, bh, bd):
                if ep[0] < CW and ep[1] < CH and ep[2] < CD:
                    extreme_points.add(ep)

    return placed


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

MAX_COPIES = 7  # max duplicates of each non-laptop object

def build_object_pool(objects_dict, container):
    """
    Duplicate each non-laptop object up to MAX_COPIES times.
    The pool is a list of (name, w, h, d) tuples.
    """
    container_vol = container[0] * container[1] * container[2]
    target_vol = 0.80 * container_vol  # 80% fill target

    other = {k: v for k, v in objects_dict.items() if k != "laptop"}

    pool = []
    for copies in range(1, MAX_COPIES + 1):
        pool = []
        total_vol = 0
        for name, dims in other.items():
            for c in range(copies):
                tag = f"{name}" if c == 0 else f"{name}_{c+1}"
                pool.append((tag, dims))
                total_vol += dims[0] * dims[1] * dims[2]
        if total_vol >= target_vol:
            break

    print(f"  Object pool: {len(pool)} items, "
          f"total volume = {total_vol:,.0f} / {container_vol:,.0f} "
          f"({100*total_vol/container_vol:.1f}%)")
    return pool


def generate_scene(container, objects_dict):
    """Generate one packed scene: laptop placed randomly first, then pack rest."""
    laptop_dims = objects_dict["laptop"]
    CW, CH, CD = container

    # Step 1: Place laptop at a random valid position with random 90 deg orientation
    placed = []
    for _ in range(1000):
        orientation = random.choice(get_orientations(*laptop_dims))
        lw, lh, ld = orientation
        if lw > CW or lh > CH or ld > CD:
            continue
        x = random.uniform(0, CW - lw)
        y = random.uniform(0, CH - lh)
        z = random.uniform(0, CD - ld)
        placed.append(("laptop", x, y, z, lw, lh, ld))
        break

    # Step 2: Build duplicated pool and pack optimally
    pool = build_object_pool(objects_dict, container)
    pool_dict = {}
    for tag, dims in pool:
        pool_dict[tag] = dims
    placed = pack_remaining(container, placed, pool_dict)
    return placed


def compute_fill(scene, container):
    """Compute fill percentage of a scene."""
    container_vol = container[0] * container[1] * container[2]
    filled = sum(obj[4] * obj[5] * obj[6] for obj in scene)
    return filled, container_vol, 100.0 * filled / container_vol


# ---------------------------------------------------------------------------
# X-ray scan color mapping
# ---------------------------------------------------------------------------
XRAY_COLORS = {
    "laptop":  ("#FF8C00", 0.90),   # orange - electronics, circuit boards
    "Charger": ("#1E90FF", 0.75),   # blue - metal prongs, copper wiring, dense
    "AirPods": ("#4169E1", 0.75),   # royal blue - metal case, lithium battery
    "Diary":   ("#FFD700", 0.55),   # yellow - paper, very low density organic
    "Pen":     ("#FFEC8B", 0.50),   # light yellow - plastic body, thin metal tip
    "buerste": ("#32CD32", 0.60),   # green - bristles + plastic, mixed material
    "comb":    ("#ADFF2F", 0.50),   # yellow-green - pure plastic, low density
    "hanger":  ("#00CED1", 0.65),   # dark cyan - metal wire, moderate density
}

def get_base_name(tag):
    """Strip the duplicate suffix to get the base object name for color lookup."""
    for base in XRAY_COLORS:
        if tag == base or tag.startswith(base + "_"):
            return base
    return None


# ---------------------------------------------------------------------------
# Visualization: 10x2 paired subplot (with laptop / without laptop)
# ---------------------------------------------------------------------------

def render_scene_to_subplot(plotter, scene, container, title, skip_laptop=False):
    """Render a single scene into the current subplot."""
    # Container wireframe
    container_mesh = pv.Box(bounds=(0, container[0],
                                    0, container[1],
                                    0, container[2]))
    plotter.add_mesh(container_mesh, style='wireframe',
                     color='white', line_width=2)

    # Draw each placed object using actual STL mesh
    for obj in scene:
        tag, x, y, z, w, h, d = obj
        base = get_base_name(tag)
        if base is None:
            continue
        if skip_laptop and base == "laptop":
            continue

        orig_mesh = load_stl_centered(base)
        orig_dims = OBJECTS[base]
        placed_dims = (w, h, d)

        mesh = orient_mesh_to_placement(orig_mesh, orig_dims, placed_dims)
        box_center = np.array([x + w/2, y + h/2, z + d/2])
        mesh.translate(box_center, inplace=True)

        color, opacity = XRAY_COLORS.get(base, ("#AAAAAA", 0.5))
        plotter.add_mesh(mesh, color=color, opacity=opacity,
                         show_edges=False, smooth_shading=True)

    plotter.add_text(title, font_size=8, color='white')
    plotter.set_background('#1a1a2e')


def visualize_paired_scenes(scenes, container):
    """10x2 grid: left = with laptop, right = without laptop."""
    n = len(scenes)
    plotter = pv.Plotter(shape=(n, 2), window_size=(1200, 4000))

    for idx, scene in enumerate(scenes):
        filled, _, pct = compute_fill(scene, container)

        # Left column: WITH laptop
        plotter.subplot(idx, 0)
        render_scene_to_subplot(
            plotter, scene, container,
            title=f"Scene {idx+1} WITH laptop  |  {len(scene)} objs  |  {pct:.1f}%",
            skip_laptop=False
        )

        # Right column: WITHOUT laptop
        plotter.subplot(idx, 1)
        scene_no_laptop = [o for o in scene if get_base_name(o[0]) != "laptop"]
        filled_nl, _, pct_nl = compute_fill(scene_no_laptop, container)
        render_scene_to_subplot(
            plotter, scene, container,
            title=f"Scene {idx+1} NO laptop  |  {len(scene_no_laptop)} objs  |  {pct_nl:.1f}%",
            skip_laptop=True
        )

    plotter.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(None)

    NUM_SCENES = 5

    print("Pre-loading STL meshes...")
    for name in STL_FILES:
        load_stl_centered(name)
        print(f"  Loaded {name}")

    print(f"\nGenerating {NUM_SCENES} scenes...")
    container_vol = CONTAINER[0] * CONTAINER[1] * CONTAINER[2]
    print(f"Container volume: {container_vol:,.0f}\n")

    scenes = []
    for i in range(NUM_SCENES):
        print(f"--- Scene {i+1} ---")
        scene = generate_scene(CONTAINER, OBJECTS)
        scenes.append(scene)
        filled, _, pct = compute_fill(scene, CONTAINER)
        print(f"  Placed {len(scene)} objects, fill = {pct:.1f}%")
        print()

    print("Opening paired visualization (with / without laptop)...")
    visualize_paired_scenes(scenes, CONTAINER)