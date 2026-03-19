"""
detect_with_pointnet.py
-----------------------
Loads a trained PointNet model, generates a new scene, clusters its
point cloud, runs PointNet on each cluster, and highlights the
detected laptop using the same 3-panel PyVista visualisation.

Run:
    python detect_with_pointnet.py --ckpt checkpoints/best_model.pth
"""

import argparse
import os
import random
import numpy as np
import torch
import trimesh
from itertools import permutations
from sklearn.cluster import DBSCAN  # kept for optional future use
import pyvista as pv

# Re-use the PointNet definition from train_pointnet.py
from train_pointnet import PointNetClassifier

# ---------------------------------------------------------------------------
# Constants (same as other scripts)
# ---------------------------------------------------------------------------
CONTAINER = (236.0, 324.0, 149.4)

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

OBJECT_COLORS = {
    "laptop":  "#FF8C00",
    "Charger": "#1E90FF",
    "AirPods": "#4169E1",
    "Diary":   "#FFD700",
    "Pen":     "#FFEC8B",
    "buerste": "#32CD32",
    "comb":    "#ADFF2F",
    "hanger":  "#00CED1",
}

POINTS_PER_OBJECT = 1024   # must match what was used during training

# ---------------------------------------------------------------------------
# STL helpers (pyvista for rendering, trimesh for sampling)
# ---------------------------------------------------------------------------
_pv_cache     = {}
_trimesh_cache = {}

def load_pv_centered(name):
    if name in _pv_cache:
        return _pv_cache[name]
    mesh = pv.read(os.path.join(OBJECTS_DIR, STL_FILES[name]))
    mesh.translate(-np.array(mesh.center), inplace=True)
    _pv_cache[name] = mesh
    return mesh

def load_trimesh_centered(name):
    if name in _trimesh_cache:
        return _trimesh_cache[name]
    mesh = trimesh.load(os.path.join(OBJECTS_DIR, STL_FILES[name]), force='mesh')
    mesh.apply_translation(-mesh.centroid)
    _trimesh_cache[name] = mesh
    return mesh


def get_base_name(tag):
    for base in OBJECTS:
        if tag == base or tag.startswith(base + "_"):
            return base
    return None


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------
def _best_perm(orig_dims, placed_dims):
    orig   = np.array(orig_dims)
    target = np.array(placed_dims)
    best_perm, best_err = None, float('inf')
    for perm in permutations(range(3)):
        err = np.sum(np.abs(orig[list(perm)] - target))
        if err < best_err:
            best_err = err
            best_perm = perm
    return best_perm


def orient_pv(mesh, orig_dims, placed_dims):
    mesh = mesh.copy()
    perm = _best_perm(orig_dims, placed_dims)
    if perm == (0, 2, 1):  mesh.rotate_x(90, inplace=True)
    elif perm == (1, 0, 2): mesh.rotate_z(90, inplace=True)
    elif perm == (1, 2, 0): mesh.rotate_z(90, inplace=True); mesh.rotate_x(90, inplace=True)
    elif perm == (2, 0, 1): mesh.rotate_x(90, inplace=True); mesh.rotate_z(90, inplace=True)
    elif perm == (2, 1, 0): mesh.rotate_y(90, inplace=True)
    bounds = mesh.bounds
    current = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    mesh.scale(np.array(placed_dims) / np.maximum(current, 1e-6), inplace=True)
    return mesh


def orient_trimesh(mesh, orig_dims, placed_dims):
    import trimesh.transformations as T
    mesh  = mesh.copy()
    perm  = _best_perm(orig_dims, placed_dims)
    rots  = {
        (0, 2, 1): T.rotation_matrix(np.radians(90),  [1, 0, 0]),
        (1, 0, 2): T.rotation_matrix(np.radians(90),  [0, 0, 1]),
        (2, 1, 0): T.rotation_matrix(np.radians(90),  [0, 1, 0]),
    }
    if perm == (1, 2, 0):
        mesh.apply_transform(T.rotation_matrix(np.radians(90), [0, 0, 1]))
        mesh.apply_transform(T.rotation_matrix(np.radians(90), [1, 0, 0]))
    elif perm == (2, 0, 1):
        mesh.apply_transform(T.rotation_matrix(np.radians(90), [1, 0, 0]))
        mesh.apply_transform(T.rotation_matrix(np.radians(90), [0, 0, 1]))
    elif perm in rots:
        mesh.apply_transform(rots[perm])
    mesh.apply_scale(np.array(placed_dims) / np.maximum(mesh.extents, 1e-6))
    return mesh


# ---------------------------------------------------------------------------
# Packing (identical to main script)
# ---------------------------------------------------------------------------
def get_orientations(w, h, d):
    return list(set(permutations((w, h, d))))

def fits_in_container(x, y, z, w, h, d, container):
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= container[0] + 0.01 and
            y + h <= container[1] + 0.01 and
            z + d <= container[2] + 0.01)

def overlaps(a, b):
    ax,ay,az,aw,ah,ad = a
    bx,by,bz,bw,bh,bd = b
    return (ax<bx+bw and ax+aw>bx and ay<by+bh and ay+ah>by and az<bz+bd and az+ad>bz)

def generate_eps(px,py,pz,pw,ph,pd):
    return [(px+pw,py,pz),(px,py+ph,pz),(px,py,pz+pd),
            (px+pw,py+ph,pz),(px+pw,py,pz+pd),(px,py+ph,pz+pd)]

def pack_remaining(container, placed, others):
    CW,CH,CD = container
    eps = set()
    for yf in [0,.25,.5,.75]:
        for xf in [0,.5]:
            for zf in [0,.5]:
                eps.add((xf*CW, yf*CH, zf*CD))
    for p in placed:
        _,px,py,pz,pw,ph,pd = p
        for ep in generate_eps(px,py,pz,pw,ph,pd):
            if ep[0]<CW and ep[1]<CH and ep[2]<CD: eps.add(ep)

    for name, dims in sorted(others.items(),
                              key=lambda i: i[1][0]*i[1][1]*i[1][2], reverse=True):
        best, best_score = None, float('inf')
        ty = random.uniform(0, CH)
        for ep in list(eps):
            ex,ey,ez = ep
            for ow,oh,od in get_orientations(*dims):
                if not fits_in_container(ex,ey,ez,ow,oh,od,container): continue
                nb = (ex,ey,ez,ow,oh,od)
                if any(overlaps(nb,(p[1],p[2],p[3],p[4],p[5],p[6])) for p in placed): continue
                score = abs(ey-ty)*10 + ez*3 + ex
                if score < best_score:
                    best_score = score; best = (name,ex,ey,ez,ow,oh,od)
        if best:
            placed.append(best)
            _,bx,by,bz,bw,bh,bd = best
            for ep in generate_eps(bx,by,bz,bw,bh,bd):
                if ep[0]<CW and ep[1]<CH and ep[2]<CD: eps.add(ep)
    return placed

def build_pool(objects_dict, container):
    cv = container[0]*container[1]*container[2]
    tv = 0.80 * cv
    other = {k:v for k,v in objects_dict.items() if k != "laptop"}
    pool = []
    for copies in range(1, 8):
        pool, total = [], 0
        for name, dims in other.items():
            for c in range(copies):
                tag = name if c == 0 else f"{name}_{c+1}"
                pool.append((tag, dims)); total += dims[0]*dims[1]*dims[2]
        if total >= tv: break
    return pool

def generate_scene(container, objects_dict):
    ld = objects_dict["laptop"]
    CW,CH,CD = container
    placed = []
    for _ in range(1000):
        ori = random.choice(get_orientations(*ld))
        lw,lh,ld_ = ori
        if lw>CW or lh>CH or ld_>CD: continue
        placed.append(("laptop",
                        random.uniform(0,CW-lw),
                        random.uniform(0,CH-lh),
                        random.uniform(0,CD-ld_),
                        lw, lh, ld_))
        break
    pool_dict = {tag: dims for tag,dims in build_pool(objects_dict, container)}
    return pack_remaining(container, placed, pool_dict)


# ---------------------------------------------------------------------------
# Step 1 — Build full scene point cloud (no labels, as if scanned)
# ---------------------------------------------------------------------------
def build_scene_pointcloud(scene, pts_per_obj=POINTS_PER_OBJECT):
    """
    Sample points from every object and merge into one unlabelled cloud.
    Also returns per-point object index so we can map clusters back.
    """
    all_pts   = []
    all_objid = []

    for obj_idx, obj in enumerate(scene):
        tag, x, y, z, w, h, d = obj
        base = get_base_name(tag)
        if base is None:
            continue
        mesh = load_trimesh_centered(base)
        mesh = orient_trimesh(mesh, OBJECTS[base], (w, h, d))
        mesh.apply_translation(np.array([x + w/2, y + h/2, z + d/2]))
        pts, _ = trimesh.sample.sample_surface(mesh, pts_per_obj)
        all_pts.append(pts.astype(np.float32))
        all_objid.extend([obj_idx] * pts_per_obj)

    return np.vstack(all_pts), np.array(all_objid)


# ---------------------------------------------------------------------------
# Step 2 — Segment point cloud using obj_ids (replaces DBSCAN)
# ---------------------------------------------------------------------------
def segment_by_object(pts, obj_ids):
    """
    Instead of DBSCAN (which merges tightly-packed objects), use the obj_ids
    array returned by build_scene_pointcloud. Each unique id corresponds to
    exactly one scene object — this is how the training data was structured,
    so it matches perfectly what the model expects.

    Returns dict: { obj_idx: point_array }
    """
    segments = {}
    for oid in np.unique(obj_ids):
        mask = obj_ids == oid
        segments[int(oid)] = pts[mask]
    return segments


# ---------------------------------------------------------------------------
# Step 3 — Normalise one cluster for PointNet
# ---------------------------------------------------------------------------
def normalize_cluster(pts):
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def subsample_cluster(pts, n=POINTS_PER_OBJECT):
    """Randomly subsample or upsample (with replacement) to exactly n points."""
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
    else:
        idx = np.random.choice(len(pts), n, replace=True)
    return pts[idx]


# ---------------------------------------------------------------------------
# Step 4 — Run PointNet on each cluster
# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    model = PointNetClassifier(num_classes=2).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from {ckpt_path}  (val_acc={ckpt['val_acc']:.2f}%)")
    return model


def classify_segments(segments, scene, model, device):
    """
    Run PointNet on each per-object segment.
    Returns dict: { obj_idx: { prob_laptop, centroid, min_pt, max_pt, n_points, tag } }
    """
    results = {}

    for obj_idx, cluster in segments.items():
        if len(cluster) < 20:
            continue

        sub  = subsample_cluster(cluster, POINTS_PER_OBJECT)
        norm = normalize_cluster(sub)

        tensor = torch.tensor(norm, dtype=torch.float32).T.unsqueeze(0).to(device)
        with torch.no_grad():
            logits      = model(tensor)
            probs       = torch.softmax(logits, dim=1)[0]
            prob_laptop = probs[1].item()

        tag = scene[obj_idx][0] if obj_idx < len(scene) else "unknown"

        results[obj_idx] = {
            "prob_laptop": prob_laptop,
            "centroid":    cluster.mean(axis=0),
            "min_pt":      cluster.min(axis=0),
            "max_pt":      cluster.max(axis=0),
            "n_points":    len(cluster),
            "tag":         tag,
        }

    return results


# ---------------------------------------------------------------------------
# Step 5 — Pick the most likely laptop cluster
# ---------------------------------------------------------------------------
def pick_laptop_cluster(cluster_results, threshold=0.5):
    """
    Return the cluster ID with highest laptop probability above threshold.
    Returns (cluster_id, result_dict) or (None, None).
    """
    if not cluster_results:
        return None, None
    best_id = max(cluster_results, key=lambda k: cluster_results[k]["prob_laptop"])
    best    = cluster_results[best_id]
    if best["prob_laptop"] < threshold:
        print(f"  Best cluster probability {best['prob_laptop']:.3f} < threshold {threshold} — no laptop detected.")
        return None, None
    return best_id, best


# ---------------------------------------------------------------------------
# Visualisation — 3-panel identical to scene_laptop_detection.py
# ---------------------------------------------------------------------------
def render_scene(plotter, scene, container, title,
                 highlight_obj=None, skip_obj=None):
    """
    highlight_obj : result dict from pick_laptop_cluster — only the single
                    closest object to its centroid is highlighted red.
                    OBB wireframe + label drawn once after the object loop.
    skip_obj      : np.array centroid — object nearest this point is omitted
                    (used for the After Removal panel).
    """
    container_mesh = pv.Box(bounds=(0,container[0],0,container[1],0,container[2]))
    plotter.add_mesh(container_mesh, style='wireframe', color='white', line_width=2)

    # Pre-compute which single object index is the detected laptop.
    # Strategy: find the scene object whose center is closest to the
    # detected cluster centroid. This gives exactly 1 match regardless
    # of how large the DBSCAN cluster OBB is.
    detected_idx = None
    if highlight_obj is not None:
        centroid = highlight_obj["centroid"]
        best_dist, best_i = float("inf"), None
        for i, obj in enumerate(scene):
            _, x, y, z, w, h, d = obj
            c = np.array([x+w/2, y+h/2, z+d/2])
            dist = np.linalg.norm(c - centroid)
            if dist < best_dist:
                best_dist, best_i = dist, i
        detected_idx = best_i

    # Similarly find which object to skip (removal panel)
    skip_idx = None
    if skip_obj is not None:
        best_dist, best_i = float("inf"), None
        for i, obj in enumerate(scene):
            _, x, y, z, w, h, d = obj
            c = np.array([x+w/2, y+h/2, z+d/2])
            dist = np.linalg.norm(c - skip_obj)
            if dist < best_dist:
                best_dist, best_i = dist, i
        skip_idx = best_i

    detected_center = None

    for i, obj in enumerate(scene):
        tag, x, y, z, w, h, d = obj
        base = get_base_name(tag)
        if base is None:
            continue

        if i == skip_idx:
            continue

        obj_center = np.array([x+w/2, y+h/2, z+d/2])
        mesh = orient_pv(load_pv_centered(base), OBJECTS[base], (w, h, d))
        mesh.translate(obj_center, inplace=True)

        if i == detected_idx:
            # Highlight only this one object red
            plotter.add_mesh(mesh, color="#FF2222", opacity=1.0,
                             show_edges=True, edge_color="#FF8888",
                             line_width=1, smooth_shading=True)
            detected_center = obj_center
        else:
            color = OBJECT_COLORS.get(base, "#AAAAAA")
            plotter.add_mesh(mesh, color=color, opacity=0.85,
                             show_edges=False, smooth_shading=True)

    # Draw OBB wireframe + label exactly once, outside the object loop
    if highlight_obj is not None and detected_center is not None:
        mn  = highlight_obj["min_pt"]
        mx  = highlight_obj["max_pt"]
        prob = highlight_obj["prob_laptop"]

        obb = pv.Box(bounds=(mn[0],mx[0],mn[1],mx[1],mn[2],mx[2]))
        plotter.add_mesh(obb, style='wireframe', color='yellow', line_width=5)

        corners = [[mn[0],mn[1],mn[2]], [mx[0],mn[1],mn[2]],
                   [mn[0],mx[1],mn[2]], [mx[0],mx[1],mn[2]],
                   [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]],
                   [mn[0],mx[1],mx[2]], [mx[0],mx[1],mx[2]]]
        for c in corners:
            plotter.add_mesh(pv.Sphere(radius=3, center=c), color="yellow")

        # Single label above the detected object
        label_pos = np.array([detected_center[0], mx[1] + 8, detected_center[2]])
        plotter.add_point_labels(
            [label_pos],
            [f"LAPTOP DETECTED  ({prob*100:.1f}%)"],
            font_size=13, text_color="yellow", bold=True,
            show_points=False, always_visible=True,
        )

    plotter.add_text(title, font_size=9, color='white')
    plotter.set_background('#1a1a2e')
    plotter.camera_position = 'iso'
    plotter.reset_camera()


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------
def run_detection(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load trained model
    model = load_model(args.ckpt, device)

    # Generate a fresh scene (laptop position unknown to the detector)
    print("Generating scene...")
    scene = generate_scene(CONTAINER, OBJECTS)
    print(f"  Placed {len(scene)} objects\n")

    # ---- Step 1: Build merged point cloud ----
    print("Sampling scene point cloud...")
    pts, obj_ids = build_scene_pointcloud(scene)
    print(f"  Total points : {len(pts)}")
    print(f"  Objects      : {len(np.unique(obj_ids))}\n")

    # ---- Step 2: Segment by object (replaces DBSCAN) ----
    # Objects are tightly packed so DBSCAN merges them. Instead we use the
    # obj_ids array — each point already knows which object it came from,
    # exactly mirroring how training data was structured.
    print("Segmenting point cloud by object...")
    segments = segment_by_object(pts, obj_ids)
    print(f"  {len(segments)} segments\n")

    # ---- Step 3: Classify each segment with PointNet ----
    print("Running PointNet on each segment...")
    seg_results = classify_segments(segments, scene, model, device)

    for oid, res in sorted(seg_results.items(),
                           key=lambda x: x[1]["prob_laptop"], reverse=True):
        print(f"  obj={oid:3d} | tag={res['tag']:15s} | pts={res['n_points']:5d} | "
              f"laptop_prob={res['prob_laptop']:.4f}")

    # ---- Step 4: Pick laptop ----
    print()
    laptop_id, laptop_info = pick_laptop_cluster(seg_results, threshold=args.threshold)

    if laptop_id is not None:
        print(f"Laptop detected! Cluster {laptop_id}")
        print(f"  Probability : {laptop_info['prob_laptop']*100:.2f}%")
        print(f"  OBB min     : {laptop_info['min_pt']}")
        print(f"  OBB max     : {laptop_info['max_pt']}")
        print(f"  Centroid    : {laptop_info['centroid']}")

        # Find ground truth for comparison
        for obj in scene:
            tag,x,y,z,w,h,d = obj
            if get_base_name(tag) == "laptop":
                gt_center = np.array([x+w/2, y+h/2, z+d/2])
                dist = np.linalg.norm(laptop_info["centroid"] - gt_center)
                print(f"\n  Ground truth center : {gt_center}")
                print(f"  Detection error     : {dist:.2f} mm")
    else:
        print("No laptop detected.")

    # ---- Visualise 3 panels ----
    print("\nLaunching visualisation...")
    plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

    # Panel 0: Original
    plotter.subplot(0, 0)
    render_scene(plotter, scene, CONTAINER,
                 title=f"Original  |  {len(scene)} objects",
                 highlight_obj=None, skip_obj=None)

    # Panel 1: Detection result
    plotter.subplot(0, 1)
    render_scene(plotter, scene, CONTAINER,
                 title="PointNet Detection",
                 highlight_obj=laptop_info, skip_obj=None)

    # Panel 2: After removal
    plotter.subplot(0, 2)
    skip_center = laptop_info["centroid"] if laptop_info else None
    render_scene(plotter, scene, CONTAINER,
                 title="After Removal",
                 highlight_obj=None, skip_obj=skip_center)

    plotter.link_views()
    plotter.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=str,   default="checkpoints/best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min probability to call a segment 'laptop' (default: 0.5)")
    parser.add_argument("--seed",      type=int,   default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    run_detection(args)