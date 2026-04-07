"""
interactive_gui.py
------------------
Real-time interactive PyVista GUI for laptop detection in packed scenes.

Three buttons:
  1. Generate New View  — random packed scene
  2. Detect Laptop      — run PointNet, show bounding box
  3. Toggle Laptop      — remove / add back the laptop mesh

Run:
    python interactive_gui.py --ckpt checkpoints/best_model.pth
"""

import argparse
import random
import numpy as np
import torch
import pyvista as pv

# Reuse everything from the detection script
from detect_with_pointnet import (
    CONTAINER, OBJECTS, OBJECT_COLORS,
    load_pv_centered, load_trimesh_centered,
    orient_pv, orient_trimesh,
    get_base_name, generate_scene,
    build_scene_pointcloud, segment_by_object,
    classify_segments, pick_laptop_cluster,
    POINTS_PER_OBJECT,
)
from train_pointnet import PointNetClassifier


# ──────────────────────────────────────────────────────────────
# Scene state — holds everything needed to render + interact
# ──────────────────────────────────────────────────────────────
class SceneState:
    def __init__(self):
        self.scene = []              # list of (tag, x,y,z, w,h,d)
        self.laptop_visible = True   # toggle state
        self.detection = None        # result dict from pick_laptop_cluster
        self.detection_ran = False   # whether Detect was clicked since last gen


# ──────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────
def render(plotter: pv.Plotter, state: SceneState, container):
    """Clear the plotter and redraw the scene from *state*."""
    plotter.clear()

    # Container wireframe
    box = pv.Box(bounds=(0, container[0], 0, container[1], 0, container[2]))
    plotter.add_mesh(box, style="wireframe", color="white", line_width=2)

    # Identify which scene object is the detected laptop (by centroid distance)
    detected_idx = None
    if state.detection_ran and state.detection is not None:
        centroid = state.detection["centroid"]
        best_dist = float("inf")
        for i, obj in enumerate(state.scene):
            _, x, y, z, w, h, d = obj
            c = np.array([x + w / 2, y + h / 2, z + d / 2])
            dist = np.linalg.norm(c - centroid)
            if dist < best_dist:
                best_dist, detected_idx = dist, i

    detected_center = None

    for i, obj in enumerate(state.scene):
        tag, x, y, z, w, h, d = obj
        base = get_base_name(tag)
        if base is None:
            continue

        # Toggle: skip laptop mesh when hidden
        if base == "laptop" and not state.laptop_visible:
            continue

        obj_center = np.array([x + w / 2, y + h / 2, z + d / 2])
        mesh = orient_pv(load_pv_centered(base), OBJECTS[base], (w, h, d))
        mesh.translate(obj_center, inplace=True)

        if i == detected_idx:
            # Highlight detected laptop in red
            plotter.add_mesh(
                mesh, color="#FF2222", opacity=1.0,
                show_edges=True, edge_color="#FF8888",
                line_width=1, smooth_shading=True,
            )
            detected_center = obj_center
        else:
            color = OBJECT_COLORS.get(base, "#AAAAAA")
            plotter.add_mesh(
                mesh, color=color, opacity=0.85,
                show_edges=False, smooth_shading=True,
            )

    # Draw detection OBB + label
    if state.detection_ran and state.detection is not None and detected_center is not None:
        det = state.detection
        mn, mx = det["min_pt"], det["max_pt"]
        prob = det["prob_laptop"]

        obb = pv.Box(bounds=(mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]))
        plotter.add_mesh(obb, style="wireframe", color="yellow", line_width=5)

        corners = [
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
        ]
        for c in corners:
            plotter.add_mesh(pv.Sphere(radius=2.5, center=c), color="yellow")

        label_pos = np.array([detected_center[0], mx[1] + 8, detected_center[2]])
        plotter.add_point_labels(
            [label_pos],
            [f"LAPTOP  ({prob * 100:.1f}%)"],
            font_size=13, text_color="yellow", bold=True,
            show_points=False, always_visible=True,
        )
    elif state.detection_ran and state.detection is None:
        # No detection
        plotter.add_text(
            "No laptop detected",
            position="upper_right", font_size=12, color="red",
        )

    # Status bar
    n_objs = len(state.scene)
    laptop_str = "VISIBLE" if state.laptop_visible else "HIDDEN"
    det_str = "DETECTED" if (state.detection_ran and state.detection) else (
        "NOT FOUND" if state.detection_ran else "—"
    )
    plotter.add_text(
        f"Objects: {n_objs}  |  Laptop: {laptop_str}  |  Detection: {det_str}",
        position="upper_left", font_size=10, color="white",
    )

    plotter.set_background("#1a1a2e")
    plotter.reset_camera()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load PointNet model once
    print("Loading PointNet model...")
    model = PointNetClassifier(num_classes=2).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded {args.ckpt}  (val_acc={ckpt['val_acc']:.2f}%)")

    # Pre-load all meshes
    print("Pre-loading STL meshes...")
    from detect_with_pointnet import STL_FILES
    for name in STL_FILES:
        load_pv_centered(name)
        load_trimesh_centered(name)

    state = SceneState()

    # ── Create plotter ──────────────────────────────────────
    plotter = pv.Plotter(title="Laptop Detection — Interactive GUI",
                         window_size=(1400, 900))
    plotter.set_background("#1a1a2e")
    plotter.camera_position = "iso"

    # Generate initial scene
    state.scene = generate_scene(CONTAINER, OBJECTS)
    render(plotter, state, CONTAINER)

    # ── Button callbacks ────────────────────────────────────
    def on_generate(_=None):
        """Generate a new random scene."""
        state.scene = generate_scene(CONTAINER, OBJECTS)
        state.laptop_visible = True
        state.detection = None
        state.detection_ran = False
        render(plotter, state, CONTAINER)
        plotter.render()
        print("[GUI] New scene generated")

    def on_detect(_=None):
        """Run PointNet detection on current scene."""
        print("[GUI] Running detection...")
        pts, obj_ids = build_scene_pointcloud(state.scene, POINTS_PER_OBJECT)
        segments = segment_by_object(pts, obj_ids)
        seg_results = classify_segments(segments, state.scene, model, device)

        _, laptop_info = pick_laptop_cluster(seg_results, threshold=args.threshold)
        state.detection = laptop_info
        state.detection_ran = True

        if laptop_info:
            print(f"  Laptop detected  ({laptop_info['prob_laptop']*100:.1f}%)")
        else:
            print("  No laptop detected")

        render(plotter, state, CONTAINER)
        plotter.render()

    def on_toggle(_=None):
        """Toggle laptop visibility."""
        state.laptop_visible = not state.laptop_visible
        render(plotter, state, CONTAINER)
        plotter.render()
        vis = "shown" if state.laptop_visible else "hidden"
        print(f"[GUI] Laptop {vis}")

    # ── Add button widgets ──────────────────────────────────
    # PyVista button widgets: (callback, value, position, size, color_on, color_off, ...)
    # Using add_key_event as a simpler fallback if button widgets misbehave.

    # Buttons stacked in the lower-right
    btn_w, btn_h = 220, 45
    margin = 12
    x_right = 10       # pixels from left edge

    plotter.add_text_slider_widget(
        callback=lambda _: None,  # dummy — we use buttons instead
        data=[""],
        value=0,
        pointa=(0, 0), pointb=(0, 0),
        style="classic",
    ) if False else None  # skip — just using buttons

    y_base = 10

    plotter.add_checkbox_button_widget(
        on_generate,
        value=False,
        position=(x_right, y_base + 2 * (btn_h + margin)),
        size=btn_h,
        border_size=3,
        color_on="#00CC66",
        color_off="#00CC66",
        background_color="#1a1a2e",
    )
    plotter.add_text(
        "  [G] Generate New View",
        position=(x_right + btn_h + 8, y_base + 2 * (btn_h + margin) + 10),
        font_size=11, color="#00CC66",
    )

    plotter.add_checkbox_button_widget(
        on_detect,
        value=False,
        position=(x_right, y_base + 1 * (btn_h + margin)),
        size=btn_h,
        border_size=3,
        color_on="#FFD700",
        color_off="#FFD700",
        background_color="#1a1a2e",
    )
    plotter.add_text(
        "  [D] Detect Laptop",
        position=(x_right + btn_h + 8, y_base + 1 * (btn_h + margin) + 10),
        font_size=11, color="#FFD700",
    )

    plotter.add_checkbox_button_widget(
        on_toggle,
        value=False,
        position=(x_right, y_base),
        size=btn_h,
        border_size=3,
        color_on="#FF6666",
        color_off="#FF6666",
        background_color="#1a1a2e",
    )
    plotter.add_text(
        "  [T] Toggle Laptop",
        position=(x_right + btn_h + 8, y_base + 10),
        font_size=11, color="#FF6666",
    )

    # Keyboard shortcuts (G / D / T)
    plotter.add_key_event("g", on_generate)
    plotter.add_key_event("d", on_detect)
    plotter.add_key_event("t", on_toggle)

    print("\n" + "=" * 55)
    print("  Interactive Laptop Detection GUI")
    print("  ─────────────────────────────────")
    print("  Click the buttons or use keyboard shortcuts:")
    print("    G  →  Generate New View")
    print("    D  →  Detect Laptop")
    print("    T  →  Toggle Laptop")
    print("=" * 55 + "\n")

    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive laptop detection GUI"
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth",
                        help="Path to trained PointNet checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    main(args)
