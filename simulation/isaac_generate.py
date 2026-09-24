"""Isaac Sim 6.1 standalone Replicator dataset generator.

Run with the Isaac Sim Python (e.g. .venv_isaac), not the project's ML virtualenv:

    .venv_isaac/Scripts/python.exe simulation/isaac_generate.py --layouts 150 --views 4

Each layout is one physical arrangement (panel curvature, paint, seams, window, antenna, tooling,
defects, clutter, floor). Each view re-samples camera and lighting. Scene geometry is sampled by
simulation/isaac_scene.py, which is plain NumPy and unit-tested.
"""

import argparse
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation import isaac_scene as scene


def parse_config() -> tuple[dict, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/isaac.json"))
    parser.add_argument("--layouts", type=int)
    parser.add_argument("--views", type=int, dest="views_per_layout")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--gui", action="store_true", help="Show the Isaac Sim window")
    parser.add_argument("--overwrite", action="store_true", help="Replace a previous render")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ["layouts", "views_per_layout", "seed", "output"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.gui:
        config["headless"] = False
    if config["layouts"] < 1 or config["views_per_layout"] < 1:
        raise ValueError("layouts and views_per_layout must be positive")
    config["depth_noise"] = {**scene.DEFAULT_DEPTH_NOISE, **config.get("depth_noise", {})}
    return config, args.overwrite


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        looks_generated = (path / "scene_metadata.json").exists() or any(path.glob("rgb_*.png"))
        if not overwrite:
            raise SystemExit(f"{path} is not empty; pass --overwrite or choose --output")
        if not looks_generated:
            raise SystemExit(f"Refusing to delete {path}: it does not look like a render output")
        shutil.rmtree(path)
    (path / "textures").mkdir(parents=True, exist_ok=True)


def main():
    config, overwrite = parse_config()
    output = Path(config["output"]).resolve()
    prepare_output(output, overwrite)
    try:
        from isaacsim import SimulationApp
    except ImportError as error:
        raise SystemExit(
            "Isaac Sim is unavailable. Run with the Isaac Sim Python environment; "
            "the non-Isaac fallback is: python -m aerosurface.cli data"
        ) from error

    app = SimulationApp({"headless": config["headless"]})
    try:
        import carb.settings
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.utils.semantics import add_labels
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

        rng = np.random.default_rng(config["seed"])
        rep.set_global_seed(config["seed"])
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        rep.orchestrator.set_capture_on_play(False)
        carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)

        def vec3(values):
            return Gf.Vec3f(*[float(v) for v in values])

        def preview(path, colour=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0, opacity=1.0):
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(vec3(colour))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat, shader

        def textured(path):
            mat, shader = preview(path)
            reader = UsdShade.Shader.Define(stage, path + "/UV")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(stage, path + "/Texture")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result"
            )
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset)
            shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
            return mat, shader, tex

        def bind(prim, mat):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

        def label(prim, name):
            add_labels(prim, [name])

        def ops(prim, rotate_y=False, yaw=False, scale=True):
            xf = UsdGeom.Xformable(prim)
            out = {"t": xf.AddTranslateOp()}
            if rotate_y:
                out["ry"] = xf.AddRotateYOp()
            if yaw:
                out["rz"] = xf.AddRotateZOp()
            if scale:
                out["s"] = xf.AddScaleOp()
            return out

        def set_ops(handles, position=None, angle=None, yaw=None, size=None):
            if position is not None:
                handles["t"].Set(Gf.Vec3d(*[float(v) for v in position]))
            if angle is not None:
                handles["ry"].Set(float(angle))
            if yaw is not None:
                handles["rz"].Set(float(yaw))
            if size is not None:
                handles["s"].Set(vec3(size))

        def visible(prim, show):
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeVisible() if show else imageable.MakeInvisible()

        def box_mesh(path, uv_scale=1.0):
            """Unit box with per-face UVs, so textures map (UsdGeom.Cube has no UVs)."""
            mesh = UsdGeom.Mesh.Define(stage, path)
            c = [(x, y, z) for z in (-0.5, 0.5) for y in (-0.5, 0.5) for x in (-0.5, 0.5)]
            faces = [
                (4, 5, 7, 6),
                (0, 2, 3, 1),
                (0, 1, 5, 4),
                (2, 6, 7, 3),
                (0, 4, 6, 2),
                (1, 3, 7, 5),
            ]
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in c])
            mesh.CreateFaceVertexCountsAttr([4] * 6)
            mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            )
            quad = [(0, 0), (1, 0), (1, 1), (0, 1)]
            st.Set([Gf.Vec2f(u * uv_scale, v * uv_scale) for _ in faces for u, v in quad])
            return mesh

        # --- Static scene graph; per-layout values are set below. ---
        floor_mat, floor_shader, floor_tex = textured("/World/Looks/Floor")
        floor = box_mesh("/World/Floor", uv_scale=8.0)
        set_ops(ops(floor.GetPrim()), position=(0, 0, -0.011), size=(10, 10, 0.02))
        bind(floor.GetPrim(), floor_mat)

        panel_mat, panel_shader, panel_tex = textured("/World/Looks/Panel")
        segments = 64
        panel = UsdGeom.Mesh.Define(stage, "/World/CurvedPanel")
        panel.CreateFaceVertexCountsAttr([4] * segments)
        panel.CreateFaceVertexIndicesAttr(
            [v for j in range(segments) for v in (j, j + 1, j + segments + 2, j + segments + 1)]
        )
        panel.CreateSubdivisionSchemeAttr("none")
        panel.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(panel).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set([Gf.Vec2f(2 * j / segments, 2 * v) for v in (0.0, 1.0) for j in range(segments + 1)])
        panel.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        label(panel.GetPrim(), "sandable_surface")
        bind(panel.GetPrim(), panel_mat)

        flat_mat, _, flat_tex = textured("/World/Looks/FlatPanel")
        flat = box_mesh("/World/FlatPanel")
        flat_ops = ops(flat.GetPrim(), yaw=True)
        label(flat.GetPrim(), "sandable_surface")
        bind(flat.GetPrim(), flat_mat)

        seam_mat, seam_shader = preview("/World/Looks/Seam", roughness=0.6)
        seam = UsdGeom.Cube.Define(stage, "/World/Seam")
        seam.CreateSizeAttr(1.0)
        seam_ops = ops(seam.GetPrim(), rotate_y=True)
        label(seam.GetPrim(), "protected_or_avoid_region")
        bind(seam.GetPrim(), seam_mat)

        chord_mat, chord_shader = preview("/World/Looks/ChordSeam", roughness=0.6)
        chord = UsdGeom.Mesh.Define(stage, "/World/ChordSeam")
        chord.CreateSubdivisionSchemeAttr("none")
        chord.CreateDoubleSidedAttr(True)
        stations = 64
        chord.CreateFaceVertexCountsAttr([4] * (3 * stations))
        indices = []
        for j in range(stations):
            a, b = 4 * j, 4 * (j + 1)
            for k in range(3):  # left side, top, right side
                indices += [a + k, b + k, b + k + 1, a + k + 1]
        chord.CreateFaceVertexIndicesAttr(indices)
        label(chord.GetPrim(), "protected_or_avoid_region")
        bind(chord.GetPrim(), chord_mat)

        glass_mat, _ = preview("/World/Looks/Glass", (0.08, 0.18, 0.25), 0.05, 0.0, 0.4)
        frame_mat, frame_shader = preview("/World/Looks/Frame", (0.12, 0.14, 0.16), 0.3, 0.65)
        window = UsdGeom.Xform.Define(stage, "/World/Window")
        window_ops = ops(window.GetPrim(), rotate_y=True, yaw=True)
        label(window.GetPrim(), "protected_or_avoid_region")
        window_parts = {}
        for name, xy, extent, mat in [
            ("Pane", (0, 0), (0.22, 0.16), glass_mat),
            ("FrameTop", (0, 0.09), (0.25, 0.02), frame_mat),
            ("FrameBottom", (0, -0.09), (0.25, 0.02), frame_mat),
            ("FrameLeft", (-0.115, 0), (0.02, 0.16), frame_mat),
            ("FrameRight", (0.115, 0), (0.02, 0.16), frame_mat),
        ]:
            part = UsdGeom.Cube.Define(stage, f"/World/Window/{name}")
            part.CreateSizeAttr(1.0)
            label(part.GetPrim(), "protected_or_avoid_region")
            bind(part.GetPrim(), mat)
            window_parts[name] = (ops(part.GetPrim()), xy, extent)

        antenna_mat, antenna_shader = preview("/World/Looks/Antenna", roughness=0.35, metallic=0.7)
        antenna = UsdGeom.Xform.Define(stage, "/World/Antenna")
        antenna_ops = ops(antenna.GetPrim(), rotate_y=True, yaw=True, scale=False)
        label(antenna.GetPrim(), "obstacle")
        mast = UsdGeom.Cylinder.Define(stage, "/World/Antenna/Mast")
        mast.CreateAxisAttr(UsdGeom.Tokens.z)
        mast_ops = ops(mast.GetPrim())
        blade = UsdGeom.Cube.Define(stage, "/World/Antenna/Blade")
        blade.CreateSizeAttr(1.0)
        blade_ops = ops(blade.GetPrim())
        for prim in (mast.GetPrim(), blade.GetPrim()):
            label(prim, "obstacle")
            bind(prim, antenna_mat)

        tooling_mat, tooling_shader = preview("/World/Looks/Tooling", roughness=0.5, metallic=0.3)
        tooling = UsdGeom.Cube.Define(stage, "/World/Tooling")
        tooling.CreateSizeAttr(1.0)
        tooling_ops = ops(tooling.GetPrim(), rotate_y=True, yaw=True)
        label(tooling.GetPrim(), "obstacle")
        bind(tooling.GetPrim(), tooling_mat)

        defect_slots = []
        for i in range(3):
            mat, shader = preview(f"/World/Looks/Defect{i}")
            prim = UsdGeom.Cube.Define(stage, f"/World/Defect{i}")
            prim.CreateSizeAttr(1.0)
            label(prim.GetPrim(), "surface_defect")
            bind(prim.GetPrim(), mat)
            defect_slots.append((prim.GetPrim(), ops(prim.GetPrim(), True, True), shader))

        clutter_slots = []
        for i in range(4):
            mat, shader = preview(f"/World/Looks/Clutter{i}", roughness=0.6)
            box = UsdGeom.Cube.Define(stage, f"/World/Clutter{i}Box")
            box.CreateSizeAttr(1.0)
            cyl = UsdGeom.Cylinder.Define(stage, f"/World/Clutter{i}Cylinder")
            cyl.CreateAxisAttr(UsdGeom.Tokens.z)
            cyl.CreateRadiusAttr(0.5)
            cyl.CreateHeightAttr(1.0)
            handles = {}
            for kind, prim in (("box", box.GetPrim()), ("cylinder", cyl.GetPrim())):
                bind(prim, mat)
                handles[kind] = (prim, ops(prim, yaw=True))
            clutter_slots.append((handles, shader))

        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key_rotation = UsdGeom.Xformable(key).AddRotateXYZOp()
        dome = UsdLux.DomeLight.Define(stage, "/World/FillLight")

        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        camera.CreateHorizontalApertureAttr(20.955)
        camera_xform = UsdGeom.Xformable(camera).AddTransformOp()
        product = rep.create.render_product("/World/Camera", (config["width"], config["height"]))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=str(output),
            rgb=True,
            distance_to_image_plane=True,
            semantic_segmentation=True,
            colorize_semantic_segmentation=False,
            instance_segmentation=True,
        )
        writer.attach([product])

        def look_at(eye, target):
            eye, target = np.asarray(eye, float), np.asarray(target, float)
            forward = (target - eye) / np.linalg.norm(target - eye)
            right = np.cross(forward, [0.0, 0.0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            rows = [*right, 0, *up, 0, *(-forward), 0, *eye, 1]
            return Gf.Matrix4d(*[float(v) for v in rows])

        def write_texture(name, pixels):
            path = output / "textures" / f"{name}.png"
            Image.fromarray(pixels).save(path)
            return str(path)

        def apply_layout(layout):
            h = layout["panel_height"]
            xs = np.linspace(-scene.A, scene.A, segments + 1)
            panel.GetPointsAttr().Set(
                [
                    Gf.Vec3f(float(x), y, float(scene.panel_z(x, h)))
                    for y in (-scene.SPAN, scene.SPAN)
                    for x in xs
                ]
            )
            panel.GetNormalsAttr().Set(
                [vec3(scene.panel_normal(float(x), h)) for _ in (0, 1) for x in xs]
            )
            panel.CreateExtentAttr(
                [Gf.Vec3f(-scene.A, -scene.SPAN, 0.0), Gf.Vec3f(scene.A, scene.SPAN, h)]
            )
            n = layout["layout"]
            paint = scene.PAINTS[layout["paint"]]
            panel_tex.GetInput("file").Set(
                write_texture(f"panel_{n:04d}", scene.paint_texture(rng, paint))
            )
            panel_shader.GetInput("roughness").Set(layout["panel_roughness"])
            panel_shader.GetInput("metallic").Set(layout["panel_metallic"])
            floor_tex.GetInput("file").Set(
                write_texture(
                    f"floor_{n:04d}",
                    scene.paint_texture(rng, scene.FLOORS[layout["floor"]], grime=3.0),
                )
            )
            floor_shader.GetInput("roughness").Set(0.8)
            dome.GetColorAttr().Set(vec3(layout["dome_colour"]))

            s = layout["seam"]
            set_ops(seam_ops, s["position"], s["angle"], size=s["size"])
            seam_shader.GetInput("diffuseColor").Set(vec3(s["colour"]))

            c = layout["chord_seam"]
            visible(chord.GetPrim(), c is not None)
            if c:
                points = []
                for x in np.linspace(-scene.A + 0.02, scene.A - 0.02, stations + 1):
                    base = np.array([x, 0.0, float(scene.panel_z(x, h))])
                    normal = scene.panel_normal(float(x), h)
                    low, high = base - 0.001 * normal, base + c["thickness"] * normal
                    for p, dy in ((low, -1), (high, -1), (high, 1), (low, 1)):
                        points.append(
                            Gf.Vec3f(float(p[0]), c["y"] + dy * c["width"] / 2, float(p[2]))
                        )
                chord.GetPointsAttr().Set(points)
                chord_shader.GetInput("diffuseColor").Set(vec3(c["colour"]))

            w = layout["window"]
            visible(window.GetPrim(), w is not None)
            if w:
                set_ops(
                    window_ops, w["position"], w["angle"], w["yaw"], (w["scale"], w["scale"], 1.0)
                )
                sag = w["sag"] + 0.001
                for name, (handles, xy, extent) in window_parts.items():
                    if name == "Pane":
                        set_ops(handles, (*xy, 0.004), size=(*extent, 0.004))
                    else:  # frame spans local z in [-sag, 0.012] so its edges meet the shell
                        set_ops(handles, (*xy, (0.012 - sag) / 2), size=(*extent, 0.012 + sag))
                w["frame_colour"] = rng.uniform(0.08, 0.5, 3).tolist()
                frame_shader.GetInput("diffuseColor").Set(vec3(w["frame_colour"]))

            a = layout["antenna"]
            visible(antenna.GetPrim(), a is not None)
            if a:
                set_ops(antenna_ops, a["position"], a["angle"], a["yaw"])
                visible(mast.GetPrim(), a["kind"] == "mast")
                visible(blade.GetPrim(), a["kind"] == "blade")
                if a["kind"] == "mast":
                    radius, height = a["size"]
                    mast.GetRadiusAttr().Set(radius)
                    mast.GetHeightAttr().Set(height)
                    set_ops(mast_ops, (0, 0, height / 2), size=(1, 1, 1))
                else:
                    set_ops(blade_ops, (0, 0, a["size"][2] / 2), size=a["size"])
                a["colour"] = rng.uniform(0.1, 0.8, 3).tolist()
                antenna_shader.GetInput("diffuseColor").Set(vec3(a["colour"]))

            t = layout["tooling"]
            visible(tooling.GetPrim(), t is not None)
            if t:
                set_ops(tooling_ops, t["position"], t["angle"], t["yaw"], t["size"])
                tooling_shader.GetInput("diffuseColor").Set(vec3(t["colour"]))

            for i, (prim, handles, shader) in enumerate(defect_slots):
                d = layout["defects"][i] if i < len(layout["defects"]) else None
                visible(prim, d is not None)
                if d:
                    set_ops(handles, d["position"], d["angle"], d["yaw"], d["size"])
                    shader.GetInput("diffuseColor").Set(vec3(d["colour"]))
                    shader.GetInput("roughness").Set(d["roughness"])
                    shader.GetInput("metallic").Set(d["metallic"])

            f = layout["flat_panel"]
            visible(flat.GetPrim(), f is not None)
            if f:
                set_ops(flat_ops, f["position"], yaw=f["yaw"], size=f["size"])
                flat_tex.GetInput("file").Set(
                    write_texture(
                        f"flat_{n:04d}", scene.paint_texture(rng, scene.PAINTS[f["paint"]])
                    )
                )

            for i, (handles, shader) in enumerate(clutter_slots):
                item = layout["clutter"][i] if i < len(layout["clutter"]) else None
                for kind, (prim, handle) in handles.items():
                    show = item is not None and item["shape"] == kind
                    visible(prim, show)
                    if show:
                        set_ops(handle, item["position"], yaw=item["yaw"], size=item["size"])
                if item:
                    shader.GetInput("diffuseColor").Set(vec3(item["colour"]))

        def apply_view(view):
            camera_xform.Set(look_at(view["camera_position"], view["camera_target"]))
            camera.GetFocalLengthAttr().Set(view["focal_length"])
            key.GetIntensityAttr().Set(view["key_intensity"])
            key.GetColorAttr().Set(vec3(view["key_colour"]))
            key_rotation.Set(vec3(view["key_rotation"]))
            dome.GetIntensityAttr().Set(view["dome_intensity"])

        key.CreateIntensityAttr(1000.0)
        key.CreateColorAttr(vec3((1, 1, 1)))
        dome.CreateIntensityAttr(200.0)
        dome.CreateColorAttr(vec3((1, 1, 1)))
        camera.CreateFocalLengthAttr(24.0)
        mast.CreateRadiusAttr(0.02)
        mast.CreateHeightAttr(0.2)
        chord.CreatePointsAttr([Gf.Vec3f(0, 0, 0)] * (4 * (stations + 1)))
        panel.CreatePointsAttr([Gf.Vec3f(0, 0, 0)] * (2 * (segments + 1)))
        panel.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * (2 * (segments + 1))))

        layouts, frames = [], []
        total = config["layouts"] * config["views_per_layout"]
        start = time.perf_counter()
        for n in range(config["layouts"]):
            layout = scene.sample_layout(rng, n)
            apply_layout(layout)
            layouts.append(layout)
            for v in range(config["views_per_layout"]):
                view = scene.sample_view(rng, layout)
                view["focal_length"] = float(rng.uniform(18.0, 32.0))
                apply_view(view)
                rep.orchestrator.step(rt_subframes=config["rt_subframes"])
                frames.append({"frame": len(frames), "layout": n, "view": v, **view})
            done = len(frames)
            rate = (time.perf_counter() - start) / done
            print(f"[isaac] {done}/{total} frames, {rate:.2f} s/frame", flush=True)
        rep.orchestrator.wait_until_complete()
        writer.detach()
        product.destroy()

        rendered = sorted(output.glob("rgb_*.png"))
        if len(rendered) != len(frames):
            raise RuntimeError(
                f"Writer produced {len(rendered)} RGB frames for {len(frames)} steps"
            )

        # Keep the clean depth; write sensor-like depth next to it. Record what is visible.
        noise_rng = np.random.default_rng(config["seed"] + 1)
        for record in frames:
            suffix = f"{record['frame']:04d}"
            raw = np.load(output / f"distance_to_image_plane_{suffix}.npy", allow_pickle=False)
            noisy = scene.depth_noise(raw, noise_rng, config["depth_noise"])
            np.save(output / f"depth_noisy_{suffix}.npy", noisy, allow_pickle=False)
            semantic = np.array(Image.open(output / f"semantic_segmentation_{suffix}.png"))
            labels = json.loads(
                (output / f"semantic_segmentation_labels_{suffix}.json").read_text()
            )
            record["class_pixels"] = scene.class_pixels(semantic, labels)
            instance = np.array(Image.open(output / f"instance_segmentation_{suffix}.png"))
            mapping = json.loads(
                (output / f"instance_segmentation_mapping_{suffix}.json").read_text()
            )
            record["prim_pixels"] = scene.instance_pixels(instance, mapping)

        metadata = {
            "generator": "simulation/isaac_generate.py",
            "isaac_sim": importlib.metadata.version("isaacsim"),
            "config": config,
            "seconds": time.perf_counter() - start,
            "layouts": layouts,
            "frames": frames,
        }
        (output / "scene_metadata.json").write_text(json.dumps(metadata, indent=1))
        stage.GetRootLayer().Export(str(output / "scene_last_layout.usda"))
        print(f"[isaac] wrote {len(frames)} frames to {output}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
