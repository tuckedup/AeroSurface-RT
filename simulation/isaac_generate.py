"""Isaac Sim 5.1 standalone Replicator recipe. NOT executed on the authoring host.

Run with Isaac Sim's python.sh/python.bat, not the project's ML virtualenv.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/isaac.json"))
    parser.add_argument("--frames", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ["frames", "seed"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["frames"] < 1:
        raise ValueError("frames must be positive")
    try:
        from isaacsim import SimulationApp
    except ImportError as error:
        raise SystemExit(
            "Isaac Sim is unavailable. Run with Isaac Sim 5.1 python.sh/python.bat; "
            "the executable fallback is: python -m aerosurface.cli data"
        ) from error

    app = SimulationApp({"headless": config["headless"]})
    try:
        import carb.settings
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.utils.semantics import add_update_semantics
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        output = Path(config["output"]).resolve()
        output.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(config["seed"])
        rep.set_global_seed(config["seed"])
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        rep.orchestrator.set_capture_on_play(False)
        carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)

        def material(path, color):
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.5)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat, shader

        panel_mat, panel_shader = material("/World/PanelMaterial", (0.6, 0.65, 0.7))
        protected_mat, _ = material("/World/ProtectedMaterial", (0.8, 0.55, 0.15))
        defect_mat, _ = material("/World/DefectMaterial", (0.15, 0.08, 0.05))
        obstacle_mat, _ = material("/World/ObstacleMaterial", (0.15, 0.2, 0.3))
        # Physically curved tessellated panel, not a texture illusion.
        mesh = UsdGeom.Mesh.Define(stage, "/World/CurvedPanel")
        count = 32
        points = []
        for y in [-0.55, 0.55]:
            for j in range(count + 1):
                x = -0.65 + 1.3 * j / count
                points.append(Gf.Vec3f(x, y, 0.18 * (1 - (x / 0.65) ** 2)))
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * count)
        mesh.CreateFaceVertexIndicesAttr(
            [v for j in range(count) for v in [j, j + 1, j + count + 2, j + count + 1]]
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        uv.Set([Gf.Vec2f(j / count, y) for y in [0.0, 1.0] for j in range(count + 1)])
        add_update_semantics(mesh.GetPrim(), "sandable_surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(panel_mat)

        def cube(name, label, position, scale, mat):
            obj = UsdGeom.Cube.Define(stage, "/World/" + name)
            obj.CreateSizeAttr(1.0)
            transform = UsdGeom.Xformable(obj)
            translate = transform.AddTranslateOp()
            translate.Set(Gf.Vec3d(*position))
            size = transform.AddScaleOp()
            size.Set(Gf.Vec3f(*scale))
            add_update_semantics(obj.GetPrim(), label)
            UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
            return translate, size

        cube("FlatPanel", "sandable_surface", (1.0, 0.0, 0.02), (0.55, 1.1, 0.04), panel_mat)
        cube(
            "Seam", "protected_or_avoid_region", (0.2, 0, 0.185), (0.015, 1.0, 0.015), protected_mat
        )
        protected, _ = cube(
            "ProtectedFixture",
            "protected_or_avoid_region",
            (0, 0, 0.25),
            (0.12, 0.15, 0.18),
            protected_mat,
        )
        defect, defect_scale = cube(
            "DefectPatch", "surface_defect", (0.3, 0.2, 0.16), (0.06, 0.025, 0.003), defect_mat
        )
        obstacle, _ = cube(
            "Obstacle", "obstacle", (-0.2, -0.2, 0.3), (0.09, 0.09, 0.22), obstacle_mat
        )
        light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        rotation = UsdGeom.Xformable(light).AddRotateXYZOp()
        dome = UsdLux.DomeLight.Define(stage, "/World/FillLight")
        dome.CreateIntensityAttr(200.0)
        camera = rep.create.camera(position=(0.3, -1.8, 1.7), look_at=(0.25, 0, 0))
        product = rep.create.render_product(camera, (config["width"], config["height"]))
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
        metadata = []
        # UV texture uses a local generated asset; no external proprietary scene dependencies.
        tex = UsdShade.Shader.Define(stage, "/World/PanelMaterial/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        reader = UsdShade.Shader.Define(stage, "/World/PanelMaterial/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        panel_shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
        for frame in range(config["frames"]):
            intensity = float(rng.uniform(700, 3500))
            light.CreateIntensityAttr(intensity)
            light_color = rng.uniform(0.75, 1.0, 3).tolist()
            light.CreateColorAttr(Gf.Vec3f(*light_color))
            angles = rng.uniform([-70, -50, -180], [-15, 50, 180]).tolist()
            rotation.Set(Gf.Vec3f(*angles))
            roughness, metallic = float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.95))
            panel_shader.GetInput("roughness").Set(roughness)
            panel_shader.GetInput("metallic").Set(metallic)
            texture = np.clip(
                rng.normal(rng.uniform(100, 200), rng.uniform(3, 20), (128, 128, 3)), 0, 255
            ).astype(np.uint8)
            texture_path = output / f"texture_{frame:04d}.png"
            Image.fromarray(texture).save(texture_path)
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))

            def position(z_offset):
                x, y = float(rng.uniform(-0.45, 0.45)), float(rng.uniform(-0.4, 0.4))
                return [x, y, 0.18 * (1 - (x / 0.65) ** 2) + z_offset]

            dpos, ppos, opos = position(0.003), position(0.09), position(0.11)
            defect.Set(Gf.Vec3d(*dpos))
            protected.Set(Gf.Vec3d(*ppos))
            obstacle.Set(Gf.Vec3d(*opos))
            size = [float(rng.uniform(0.02, 0.09)), float(rng.uniform(0.01, 0.04)), 0.003]
            defect_scale.Set(Gf.Vec3f(*size))
            camera_pos = rng.uniform([-0.25, -2.2, 1.2], [0.7, -1.1, 2.2]).tolist()
            with camera:
                rep.modify.pose(position=tuple(camera_pos), look_at=(0.25, 0, 0.1))
            rep.orchestrator.step(rt_subframes=config["rt_subframes"])
            metadata.append(
                {
                    "frame": frame,
                    "source": "Isaac_Sim_Replicator",
                    "seed": config["seed"],
                    "intensity": intensity,
                    "light_color": light_color,
                    "light_rotation": angles,
                    "roughness": roughness,
                    "metallic": metallic,
                    "camera_position": camera_pos,
                    "defect_position": dpos,
                    "defect_size": size,
                    "protected_position": ppos,
                    "obstacle_position": opos,
                }
            )
        rep.orchestrator.wait_until_complete()
        writer.detach()
        product.destroy()
        (output / "scene_metadata.json").write_text(json.dumps(metadata, indent=2))
        stage.GetRootLayer().Export(str(output / "scene.usda"))
    finally:
        app.close()


if __name__ == "__main__":
    main()
