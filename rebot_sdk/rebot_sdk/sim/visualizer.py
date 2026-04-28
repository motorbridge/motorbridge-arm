from __future__ import annotations


class MeshCatArmVisualizer:
    """Optional MeshCat visualizer for SimArm trajectories."""

    def __init__(self, urdf_path: str, open_browser: bool = True) -> None:
        try:
            import meshcat
            import meshcat.geometry as mcg
            import numpy as np
            import pinocchio as pin
            from pinocchio.visualize import MeshcatVisualizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "MeshCat visualization dependencies are missing. "
                "Install extras: uv sync --extra full"
            ) from exc

        self._mcg = mcg
        self._np = np
        self._pin = pin

        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)
        visual_data = visual_model.createData()

        viewer = meshcat.Visualizer(zmq_url=None)
        viz = MeshcatVisualizer(model, collision_model=None, visual_model=visual_model, data=data, visual_data=visual_data)
        viz.initViewer(viewer, loadModel=False)
        viz.loadViewerModel()

        if open_browser:
            print(f"MeshCat URL: {viewer.url()}")

        self._model = model
        self._viewer = viewer
        self._viz = viz

    @property
    def model(self):
        return self._model

    def update(self, q: list[float]) -> None:
        self._viz.display(self._np.asarray(q, dtype=float))

    def clear_paths(self) -> None:
        for name in ("traj_path/ref", "traj_path/actual"):
            try:
                del self._viewer[name]
            except Exception:
                pass

    def draw_path(self, points_xyz: list[list[float]], name: str, color: int) -> None:
        if len(points_xyz) < 2:
            return
        pts = self._np.array(points_xyz, dtype=self._np.float32).T
        line = self._mcg.Line(
            self._mcg.PointsGeometry(pts),
            self._mcg.LineBasicMaterial(color=color, linewidth=2),
        )
        self._viewer[name].set_object(line)

    def draw_ref_path(self, points_xyz: list[list[float]]) -> None:
        self.draw_path(points_xyz, "traj_path/ref", color=0x888888)

    def draw_actual_path(self, points_xyz: list[list[float]]) -> None:
        self.draw_path(points_xyz, "traj_path/actual", color=0x00CC44)
