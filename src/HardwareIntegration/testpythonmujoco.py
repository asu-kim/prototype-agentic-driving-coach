#!/usr/bin/env python3
"""Quick MuJoCo viewer for src/HardwareIntegration/models/car.xml.

Run from the repository root:
    python3 src/HardwareIntegration/testpythonmujoco.py

Keyboard speed is in real km/h. The XML uses 1 MuJoCo meter = 100 real
meters, so the script converts real speed to scaled MuJoCo motion.

Optional:
    python3 src/HardwareIntegration/testpythonmujoco.py --speed-step-kmh 10
    python3 src/HardwareIntegration/testpythonmujoco.py --max-speed-kmh 120
    python3 src/HardwareIntegration/testpythonmujoco.py --visual-speedup 3
    python3 src/HardwareIntegration/testpythonmujoco.py --fps 60
    python3 src/HardwareIntegration/testpythonmujoco.py --chase-camera
    python3 src/HardwareIntegration/testpythonmujoco.py --free-camera
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MuJoCo Python package is not installed in this environment.\n"
        "Install it with: python3 -m pip install mujoco"
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "car.xml"
REAL_METERS_PER_SIM_METER = 100.0


@dataclass
class KeyboardControl:
    speed_kmh: float = 0.0
    heading_rad: float = 0.0
    speed_step_kmh: float = 10.0
    steer_step_rad: float = 0.04
    max_speed_kmh: float = 120.0
    max_heading_rad: float = 0.5

    def clamp(self) -> None:
        self.speed_kmh = max(-self.max_speed_kmh, min(self.max_speed_kmh, self.speed_kmh))
        self.heading_rad = max(-self.max_heading_rad, min(self.max_heading_rad, self.heading_rad))

    def handle_key(self, keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""

        if key == "w":
            self.speed_kmh += self.speed_step_kmh
        elif key == "s":
            self.speed_kmh -= self.speed_step_kmh
        elif key == "a":
            self.heading_rad += self.steer_step_rad
        elif key == "d":
            self.heading_rad -= self.steer_step_rad
        elif key == " ":
            self.speed_kmh = 0.0
            self.heading_rad = 0.0
        elif key == "r":
            self.speed_kmh = 0.0
        elif key == "c":
            self.heading_rad = 0.0

        self.clamp()

    @property
    def sim_speed_mps(self) -> float:
        real_mps = self.speed_kmh / 3.6
        return real_mps / REAL_METERS_PER_SIM_METER


def name_exists(model: mujoco.MjModel, obj_type: int, name: str) -> bool:
    return mujoco.mj_name2id(model, obj_type, name) >= 0


def print_scene_check(model: mujoco.MjModel) -> None:
    checks = [
        (mujoco.mjtObj.mjOBJ_BODY, "car", "ego car at center lane start"),
        (mujoco.mjtObj.mjOBJ_BODY, "csv_right_lane_car", "right-lane car at 0.5 km"),
        (mujoco.mjtObj.mjOBJ_GEOM, "exit_sign_board", "exit sign at 1.6 km"),
        (mujoco.mjtObj.mjOBJ_GEOM, "road_asphalt", "4 km straight road"),
    ]

    print("Scene check:")
    for obj_type, name, label in checks:
        status = "OK" if name_exists(model, obj_type, name) else "MISSING"
        print(f"  {status:7s} {name:20s} {label}")
    print()


def set_driver_camera(viewer: mujoco.viewer.Handle, model: mujoco.MjModel) -> bool:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "driver_cam")
    if camera_id < 0:
        return False
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
    return True


def set_chase_camera(viewer: mujoco.viewer.Handle, model: mujoco.MjModel) -> None:
    car_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = car_id
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -18


def set_car_pose(data: mujoco.MjData, x: float, y: float, heading_rad: float) -> None:
    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = 0.03
    data.qpos[3] = math.cos(heading_rad / 2.0)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = math.sin(heading_rad / 2.0)
    data.qvel[:6] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="View and test the highway MuJoCo scene.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to car.xml")
    parser.add_argument("--speed-step-kmh", type=float, default=10.0, help="Real km/h increment per W/S press")
    parser.add_argument("--steer-step", type=float, default=0.04, help="Heading increment per A/D press, radians")
    parser.add_argument("--max-speed-kmh", type=float, default=120.0, help="Maximum target real speed")
    parser.add_argument("--visual-speedup", type=float, default=1.0, help="Multiply visual motion while preserving printed real speed")
    parser.add_argument("--fps", type=float, default=60.0, help="Viewer update rate; lower this if the display jitters")
    parser.add_argument("--chase-camera", action="store_true", help="Start in third-person tracking view")
    parser.add_argument("--free-camera", action="store_true", help="Start with the default free camera")
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    forward_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "forward")
    turn_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "turn")
    if forward_id < 0 or turn_id < 0:
        raise SystemExit("Missing expected actuators: forward and/or turn")

    print(f"Loaded model: {model_path}")
    print_scene_check(model)
    print("Controls:")
    print("  Close the MuJoCo viewer window to stop.")
    print("  Starts in driver camera view, looking forward down the road.")
    print("  Mouse drag/scroll controls the camera after switching/free camera interactions.")
    print("  W/S: increase/decrease target real speed in km/h.")
    print("  A/D: steer left/right.")
    print("  Space: stop and center steering.")
    print("  R: zero speed. C: center steering.\n")
    print(f"Scale: 1 sim meter = {REAL_METERS_PER_SIM_METER:.0f} real meters")
    print(f"Visual speedup: {args.visual_speedup:.2f}x")
    print(f"Viewer update rate: {args.fps:.1f} FPS\n")

    control = KeyboardControl(
        speed_step_kmh=args.speed_step_kmh,
        steer_step_rad=args.steer_step,
        max_speed_kmh=args.max_speed_kmh,
    )
    sim_x = 0.0
    sim_y = 0.0
    frame_dt = 1.0 / max(1.0, args.fps)

    with mujoco.viewer.launch_passive(model, data, key_callback=control.handle_key) as viewer:
        if args.chase_camera:
            set_chase_camera(viewer, model)
        elif not args.free_camera:
            if not set_driver_camera(viewer, model):
                set_chase_camera(viewer, model)

        last_print = time.monotonic()
        while viewer.is_running():
            step_start = time.monotonic()

            sim_speed = control.sim_speed_mps * args.visual_speedup
            sim_x += math.cos(control.heading_rad) * sim_speed * frame_dt
            sim_y += math.sin(control.heading_rad) * sim_speed * frame_dt
            sim_y = max(-0.25, min(0.25, sim_y))

            data.ctrl[forward_id] = 0.0
            data.ctrl[turn_id] = max(-0.5, min(0.5, -control.heading_rad))
            set_car_pose(data, sim_x, sim_y, control.heading_rad)
            mujoco.mj_forward(model, data)

            now = time.monotonic()
            if now - last_print >= 1.0:
                real_distance_m = sim_x * REAL_METERS_PER_SIM_METER
                print(
                    f"speed={control.speed_kmh:6.1f} km/h | "
                    f"ego x={sim_x:6.2f} sim m | distance={real_distance_m:7.1f} real m | "
                    f"lane y={sim_y: .2f}"
                )
                last_print = now

            viewer.sync()

            sleep_time = frame_dt - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
