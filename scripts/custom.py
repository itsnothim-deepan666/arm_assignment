"""
    python test.py
    python test.py --xml so101.xml
    python test.py --sinusoid
    python test.py --steps 1000 --headless
"""

import argparse
import os
import sys
import time
import mujoco
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex")
EE_SITE = "gripperframe"


def load_model(xml_path):
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(ROOT_DIR, xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def set_keyframe(model, data, name):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    
    if key_id < 0:
        names = []
        for i in range(model.nkey):
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i))
        print(f"Keyframe '{name}' not found. Available ones: {names}")
        return

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def site_position(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def validate_model(model):
    errors = []
    if model.nq != 3 or model.nv != 3:
        errors.append(f"expected nq=3 and nv=3, got nq={model.nq}, nv={model.nv}")
    if model.nu != 3:
        errors.append(f"expected nu=3, got nu={model.nu}")
    if model.njnt != 3:
        errors.append(f"expected 3 joints, got njnt={model.njnt}")

    for name in JOINT_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            errors.append(f"missing joint '{name}'")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE) < 0:
        errors.append(f"missing site '{EE_SITE}'")

    if errors:
        raise RuntimeError("Model structure validation failed:\n  " + "\n  ".join(errors))


def apply_sinusoid(model, data):
    if model.nu == 0:
        return

    ranges = model.actuator_ctrlrange[:model.nu]
    centers = 0.5 * (ranges[:, 0] + ranges[:, 1])
    amplitudes = 0.35 * (ranges[:, 1] - ranges[:, 0])

    freqs = np.array([0.12, 0.17, 0.22])[:model.nu]

    data.ctrl[:] = centers + amplitudes * np.sin(2.0 * np.pi * freqs * data.time)
    print(data.ctrl)


def print_summary(model):
    print("Model Summary")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, ngeom={model.ngeom}")
    
    print("Joints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = np.degrees(model.jnt_range[i])
        print(f"  {i}: {name} -> Limits: [{lo:.2f}, {hi:.2f}] deg")
        
    print("Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {name}")
    print("\n")


def print_state(model, data, step):
    ee = site_position(model, data, EE_SITE)
    q_deg = np.degrees(data.qpos[:model.nq])
    
    q_text = ", ".join([f"{q:+.2f}" for q in q_deg])
    #print(f"[Step {step:5d}] Time: {data.time:7.3f}s | q_deg: [{q_text}] | EE XYZ: [{ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f}]")


def run_headless(model, data, steps, sinusoid):
    print("Running in headless mode...")
    for step in range(steps):
        if sinusoid:
            apply_sinusoid(model, data)
        mujoco.mj_step(model, data)
        
        if step == 0 or (step + 1) % 100 == 0:
            print_state(model, data, step + 1)


def run_viewer(model, data, sinusoid):
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(model, data)
    
    viewer.cam.azimuth = 140
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.65
    viewer.cam.lookat[:] = [0.02, 0.0, 0.13]

    last_print = 0.0
    step = 0
    
    while viewer.is_running():
        if sinusoid:
            apply_sinusoid(model, data)

        mujoco.mj_step(model, data)
        viewer.sync()

        now = time.time()
        if now - last_print >= 0.25:
            print_state(model, data, step)
            last_print = now
        step += 1

def custom(model, data, ctrl_inputs):
    data.ctrl[:] = ctrl_inputs


def main():
    parser = argparse.ArgumentParser(description="SO101 3-DOF MuJoCo Test Script")
    parser.add_argument("--xml", default=os.path.join("assets", "scene.xml"), help="Path to XML model file")
    parser.add_argument("--keyframe", default=None, help="Initial position keyframe")
    parser.add_argument("--sinusoid", action="store_true", help="Move joints using sine waves")
    parser.add_argument("--headless", action="store_true", help="Run simulation without the UI window")
    parser.add_argument("--steps", type=int, default=500, help="Steps to run if headless")
    #parser.add_argument("--ctrl", nargs=3, type=float, default=[0, 0, 0], help="Control inputs for the actuators")
    args = parser.parse_args()
    model, data = load_model(args.xml)
    validate_model(model)

    if args.keyframe is not None:
        set_keyframe(model, data, args.keyframe)
    else:
        mujoco.mj_forward(model, data)
    
    print_summary(model)
    print_state(model, data, 0)

    # whether to run UI or headless
    if args.headless:
        run_headless(model, data, args.steps, args.sinusoid)
    else:
        run_viewer(model, data, args.sinusoid)



if __name__ == "__main__":
    main()
