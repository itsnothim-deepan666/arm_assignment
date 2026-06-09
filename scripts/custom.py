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


def run_viewer(model, data):
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(model, data)
    
    viewer.cam.azimuth = 140
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.65
    viewer.cam.lookat[:] = [0.02, 0.0, 0.13]

    last_print = 0.0
    step = 0
    
    while viewer.is_running():

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

    model, data = load_model(os.path.join("assets", "scene.xml"))
    validate_model(model)

    mujoco.mj_forward(model, data)
    
    print_summary(model)
    print_state(model, data, 0)

    # whether to run UI or headless
    run_viewer(model, data)



if __name__ == "__main__":
    main()
