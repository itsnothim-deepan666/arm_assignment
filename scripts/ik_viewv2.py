"""
Solve pure DH IK for a target and visualize the resulting pose moving from folded in MuJoCo.
"""

import argparse
import os
import sys
import time

import mujoco
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from planning import ik_logicv2
from planning import fk_logic
from planning import dh

def run_viewer(model, data, target_q, title):
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = 140
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.65
    viewer.cam.lookat[:] = [0.02, 0.0, 0.13]

    print(title)
    print("Close the viewer window to exit.")
    
    # Set the actuators' control signals to the target IK solution
    for name, q_val in zip(fk_logic.JOINT_NAMES, target_q):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id >= 0:
            data.ctrl[act_id] = q_val
            
    while viewer.is_running():
        # Step the physics engine so the arm moves to the target
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


def main():
    parser = argparse.ArgumentParser(description="Visualize a pure DH IK solution in MuJoCo moving from folded keyframe")
    parser.add_argument("--xml", default=os.path.join("assets", "scene.xml"), help="MJCF file to load")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--seed", nargs=3, type=float, metavar=("Q1", "Q2", "Q3"))
    parser.add_argument("--elbow", choices=("down", "up"), default="down")
    args = parser.parse_args()

    # Load MuJoCo model for visualization
    model, data = fk_logic.load_model(args.xml)
    
    target = np.array(args.target, dtype=float)
    seed = np.array(args.seed, dtype=float) if args.seed is not None else None

    # Pure DH IK solve
    sol = ik_logicv2.solve(target, seed, elbow=args.elbow)
    fk_pos = dh.position(sol.q)
    ik_logicv2.print_solution(target, sol, fk_pos)

    # Set initial state to the 'folded' keyframe
    try:
        folded_q = fk_logic.keyframe_qpos(model, "folded")
        fk_logic.set_qpos(model, data, folded_q)
        # Initialize controls to the folded position so it doesn't jerk violently on frame 1
        for name, q_val in zip(fk_logic.JOINT_NAMES, folded_q):
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id >= 0:
                data.ctrl[act_id] = q_val
    except Exception as e:
        print(f"Warning: Could not set 'folded' keyframe: {e}")

    # Reset velocities
    data.qvel[:] = 0.0

    run_viewer(
        model,
        data,
        sol.q,
        f"IK visualization for target [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}] m",
    )


if __name__ == "__main__":
    main()
