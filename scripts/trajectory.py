import argparse
import os
import sys
import time
import mujoco
import numpy as np
from mujoco import viewer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from planning import dh_target as dh
from planning import ik_logic_target as ik
from planning import trajectory as traj


def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def validate_model(model):
    errors = []

    if model.nq != 3 or model.nv != 3:
        errors.append(
            f"expected nq=3 and nv=3, got nq={model.nq}, nv={model.nv}"
        )

    if model.nu != 3:
        errors.append(f"expected nu=3, got nu={model.nu}")

    if model.njnt != 3:
        errors.append(f"expected 3 joints, got njnt={model.njnt}")

    for name in dh.JOINT_NAMES:
        if mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name
        ) < 0:
            errors.append(f"missing joint '{name}'")

    if mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripperframe"
    ) < 0:
        errors.append("missing site 'gripperframe'")

    if errors:
        raise RuntimeError(
            "Model structure validation failed:\n  " + "\n  ".join(errors)
        )


def print_summary(model):
    print("Model summary:")
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}")

    print(
        "  Joint names:",
        [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                i
            )
            for i in range(model.njnt)
        ]
    )

    print(
        "  Actuator names:",
        [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                i
            )
            for i in range(model.nu)
        ]
    )

    print(
        "  Site names:",
        [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                i
            )
            for i in range(model.nsite)
        ]
    )

    print(
        "  Keyframe names:",
        [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_KEY,
                i
            )
            for i in range(model.nkey)
        ]
    )


def print_state(model, data, step):
    ee_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripperframe"
    )

    ee_pos = data.site_xpos[ee_site].copy()
    qdeg = np.degrees(data.qpos[:model.nq])

    print(
        f"step={step:3d} "
        f"q_deg=[{qdeg[0]:+.2f}, {qdeg[1]:+.2f}, {qdeg[2]:+.2f}] "
        f"ee=[{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}]"
    )


def apply_actuator_controls(model, data, q):
    q = np.asarray(q, dtype=float).reshape(3)

    for name, q_val in zip(dh.JOINT_NAMES, q):
        act_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name
        )

        if act_id < 0:
            raise RuntimeError(
                f"Actuator for joint '{name}' not found"
            )

        data.ctrl[act_id] = q_val


def traj_planning(
    model,
    data,
    joint,
    start,
    end,
    steps=100,
    gui=True
):
    if joint:
        trajectory = traj.trajectory_joint(
            start,
            end,
            steps
        )
    else:
        trajectory = traj.trajectory_cartesian(
            start,
            end,
            steps
        )

    if trajectory.shape != (steps, 3):
        raise RuntimeError(
            "Trajectory output must be shape (steps, 3)"
        )

    # Initialize first configuration
    data.qpos[:model.nq] = trajectory[0]
    mujoco.mj_forward(model, data)

    if gui:
        v = viewer.launch_passive(model, data)

        v.cam.azimuth = 140
        v.cam.elevation = -25
        v.cam.distance = 0.65
        v.cam.lookat[:] = [0.02, 0.0, 0.13]

        print("Starting trajectory execution in GUI...")

        for step, q in enumerate(trajectory, start=1):
            if not v.is_running():
                break

            apply_actuator_controls(model, data, q)
            mujoco.mj_step(model, data)
            v.sync()

            if (
                step == 1
                or step == steps
                or step % max(1, steps // 10) == 0
            ):
                print_state(model, data, step)

        while v.is_running():
            v.sync()

    else:
        print("Starting trajectory execution headless...")

        for step, q in enumerate(
            trajectory,
            start=1
        ):
            apply_actuator_controls(model, data, q)
            mujoco.mj_step(model, data)

            if (
                step == 1
                or step == steps
                or step % max(1, steps // 10) == 0
            ):
                print_state(model, data, step)

            time.sleep(model.opt.timestep)


def main():
    parser = argparse.ArgumentParser(
        description="Run a joint or Cartesian trajectory in MuJoCo"
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--joint_start_and_end",
        nargs=6,
        type=float,
        metavar=(
            "Q1_start",
            "Q2_start",
            "Q3_start",
            "Q1_end",
            "Q2_end",
            "Q3_end"
        ),
        help="Start and end joint angles in radians"
    )

    group.add_argument(
        "--cartesian_start_and_end",
        nargs=6,
        type=float,
        metavar=(
            "X_start",
            "Y_start",
            "Z_start",
            "X_end",
            "Y_end",
            "Z_end"
        ),
        help="Start and end Cartesian positions in meters"
    )

    parser.add_argument(
        "--xml",
        default="assets/scene.xml",
        help="Path to MuJoCo XML model"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of trajectory steps"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI"
    )

    args = parser.parse_args()

    if args.joint_start_and_end:
        start_q = np.array(
            args.joint_start_and_end[:3],
            dtype=float
        )

        end_q = np.array(
            args.joint_start_and_end[3:],
            dtype=float
        )

        is_joint = True

    else:
        start_pos = np.array(
            args.cartesian_start_and_end[:3],
            dtype=float
        )

        end_pos = np.array(
            args.cartesian_start_and_end[3:],
            dtype=float
        )

        start_q = ik.solve(start_pos).q
        end_q = ik.solve(end_pos).q
        is_joint = False

    if (
        np.any(np.isnan(start_q))
        or np.any(np.isnan(end_q))
    ):
        raise RuntimeError(
            "IK failed for one of the Cartesian endpoints"
        )

    model, data = load_model(args.xml)

    validate_model(model)
    print_summary(model)

    traj_planning(
        model=model,
        data=data,
        joint=is_joint,
        start=start_q,
        end=end_q,
        steps=args.steps,
        gui=not args.headless
    )


if __name__ == "__main__":
    main()
