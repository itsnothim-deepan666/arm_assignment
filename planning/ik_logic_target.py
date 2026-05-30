import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from planning import dh

class IKSolution:
    def __init__(self, q, converged, iterations, position_error):
        self.q = q
        self.converged = converged
        self.iterations = iterations
        self.position_error = position_error

def clamp_to_limits(q, limits):
    return np.clip(q, limits[:, 0], limits[:, 1])

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def within_limits(q, limits):
    return bool(np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]))

def solve(target, q0=None, elbow="down"):
    """
    TODO: Implement the analytical inverse kinematics for the 3-DOF arm.
    """
    target = np.asarray(target, dtype=float).reshape(3)
    limits = dh.JOINT_LIMITS

    # Placeholder: just return zeros
    q = np.zeros(3)
    err = float(np.linalg.norm(target - dh.position(q)))
    return IKSolution(q=q, converged=False, iterations=0, position_error=err)

def print_solution(target, sol, fk_pos):
    p = fk_pos
    print(f"target      = [{target[0]:+.6f} {target[1]:+.6f} {target[2]:+.6f}] m")
    print(f"converged   = {sol.converged}")
    print(f"iterations  = {sol.iterations}")
    print(f"q           = [{sol.q[0]:+.6f} {sol.q[1]:+.6f} {sol.q[2]:+.6f}] rad")
    print(f"q_deg       = [{np.degrees(sol.q[0]):+.2f} {np.degrees(sol.q[1]):+.2f} {np.degrees(sol.q[2]):+.2f}] deg")
    print(f"dh_fk(q)    = [{p[0]:+.6f} {p[1]:+.6f} {p[2]:+.6f}] m")
    print(f"pos_error   = {sol.position_error * 1000:.6f} mm")

def run_sweep(n):
    limits = dh.JOINT_LIMITS
    rng = np.random.default_rng(0)
    qs = rng.uniform(limits[:, 0], limits[:, 1], size=(n, 3))
    errors = []
    for q_true in qs:
        target = dh.position(q_true)
        sol = solve(target, elbow="down")
        errors.append(sol.position_error)

    errors = np.array(errors)
    print(f"Sweep: {n} pure-DH targets")
    print(f"  max error  = {errors.max() * 1000:.6f} mm")
    print(f"  mean error = {errors.mean() * 1000:.6f} mm")
    print(f"  all < 1 mm = {'YES' if np.all(errors < 1e-3) else 'NO'}")
