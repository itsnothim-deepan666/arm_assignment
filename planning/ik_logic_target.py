import os
import sys
import numpy as np
import csv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from planning import dh_target as dh

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
    
    q1 = np.atan2(target[1], target[0])
    r = np.hypot(target[0], target[1])
    D = (r**2 + (target[2] - dh.DH_PARAMS[0][1])**2 - dh.DH_PARAMS[1][0]**2 - (dh.DH_PARAMS[2][0] + dh.EE_OFFSET)**2)/(2*dh.DH_PARAMS[1][0]*(dh.DH_PARAMS[2][0] + dh.EE_OFFSET))
    if abs(D) > 1.0:
        q = np.array([np.nan, np.nan, np.nan])
        err = float('inf')
        return IKSolution(q=q, converged=False, iterations=0, position_error=err)

    q3 = np.arctan2(np.sqrt(1 - D**2), D) if elbow == "up" else np.arctan2(-np.sqrt(1 - D**2), D)
    q2 = np.arctan2(target[2] - dh.DH_PARAMS[0][1], r) - np.arctan2((dh.DH_PARAMS[2][0] + dh.EE_OFFSET) * np.sin(q3), dh.DH_PARAMS[1][0] + (dh.DH_PARAMS[2][0] + dh.EE_OFFSET) * np.cos(q3))
    q1 = wrap_to_pi(q1)
    q = np.array([q1, q2, q3])
    
    #q = clamp_to_limits(q, limits)
    err = np.linalg.norm(dh.position(q) - target)

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
