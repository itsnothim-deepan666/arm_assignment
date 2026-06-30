import numpy as np
from planning import dh_target as dh
from planning import ik_logic_target as ik
from planning import fk_logic_target as fk

def trajectory_joint(start_q, end_q, num_steps):
    
    trajectory = np.linspace(start_q, end_q, num_steps)
    return trajectory

def trajectory_cartesian(start_pos, end_pos, num_steps):
    
    trajectory = np.linspace(start_pos, end_pos, num_steps)
    trajectory = np.array([ik.solve(pos).q for pos in trajectory])
    return trajectory