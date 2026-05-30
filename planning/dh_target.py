import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex")

# TODO: Fill in the DH Parameters (a, d, alpha)
DH_PARAMS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
], dtype=float)

EE_OFFSET = 0.0100

JOINT_LIMITS = np.array([
    [-1.91986, 1.91986],
    [-1.7453293, 1.7453293],
    [-1.69, 1.69],
], dtype=float)


def dh_transform(a, d, alpha, theta):
    """
    TODO: Implement the standard DH transformation matrix.
    Return a 4x4 numpy array.
    """
    return np.eye(4)


def fk(q):
    """
    TODO: Implement the full forward kinematics using DH_PARAMS and EE_OFFSET.
    Return the 4x4 end-effector transformation matrix.
    """
    return np.eye(4)


def position(q):
    """
    TODO: Return just the (3,) position vector [X, Y, Z] from the FK matrix.
    """
    return np.zeros(3)


def print_table():
    print("Teaching DH table")
    print("  i  joint            theta     d (m)    a (m)    alpha (rad)")
    for i, (name, (a, d, alpha)) in enumerate(zip(JOINT_NAMES, DH_PARAMS), start=1):
        print(f"  {i:<2} {name:<15} q{i:<7} {d:8.4f} {a:8.4f} {alpha:12.6f}")
    print(f"  end-effector offset along final x-axis: {EE_OFFSET:.4f} m")
