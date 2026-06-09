import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex")

# TODO: Fill in the DH Parameters (a, d, alpha)
DH_PARAMS = np.array([
    [0.0, 0.0595, np.radians(90)], #a1, d1, alpha1
    [0.1175, 0.0, 0.0], #a2, d2, alpha2
    [0.095, 0.0, 0.0]   #a3, d3, alpha3
], dtype=float)

EE_OFFSET = 0.0100

JOINT_LIMITS = np.array([
    [-1.91986, 1.91986],
    [-1.7453293, 1.7453293],
    [-1.69, 1.69],
], dtype=float)


def dh_transform(a, d, alpha, theta):
    out = [
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ]
    return np.array(out)

def fk(q):
    """
    TODO: Implement the full forward kinematics using DH_PARAMS and EE_OFFSET.
    Return the 4x4 end-effector transformation matrix.
    T = T0@T1@T2@Tend
    """
    T0 = dh_transform(*DH_PARAMS[0], q[0])
    T1 = dh_transform(*DH_PARAMS[1], q[1])
    T2 = dh_transform(*DH_PARAMS[2], q[2])
    Tend = np.eye(4)
    Tend[:3, 3] = EE_OFFSET * np.array([1, 0, 0])
    return T0 @ T1 @ T2 @ Tend


def position(q):
    """
    TODO: Return just the (3,) position vector [X, Y, Z] from the FK matrix.
    """
    T = fk(q)
    return T[:3, 3]


def print_table():
    print("Teaching DH table")
    print("  i  joint            theta     d (m)    a (m)    alpha (rad)")
    for i, (name, (a, d, alpha)) in enumerate(zip(JOINT_NAMES, DH_PARAMS), start=1):
        print(f"  {i:<2} {name:<15} q{i:<7} {d:8.4f} {a:8.4f} {alpha:12.6f}")
    print(f"  end-effector offset along final x-axis: {EE_OFFSET:.4f} m")

#print_table()
#print(dh_transform(0.1175, 0.0, 0.0, np.radians(45)))