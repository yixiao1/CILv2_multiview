import numpy as np

from scipy.spatial.transform import Rotation as R

def rotation_euler_to_matrix33(rotation_euler):
    matrix33 = R.from_euler('xyz', rotation_euler, degrees=True).as_matrix()
    return matrix33

def rotation33_and_position_to_matrix44(rotation, position):
    matrix44 = np.empty(shape=(4,4))
    matrix44[:3,:3] = rotation
    matrix44[:3,3] = position
    matrix44[3,:3] = 0
    matrix44[3,3] = 1
    return matrix44