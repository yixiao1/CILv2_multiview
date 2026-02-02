import time
import torch
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import DataParallel
from copy import deepcopy
import cv2

########################################################
### Color plate
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_SCARLET_RED_0 = (255, 0, 0)
COLOR_SKY_BLUE_0 = (0, 0, 255)
COLOR_GREEN_0 = (0, 255, 0)
COLOR_LIGHT_GRAY = (196, 196, 196)
COLOR_PINK = (255,19, 203)
COLOR_BUTTER_0 = (252, 233, 79)
COLOR_ORANGE_0 = (252, 175, 62)
COLOR_CHOCOLATE_0 = (233, 185, 110)
COLOR_CHAMELEON_0 = (138, 226, 52)
COLOR_PLUM_0 = (173, 127, 168)
COLOR_ALUMINIUM_0 = (238, 238, 236)

color_plate = {
    '0': COLOR_SCARLET_RED_0,
    '1': COLOR_GREEN_0,
    '2': COLOR_SKY_BLUE_0,
    '3': COLOR_ALUMINIUM_0,
    '4': COLOR_CHAMELEON_0,
    '5': COLOR_CHOCOLATE_0,
    '6': COLOR_LIGHT_GRAY,
    '7': COLOR_PLUM_0,
    '8': COLOR_ORANGE_0,
    '9': COLOR_BUTTER_0
}

########################################################

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    l.sort(key=alphanum_key)

def experiment_log_path(experiment_path, dataset_name):
    # WARNING if the path exist without checkpoints it breaks
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    return os.path.join(experiment_path, dataset_name + '_result.csv')

class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __len__(self):
        return len(self.module)
#@timeit
def print_train_info(log_frequency, final_epoch, batch_size, model,
                     time_start, acc_time, loss_data, steer_loss_data, acc_loss_data, brake_loss_data=None, wp_loss_data=None):

    epoch = model._current_iteration * batch_size / len(model)

    time_end = time.time()
    batch_time = time_end - time_start
    acc_time += batch_time

    if model._current_iteration % log_frequency == 0:

        rem_iterations = ((float(final_epoch) - epoch) * len(model)) / batch_size
        rem_time = rem_iterations / (log_frequency / acc_time)
        hours = int(rem_time / 60 / 60)
        minutes = int(rem_time / 60) - hours * 60
        seconds = int(rem_time) - hours * 60 * 60 - minutes * 60

        if brake_loss_data is not None:
            print ("Training epoch {:.2f}, iteration {}, Loss {:.3f}, Steer Loss {:.3f}, Throttle Loss {:.3f}, , Brake Loss {:.3f}, {:.2f} steps/s, ETA: {:0>2d}H:{:0>2d}M:{:0>2d}S".format(
                epoch, model._current_iteration, loss_data.item(), steer_loss_data.item(), acc_loss_data.item(), brake_loss_data.item(), (log_frequency / acc_time), hours, minutes, seconds))
        elif wp_loss_data is not None:
            print ("Training epoch {:.2f}, iteration {}, Loss {:.3f}, Steer Loss {:.3f}, Acc Loss {:.3f}, WP Loss {:.3f}, {:.2f} steps/s, ETA: {:0>2d}H:{:0>2d}M:{:0>2d}S".format(
                epoch, model._current_iteration, loss_data.item(), steer_loss_data.item(), acc_loss_data.item(), wp_loss_data.item(),(log_frequency / acc_time), hours, minutes, seconds))
        else:
            print ("Training epoch {:.2f}, iteration {}, Loss {:.3f}, Steer Loss {:.3f}, Acc Loss {:.3f}, {:.2f} steps/s, ETA: {:0>2d}H:{:0>2d}M:{:0>2d}S".format(
                epoch, model._current_iteration, loss_data.item(), steer_loss_data.item(), acc_loss_data.item(),(log_frequency / acc_time), hours, minutes, seconds))
        acc_time = 0.0

    return acc_time


#@timeit
def test_stop(number_of_data, iterated_data):

    if number_of_data != 0 and \
            iterated_data >= number_of_data:
        return True
    return False


def generate_specific_rows(filePath, row_indices=[]):
    with open(filePath) as f:

        # using enumerate to track line no.
        for i, line in enumerate(f):

            # if line no. is in the row index list, then return that line
            if i in row_indices:
                yield line

def read_results(result_file, metric=''):
    head = np.genfromtxt(generate_specific_rows(result_file, row_indices=[0]), delimiter=',', dtype='str')
    col = [h.strip() for h in list(head)].index(metric)
    res = np.loadtxt(result_file, delimiter=",", skiprows=1)
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)

    return res[:, col]


def draw_offline_evaluation_results(experiment_path, metrics_list, x_range=[0, 10]):
    for metric in metrics_list:
        print('drawing results graph for ', experiment_path, 'of', metric)
        results_files = glob.glob(os.path.join(experiment_path, '*.csv'))
        for results_file in results_files:
            plt.figure()
            output_path = os.path.join(experiment_path, results_file.split('/')[-1].split('.')[-2]+ '_' + metric+'.jpg')
            results_list = read_results(results_file, metric=metric)
            epochs_list = read_results(results_file, metric='epoch')
            plt.ylabel(metric, fontsize=15)
            plt.plot(epochs_list, results_list)
            for i in range(len(results_list)):
                if results_list[i] == min(results_list):
                    plt.text(epochs_list[i], results_list[i], str(results_list[i]), color='blue',
                             fontweight='bold')
                    plt.plot(epochs_list[i], results_list[i], color='blue', marker='*')
            plt.xlabel('Epoch', fontsize=15)
            plt.xlim(left=x_range[0], right=x_range[-1])
            plt.title(results_file.split('/')[-1].split('.')[-2])
            plt.savefig(output_path)
            plt.close()


def write_model_results(experiment_path, model_name, results_dict, acc_as_action=False):
    for dataset_name, results in results_dict.items():
        results_file_csv = experiment_log_path(experiment_path, dataset_name)
        new_row = ""
        # first row if file doest exist
        if not os.path.exists(results_file_csv):
            new_row += "iteration, epoch, "
            if acc_as_action:
                new_row += "MAE_steer, MAE_acceleration, MAE"
            else:
                new_row += "MAE_steer, MAE_throttle, MAE_brake, MAE"

            new_row += "\n"
        with open(results_file_csv, 'a') as f:
            new_row += "{}, {:.2f}, ".format(results['iteration'], results['epoch'])
            if acc_as_action:
                new_row += "{:.4f}, {:.4f}, {:.4f}".format(results[model_name]['MAE_steer'],
                                                           results[model_name]['MAE_acceleration'],
                                                           results[model_name]['MAE'])
            else:
                new_row += "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(results[model_name]['MAE_steer'], results[model_name]['MAE_throttle'],
                                                                   results[model_name]['MAE_brake'], results[model_name]['MAE'])

            new_row += "\n"
            f.write(new_row)
        print (" The results have been saved in: ", results_file_csv)

def eval_done(experiment_path, dataset_paths, epoch):
    results_files = glob.glob(os.path.join(experiment_path, '*.csv'))
    for dataset_path in dataset_paths:
        if os.path.join(experiment_path, dataset_path.split('/')[-1]+'_result.csv') not in results_files:
            return False
        else:
            epochs_list = read_results(os.path.join(experiment_path, dataset_path.split('/')[-1]+'_result.csv'), metric='epoch')
            if float(epoch) in epochs_list:
                return True
            else:
                return False


def is_result_better(experiment_path, model_name, dataset_name):
    results_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='MAE')
    iter_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='iteration')
    epoch_list= read_results(os.path.join(experiment_path, dataset_name + '_result.csv'), metric='epoch')
    if len(results_list) == 1:  # There is just one result so we save the check sure.
        return True
    if results_list[-1] < min(results_list[:-1]):
        print("Result for {} at iteration {} / epoch {} is better than the previous one. SAVE".format(model_name, iter_list[-1], epoch_list[-1]))
        return True
    return False


def extract_targets(data, targets=[], ignore=[]):

    """
    Method used to get to know which positions from the dataset are the targets
    for this experiments
    Args:

    Returns:
        the float data that is actually targets

    Raises
        value error when the configuration set targets that didn't exist in metadata
    """

    targets_vec = []
    for target_name in targets:
        if target_name in ignore:
            continue
        targets_vec.append(data[target_name])

    return torch.stack(targets_vec, 1).float().squeeze()


def extract_other_inputs(data, other_inputs=[], ignore=[]):
    """
    Method used to get to know which positions from the dataset are the inputs
    for this experiments
    Args:

    Returns:
        the float data that is actually targets

    Raises
        value error when the configuration set targets that didn't exist in metadata
    """

    inputs_vec = []
    for input_name in other_inputs:
        if input_name in ignore:
            continue
        inputs_vec.append(data[input_name])
    return torch.stack(inputs_vec, 1).float()


def extract_commands(data):
    return torch.stack(data, 1).float()


##################################################
#################### Draw Path ###################
##################################################


def project_rect_to_unitsphere(pc, xi, mode='numpy'):
    # christopher mei
    # unit sphere (X_sm = X/norm(X))
    if mode == 'numpy':
        pc = pc / np.expand_dims(np.sqrt(pc[:, 0] * pc[:, 0] + pc[:, 1] * pc[:, 1] + pc[:, 2] * pc[:, 2]), -1)
    else:
        pc = pc / torch.unsqueeze(torch.sqrt(pc[:, 0] * pc[:, 0] + pc[:, 1] * pc[:, 1] + pc[:, 2] * pc[:, 2]), -1)
    # new reference frame (new center) (X_s = X_sp = X_sm + xi)
    pc[:, 2] = pc[:, 2] + xi
    # normalized plane (m=h(X_sp))
    if mode == 'numpy':
        pc = pc / np.expand_dims(pc[:, 2], -1)
    else:
        pc = pc / torch.unsqueeze(pc[:, 2], -1)
    return pc


def project_rect_to_image(pts_3d_rect, P_T, xi=None, mode='numpy'):
    """Input: nx3 points in rect camera coord.
    Output: nx2 points in image2 coord.
    """
    if xi is not None:
        pts_3d_rect = project_rect_to_unitsphere(pts_3d_rect, xi, mode)
    n = pts_3d_rect.shape[0]

    if mode == 'numpy':
        ones = np.ones((n, 1))
        pts_3d_rect = np.hstack((pts_3d_rect, ones))
        pts_2d = np.matmul(pts_3d_rect, P_T.T)  # nx3
    else:
        ones = torch.ones((n, 1), device=pts_3d_rect.device)
        pts_3d_rect = torch.hstack((pts_3d_rect, ones))
        pts_2d = torch.matmul(pts_3d_rect, P_T.T)  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def ref_to_kitti(points):
    if len(points) > 0:
        points_ = deepcopy(points)
        points_[:, 0], points_[:, 1], points_[:, 2] = -points[:, 1], -points[:, 2], points[:, 0]
        return points_

    return points


def draw_poly(img, uv_list, color_bgr, w_img, h_img, transperency=0.5):
    pts = [pt for pt in uv_list if pt is not None and 0 <= pt[0] < w_img and 0 <= pt[1] < h_img and -1. < pt[2]]
    if len(pts) >= 3:
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 3)
        white_rect = img.copy()  # np.ones(img.shape, dtype=np.uint8) * 255
        cv2.fillPoly(white_rect, np.array([pts_np[:, :2]], dtype=np.int32), color=color_bgr)
        img = cv2.addWeighted(img, 1 - transperency, white_rect, transperency, 1.0)

    return img


def project_points(pts_cam3d, P_T, xi=None):
    pts_cam2d = []
    if pts_cam3d is not None:
        # move pointcloud to camera-origin center
        if len(pts_cam3d) > 0:
            # pc_velo = np.concatenate([pc_velo, np.ones((len(pc_velo), 1))], axis=1)
            # transform pointcloud to image coordinates
            pts_cam3d = ref_to_kitti(pts_cam3d)
            pts_cam2d = project_rect_to_image(pts_cam3d[:, :3], P_T, xi, mode='numpy')
            pts_cam2d = np.concatenate([pts_cam2d, np.expand_dims(pts_cam3d[:, 2], -1)], axis=1)

    return pts_cam2d


def transform_3d(pt, cam, cam_plus):
    pt = (cam_plus @ pt.T).T
    pt = (np.linalg.inv(cam) @ pt.T).T
    pt = pt[:, :3] / np.expand_dims(pt[:, 3], -1)
    return pt


def build_pose_matrix(rotation, translation):
    if len(rotation) == 3:
        R = create_rot_matrix(rotation[0], rotation[1], rotation[2], True)
    elif len(rotation) == 9:
        R = np.asarray(rotation).reshape(3, 3)

    T = np.asarray(translation)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = T

    return M


def rotate_y(angle):
    """
    Create a Y-rotation matrix based on a given angle.
        angle: angle value in degrees.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    return r


def rotate_x(angle):
    """
    Create a X-rotation matrix based on a given angle.
        angle: angle value in degrees.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    return r


def rotate_z(angle):
    """
    Create a Z-rotation matrix based on a given angle.
        angle: angle value in degrees.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return r


def create_rot_matrix(rot_x=0.0, rot_y=0.0, rot_z=0.0, degrees=True):
    """
    Create a rotation matrix based on a given angle for each axis.
        rot_x: X-angle value in degrees.
        rot_y: Y-angle value in degrees.
        rot_z: Z-angle value in degrees.
    """
    m_rot_x = rotate_x(np.pi * rot_x / 180.0) if degrees else rotate_x(rot_x)
    m_rot_y = rotate_y(np.pi * rot_y / 180.0) if degrees else rotate_y(rot_y)
    m_rot_z = rotate_z(np.pi * rot_z / 180.0) if degrees else rotate_z(rot_z)

    P3 = np.matmul(m_rot_x, m_rot_y)
    P3 = np.matmul(P3, m_rot_z)

    return P3


def draw_vehicle_path_on_image(
    img,
    acceleration,        # m/s^2, scalar (longitudinal acceleration of vehicle center)
    steering_rad,        # steering angle in degrees (front wheel steer)
    cam_T,               # camera extrinsic: vehicle_frame -> camera_frame (4x4) or (3x4)
    cam_K,               # camera extrinsic: vehicle_frame -> camera_frame (4x4) or (3x4)
    width,               # vehicle track width (meters) distance between left and right wheels
    wheelbase=1.4,       # meters, distance between axle centers (default typical car)
    initial_speed=0.0,   # m/s
    horizon=5.0,         # seconds to simulate forward
    dt=1.0,              # simulation timestep (s)
    w_img=300,
    h_img=300,
    xi=None,
):
    """
    Simulate a kinematic bicycle model and draw the predicted path and wheel tracks
    on the front camera image.

    Notes on inputs:
      - cam_T must map homogeneous vehicle-frame points -> camera-frame points:
          [Xc, Yc, Zc, 1].T = cam_T @ [Xv, Yv, Zv, 1].T
        Accepts 4x4 or 3x4. If you have camera->vehicle, invert it first.
    """
    # image copy to draw onto
    h_img_i, w_img_i = img.shape[:2]
    if w_img != w_img_i or h_img != h_img_i:
        img = cv2.resize(img, (w_img, h_img))

    # simulation variables
    steps = int(np.ceil(horizon / dt))
    v = float(initial_speed)
    a = float(acceleration)
    # delta = float(np.deg2rad(steering_rad))
    delta = float(steering_rad)

    # initial state: center of vehicle at origin, heading = 0 rad (pointing +X)
    x = 0.0
    y = 0.0
    theta = 0.0

    # Axles positions in vehicle frame relative to vehicle center:
    # assuming vehicle center is mid-point between axles:
    front_axle_x = wheelbase / 2.0

    # storage for world points (vehicle frame)
    pts_cam3d = []
    pts_cam3d_center = []

    # center_path_bev = np.array([x, y, 0.0, 1.0])
    # compute wheel positions in vehicle frame (before rotation/translation)
    # left is +y, right is -y (we defined +Y left)
    pts_local = np.array([[front_axle_x, width / 2.0, 0.0, 1.0], [front_axle_x, -width / 2.0, 0.0, 1.0]])
    pts_local_center = np.array([[front_axle_x, 0, 0.0, 1.0]])

    for i in range(steps):
        cam_T_plus = build_pose_matrix(rotation=[0, 0, theta], translation=[x, y, 0.0])

        # # transform wheels to world (vehicle frame positioned at x,y,theta)
        pts_cam3d.append(transform_3d(pts_local, cam_T, cam_T_plus))
        pts_cam3d_center.append(transform_3d(pts_local_center, cam_T, cam_T_plus))

        # integrate kinematic bicycle model (simple Euler)
        # theta_dot = v / L * tan(delta)
        if wheelbase == 0:
            theta_dot = 0.0
        else:
            theta_dot = (v / wheelbase) * np.tan(delta)

        # vehicle center velocity components
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)

        # Euler integration
        x += x_dot * dt
        y += y_dot * dt
        theta += theta_dot * dt
        v += a * dt

    pts_cam3d = np.array(pts_cam3d, dtype=float).reshape(-1, 3)
    pts_cam3d_center = np.array(pts_cam3d_center, dtype=float).reshape(-1, 3)
    pts_cam2d = project_points(pts_cam3d, cam_K, xi)
    pts_cam2d_center = project_points(pts_cam3d_center, cam_K, xi)

    # draw polygon
    # pts_cam2d = np.array(pts_cam2d, dtype=int).reshape(-1, 2)
    pts_cam2d = np.array(pts_cam2d, dtype=int).reshape(-1, 2, 3)
    pts_cam2d[:, 1, :] = np.flip(pts_cam2d[:, 1, :], axis=0)
    pts_cam2d = pts_cam2d.swapaxes(0, 1)
    pts_cam2d = np.array(pts_cam2d, dtype=int).reshape(-1, 3)
    img = draw_poly(img, pts_cam2d, color_bgr=(255, 222, 33), w_img=w_img, h_img=h_img, transperency=0.2)

    # draw line
    pts_cam2d_center = np.array(pts_cam2d_center, dtype=int).reshape(-1, 2, 3)
    pts_cam2d_center[:, 1, :] = np.flip(pts_cam2d_center[:, 1, :], axis=0)
    pts_cam2d_center = pts_cam2d_center.swapaxes(0, 1)
    pts_cam2d_center = np.array(pts_cam2d_center, dtype=int).reshape(-1, 3)
    for (u, v, _) in pts_cam2d_center:
        if 0 <= u < w_img and 0 <= v < h_img:
            cv2.drawMarker(
                img,
                (u, v),
                color=(0, 0, 255),           # red cross
                markerType=cv2.MARKER_CROSS, # or cv2.MARKER_TILTED_CROSS
                markerSize=6,
                thickness=1,
                line_type=cv2.LINE_AA,
            )


    if w_img != w_img_i or h_img != h_img_i:
        img = cv2.resize(img, (w_img_i, h_img_i))

    return img


def add_wp_to_image(src_images, acceleration, steering_rad, speed, cam_T, cam_K, cam_size, xi=None):

    ret_images = []
    for indx, img in enumerate(src_images):
        img_wp = draw_vehicle_path_on_image(img, acceleration, steering_rad, cam_T=cam_T[indx], cam_K=cam_K[indx], 
            width=1.7, wheelbase=1.45, initial_speed=speed, horizon=8.0, dt=0.5,
            xi=xi[indx], w_img=cam_size[indx][0], h_img=cam_size[indx][1])
        ret_images.append(img_wp)
    return ret_images
