import os
import numpy as np
import pygame
import cv2

from train_rl.visualization.src.camara_manager import CameraManager
from train_rl.visualization.src.hud import HUD
from train_rl.utilities.transformations import rotation33_and_position_to_matrix44, rotation_euler_to_matrix33
from train_rl.utilities.configs import get_env_config_from_train_test_config, get_config
from train_rl.utilities.visualization import save_video_from_images


class Monitor(object):
    def __init__(self, experiment_name, episode, eval_idx, fps, dim, save, train_test_config_name, rlad_video=False):
        self.experiment_name = experiment_name
        self.experiment_path = f"{os.getenv('HOME')}/results/CILv2_multiview/{experiment_name}"
        self.train_test_config_name = train_test_config_name
        self.dim = dim
        self.episode = episode
        self.eval_idx = eval_idx
        self.fps = fps
        self.save = save

        self.log = self.loadLogFile()
        self.env_config = get_env_config_from_train_test_config(
            experiment_name, self.train_test_config_name)[0]

        # limits of the steps in this episode.
        self.step_min, self.step_max = self.getStepsLimits()
        self.step_idx = self.step_min-1

        self.camera_info = self.getCameraInfo()
        self.camera_transformation = self.computeCameraTransformation()

        waypoints_3d = self.getWaypoints()
        self.projected_waypoints = self.projectWaypoints(waypoints_3d)

        if train_test_config_name == 'train.yaml':
            monitor_path = f"{self.experiment_path}/monitor"
        else:
            monitor_path = f"{self.experiment_path}/{train_test_config_name.split('.')[0]}/monitor"

        self.camera_manager = CameraManager(monitor_path=monitor_path, episode=self.episode, eval_idx=self.eval_idx,
                                            dim=self.dim, projected_waypoints=self.projected_waypoints, step_min=self.step_min, step_max=self.step_max)
        self.hud = HUD(dim=self.dim, log=self.log,
                       env_config=self.env_config, episode=episode, fps=self.fps, rlad_video=rlad_video)

        self.video_name = f"{monitor_path}/{eval_idx}_{episode:04d}_ann.mp4"
        self.annotated_images = []

    def loadLogFile(self):
        if self.train_test_config_name == 'train.yaml':
            log_episode = get_config(f"{self.experiment_path}/logs/{self.eval_idx}_{self.episode:04d}.json")
        else:
            log_episode = get_config(f"{self.experiment_path}/{self.train_test_config_name.split('.')[0]}/logs/{self.eval_idx}_{self.episode:04d}.json")

        if log_episode['stats'] == {}:
            raise Exception(f'Episode {self.episode} has no information.')
       
        return log_episode

    def getStepsLimits(self):
        steps = list(self.log['steps'].keys())
        step_min = int(steps[0])
        step_max = int(steps[-1])
        return step_min, step_max

    def getCameraInfo(self):
        obs_config = get_config(
            f"{self.experiment_path}/configs/observation.yaml")

        camera_info = {'position': obs_config['rgb_backontop']['location'],
                       'rotation': obs_config['rgb_backontop']['rotation'],
                       'fov': obs_config['rgb_backontop']['fov'],
                       'width': obs_config['rgb_backontop']['width'],
                       'height': obs_config['rgb_backontop']['height']}
        return camera_info

    def computeCameraTransformation(self):
        rotation33 = rotation_euler_to_matrix33(self.camera_info['rotation'])
        camera_transformation = rotation33_and_position_to_matrix44(
            rotation33, self.camera_info['position'])

        return camera_transformation

    def getWaypoints(self):
        waypoints = {'steps': {}}

        for step_idx, data in self.log['steps'].items():
            waypoints['steps'][int(step_idx)] = np.asarray(
                data['observation']['waypoints']['location']).T  # 3xN

        return waypoints

    def projectWaypoints(self, waypoints_3d):

        K = self.computeCameraIntrinsics()
        projected_waypoints = {'steps': {}}

        for step_idx, waypoints in waypoints_3d['steps'].items():

            waypoints = np.r_[waypoints, [np.ones(waypoints.shape[1])]]

            # transform waypoints coordinates from vheicle frame to sensor frame.
            waypoints_wrt_camera = np.dot(
                np.linalg.inv(self.camera_transformation), waypoints)

            # change from the UE4s coordinate system to standard camera coordinate system.
            matrix_standard = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])

            waypoints_wrt_camera = np.array(
                [waypoints_wrt_camera[0], waypoints_wrt_camera[1], waypoints_wrt_camera[2]])

            waypoints_wrt_camera = np.dot(
                matrix_standard, waypoints_wrt_camera)

            # Apply projection equation: K*Points.
            points_2d = np.dot(K, waypoints_wrt_camera)

            # normalize coordinates using third dimention.
            points_2d = np.array([
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :]])

            # discard points outside of the screen.
            points_2d = points_2d.T
            points_in_canvas_mask = \
                (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < self.camera_info['width']) & \
                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < self.camera_info['height']) & \
                (points_2d[:, 2] > 0.0)
            points_2d = points_2d[points_in_canvas_mask]

            # Extract the screen coords (uv) as integers.
            u_coord = points_2d[:, 0].astype(np.int)
            v_coord = points_2d[:, 1].astype(np.int)

            projected_waypoints[step_idx] = {'u': u_coord,
                                             'v': v_coord}

        return projected_waypoints

    def computeCameraIntrinsics(self):
        focal = self.camera_info['width'] / \
            (2.0 * np.tan(self.camera_info['fov'] * np.pi / 360.0))

        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = self.camera_info['width'] / 2.0
        K[1, 2] = self.camera_info['height'] / 2.0

        return K

    def addAnnotatedImage(self):
        image_display = pygame.display.get_surface()
        # necessary to unlock pygame surface.
        image_display_copy = pygame.Surface.copy(image_display)
        del image_display
        rgb_image = cv2.cvtColor(pygame.surfarray.pixels3d(
            image_display_copy), cv2.COLOR_BGR2RGB)
        self.annotated_images.append(rgb_image.swapaxes(0, 1))

    def saveVideoFromImages(self):
        save_video_from_images(filename=self.video_name,
                               images=self.annotated_images, fps=self.fps)
        self.annotated_images = []

    def tick(self):
        done = False

        if self.step_idx < self.step_max:
            self.step_idx += 1
        else:
            self.step_idx = self.step_min
            if self.save:
                done = True
                self.saveVideoFromImages()

        self.camera_manager.tick(step_idx=self.step_idx)
        self.hud.tick(step_idx=self.step_idx)

        percentage_episode = (self.step_idx - self.step_min) / \
            (self.step_max - self.step_min)
        print(f"Percentage Episode: {percentage_episode:.2f}", end="\r")

        return done

    def render(self, display):
        self.camera_manager.render(display=display)
        self.hud.render(display=display)

        if self.save:
            self.addAnnotatedImage()
