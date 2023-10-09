import cv2
import pygame

class CameraManager(object):
    def __init__(self, monitor_path, episode, eval_idx, dim, projected_waypoints, step_min, step_max):
        self.monitor_path = monitor_path
        self.episode = episode
        self.eval_idx = eval_idx
        self.dim = dim
        self.projected_waypoints = projected_waypoints
        self.step_min, self.step_max = step_min, step_max

        # surface to use in the display.
        self.surface = None
        self.image = None

        self.buffer_images = self.getImages()

    def getImages(self):
        buffer_images = {}
        cap = cv2.VideoCapture(
            f"{self.monitor_path}/{self.eval_idx}_{int(self.episode):04d}.mp4")

        step_idx = self.step_min
        while (cap.isOpened()):
            success, frame = cap.read()
            if success:
                # read frame.
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # draw points.
                u = self.projected_waypoints[step_idx]['u']
                v = self.projected_waypoints[step_idx]['v']
                for i in range(len(u)):
                    frame = cv2.circle(
                        frame, (u[i], v[i]), 2, (0, 255, 0), -1)

                # # resize to self.dim.
                # frame = cv2.resize(
                #     frame, self.dim, interpolation=cv2.INTER_AREA)

                buffer_images[step_idx] = frame
                step_idx += 1
            else:
                break

        return buffer_images

    def tick(self, step_idx):
        self.image = self.buffer_images[step_idx]
    
    def render(self, display):
        self.surface = pygame.surfarray.make_surface(self.image.swapaxes(0, 1))
        display.blit(self.surface, (0, 0))
