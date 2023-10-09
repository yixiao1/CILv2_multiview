import carla
import pygame
import os

from train_rl.utilities.conversions import command_to_text

class HUD(object):
    def __init__(self, dim, log, env_config, episode, fps, rlad_video=False):
        self.dim = dim
        self.log = log
        self.env_config = env_config['env_configs']
        self.episode = episode
        self.fps = fps
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self.rlad_video = rlad_video

    def tick(self, step_idx):
        step_data = self.log['steps'][str(step_idx)]
        velocity_kmh = float(step_data['observation']['SPEED']['forward_speed'][0])*3.6
        control = carla.VehicleControl(
        throttle=step_data['observation']['control']['throttle'][0], steer=step_data['observation']['control']['steer'][0], brake=step_data['observation']['control']['brake'][0])        

        reward = step_data['reward']
        reward_debug = step_data['info']['reward_debug']['debug_texts']
        r_speed = reward_debug['r_speed']
        r_position = reward_debug['r_position']
        r_rotation = reward_debug['r_rotation']
        r_action = reward_debug['r_action']
        r_terminal = reward_debug['r_terminal']
        desired_speed = round(float(reward_debug['desired_speed'])*3.6,2)
        
        command = command_to_text(step_data['observation']['GPS']['command'][0])


        self._info_text = [
            'Video:  % 6.0f FPS' % self.fps,
            '',
            'Vehicle: % 6s' % 'lincoln.mkz_2017', #TODO: We should save this info automatically.
            'Map:     % 6s' % self.env_config['carla_map_val']]
        
        if not self.rlad_video:
            self._info_text += [f'Episode: {self.episode:6d}',
                                '',
                                '']
        else:
            self._info_text += ['']
                                
        
        self._info_text += ['Speed:   % 8.0f km/h' % (velocity_kmh),
            f'command: {command}']
            # f'light_distance: {round(light_distance,2)}']
        
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0)]
        
        if not self.rlad_video:
            
            self._info_text += [
                f'Reward: {reward:.2f}',
                f'r_speed: {r_speed}',
                f'r_position: {r_position}',
                f'r_rotation: {r_rotation}',
                f'r_action: {r_action}',
                f'r_terminal: {r_terminal}',
                f'desired speed: {desired_speed}']
        
    def render(self, display):
        info_surface = pygame.Surface((185, self.dim[1]))
        info_surface.set_alpha(100)
        display.blit(info_surface, (0, 0))
        v_offset = 4
        bar_h_offset = 100
        bar_width = 60
        for item in self._info_text:
            if v_offset + 18 > self.dim[1]:
                break
            if isinstance(item, list):
                if len(item) > 1:
                    points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                    pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                item = None
                v_offset += 18
            elif isinstance(item, tuple):
                if isinstance(item[1], bool):
                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    f = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                item = item[0]
            if item:  # At this point has to be a str.
                surface = self._font_mono.render(item, True, (255, 255, 255))
                display.blit(surface, (8, v_offset))
            v_offset += 18
        


