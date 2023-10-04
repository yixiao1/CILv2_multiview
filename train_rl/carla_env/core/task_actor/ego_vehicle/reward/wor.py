import numpy as np
import carla

import carla_env.utils.transforms as trans_utils
from carla_env.core.obs_manager.object_finder.vehicle import ObsManager as OmVehicle
from carla_env.core.obs_manager.object_finder.pedestrian import ObsManager as OmPedestrian

from carla_env.utils.traffic_light import TrafficLightHandler
from carla_env.utils.hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker

from utilities.networks import encode_traffic_light_state

class ValeoAction(object):

    def __init__(self, ego_vehicle, maximum_speed=6.0):
        self._ego_vehicle = ego_vehicle

        self.om_vehicle = OmVehicle({'max_detection_number': 10, 'distance_threshold': 15})
        self.om_pedestrian = OmPedestrian({'max_detection_number': 10, 'distance_threshold': 15})
        self.om_vehicle.attach_ego_vehicle(self._ego_vehicle)
        self.om_pedestrian.attach_ego_vehicle(self._ego_vehicle)

        self.maximum_speed = maximum_speed
        self._last_steer = 0.0 
        self._tl_offset = -0.8 * self._ego_vehicle.vehicle.bounding_box.extent.x

        self.brake_reward = True
        
    def get(self, terminal_reward):
        ev_transform = self._ego_vehicle.vehicle.get_transform()
        ev_control = self._ego_vehicle.vehicle.get_control()
        ev_vel = self._ego_vehicle.vehicle.get_velocity()
        ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))

        # action
        if abs(ev_control.steer - self._last_steer) > 0.01:
            r_action = -0.1
        else:
            r_action = 0.0
        self._last_steer = ev_control.steer

        # desired_speed
        obs_vehicle = self.om_vehicle.get_observation()
        obs_pedestrian = self.om_pedestrian.get_observation()

        # all locations in ego_vehicle coordinate
        hazard_vehicle_loc = lbc_hazard_vehicle(obs_vehicle, proximity_threshold=9.5)
        hazard_ped_loc = lbc_hazard_walker(obs_pedestrian, proximity_threshold=9.5)
        light_state, light_loc, _ = TrafficLightHandler.get_light_state(self._ego_vehicle.vehicle,
                                                                        offset=self._tl_offset, dist_threshold=18.0)

        desired_spd_veh = desired_spd_ped = desired_spd_rl = desired_spd_stop = self.maximum_speed

        dist_veh = None
        if hazard_vehicle_loc is not None:
            dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2])-8.0)
            desired_spd_veh = self.maximum_speed * np.clip(dist_veh, 0.0, 5.0)/5.0

        dist_ped = None
        if hazard_ped_loc is not None:
            dist_ped = max(0.0, np.linalg.norm(hazard_ped_loc[0:2])-6.0)
            desired_spd_ped = self.maximum_speed * np.clip(dist_ped, 0.0, 5.0)/5.0

        dist_rl = -1
        if (light_state == carla.TrafficLightState.Red or light_state == carla.TrafficLightState.Yellow):
            dist_rl = max(0.0, np.linalg.norm(light_loc[0:2])-5.0)
            desired_spd_rl = self.maximum_speed * np.clip(dist_rl, 0.0, 5.0)/5.0
        
        # if light_state == carla.TrafficLightState.Green:
        #     dist_rl = max(0.0, np.linalg.norm(light_loc[0:2])-5.0)

        # stop sign
        stop_sign = self._ego_vehicle.criteria_stop._target_stop_sign
        stop_loc = None
        if (stop_sign is not None) and (not self._ego_vehicle.criteria_stop._stop_completed):
            trans = stop_sign.get_transform()
            tv_loc = stop_sign.trigger_volume.location
            loc_in_world = trans.transform(tv_loc)
            loc_in_ev = trans_utils.loc_global_to_ref(loc_in_world, ev_transform)
            stop_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
            dist_stop = max(0.0, np.linalg.norm(stop_loc[0:2])-5.0)
            desired_spd_stop = self.maximum_speed * np.clip(dist_stop, 0.0, 5.0)/5.0

        desired_speed = min(self.maximum_speed, desired_spd_veh, desired_spd_ped, desired_spd_rl, desired_spd_stop)

        # # r_speed
        # if ev_speed > self.maximum_speed:
        #     # r_speed = 0.0
        #     r_speed = 1.0 - np.abs(ev_speed-desired_speed) / self.maximum_speed
        # else:
        #     r_speed = 1.0 - np.abs(ev_speed-desired_speed) / self.maximum_speed
            
        
        # r_position
        wp_transform = self._ego_vehicle.get_route_transform()

        d_vec = ev_transform.location - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)

        lateral_distance = np.abs(np.dot(np_wp_unit_right, np_d_vec))
        # r_position = -1.0 * (lateral_distance / 2.0)
        r_position = 1 - (lateral_distance / 2.0)

        # r_rotation
        angle_difference = np.deg2rad(np.abs(trans_utils.cast_angle(
            ev_transform.rotation.yaw - wp_transform.rotation.yaw)))
        # r_rotation = -1.0 * (angle_difference / np.pi)
        # r_rotation = -1.0 * angle_difference
        r_rotation = 1 - (angle_difference)
        
        
        # zero speed zone.
        r_brake = 0.0
        if desired_speed < 0.01:
            if ev_speed < 0.01 and self.brake_reward == True:
                r_brake = 5.0
                self.brake_reward = False
            
            r_speed = 1.0 - np.abs(ev_speed-desired_speed) / self.maximum_speed
            r_position = 0.0
            r_rotation = 0.0
        else:
            r_speed = 1.0 - np.abs(ev_speed-desired_speed) / self.maximum_speed
                
                
        if ev_speed > 0.1:
            self.brake_reward = True
        


        
        reward = r_speed + r_position + r_rotation + terminal_reward + r_action + r_brake
   
        if light_state == carla.TrafficLightState.Red or light_state == carla.TrafficLightState.Yellow:
            light_state_encoded = True
        else:
            light_state_encoded = False
        
        
        debug_texts = {'r_speed':f'{r_speed:.2f}',
                       'r_position':f'{r_position:.2f}',
                       'r_rotation':f'{r_rotation:.2f}',
                       'r_action':f'{r_action:.2f}',
                       'r_terminal':f'{terminal_reward:.2f}',
                       'desired_speed':f'{desired_speed:.2f}',
                       'light_state':f'{light_state_encoded}'}
        
        reward_debug = {
            'debug_texts': debug_texts
        }
        return reward, reward_debug
