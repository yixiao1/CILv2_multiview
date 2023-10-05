import carla
import numpy as np

from utilities.conversions import convert_speed_01, convert_01, convert_11, step_function

def carla_control(action):
    control = carla.VehicleControl(
        throttle= float(action[0]), steer=float(action[1]), brake=float(action[2]))
    return control

def parse_action3(action, brake_threshold):
    throttle =  convert_01(action[0])
    steer = action[1]
    brake = step_function(action[2], threshold=brake_threshold)
    
    return [throttle, steer, brake]
    
def heuristic_tl_dist_danger(action, tl, dist_danger):
    velocity_01 = convert_speed_01(action[0])
    
    if (not tl) and (not dist_danger) and velocity_01 < 0.05:
        # increase velocity.
        velocity_01 = 0.1
        heuristic = 1.0
    else:
        heuristic = 0.0
        
    velocity_11 = convert_11(velocity_01)
    action[0] = velocity_11
    return action, heuristic
   
def heuristic_tl_dist_danger2(action, tl, dist_danger):
    velocity_01 = convert_speed_01(action[0])
    
    heuristic = 0.0
    
    if (not tl) and (not dist_danger) and velocity_01 < 0.05:
        # increase velocity.
        velocity_01 = 0.1
        heuristic = 1.0
    
    if tl or dist_danger:
        velocity_01 = 0.0
        heuristic = 1.0
    
    velocity_11 = convert_11(velocity_01)
    action[0] = velocity_11
    return action, heuristic 
    

 
class PID():
    def __init__(self, kp, ki, kd, dt, maximum_speed):
        self.kp = kp 
        self.ki = ki 
        self.kd = kd 
        self.dt = dt
        self.e = 0
        self.e_prev = 0
        self.maximum_speed = maximum_speed
    
    def get(self, action, velocity):
        # desired_velocity = action[0] * self.maximum_speed
        desired_velocity = convert_speed_01(action[0])
        velocity = velocity / self.maximum_speed
        steer = action[1]
    
        error = desired_velocity - velocity
        
        proportional = self.kp * error 
        self.e = self.e + error * self.dt
        integral = self.ki * self.e 
        derivative = self.kd * ((error - self.e_prev) / self.dt)
        
        u = proportional + integral + derivative
        
        if u>= 0:
            throttle = max(min(u, 1), 0)
            brake = 0
        elif u<0:
            throttle = 0
            brake = -max(min(u, 0), -1)
        
        self.e_prev = self.e
        
        # safety brake.
        if desired_velocity<0.01:
            throttle=0
            brake=1
            
        return [throttle, steer, brake]
    
    def reset(self):
        self.e = 0
        self.e_prev = 0
        
def acc_controller(action):
    """
    converting acceleration into throttle and brake.
    """
    acc = action[0]
    steer = action[1]
    
    if acc>= 0.0:
        throttle = acc 
        brake = 0.0
    else:
        throttle = 0
        brake = np.abs(acc)
    
    throttle = np.clip(throttle, 0, 1)
    brake = np.clip(brake, 0, 1)
    steer = np.clip(steer, -1, 1)
    
    return [throttle, steer, brake]
    
    
