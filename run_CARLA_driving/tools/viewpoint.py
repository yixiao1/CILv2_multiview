#!/usr/bin/env python

import carla
import sys

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) # seconds
    world = client.get_world()

    view_type = 'top' #rear, top
    actor_lists = []
    
    debug_telemetry_on = False
    print ("Searching Actor")
    actor=None

    while len(actor_lists) == 0:
        print ("*", end=" ")
        actor_list = world.get_blueprint_library()
        lists = actor_list.filter('vehicle.lincoln.mkz2017')
        print(lists)

        spawn_points = list(world.get_map().get_spawn_points())
        id = 0
        while actor==None:
            actor = world.try_spawn_actor(lists[0], spawn_points[id])
            id+=1
        print(actor)
        actor_lists.append(actor)

    distance = 0.5
    map = world.get_map()
    waypoints = map.generate_waypoints(distance)
    print ("waypoints", waypoints[0])
    

    for w in waypoints:
        world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                           color=carla.Color(r=255, g=0, b=0), 
                                           persistent_lines=False)


    # weather = carla.WeatherParameters(
    # cloudiness=0.0,
    # precipitation=30.0,
    # sun_altitude_angle=0.0)

    # world.set_weather(weather)

    print(world.get_weather())

    print ("\n Actor obtained... transforming")
    vehicle = actor_lists[0]
    print_vehicle_parameters(vehicle)
    if debug_telemetry_on:
        vehicle.show_debug_telemetry(enabled=True)

    print_vehicle_parameters(vehicle)
    while (True and len(actor_lists) > 0):
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        # print ("vehicle_transform", transform)
        if view_type == 'top':
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
            carla.Rotation(pitch=-90)))

        if view_type == 'rear':
            spectator.set_transform(carla.Transform(transform.location + carla.Location(x = 5, y= 4, z=2.5),
            carla.Rotation(yaw=45+180)))

        world.debug.draw_string(transform.location, 'O', draw_shadow=False,
                                           color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                                           persistent_lines=True)

def print_vehicle_parameters(vehicle):
    vehicle_physics = vehicle.get_physics_control()

    print ("\n torque_curve", [[val.x, val.y] for val in vehicle_physics.torque_curve], \
    "\n max_rpm", vehicle_physics.max_rpm, \
    "\n moi", vehicle_physics.moi, \
    "\n damping_rate_full_throttle", vehicle_physics.damping_rate_full_throttle, \
    "\n damping_rate_zero_throttle_clutch_engaged", vehicle_physics.damping_rate_zero_throttle_clutch_engaged, \
    "\n damping_rate_zero_throttle_clutch_disengaged", vehicle_physics.damping_rate_zero_throttle_clutch_disengaged, \
    "\n use_gear_autobox", vehicle_physics.use_gear_autobox, \
    "\n gear_switch_time", vehicle_physics.gear_switch_time, \
    "\n clutch_strength", vehicle_physics.clutch_strength, \
    "\n final_ratio", vehicle_physics.final_ratio)
    
    for i, gear in enumerate(vehicle_physics.forward_gears): 
        print ("gear number",i, '\n', "gear.ratio", gear.ratio, 'gear.down_ratio', gear.down_ratio, "gear.up_ratio", gear.up_ratio)

    print ("\n mass", vehicle_physics.mass, \
    "\n drag_coefficient", vehicle_physics.drag_coefficient, \
    "\n center_of_mass", vehicle_physics.center_of_mass)
    print("\n steering_curve", [[val.x, val.y] for val in vehicle_physics.steering_curve], \
    #"\n use_sweep_wheel_collision", vehicle_physics.use_sweep_wheel_collision)
    "\n use_sweep_wheel_collision", 'NoImplement')
    
    for wheel, ids in zip(vehicle_physics.wheels, ['front left', 'front right', 'back left', 'back right']):
        print ('\n \n', ids, \
        '\n tire_friction', wheel.tire_friction, \
        '\n damping_rate', wheel.damping_rate, \
        '\n max_steer_angle', wheel.max_steer_angle, \
        '\n radius', wheel.radius, \
        '\n max_brake_torque', wheel.max_brake_torque, \
        '\n max_handbrake_torque', wheel.max_handbrake_torque, \
        '\n position', wheel.position, \
        #'\n long_stiff_value', wheel.long_stiff_value, \
        '\n long_stiff_value', 'NoImplement', \
        #'\n lat_stiff_max_load', wheel.lat_stiff_max_load, \
        '\n lat_stiff_max_load', 'NoImplement',\
        #'\n lat_stiff_value', wheel.lat_stiff_value)
        '\n lat_stiff_value', 'NoImplement')

    print ("\n ===================================================")


if __name__ == '__main__':
    main()

