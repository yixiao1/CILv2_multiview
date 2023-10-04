from importlib import import_module
import json
import xml.etree.ElementTree as ET
import carla
import numpy as np

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def load_obs_configs(agent_configs_dict):
    obs_configs = {}
    for actor_id, cfg in agent_configs_dict.items():
        obs_configs[actor_id] = json.load(open(cfg['path_to_conf_file'], 'r'))['obs_configs']
    return obs_configs


def init_agents(agent_configs_dict, **kwargs):
    agents_dict = {}
    for actor_id, cfg in agent_configs_dict.items():
        AgentClass = load_entry_point(cfg['entry_point'])
        agents_dict[actor_id] = AgentClass(cfg['path_to_conf_file'], **kwargs)
    return agents_dict


def parse_routes_file(routes_xml_filename):
    route_descriptions_dict = {}
    tree = ET.parse(routes_xml_filename)

    for route in tree.iter("route"):

        route_id = int(route.attrib['id'])

        route_descriptions_dict[route_id] = {}

        for actor_type in ['ego_vehicle', 'scenario_actor']:
            route_descriptions_dict[route_id][actor_type+'s'] = {}
            for actor in route.iter(actor_type):
                actor_id = actor.attrib['id']

                waypoint_list = []  # the list of waypoints that can be found on this route for this actor
                for waypoint in actor.iter('waypoint'):
                    location = carla.Location(
                        x=float(waypoint.attrib['x']),
                        y=float(waypoint.attrib['y']),
                        z=float(waypoint.attrib['z']))
                    rotation = carla.Rotation(
                        roll=float(waypoint.attrib['roll']),
                        pitch=float(waypoint.attrib['pitch']),
                        yaw=float(waypoint.attrib['yaw']))
                    waypoint_list.append(carla.Transform(location, rotation))

                route_descriptions_dict[route_id][actor_type+'s'][actor_id] = waypoint_list

    return route_descriptions_dict


def get_single_route(routes_xml_filename, route_id):
    tree = ET.parse(routes_xml_filename)
    route = tree.find(f'.//route[@id="{route_id}"]')

    route_dict = {}
    for actor_type in ['ego_vehicle', 'scenario_actor']:
        route_dict[actor_type+'s'] = {}
        for actor in route.iter(actor_type):
            actor_id = actor.attrib['id']

            waypoint_list = []  # the list of waypoints that can be found on this route for this actor
            for waypoint in actor.iter('waypoint'):
                location = carla.Location(
                    x=float(waypoint.attrib['x']),
                    y=float(waypoint.attrib['y']),
                    z=float(waypoint.attrib['z']))
                rotation = carla.Rotation(
                    roll=float(waypoint.attrib['roll']),
                    pitch=float(waypoint.attrib['pitch']),
                    yaw=float(waypoint.attrib['yaw']))
                waypoint_list.append(carla.Transform(location, rotation))

            route_dict[actor_type+'s'][actor_id] = waypoint_list
    return route_dict


def to_camel_case(snake_str, init_capital=False):
    # agent_class_str = to_camel_case(agent_module_str.split('.')[-1], init_capital=True)
    components = snake_str.split('_')
    if init_capital:
        init_letter = components[0].title()
    else:
        init_letter = components[0]
    return init_letter + ''.join(x.title() for x in components[1:])


def get_free_tcp_port():
    s = socket.socket()
    s.bind(("", 0))  # Request the sys to provide a free port dynamically
    server_port = s.getsockname()[1]
    s.close()
    # 2000 works fine for now
    server_port = 2000
    return server_port
