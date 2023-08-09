#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import carla
import signal
import time
import json
import random
import numpy as np
import subprocess

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from driving.scenarios.scenario_manager import ScenarioManager
from driving.scenarios.route_scenario import RouteScenario
from driving.envs.sensor_interface import SensorConfigurationInvalid
from driving.autoagents.agent_wrapper import  AgentWrapper, AgentError
from driving.utils.statistics_manager import StatisticsManager
from driving.utils.route_indexer import RouteIndexer
from driving.utils.server_manager import ServerManagerDocker, find_free_port

sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.camera.depth':      'carla_depth',
    'sensor.camera.semantic_segmentation':  'carla_ss',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer',
    'sensor.can_bus':           'carla_canbus'
}

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

class Evaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds

    def __init__(self, args, statistics_manager, ServerDocker=None):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.ServerDocker = ServerDocker
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        self.frame_rate = float(args.fps)  # in Hz

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        # If using docker, the free port will be allocated automatically
        if self.ServerDocker is not None:
            args.port = find_free_port()
            self.ServerDocker.reset(args.host, args.port)
            args.trafficManagerPort=find_free_port()

        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world
        if hasattr(self, 'client') and self.client:
            del self.client
        if hasattr(self, 'traffic_manager') and self.traffic_manager:
            del self.traffic_manager

    def _cleanup(self):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            self.statistics_manager.scenario = None

    def _load_and_wait_for_world(self, args, config):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        town = config.town
        print('  port:', args.port)
        self.world = self.client.load_world(town)
        self.world.set_weather(config.weather)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.set_pedestrians_seed(int(args.PedestriansSeed))
        print('Set seed for pedestrians:', str(int(args.PedestriansSeed)))

        self.world.reset_all_traffic_lights()
        if hasattr(config, 'scenarios'):
            if 'background_activity' in list(config.scenarios.keys()):
                if 'cross_factor' in list(config.scenarios['background_activity'].keys()):
                    self.world.set_pedestrians_cross_factor(config.scenarios['background_activity']['cross_factor'])
                else:
                    self.world.set_pedestrians_cross_factor(1.0)

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(5.0)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))
        print('Set seed for traffic manager:', str(int(args.trafficManagerSeed)))
        seed_everything(seed=int(args.trafficManagerSeed))
        print('Set seed for numpy:', str(int(args.trafficManagerSeed)))
        CarlaDataProvider.set_random_state_seed(int(args.trafficManagerSeed))
        print('Set seed for random state:', str(int(args.trafficManagerSeed)))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name.split('/')[-1] != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        current_stats_record = self.statistics_manager.compute_route_statistics(
            config,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message
        )

        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()

            # for data collection
            if args.data_collection:
                vision_save_path = os.path.join(os.environ['DATASET_PATH'], config.package_name, config.name)
                self.agent_instance = getattr(self.module_agent, agent_class_name) \
                    (args.agent_config, save_driving_vision=vision_save_path)

            # for saving benchmark driving episodes
            else:
                vision_save_path = os.path.join(os.environ['SENSOR_SAVE_PATH'], config.package_name,
                                                args.checkpoint.split('/')[-1].split('.')[-2], config.name,
                                                str(config.repetition_index)) if args.save_driving_vision else False
                self.agent_instance = getattr(self.module_agent, agent_class_name) \
                    (args.agent_config, save_driving_vision=vision_save_path,
                     save_driving_measurement=args.save_driving_measurement)

            config.agent = self.agent_instance

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the execution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> set the execution to rejected and stop
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"
            entry_status = "Rejected"
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args, config)
            self.agent_instance.set_world(self.world)
            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self.statistics_manager.set_scenario(scenario.scenario)

            self.agent_instance.set_ego_vehicle(scenario._ego_vehicle)

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)

        except Exception as e:
            # The scenario is wrong -> set the execution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            self.manager.run_scenario()

        except AgentError as e:
            # The agent has failed -> set the execution to crashed and stop
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent crashed"
            entry_status = "Crashed"
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

    def run(self, args):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)

        if args.resume:
            route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        while route_indexer.peek():
            # setup
            config = route_indexer.next()
            # run
            self._load_and_run_scenario(args, config)

            route_indexer.save_state(args.checkpoint)

        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint)
        if self.ServerDocker is not None:
            self.ServerDocker.stop()

def main():
    description = "CARLA Evaluation: evaluate your Agent in CARLA simulator\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--PedestriansSeed', default='0',
                        help='Seed used by the Pedestrians setting (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="180.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes', help='Name of the route to be executed. Point to the route_xml_file to be executed.', required=True)
    parser.add_argument('--scenarios', help='Name of the scenario annotation file to be mixed with the route.', required=True)
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json', help="Path to checkpoint used for saving statistics and resuming")

    parser.add_argument('--docker', type=str, default='', help='Use docker to run CARLA off-screen, this is typically for running CARLA on server')
    parser.add_argument('--gpus', nargs='+', dest='gpus', type=str, default=0, help='The GPUs used for running the agent model. The firtst one will be used for running docker')

    parser.add_argument('--save-driving-vision', action="store_true", help=' to save the driving visualization')
    parser.add_argument('--save-driving-measurement', action="store_true", help=' to save the driving measurements')
    parser.add_argument('--data-collection', action="store_true", help=' to collect dataset')
    parser.add_argument('--fps', default=10, help='The frame rate of CARLA world')

    arguments = parser.parse_args()

    gpus=[]
    if arguments.gpus:
        # Check if the vector of GPUs passed are valid.
        for gpu in arguments.gpus[0].split(','):
            try:
                int(gpu)
                gpus.append(int(gpu))
            except ValueError:  # Reraise a meaningful error.
                raise ValueError("GPU is not a valid int number")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(arguments.gpus)  # This must to be ahead of the whole excution
        arguments.gpus = gpus
    else:
        raise ValueError('You need to define the ids of GPU you want to use by adding: --gpus')

    if arguments.save_driving_vision or arguments.save_driving_measurement:
        if not os.environ['SENSOR_SAVE_PATH']:
            raise RuntimeError('environemnt argument SENSOR_SAVE_PATH need to be setup for saving data')


    if not os.path.exists(os.path.join(arguments.checkpoint, arguments.scenarios.split('/')[-1].split('.')[-2])):
        os.makedirs(os.path.join(arguments.checkpoint, arguments.scenarios.split('/')[-1].split('.')[-2]))

    f = open(arguments.agent_config, 'r')
    _json = json.loads(f.read())
    arguments.checkpoint = '/'.join([arguments.checkpoint, arguments.scenarios.split('/')[-1].split('.')[-2], '_'.join([arguments.agent_config.split('/')[-3],
                                                                                                                        arguments.agent_config.split('/')[-2],
                                                                                                                        str(_json['checkpoint']), 'Seed'+str(arguments.PedestriansSeed),
                                                                                                                        arguments.fps+'FPS.json'])])

    statistics_manager = StatisticsManager()

    ServerDocker=None
    if arguments.docker:
        docker_params={'docker_name':arguments.docker, 'gpu':arguments.gpus[0], 'quality_level':'Epic'}
        ServerDocker = ServerManagerDocker(docker_params)

    try:
        evaluator = Evaluator(arguments, statistics_manager, ServerDocker)
        evaluator.run(arguments)
    except Exception as e:
        traceback.print_exc()
        sys.exit(-1)
    finally:
        del evaluator


if __name__ == '__main__':
    main()
    print('Finished all episode. Goodbye!')
