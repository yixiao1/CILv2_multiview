#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from driving.autoagents.agent_wrapper import AgentWrapper, AgentError
from driving.envs.sensor_interface import SensorReceivedNoData
from driving.utils.result_writer import ResultOutputProvider


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """


    def __init__(self, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        self.veh_junction_stopping_time = {}

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)
        self.route = scenario.route

        self._agent.setup_sensors(self.ego_vehicles[0], route=self.route)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        world = CarlaDataProvider.get_world()
        self.vehicles_list = world.get_actors().filter('*vehicle*')
        self.walkers_list = world.get_actors().filter('*walker*')
        self.invalid_list = [self.ego_vehicles[0].id]
        self.force_stop = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)

        print(' Vehicles number: ', len(self.vehicles_list))
        print(' Walkers number: ', len(self.walkers_list))

        while self._running:
            timestamp = None
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                ego_action = self._agent(timestamp)

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            # replan the route if the agent doesn't follow the route command, and we set the timeout longer
            for criteria_node in self.scenario.get_criteria():
                if criteria_node.name == 'InRouteTest':
                    if criteria_node.test_status == "FAILURE":
                        print('reset route')
                        new_route = self._agent._agent.reset_global_plan()
                        new_route = convert_transform_to_location(new_route)
                        for node in self.scenario.get_criteria():
                            if hasattr(node, '_route'):
                                node.reset(new_route)
                                self.scenario.timeout_node.reset()

            self.next_junction_id = CarlaDataProvider.get_next_junction_id(self.ego_vehicles[0])
            for veh in self.vehicles_list:
               if veh.id not in self.invalid_list:
                   veh_wp = CarlaDataProvider._map.get_waypoint(CarlaDataProvider._carla_actor_pool[veh.id].get_location(), lane_type=carla.LaneType.Driving)
                   if veh_wp.is_junction and veh_wp.get_junction().id == self.next_junction_id:
                       # vehicles stopping at junction case
                       if CarlaDataProvider.get_velocity(veh) < 0.1:
                           if str(veh.id) not in self.veh_junction_stopping_time.keys():
                               self.veh_junction_stopping_time.update({str(veh.id): 0})
                           else:
                               # add FPS
                               self.veh_junction_stopping_time[str(veh.id)] += 0.2
                               #print(str(veh.id), ' stops at intersection for ', self.veh_junction_stopping_time[str(veh.id)], 'sec')
                               if self.veh_junction_stopping_time[str(veh.id)] > 60.0:
                                   print("")
                                   print('  Clear intersection deadlock vehicle:', str(veh.id))
                                   CarlaDataProvider.remove_actor_by_id(veh.id)
                                   self.invalid_list.append(veh.id)
                       else:
                           if str(veh.id) in self.veh_junction_stopping_time.keys():
                               self.veh_junction_stopping_time.pop(str(veh.id), None)
                   else:
                       if str(veh.id) in self.veh_junction_stopping_time.keys():
                           self.veh_junction_stopping_time.pop(str(veh.id), None)
               else:
                   pass

            
            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'
        self.global_result = 'SUCCESS'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m'+'FAILURE'+'\033[0m'
                self.global_result = 'FAILURE'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m'+'FAILURE'+'\033[0m'
            self.global_result = 'FAILURE'

        ResultOutputProvider(self, global_result)
