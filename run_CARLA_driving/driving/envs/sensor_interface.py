import copy
import logging
import numpy as np
import os
import time
from threading import Thread

from queue import Queue
from queue import Empty

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0, other_actors_dict=None, route=None):
        self._vehicle = vehicle
        self._route_to_finish = route
        self._other_actors_dict = other_actors_dict
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()

    def __call__(self):
        pass

    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time + 0.0025 > (1 / self._reading_frequency) or first_time:
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class SpeedometerReader(BaseReader):
    """
    Sensor to measure the speed of the vehicle.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        return {'speed': self._get_forward_speed(transform=transform, velocity=velocity)}


class OpenDriveMapReader(BaseReader):
    def __call__(self):
        return {'opendrive': CarlaDataProvider.get_map().to_opendrive()}


class CanbusReader(BaseReader):
    """
    Sensor to measure the location of the vehicle and the navigation waypoint.
    """

    def _truncate_global_route_till_local_target(self, ego_location, windows_size=5):
        ev_location = ego_location
        closest_idx = 0

        for i in range(len(self._route_to_finish) - 1):
            if i > windows_size:
                break

            loc0 = self._route_to_finish[i][0].location
            loc1 = self._route_to_finish[i + 1][0].location

            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                closest_idx = i + 1

        latest_wp_transform, _ = self._route_to_finish[closest_idx]
        self._route_to_finish = self._route_to_finish[closest_idx:]
        return latest_wp_transform

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """
        transform = self._vehicle.get_transform()
        latest_navigate_wp_transform = self._truncate_global_route_till_local_target(transform.location)

        can_bus_dict = {
            'ego_location': [transform.location.x, transform.location.y, transform.location.z],
            'ego_rotation': [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
            'navigate_wp_location': [latest_navigate_wp_transform.location.x, latest_navigate_wp_transform.location.y,
                                     latest_navigate_wp_transform.location.z],
            'navigate_wp_rotation': [latest_navigate_wp_transform.rotation.pitch,
                                     latest_navigate_wp_transform.rotation.yaw,
                                     latest_navigate_wp_transform.rotation.roll]
        }

        """

        scenario_canbus_dict = {}
        for scenario_name, actors in self._other_actors_dict.items():
            scenario_canbus_dict[scenario_name]={}
            for actor_id in actors.keys():
                if actors[actor_id].is_alive:
                    actor_waypoint = CarlaDataProvider.get_map().get_waypoint(actors[actor_id].get_location())
                    actor_transform = actors[actor_id].get_transform()
                    actor_location = [actor_transform.location.x, actor_transform.location.y, actor_transform.location.z]
                    actor_rotation = [actor_transform.rotation.pitch, actor_transform.rotation.yaw, actor_transform.rotation.roll]
                    scenario_canbus_dict[scenario_name].update({actor_id+'_location': actor_location})
                    scenario_canbus_dict[scenario_name].update({actor_id+'_rotation': actor_rotation})
                    scenario_canbus_dict[scenario_name].update({actor_id + '_wp_location': [actor_waypoint.transform.location.x, actor_waypoint.transform.location.y,
                                                                     actor_waypoint.transform.location.z]})
                    scenario_canbus_dict[scenario_name].update({actor_id + '_wp_rotation': [actor_waypoint.transform.rotation.pitch, actor_waypoint.transform.rotation.yaw,
                                                                     actor_waypoint.transform.rotation.roll]})
        can_bus_dict.update(scenario_canbus_dict)

        """

        return can_bus_dict

class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        if 'ss' in tag:
            array = image
            #image.save_to_disk('tutorial/new_sem_output/%.6d.jpg' % image.frame, carla.ColorConverter.CityScapesPalette)
        elif 'depth' in tag:
            image.convert(carla.ColorConverter.LogarithmicDepth)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = copy.deepcopy(array)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
        else:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = copy.deepcopy(array)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 10

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None


    def register_sensor(self, tag, sensor_type, sensor):
        if tag in self._sensors_objects:
            raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor

        if sensor_type == 'sensor.opendrive_map': 
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, timestamp):
        if tag not in self._sensors_objects:
            raise SensorConfigurationInvalid("The sensor with tag [{}] has not been created!".format(tag))

        self._new_data_buffers.put((tag, timestamp, data))

    def get_data(self):
        try: 
            data_dict = {}
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):

                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    break

                sensor_data = self._new_data_buffers.get(True, self._queue_timeout)
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

                # print(len(data_dict.keys()), len(self._sensors_objects.keys()))
                # print(data_dict.keys(), self._sensors_objects.keys())

        except Empty:
            raise SensorReceivedNoData("A sensor took too long to send their data")

        return data_dict
