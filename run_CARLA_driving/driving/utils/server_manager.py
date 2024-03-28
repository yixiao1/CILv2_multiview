from enum import Enum
import fcntl
import logging
import os
import psutil
import random
import string
import subprocess
import time
import carla

import socket
from contextlib import closing


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]



class ServerManager(object):
    def __init__(self, opt_dict):
        log_level = logging.INFO
        logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

        self._proc = None
        self._outs = None
        self._errs = None

    def reset(self, host="127.0.0.1", port=2000):
        raise NotImplementedError("This function is to be implemented")

    def wait_until_ready(self, wait=10.0):
        time.sleep(wait)


class ServerManagerDocker(ServerManager):

    def __init__(self, opt_dict):
        super(ServerManagerDocker, self).__init__(opt_dict)
        self._docker_name = opt_dict['docker_name']
        self._gpu = opt_dict['gpu']
        self._quality_level = opt_dict['quality_level']
        self._docker_id = ''

    def reset(self, host="127.0.0.1", port=2000):
        # first we check if there is need to clean up
        if self._proc is not None:
            logging.info('Stopping previous server [PID=%s]', self._proc.pid)
            self.stop()
            self._proc.kill()
            self._outs, self._errs = self._proc.communicate()

        self._docker_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(64))
        # temporary config file
        # TODO quality level to be set here
        my_env = os.environ.copy()
        my_env["NV_GPU"] = str(self._gpu)
        if not os.path.exists(os.path.join(os.getcwd(),'CARLA_recorder')):
            os.makedirs(os.path.join(os.getcwd(),'CARLA_recorder'))
        # for running docker for CARLA version in prior to 0.9.9
        if self._docker_name in ['carla_099_t10']:
            self._proc = subprocess.Popen(['docker', 'run', '--name', self._docker_id, '--rm', '-d', '-p',
                                           f'{port}-{port+2}:{port}-{port+2}', '--runtime=nvidia', '-e',
                                           f'NVIDIA_VISIBLE_DEVICES={self._gpu}', self._docker_name, '/bin/bash',
                                           'CarlaUE4.sh', f'-quality-level={self._quality_level}', f'-carla-port={port}'],
                                          shell=False, stdout=subprocess.PIPE, env=my_env)

        # for running docker for CARLA 0.9.10. Notice that the Dockerfile needs to be modified to match the docker running
        elif self._docker_name in ['carla_0910_t10', 'carla_0910_t10:0.9.10', 'carla_0911:0.9.11']:
            self._proc = subprocess.Popen(['docker', 'run', '--name', self._docker_id, '--rm', '-d', '-p',
                                           f'{port}-{port+2}:{port}-{port+2}', '--gpus', f'device={self._gpu}', '--cpus', '5.0',
                                           '-it',
                                           '-e', f'DISPLAY=:0.{self._gpu}',
                                           self._docker_name, '/bin/bash', './CarlaUE4.sh', '-opengl',
                                           f'-quality-level={self._quality_level}', f'-carla-port={port}'],
                                          shell=False, stdout=subprocess.PIPE, env=my_env)
        # for running docker for CARLA 0.9.13
        elif self._docker_name in ['carlasim/carla:0.9.13', 'carlasim/carla_allmaps:0.9.13', 'carlasim/carla_allmaps:0.9.14',
                                   'carlasim/carla_allmaps:0.9.15']:
            self._proc = subprocess.Popen(['docker', 'run', '--name', self._docker_id, '--rm', '-d', '-p',
                                           f'{port}-{port+2}:{port}-{port+2}', '--gpus', f'device={self._gpu}', '--cpus', '5.0',
                                           '-it', self._docker_name, '/bin/bash', './CarlaUE4.sh', '-RenderOffscreen',
                                           f'-quality-level={self._quality_level}', f'-carla-port={port}'],
                                          shell=False, stdout=subprocess.PIPE, env=my_env)

        else:
            raise KeyError("You need to add your built docker name into different cases")

        (out, err) = self._proc.communicate()

        logging.debug(" Starting a docker server of id %s at port %d" % (self._docker_id, port))

        time.sleep(20)

    def stop(self):
        logging.debug("Killed a docker of id %s " % self._docker_id)
        exec_command = ['docker', 'kill', '{}'.format(self._docker_id)]
        self._proc = subprocess.Popen(exec_command)




def start_test_server(port=6666, gpu=0, docker_name='carlalatest:latest'):

    params = {'docker_name': docker_name,
              'gpu': gpu
              }

    docker_server = ServerManagerDocker(params)
    docker_server.reset(port=port)

    return docker_server



def check_test_server(port):

    # Check if a server is open at some port

    try:
        print ( " TRYING TO CONNECT ", port)
        client = carla.Client(host='localhost', port=port)
        print ( "GETTING VERSION ")
        client.get_server_version()
        del client
        return True
    except:
        return False