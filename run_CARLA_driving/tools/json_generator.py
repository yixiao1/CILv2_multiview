
import json
import os


root_route_file_position = '/home/yixiao/leaderboard/data/nocrash'

if not os.path.exists(root_route_file_position):
    os.makedirs(root_route_file_position)

# For each of the routes to be evaluated.
# Tows to be generated
town_sets = {'Town01': 'Town01_navigation.xml',
             'Town02': 'Town02_navigation.xml'}

# Weathers to be generated later
weather_sets = {'training': ["ClearNoon",
                              "WetNoon",
                              "HardRainNoon",
                               "ClearSunset"],
                'new_weather':  ["WetSunset",
                                "SoftRainSunset"],
                'single_weather': ["ClearNoon"]
                }

tasks = {'empty': { 'Town01': {},
                    'Town02': {}
                    },
         'regular': { 'Town01': {'background_activity': {"vehicle.*": 20,
                                                        "walker.*": 50,
                                                         "cross_factor": 0.1}} ,
                      'Town02': {'background_activity': {"vehicle.*": 15,
                                                          "walker.*": 50,
                                                         "cross_factor": 0.1}}

         },
         'dense': {'Town01': {'background_activity': {"vehicle.*": 100,
                                                         "walker.*": 250,
                                                      "cross_factor": 0.1}},
                     'Town02': {'background_activity': {"vehicle.*": 70,
                                                         "walker.*": 150,
                                                        "cross_factor": 0.1}}

                     }

}


name_dict = {'training':{'Town01': 'training',
                         'Town02': 'newtown'
                         },
             'new_weather': {'Town01': 'newweather',
                             'Town02': 'newweathertown'

             },
             'single_weather': {'Town01': 'trainingsw',
                          'Town02': 'newtownsw'
                          },
}

for task_name in tasks.keys():

    for town_name in town_sets.keys():

        for w_set_name in weather_sets.keys():
            # get the actual set  from th name
            w_set = weather_sets[w_set_name]
            new_json = {"envs": {},
                        "package_name": 'nocrash_' + name_dict[w_set_name][town_name] + '_'
                                        + task_name + '_' + town_name}

            for weather in w_set:

                for env_number in range(25):

                    env_dict = {
                        "route": {
                            "file":  town_sets[town_name],
                            "id": env_number
                        },
                        "scenarios": tasks[task_name][town_name],
                        "town_name": town_name,
                        "vehicle_model": "vehicle.lincoln.mkz_2017",
                        "weather_profile": weather,
                        "repetitions": 1
                    }

                    new_json["envs"].update({weather + '_route' + str(env_number).zfill(5): env_dict})

            filename = os.path.join(root_route_file_position, 'nocrash_' + name_dict[w_set_name][town_name] + '_'
                                                               + task_name + '_' + town_name + '.json')

            print(filename)

            fo = open(filename, 'w+')
                # with open(os.path.join(root_route_file_position, 'all_towns_traffic_scenarios3_4.json'), 'w') as fo:
            fo.write(json.dumps(new_json, sort_keys=True, indent=4))
            fo.close()