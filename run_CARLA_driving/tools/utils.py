
import matplotlib.pyplot as plt

import carla


COLOR_BUTTER_0 = (252/ 255.0, 233/ 255.0, 79/ 255.0)
COLOR_BUTTER_1 = (237/ 255.0, 212/ 255.0, 0/ 255.0)
COLOR_BUTTER_2 = (196/ 255.0, 160/ 255.0, 0/ 255.0)

COLOR_ORANGE_0 = (252/ 255.0, 175/ 255.0, 62/ 255.0)
COLOR_ORANGE_1 = (245/ 255.0, 121/ 255.0, 0/ 255.0)
COLOR_ORANGE_2 = (209/ 255.0, 92/ 255.0, 0/ 255.0)

COLOR_CHOCOLATE_0 = (233/ 255.0, 185/ 255.0, 110/ 255.0)
COLOR_CHOCOLATE_1 = (193/ 255.0, 125/ 255.0, 17/ 255.0)
COLOR_CHOCOLATE_2 = (143/ 255.0, 89/ 255.0, 2/ 255.0)

COLOR_CHAMELEON_0 = (138/ 255.0, 226/ 255.0, 52/ 255.0)
COLOR_CHAMELEON_1 = (115/ 255.0, 210/ 255.0, 22/ 255.0)
COLOR_CHAMELEON_2 = (78/ 255.0, 154/ 255.0, 6/ 255.0)

COLOR_GREEN_0 = (0.0/ 255.0, 255.0/ 255.0, 0.0/ 255.0)

COLOR_SKY_BLUE_0 = (114/ 255.0, 159/ 255.0, 207/ 255.0)
COLOR_SKY_BLUE_1 = (52/ 255.0, 101/ 255.0, 164/ 255.0)
COLOR_SKY_BLUE_2 = (32/ 255.0, 74/ 255.0, 135/ 255.0)

COLOR_PLUM_0 = (173/ 255.0, 127/ 255.0, 168/ 255.0)
COLOR_PLUM_1 = (117/ 255.0, 80/ 255.0, 123/ 255.0)
COLOR_PLUM_2 = (92/ 255.0, 53/ 255.0, 102/ 255.0)

COLOR_SCARLET_RED_0 = (239/ 255.0, 41/ 255.0, 41/ 255.0)
COLOR_SCARLET_RED_1 = (204/ 255.0, 0/ 255.0, 0/ 255.0)
COLOR_SCARLET_RED_2 = (164/ 255.0, 0/ 255.0, 0/ 255.0)

COLOR_ALUMINIUM_0 = (238/ 255.0, 238/ 255.0, 236/ 255.0)
COLOR_ALUMINIUM_1 = (211/ 255.0, 215/ 255.0, 207/ 255.0)
COLOR_ALUMINIUM_2 = (186/ 255.0, 189/ 255.0, 182/ 255.0)
COLOR_ALUMINIUM_3 = (136/ 255.0, 138/ 255.0, 133/ 255.0)
COLOR_ALUMINIUM_4 = (85/ 255.0, 87/ 255.0, 83/ 255.0)
COLOR_ALUMINIUM_4_5 = (66/ 255.0, 62/ 255.0, 64/ 255.0)
COLOR_ALUMINIUM_5 = (46/ 255.0, 52/ 255.0, 54/ 255.0)

COLOR_WHITE = (255/ 255.0, 255/ 255.0, 255/ 255.0)
COLOR_BLACK = (0/ 255.0, 0/ 255.0, 0/ 255.0)
COLOR_LIGHT_GRAY = (196/ 255.0, 196/ 255.0, 196/ 255.0)
COLOR_PINK = (255/255.0,192/255.0,203/255.0)
pixels_per_meter = 12
SCALE = 1.0
precision = 0.05
world_offset = [0, 0]

def draw_point(location, result_color, size, alpha=None, fig=None, text=None):

    pixel = world_to_pixel(location)
    circle = plt.Circle((pixel[0], pixel[1]), size, fc=result_color, alpha=alpha)

    # if the specific fig was defined, we add to that figure
    if fig:
        plt.figure(fig.number)

    plt.gca().add_patch(circle)

    if text:
        plt.text(pixel[0]+20, pixel[1]+20, text, color=result_color, fontsize='x-small')

def draw_point_data(datapoint, fig = None, color=COLOR_BLACK, size=12, alpha=None, text=None):
    """
    We draw in a certain position at the map
    :param position:
    :param color:
    :return:
    """

    world_pos = datapoint
    location = carla.Location(x=world_pos[0], y=world_pos[1], z=world_pos[2])
    draw_point(location, color, size, alpha, fig=fig, text=text)

def world_to_pixel(location, offset=(0, 0)):
    x = SCALE * pixels_per_meter * (location.x - world_offset[0])
    y = SCALE * pixels_per_meter * (location.y - world_offset[1])
    return [int(x - offset[0]), int(y - offset[1])]

def lateral_shift(transform, shift):
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()

def draw_roads(set_waypoints):
    for waypoints in set_waypoints:
        waypoint = waypoints[0]
        road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
        road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

        polygon = road_left_side + [x for x in reversed(road_right_side)]
        polygon = [world_to_pixel(x) for x in polygon]

        if len(polygon) > 2:
            polygon = plt.Polygon(polygon, edgecolor=COLOR_WHITE)
            plt.gca().add_patch(polygon)

def draw_lane(lane, color):
    for side in lane:
        lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
        lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

        polygon = lane_left_side + [x for x in reversed(lane_right_side)]
        polygon = [world_to_pixel(x) for x in polygon]

        if len(polygon) > 2:
            polygon = plt.Polygon(polygon, edgecolor=color)

            plt.gca().add_patch(polygon)

def draw_topology(carla_topology, index):
    topology = [x[index] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    set_waypoints = []
    for waypoint in topology:
        # if waypoint.road_id == 150 or waypoint.road_id == 16:
        waypoints = [waypoint]

        nxt = waypoint.next(precision)
        if len(nxt) > 0:
            nxt = nxt[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                else:
                    break
        set_waypoints.append(waypoints)
        # Draw Shoulders, Parkings and Sidewalks
        PARKING_COLOR = COLOR_ALUMINIUM_4_5
        SHOULDER_COLOR = COLOR_ALUMINIUM_5
        SIDEWALK_COLOR = COLOR_ALUMINIUM_3

        shoulder = [[], []]
        parking = [[], []]
        sidewalk = [[], []]

        for w in waypoints:
            l = w.get_left_lane()
            while l and l.lane_type != carla.LaneType.Driving:

                if l.lane_type == carla.LaneType.Shoulder:
                    shoulder[0].append(l)

                if l.lane_type == carla.LaneType.Parking:
                    parking[0].append(l)

                if l.lane_type == carla.LaneType.Sidewalk:
                    sidewalk[0].append(l)

                l = l.get_left_lane()

            r = w.get_right_lane()
            while r and r.lane_type != carla.LaneType.Driving:

                if r.lane_type == carla.LaneType.Shoulder:
                    shoulder[1].append(r)

                if r.lane_type == carla.LaneType.Parking:
                    parking[1].append(r)

                if r.lane_type == carla.LaneType.Sidewalk:
                    sidewalk[1].append(r)

                r = r.get_right_lane()

        draw_lane(shoulder, SHOULDER_COLOR)
        draw_lane(parking, PARKING_COLOR)
        draw_lane(sidewalk, SIDEWALK_COLOR)

    draw_roads(set_waypoints)

def draw_map(world):

    topology = world.get_map().get_topology()
    draw_topology(topology, 0)