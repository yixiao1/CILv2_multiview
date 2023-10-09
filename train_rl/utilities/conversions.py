def convert_speed_01(x):
    return (x + 1) / 2

def convert_01(x):
    return (x + 1) / 2

def convert_11(x):
    # convert from [0,1] to [-1,1]
    return (x * 2) - 1

def step_function(x, threshold):
    if x < threshold:
        return 0
    else:
        return 1
    
def command_to_text(command):
    # TURN_LEFT
    if float(command) == 1.0:
        return 'turn left'
    # TURN_RIGHT
    elif float(command) == 2.0:
        return 'turn right'
    # GO_STRAIGHT
    elif float(command) == 3.0:
        return 'go straight'
    # FOLLOW_LANE
    elif float(command) == 4.0:
        return 'follow lane'
    # CHANGELANE_LEFT
    elif float(command) == 5.0:
        return 'change lane left'
    # CHANGELANE_RIGHT
    elif float(command) == 6.0:
        return 'change lane right'
    else:
        raise ValueError("Unexpcted direction identified %s" % str(command))