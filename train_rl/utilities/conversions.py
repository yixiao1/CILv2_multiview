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