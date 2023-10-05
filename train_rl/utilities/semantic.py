import numpy as np  

def semantic_image_to_labels(image):
    return image[:,:,2]

# see labels meaning here: https://carla.readthedocs.io/en/0.9.10/ref_sensors/#rgb-camera
# only using 6 labels
def filter_semantic_labels(image_labels):
    mapping = {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 0,
        5 : 0,
        9 : 0,
        11: 0,
        12: 0,
        13: 0,
        14: 0,
        15: 0,
        16: 0,
        17: 0,
        19: 0,
        20: 0,
        21: 0,
        22: 0,
        4 : 1,
        10: 1,
        18: 2,
        6 : 3,
        7 : 4,
        8 : 5}
    
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))
    
    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype) #k,v from approach #1
    mapping_ar[k] = v

    image_labels_filtered = mapping_ar[image_labels]
    
    return image_labels_filtered

def semantic_labels_to_image(array):
   
    classes = {
        0: [70, 70, 70],   # static
        1: [0, 0, 142],    # dynamic
        2: [250, 170, 30], # traffic light
        3: [157, 234, 50], # road lines
        4: [128, 64, 128], # road
        5: [244, 35, 232]  # side walks
    }
        
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
        
    result = result/255.0
    result = result.astype(np.float32)
    return result
