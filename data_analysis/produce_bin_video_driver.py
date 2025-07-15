import numpy as np 
import cv2
# import moviepy.video.io.ImageSequenceClip
import imageio.v2 as io
import glob 
import settings as SE
import argparse
import os
from matplotlib import cm
import traceback
import json

import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def parse_args():
    parser = argparse.ArgumentParser()

    # Results options
    parser.add_argument('--scenario', type=str, default="H2/H2_1_1")
    # Pipeline options
    #parser.add_argument('--pipeline_compose', type=str, default='$ADRE_ROOT/launch/complete_pipeline.yml')
    #parser.add_argument('--pipeline_gpu', type=int, default=0)
    #parser.add_argument('--camera_name', type=str, default='sekonix_60')

    # ROS options
    #parser.add_argument('--ros_domain_id', type=int, default=0)

    args = parser.parse_args()
    return args



def joinScreenImages(path, cnt, image_size=(540,960,3), mirror_size=(360,640,3), RGB=False):#mirror_left, mirror_right, 

    #mirror_left = io.imread(mirror_left)[:,:,:3]
    #mirror_right = io.imread(mirror_right)[:,:,:3]
    all_images = True
    try:
        '''print("AAAAAA")
        print(path.replace("*", f"{cnt:06}").replace("camera", "left"))
        print("BBBBBB")'''
        left = io.imread(path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "Left"))[:,:,:3]
        if RGB:
            attention_path = path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "Left").replace("/ClearNoon", "").replace("_ClearNoon", "").replace("RGB", "AETBin").replace(".jpg", ".npz")
            attention_image = np.load(attention_path)["attention_binary"]
            attention_image_rgb = np.uint8(cm.jet(attention_image*256)*255)[:,:,:3]
            kernel = np.ones((5,5), dtype=np.uint8)
            dilated_attention = cv2.dilate(np.uint8(attention_image)*255, kernel, iterations=1) 
            attention_image_rgb[np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)] = [64,0,0]
            inner_area = np.where(attention_image>0)
            outer_area = np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)
            left[inner_area] = left[inner_area]*0.5+attention_image_rgb[inner_area]*0.5
            left[outer_area] = attention_image_rgb[outer_area]
            attention_image = np.stack([attention_image]*3, axis=-1)  # Convert to 3 channels
            left = np.uint8(left*0.75 + attention_image*0.25)

    except Exception as e:
        left = np.zeros(image_size, dtype=np.uint8)
        all_images = False
        print(traceback.format_exc())
    try:
        central = io.imread(path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "Central"))[:,:,:3]
        if RGB:
            attention_path = path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "Central").replace("/ClearNoon", "").replace("_ClearNoon", "").replace("RGB", "AETBin").replace(".jpg", ".npz")
            attention_image = np.load(attention_path)["attention_binary"]
            attention_image_rgb = np.uint8(cm.jet(attention_image*256)*255)[:,:,:3]
            kernel = np.ones((5,5), dtype=np.uint8)
            dilated_attention = cv2.dilate(np.uint8(attention_image)*255, kernel, iterations=1) 
            attention_image_rgb[np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)] = [64,0,0]
            inner_area = np.where(attention_image>0)
            outer_area = np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)
            central[inner_area] = central[inner_area]*0.5+attention_image_rgb[inner_area]*0.5
            central[outer_area] = attention_image_rgb[outer_area]
            attention_image = np.stack([attention_image]*3, axis=-1)
            central = np.uint8(central*0.75 + attention_image*0.25)
    except Exception as e:
        central = np.zeros(image_size, dtype=np.uint8)
        all_images = False
        print(traceback.format_exc())
    try:
        right = io.imread(path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "Right"))[:,:,:3]
        if RGB:
            attention_path = path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "Right").replace("/ClearNoon", "").replace("_ClearNoon", "").replace("RGB", "AETBin").replace(".jpg", ".npz")
            attention_image = np.load(attention_path)["attention_binary"]
            attention_image_rgb = np.uint8(cm.jet(attention_image*256)*255)[:,:,:3]
            kernel = np.ones((5,5), dtype=np.uint8)
            dilated_attention = cv2.dilate(np.uint8(attention_image)*255, kernel, iterations=1) 
            attention_image_rgb[np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)] = [64,0,0]
            inner_area = np.where(attention_image>0)
            outer_area = np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)
            right[inner_area] = right[inner_area]*0.5+attention_image_rgb[inner_area]*0.5
            right[outer_area] = attention_image_rgb[outer_area]
            attention_image = np.stack([attention_image]*3, axis=-1)
            right = np.uint8(right*0.75 + attention_image*0.25)
    except Exception as e:
        right = np.zeros(image_size, dtype=np.uint8)
        all_images = False
        print(traceback.format_exc())
    try:
        mirror_left = io.imread(path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "MLeft"))[:,:,:3]
        if RGB:
            attention_path = path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "MLeft").replace("/ClearNoon", "").replace("_ClearNoon", "").replace("RGB", "AETBin").replace(".jpg", ".npz")
            attention_image = np.load(attention_path)["attention_binary"]
            attention_image_rgb = np.uint8(cm.jet(attention_image*256)*255)[:,:,:3]
            kernel = np.ones((5,5), dtype=np.uint8)
            dilated_attention = cv2.dilate(np.uint8(attention_image)*255, kernel, iterations=1) 
            attention_image_rgb[np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)] = [64,0,0]
            inner_area = np.where(attention_image>0)
            outer_area = np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)
            mirror_left[inner_area] = mirror_left[inner_area]*0.5+attention_image_rgb[inner_area]*0.5
            mirror_left[outer_area] = attention_image_rgb[outer_area]
            attention_image = np.stack([attention_image]*3, axis=-1)
            mirror_left = np.uint8(mirror_left*0.75 + attention_image*0.25)
    except Exception as e:
        mirror_left = np.zeros(mirror_size, dtype=np.uint8)
        all_images = False
        print(traceback.format_exc())
    try:
        mirror_right = io.imread(path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "MRight"))[:,:,:3]
        if RGB:
            attention_path = path.replace("*", f"{SE.frameToStamp(cnt)}").replace("camera", "MRight").replace("/ClearNoon", "").replace("_ClearNoon", "").replace("RGB", "AETBin").replace(".jpg", ".npz")
            attention_image = np.load(attention_path)["attention_binary"]
            attention_image_rgb = np.uint8(cm.jet(attention_image*256)*255)[:,:,:3]
            kernel = np.ones((5,5), dtype=np.uint8)
            dilated_attention = cv2.dilate(np.uint8(attention_image)*255, kernel, iterations=1) 
            attention_image_rgb[np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)] = [64,0,0]
            inner_area = np.where(attention_image>0)
            outer_area = np.where(np.minimum(dilated_attention>0 , attention_image <1)>0)
            mirror_right[inner_area] = mirror_right[inner_area]*0.5+attention_image_rgb[inner_area]*0.5
            mirror_right[outer_area] = attention_image_rgb[outer_area]
            attention_image = np.stack([attention_image]*3, axis=-1)
            mirror_right = np.uint8(mirror_right*0.75 + attention_image*0.25)
    except Exception as e:
        mirror_right = np.zeros(mirror_size, dtype=np.uint8)
        all_images = False
        print(traceback.format_exc())

    mirror_right = cv2.resize(mirror_right, (640,360))
    mirror_left = cv2.resize(mirror_left, (640,360))
    height = central.shape[0]
    image_size = (height, central.shape[1]*3+mirror_left.shape[1]*2, 3)#
    #print(image_size)
    joint_image = np.zeros(image_size, dtype=np.uint8)
    
    if True:
        joint_image[height-mirror_left.shape[0]:, :mirror_left.shape[1]] = mirror_left
        #print("1")
        #plt.imshow(joint_image)
        joint_image[height-mirror_right.shape[0]:, image_size[1]-mirror_right.shape[1]:] = mirror_right
        #print("2")
        base = mirror_left.shape[1]
        joint_image[:, base:base+left.shape[1]] = left
        #print("3")
        base = base+left.shape[1]
        joint_image[:, base:base+central.shape[1]] = central
        #print("4")
        base = base+central.shape[1]
        #plt.imshow(right)
        #plt.show()
        #print(right.shape, left.shape)
        #print(path.replace("*", f"{cnt:06}").replace("camera", "right"))
        joint_image[:, base:base+right.shape[1]] = right
    #print("b")

    proportion = float(1920)/float(joint_image.shape[1])
    resize_size = (1920,int(height*proportion))
    #print(resize_size)
    joint_image = cv2.resize(joint_image, resize_size)
    return joint_image

def createFrame(rgb_image, attention_image,video_frame, direction, speed, acceleration, steering):
    directionDic = {1: "Left",
                    2: "Right",
                    3: "Straight",
                    4: "Follow Lane",
                    5: "Straight",
                    6: "Straight"
                    }
    if direction == 0:
        direction = 4
    if direction >6:
        direction = 3

    frame = np.zeros((249,1920,3), dtype=np.uint8)
    frame[:rgb_image.shape[0]] = rgb_image
    frame = cv2.putText(frame, f'Speed: {speed:.2f} km/h', (1635,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)
    frame = cv2.putText(frame, f'Command: {directionDic[direction]}', (1635,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)
    frame = cv2.putText(frame, f'Acceleration: {acceleration:.2f}', (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)
    frame = cv2.putText(frame, f'Steering: {steering:.2f}', (5,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)
    return frame


# def process_data(input_file, directions_file, total_frames, rgb_images_path, attention_images_path, framerate, scenario, original_scenario, user, town, route, base_user):
    
#     fps=25#25
#     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#     print(f'/data/121-2/Experiments/dporres/{user}/{town}/{route}/VIDEO')
#     if not os.path.exists(f'/data/121-2/Experiments/dporres/{user}/{town}/{route}/VIDEO'):
#         os.makedirs(f'/data/121-2/Experiments/dporres/{user}/{town}/{route}/VIDEO')
#     out = cv2.VideoWriter(f'/data/121-2/Experiments/dporres/{user}/{town}/{route}/VIDEO/{user}_{town}_{route}_video_AM-CPP-Baseline-400x225_bin.avi', fourcc, fps, (1920,360))
    
#     #sizes = {}
#     printProgressBar (0, total_frames,prefix = 'Progress:', suffix = f'{1}/{total_frames} Complete', length = 50)
#     main_folder = f"{SE.DATA_ROOT}/{original_scenario}/"
#     # eyetracker_folder = glob.glob(main_folder+"*Z")[0].replace("\\", "/")+"/"
#     # et_vid = io.get_reader(eyetracker_folder+"scenevideo.mp4",  'ffmpeg')
#     # et_fps = SE.EYETRACKER_FRAMERATE#24.93#24.95
#     # eyetracker_offset = SE.route_data[original_scenario]["start"]
#     data_offset = 0.44#float(SE.frame_count_dict[base_user][town][route])/SE.FRAMERATE - float(input_file[-1, 0])
#     data_offset *= 2
#     for cnt in range(2,total_frames):#total_frames
        
#         # current_time = float(cnt)/float(framerate)
#         #print(current_time)
        
#         # et_frame_index = round(eyetracker_offset+current_time*et_fps)
#         # et_frame = et_vid.get_data(et_frame_index)

#         json_data = None
#         json_name = f"{SE.EXPORT_ROOT}/{user}/{town}/{route}/CB/{user}_{town}_{route}_CB_{SE.frameToStamp(cnt)}.json"
#         try:
            
#             with open(json_name, "r") as json_file:
#                 json_data = json.load(json_file)
#         except Exception as e:
#             print(f"Error with file {json_name}")
#             print(e)
#         direction = int(json_data["direction"])
#         speed = json_data["speed"]
#         acceleration = json_data["acceleration"]
#         steering = json_data["steer"]
       
#         '''print("++++++++++")
#         print(rgb_images[0][cnt])
#         print(rgb_images[1][cnt])
#         print(rgb_images[2][cnt])
#         print(rgb_images[3][cnt])
#         print(rgb_images[4][cnt])
#         print("----------")'''
#         #rgb_image = None
#         try:
#             rgb_image = joinScreenImages(rgb_images_path, cnt, RGB=True)#, rgb_images[3][cnt], rgb_images[4][cnt]
#             # attention_image = joinScreenImages(attention_images_path, cnt)#, attention_images[3][cnt], attention_images[4][cnt]
#             # attention_image = np.zeros((1, 1, 3))
#         except Exception as e:
#             print(e)
#             break
#         frame = createFrame(rgb_image, None, None, direction, speed, acceleration, steering)
#         #plt.imsave(f"temp_frames/{scenario}/{SE.frameToStamp(cnt)}_frame.png", frame)
#         out.write(frame[:,:,[2,1,0]])
#         #shape = frame.shape
#         '''if shape in sizes.keys():
#             sizes[shape]+=1
#         else:
#             sizes[shape] = 0
#     print(sizes)'''
#         printProgressBar (cnt+1, total_frames, prefix = 'Progress:', suffix = f'{cnt+1}/{total_frames} Complete', length = 50)
    
    
    
    # image_files = []#sorted(glob.glob(f"temp_frames/{scenario}/*_frame.png"))
    # #print(image_files)
    # base_size = np.zeros((1080,1920,4)).shape
    # for image in image_files:
    #     img = io.imread(image)
    #     image_size = img.shape
    #     if not image_size == base_size:
    #         #print(image_size, image)
    #         image_files.remove(image)
    #     else:
    #         #print(image)
    #         out.write(img[:,:,[2,1,0]])
    # out.release()
    # #clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    # #clip.write_videofile(f'{scenario}_test_video_a.mp4')
    # return


def process_single_frame(frame_data, rgb_images_path, eyetracker_offset, et_fps, et_vid, user, town, route):
    cnt, framerate = frame_data
    
    # current_time = float(cnt)/float(framerate)
    # et_frame_index = round(eyetracker_offset+current_time*et_fps)
    # et_frame = et_vid.get_data(et_frame_index)

    json_name = f"{SE.EXPORT_ROOT}/{user}/{town}/{route}/CB/{user}_{town}_{route}_CB_{SE.frameToStamp(cnt)}.json"
    try:
        with open(json_name, "r") as json_file:
            json_data = json.load(json_file)
    except Exception as e:
        print(f"Error with file {json_name}")
        print(e)
        return None

    direction = int(json_data["direction"])
    speed = json_data["speed"]
    acceleration = json_data["acceleration"]
    steering = json_data["steer"]

    try:
        rgb_image = joinScreenImages(rgb_images_path, cnt, RGB=True)
        # attention_image = np.zeros((1, 1, 3))
        frame = createFrame(rgb_image, None, None, direction, speed, acceleration, steering)
        return frame[:,:,[2,1,0]]
    except Exception as e:
        print(e)
        return None

def process_data(input_file, directions_file, total_frames, rgb_images_path, attention_images_path, framerate, scenario, original_scenario, user, town, route, base_user):
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    output_dir = f'/data/121-2/Experiments/dporres/TED_AttMaps/{user}/{town}/{route}/VIDEO'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = f'{output_dir}/{user}_{town}_{route}_video_bin.avi'
    print(f"Creating video: {output_path}")
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920,249))

    # main_folder = f"{SE.DATA_ROOT}/{original_scenario}/"
    # eyetracker_folder = glob.glob(main_folder+"*Z")[0].replace("\\", "/")+"/"
    # et_vid = io.get_reader(eyetracker_folder+"scenevideo.mp4", 'ffmpeg')
    et_fps = SE.EYETRACKER_FRAMERATE
    eyetracker_offset = SE.route_data[original_scenario]["start"]
    
    # Create frame data for parallel processing
    frame_data = [(cnt, framerate) for cnt in range(2, total_frames)]
    
    # Create a partial function with fixed parameters
    process_frame = partial(
        process_single_frame,
        rgb_images_path=rgb_images_path,
        eyetracker_offset=eyetracker_offset,
        et_fps=et_fps,
        et_vid=None,
        user=user,
        town=town,
        route=route
    )
    
    # Use multiprocessing pool
    num_cores = cpu_count() - 1  # Leave one core free
    with Pool(num_cores) as pool:
        # Process frames in parallel with tqdm progress bar
        for frame in tqdm.tqdm(
            pool.imap(process_frame, frame_data),
            total=len(frame_data),
            desc="Processing frames",
            unit="frame"
        ):
            if frame is not None:
                out.write(frame)
    
    out.release()
    print(f"\nVideo saved to: {output_path}")




def process_folder(scenario):
    main_path = f"{SE.DATA_ROOT}/{scenario}/"
    data = scenario.split("/")[-1].split("_")
    #print(data)
    user = SE.getUser(data[0])
    town = SE.getTown(data[1])
    route = SE.getRoute(data[2])
    #print(user, town, route)
    framerate = SE.FRAMERATE
    original_scenario = scenario
    scenario = scenario.split("/")[-1]
    #if not os.path.exists(f"temp_frames/{scenario}"):
    #    os.makedirs(f"temp_frames/{scenario}")
    attention_path = f"{SE.EXPORT_ROOT}/{user}/{town}/{route}/AETOverlay/camera/{user}_{town}_{route}_AETOverlay_camera_*.jpg"
    RGB_path = f"{SE.EXPORT_ROOT}/{user}/{town}/{route}/RGB/camera/ClearNoon/{user}_{town}_{route}_RGB_camera_ClearNoon_*.jpg"
    #print(f"/data-net/ted/{scenario}/*.json")
    frame_range = SE.frame_count_dict[data[0]][town][route]#getVideoFrames(user, town, route)
    input_file = np.loadtxt(glob.glob(main_path+"*_input_data.txt")[0], delimiter="\t", skiprows=1, dtype=np.unicode_)
    directions_file = np.loadtxt(glob.glob(main_path+"*_directions_data.txt")[0], delimiter="\t")
    #attention_path+"/rgb_camera*_warped.png"#semantic_segmentation(png)#main_path+"input_frames/rgb_camera*.jpg#attention_path+"/rgb_camera*_attention_overlay.png"
    process_data(input_file, directions_file, frame_range, RGB_path, attention_path, framerate, scenario,original_scenario, user, town, route, data[0])


if __name__ == "__main__":
    args = parse_args()
    process_folder(args.scenario)
    