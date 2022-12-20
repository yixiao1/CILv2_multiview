import os
import json
from json.decoder import JSONDecodeError

class Writer(object):
    """
        Organizing the writing process, note that the sensors are written on a separate thread.
        directly on the sensor interface.
    """

    def __init__(self, full_path):
        """
        """
        # path for the writter for this specific client
        self._full_path = os.path.join(full_path)
        self._latest_id = 0
        if not os.path.exists(self._full_path):
            os.makedirs(self._full_path)

    def update_latest_id(self):
        self._latest_id +=1

    def write_image(self, image, tag):
        image.save_to_disk(os.path.join(self._full_path, tag + '%06d.png' % self._latest_id))

    def write_pseudo(self, pseudo_data, tag):
        try:
            with open(os.path.join(self._full_path, tag + str(self._latest_id).zfill(6) + '.json'), 'r+') as fo:
                jsonObj = json.load(fo)
                jsonObj.update(pseudo_data)
                fo.seek(0)
                fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))
        except FileNotFoundError:
            with open(os.path.join(self._full_path, tag + str(self._latest_id).zfill(6) + '.json'), 'w') as fo:
                jsonObj = {}
                jsonObj.update(pseudo_data)
                fo.seek(0)
                fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

    # in principle these are not needed.
    def write_gnss(self, gnss, tag):
        try:
            with open(os.path.join(self._full_path, tag + str(self._latest_id).zfill(6) + '.json'), 'r+') as fo2:
                jsonObj = json.load(fo2)
                jsonObj.update(gnss)
                fo2.seek(0)
                fo2.write(json.dumps(jsonObj, sort_keys=True, indent=4))
        except FileNotFoundError:
            with open(os.path.join(self._full_path, tag + str(self._latest_id).zfill(6) + '.json'), 'w') as fo2:
                jsonObj = {}
                jsonObj.update(gnss)
                fo2.seek(0)
                fo2.write(json.dumps(jsonObj, sort_keys=True, indent=4))

    def write_lidar(self, lidar, tag, visualization=False):
        # TODO: NOT YET DEFINED FOR LIDAR DATA
        pass

    def write_2d_bbox(self, data, tag, visualization=False):
        with open(os.path.join(self._full_path, tag + str(self._latest_id).zfill(6) + '.json'), 'w') as fo:
            jsonObj = {}
            jsonObj.update(data)
            fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

        if visualization:
            save_path = os.path.join(self._full_path, 'bbox_patches')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # we check if the bbox data and rgb images are ready for drawing pathces
            exist_rgb_idx = [img[-10:-4] for img in sorted(os.listdir(self._full_path)) if img.startswith('rgb_'+tag.split('_')[-1]) and img.endswith('.png')]
            exist_bbox_idx = [json_f[-11:-5] for json_f in sorted(os.listdir(self._full_path)) if json_f.startswith(tag) and json_f.endswith('.json')]
            exist_bbox_img_idx = [bbox_img[-10:-4] for bbox_img in sorted(os.listdir(save_path)) if bbox_img.startswith(tag)]
            to_do_list = [idx for idx in exist_rgb_idx if idx in exist_bbox_idx and idx not in exist_bbox_img_idx]
            draw_bbox(to_do_list, tag, self._full_path, save_path)



