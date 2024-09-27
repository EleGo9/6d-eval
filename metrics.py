import numpy as np
import yaml
import json
import os.path as osp
from utils import inout, misc
import time
from utils.pose_error import re, te, add, adi
from utils.pose_utils import get_closest_rot
import argparse

class Metrics:
    def __init__(self, conf_path):
        with open(conf_path, 'r') as file:
            config = yaml.safe_load(file)
            self.info = self._init_info(config['info_path'])
            self.diameters = self._get_diameters()
            self.models = [osp.join(self.info["model_dir"], f"obj_{obj_id+1:06d}.ply") for obj_id in
                           self.info["obj2id"].values()]
            self.models_3d = [inout.load_ply(self.models[i], vertex_scale=0.001)["pts"] for i in
                              range(len(self.models))]
            self.sym_infos = self._sym_infos(self.info)
            self.metrics = [{"re": [], "te": [], "add": []} for _ in range(len(self.info["objects"]))]
            self.predictions_path = config['predictions']
            self.ground_truth_path = config['ground_truth']

    def _init_info(self, info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        return info

    def _get_diameters(self):
        with open(self.info["model_dir"] + "/models_info.json", "r") as f:
            models_info = json.load(f)
        diameters = {}
        for i in range(len(models_info.keys())):
            diameters[i] = models_info[str(i)]["diameter"] / 1000
        return diameters

    def _sym_infos(self, info):
        sym_infos = {}
        with open(info["model_dir"] + "/models_info.json", "r") as f:
            models_info = json.load(f)
        for i in range(len(models_info.keys())):
            if "symmetries_discrete" in models_info[str(i)] or "symmetries_continuous" in models_info[str(i)]:
                sym_transforms = misc.get_symmetry_transformations(models_info[str(i)], max_sym_disc_step=0.01)
                sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
            else:
                sym_info = None
            sym_infos[str(i)] = sym_info
        return sym_infos

    def sort_boxes_by_iou(self, boxes1, boxes2):
        """
        Sort two lists of bounding boxes by Intersection over Union (IOU).
        Arguments:
        boxes1, boxes2: Lists of dictionaries with 'bbox' key containing bounding box.
        Returns:
        Sorted lists of bounding boxes based on IOU.
        """
        sorted_boxes1 = []
        sorted_boxes2 = []

        for box1 in boxes1:
            max_iou = -1
            best_box2 = None
            for box2 in boxes2:
                box1xywh = box1['obj_bb']
                box2xywh = box2['obj_bb'] # if 'o' in box2 else box2['bbox_est']
                box1_ = [box1xywh[0], box1xywh[1], box1xywh[0] + box1xywh[2], box1xywh[1] + box1xywh[3]]
                box2_ = [box2xywh[0], box2xywh[1], box2xywh[0] + box2xywh[2], box2xywh[1] + box2xywh[3]]
                iou = self.calculate_iou(box1_, box2_)
                if iou > max_iou:
                    max_iou = iou
                    best_box2 = box2
            sorted_boxes1.append(box1)
            sorted_boxes2.append(best_box2)

        return sorted_boxes1, sorted_boxes2

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IOU) between two bounding boxes.
        Arguments:
        box1, box2: Bounding boxes in the format [x1, y1, x2, y2].
        Returns:
        IOU value.
        """
        # Determine coordinates of intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # Calculate intersection area
        if x_right < x_left or y_bottom < y_top:
            intersection_area = 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate area of each bounding box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union area
        union_area = box1_area + box2_area - intersection_area

        # Calculate IOU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou



    def compute_metrics(self, predictions, annotations):
        # given an image, compute the pose metrics
        predictions, annotations = self.sort_boxes_by_iou(predictions, annotations)
        for i in range(len(predictions)):
            pred = predictions[i]
            annot = annotations[i]
            R_pred = np.array(pred['cam_R_m2c']).reshape((3, 3))
            t_pred = np.array(pred['cam_t_m2c']).reshape((3, 1))  # * (self.info["cam"][0] / 608.219957982)
            R_gt = np.array(annot['cam_R_m2c']).reshape((3, 3))
            t_gt = np.array(annot['cam_t_m2c']).reshape((3, 1))

            te_error = te(t_pred, t_gt)
            cat = pred['obj_id']
            if self.info["id2obj"][str(cat)] in self.info['sym_obj']:
                R_gt_sym = get_closest_rot(R_pred, R_gt, self.sym_infos[str(cat)])
                re_error = re(R_pred, R_gt_sym)
                ad_error = adi(
                    R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[cat]
                )
            else:
                re_error = re(R_pred, R_gt)

                ad_error = add(
                    R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[cat]
                )
            self.metrics[cat]["re"].append(re_error)
            self.metrics[cat]["te"].append(te_error)
            self.metrics[cat]["add"].append(float(ad_error < 0.1 * self.diameters[cat]))

    def print_metrics(self):
        for i in range(len(self.info["objects"])):
            if len(self.metrics[i]['re']) == 0:
                continue
            print(f"Object {self.info['objects'][i]}")
            print(f"re: {np.mean(self.metrics[i]['re']):.2f}, te: {np.mean(self.metrics[i]['te']):.2f}, add: {np.mean(self.metrics[i]['add'])* 100 :.2f}")
            print("\n\n")

    def extract_obj_id_n(self, data, n):
        extracted_data = {}

        # Iterate through the keys in the main dictionary
        for key in data:
            # Filter out only the objects with obj_id == 2
            filtered_objects = [entry for entry in data[key] if entry["obj_id"] == n]

            # If there are filtered objects, add them to the result dictionary
            if filtered_objects:
                extracted_data[key] = filtered_objects

        return extracted_data

    def run(self):
        with open(self.predictions_path, 'r') as f:
            print(self.predictions_path)
            tot_predictions = json.load(f)
        with open(self.ground_truth_path, 'r') as f:
            print(self.ground_truth_path)
            tot_poses = json.load(f)
        for i in range(len(self.info['objects'])):
            predictions = self.extract_obj_id_n(tot_predictions, i)
            poses = self.extract_obj_id_n(tot_poses, i)
            for num in predictions:
                print('NUM', num)
                # annotation = annotations[num]
                pose = poses[num]
                prediction = predictions[num]
                self.compute_metrics(prediction, pose)
        self.print_metrics()








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default= '/home/elena/repos/6d-eval/config/metrics_cfg.yml' )
    args = parser.parse_args()
    metric_computation = Metrics(conf_path=args.conf_path)
    metric_computation.run()