import numpy as np
import torch
import yaml
import json
import os.path as osp
from utils import inout, misc
import time
from utils.pose_utils import get_closest_rot
import argparse
from mean_average_precision import MetricBuilder
from typing import List, Dict, Tuple, Optional
import warnings

class GPUMetrics:
    def __init__(self, conf_path: str, device: Optional[str] = None):
        """
        GPU-accelerated metrics computation for 6D pose estimation.

        Args:
            conf_path: Path to configuration YAML file
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # GPU memory management
        self.max_memory_usage = 0.8  # Use max 80% of GPU memory
        if torch.cuda.is_available() and self.device.type == 'cuda':
            self.gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory
            print(f"GPU memory: {self.gpu_memory_total / 1024**3:.2f} GB")

            # Set memory fraction to avoid OOM
            if self.device.index is not None:
                torch.cuda.set_per_process_memory_fraction(self.max_memory_usage, device=self.device.index)
            else:
                torch.cuda.set_per_process_memory_fraction(self.max_memory_usage, device=0)
        else:
            self.gpu_memory_total = None

        with open(conf_path, 'r') as file:
            config = yaml.safe_load(file)
            self.info = self._init_info(config['info_path'])
            self.diameters = self._get_diameters()
            self.models = [osp.join(self.info["model_dir"], f"obj_{obj_id:06d}.ply") for obj_id in
                           self.info["obj2id"].values()]
            self.models_3d = [inout.load_ply(self.models[i], vertex_scale=0.001)["pts"] for i in
                              range(len(self.models))]

            # Convert model points to GPU tensors
            self.models_3d_gpu = [torch.from_numpy(pts.astype(np.float32)).to(self.device)
                                 for pts in self.models_3d]

            self.sym_infos = self._sym_infos(self.info)
            self.metrics = [{"re": [], "te": [], "add": []} for _ in range(len(self.info["objects"]))]
            self.predictions_path = config['predictions']
            self.ground_truth_path = config['ground_truth']
            print('metric list', MetricBuilder.get_metrics_list())
            self.number_class = len(self.models)
            self.metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=self.number_class)

            # Convert diameters to GPU tensor
            diameter_list = [self.diameters[i] for i in range(len(self.diameters))]
            self.diameters_gpu = torch.tensor(diameter_list, device=self.device, dtype=torch.float32)

    def _init_info(self, info_path: str) -> Dict:
        with open(info_path, "r") as f:
            info = json.load(f)
        return info

    def _get_diameters(self) -> Dict:
        with open(self.info["model_dir"] + "/models_info.json", "r") as f:
            models_info = json.load(f)
        diameters = {}
        for i in range(len(models_info.keys())):
            diameters[i] = models_info[str(i)]["diameter"]/1000
        return diameters

    def _sym_infos(self, info: Dict) -> Dict:
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

    def calculate_iou_batch_gpu(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Vectorized IOU calculation on GPU for batches of bounding boxes.

        Args:
            boxes1: Tensor of shape (N, 4) in format [x1, y1, x2, y2]
            boxes2: Tensor of shape (M, 4) in format [x1, y1, x2, y2]

        Returns:
            Tensor of shape (N, M) with IOU values
        """
        # Expand dimensions for broadcasting
        boxes1 = boxes1.unsqueeze(1)  # (N, 1, 4)
        boxes2 = boxes2.unsqueeze(0)  # (1, M, 4)

        # Calculate intersection coordinates
        x_left = torch.max(boxes1[..., 0], boxes2[..., 0])
        y_top = torch.max(boxes1[..., 1], boxes2[..., 1])
        x_right = torch.min(boxes1[..., 2], boxes2[..., 2])
        y_bottom = torch.min(boxes1[..., 3], boxes2[..., 3])

        # Calculate intersection area
        intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)

        # Calculate areas of each box
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # Calculate union area
        union_area = boxes1_area + boxes2_area - intersection_area

        # Calculate IOU
        iou = intersection_area / (union_area + 1e-8)  # Add small epsilon to avoid division by zero

        return iou

    def sort_boxes_by_iou_gpu(self, boxes1: List[Dict], boxes2: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        GPU-accelerated version of box sorting by IOU.
        """
        if not boxes1 or not boxes2:
            return boxes1, boxes2

        # Convert to tensors
        boxes1_tensor = torch.zeros((len(boxes1), 4), device=self.device)
        boxes2_tensor = torch.zeros((len(boxes2), 4), device=self.device)

        for i, box1 in enumerate(boxes1):
            xywh = box1['obj_bb']
            boxes1_tensor[i] = torch.tensor([xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]],
                                          device=self.device)

        for i, box2 in enumerate(boxes2):
            xywh = box2['obj_bb']
            boxes2_tensor[i] = torch.tensor([xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]],
                                          device=self.device)

        # Calculate IOU matrix
        iou_matrix = self.calculate_iou_batch_gpu(boxes1_tensor, boxes2_tensor)

        # Find best matches
        _, best_indices = torch.max(iou_matrix, dim=1)

        sorted_boxes1 = boxes1.copy()
        sorted_boxes2 = [boxes2[idx.item()] for idx in best_indices]

        return sorted_boxes1, sorted_boxes2

    def transform_pts_rt_gpu(self, pts: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated 3D point transformation.

        Args:
            pts: Tensor of shape (N, 3) with 3D points
            R: Tensor of shape (3, 3) rotation matrix
            t: Tensor of shape (3, 1) translation vector

        Returns:
            Transformed points of shape (N, 3)
        """
        return (R @ pts.T + t.reshape(3, 1)).T

    def re_gpu(self, R_est: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated rotational error computation.
        """
        rotation_diff = torch.mm(R_est, R_gt.T)
        trace = torch.trace(rotation_diff)
        trace = torch.clamp(trace, max=3.0)
        error_cos = torch.clamp(0.5 * (trace - 1.0), min=-1.0, max=1.0)
        rd_deg = torch.rad2deg(torch.acos(error_cos))
        return rd_deg

    def te_gpu(self, t_est: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated translational error computation.
        """
        return torch.norm(t_gt.flatten() - t_est.flatten())

    def add_gpu(self, R_est: torch.Tensor, t_est: torch.Tensor,
                R_gt: torch.Tensor, t_gt: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated ADD (Average Distance of Model Points) computation.
        """
        pts_est = self.transform_pts_rt_gpu(pts, R_est, t_est)
        pts_gt = self.transform_pts_rt_gpu(pts, R_gt, t_gt)
        return torch.norm(pts_est - pts_gt, dim=1).mean()

    def adi_gpu(self, R_est: torch.Tensor, t_est: torch.Tensor,
                R_gt: torch.Tensor, t_gt: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient GPU-accelerated ADI computation.
        """
        pts_est = self.transform_pts_rt_gpu(pts, R_est, t_est)
        pts_gt = self.transform_pts_rt_gpu(pts, R_gt, t_gt)

        # Memory-efficient chunked distance computation
        n_pts = pts_gt.shape[0]

        # Adaptive chunk size based on available GPU memory and number of points
        if self.device.type == 'cuda' and self.gpu_memory_total:
            # Estimate memory needed per chunk (rough heuristic)
            bytes_per_point = 4 * 3 * 2  # float32 * 3 coords * 2 point clouds
            available_memory = self.gpu_memory_total * 0.3  # Use 30% for distance computation
            max_chunk_size = int(available_memory / (bytes_per_point * n_pts))
            chunk_size = min(max(100, max_chunk_size), 2000)  # Between 100 and 2000 points
        else:
            chunk_size = min(1000, n_pts)  # Conservative default for CPU or unknown GPU

        chunk_size = min(chunk_size, n_pts)  # Don't exceed total points

        nn_dists = torch.zeros(n_pts, device=self.device)

        for i in range(0, n_pts, chunk_size):
            end_idx = min(i + chunk_size, n_pts)
            chunk_gt = pts_gt[i:end_idx]

            # Compute distances for this chunk
            # Using broadcasting instead of cdist for better memory control
            chunk_gt_expanded = chunk_gt.unsqueeze(1)  # (chunk_size, 1, 3)
            pts_est_expanded = pts_est.unsqueeze(0)    # (1, n_pts, 3)

            # Compute squared distances manually to control memory usage
            diff = chunk_gt_expanded - pts_est_expanded  # (chunk_size, n_pts, 3)
            distances_sq = torch.sum(diff * diff, dim=2)  # (chunk_size, n_pts)
            distances = torch.sqrt(distances_sq + 1e-8)   # Add small epsilon for numerical stability

            # Find minimum distance for each point in the chunk
            nn_dists[i:end_idx] = torch.min(distances, dim=1)[0]

            # Clear intermediate tensors to save memory
            del chunk_gt_expanded, pts_est_expanded, diff, distances_sq, distances

        return nn_dists.mean()

    def compute_metrics_batch_gpu(self, predictions: List[Dict], annotations: List[Dict], class_id: int):
        """
        GPU-accelerated batch processing of metrics computation.
        """
        if not predictions or not annotations:
            return

        predictions, annotations = self.sort_boxes_by_iou_gpu(predictions, annotations)

        # Prepare batch tensors
        batch_size = len(predictions)
        R_preds = torch.zeros((batch_size, 3, 3), device=self.device)
        t_preds = torch.zeros((batch_size, 3, 1), device=self.device)
        R_gts = torch.zeros((batch_size, 3, 3), device=self.device)
        t_gts = torch.zeros((batch_size, 3, 1), device=self.device)

        # Fill batch tensors
        for i in range(batch_size):
            pred = predictions[i]
            annot = annotations[i]

            R_preds[i] = torch.from_numpy(np.array(pred['cam_R_m2c']).reshape((3, 3)).astype(np.float32)).to(self.device)
            t_preds[i] = torch.from_numpy(np.array(pred['cam_t_m2c']).reshape((3, 1)).astype(np.float32)).to(self.device)
            R_gts[i] = torch.from_numpy(np.array(annot['cam_R_m2c']).reshape((3, 3)).astype(np.float32)).to(self.device)
            t_gts[i] = torch.from_numpy(np.array(annot['cam_t_m2c']).reshape((3, 1)).astype(np.float32)).to(self.device)

            # Handle mAP computation (still using CPU as MetricBuilder requires numpy)
            xyxy_pred = self.xywh2xyxy(pred["obj_bb"])
            xyxy_gt = self.xywh2xyxy(annot["obj_bb"])
            xyxy_pred.extend([float(class_id), 1.0])
            xyxy_gt.extend([float(class_id), 0.0, 0.0])
            self.metric_fn.add(np.array([xyxy_pred]), np.array([xyxy_gt]))

        # Batch compute metrics
        cat = predictions[0]['obj_id']  # Assuming all predictions have same obj_id
        pts_gpu = self.models_3d_gpu[cat]

        # Process each sample in the batch
        for i in range(batch_size):
            te_error = self.te_gpu(t_preds[i], t_gts[i])

            if self.info["id2obj"][str(cat)] in self.info['sym_obj']:
                # Handle symmetry (fallback to CPU for complex symmetry operations)
                R_pred_np = R_preds[i].cpu().numpy()
                R_gt_np = R_gts[i].cpu().numpy()
                R_gt_sym = get_closest_rot(R_pred_np, R_gt_np, self.sym_infos[str(cat)])
                R_gt_sym_gpu = torch.from_numpy(R_gt_sym.astype(np.float32)).to(self.device)

                re_error = self.re_gpu(R_preds[i], R_gt_sym_gpu)
                ad_error = self.adi_gpu(R_preds[i], t_preds[i],
                                       torch.from_numpy(R_gt_sym.astype(np.float32)).to(self.device),
                                       t_gts[i], pts_gpu)
            else:
                re_error = self.re_gpu(R_preds[i], R_gts[i])
                ad_error = self.adi_gpu(R_preds[i], t_preds[i], R_gts[i], t_gts[i], pts_gpu)

            # Convert back to CPU for storage
            self.metrics[cat]["re"].append(re_error.cpu().item())
            self.metrics[cat]["te"].append(te_error.cpu().item())
            self.metrics[cat]["add"].append(float(ad_error.cpu().item() < 0.1 * self.diameters[cat]))

    def xywh2xyxy(self, xywh: List[float]) -> List[float]:
        """Convert xywh format to xyxy format."""
        return [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]

    def print_metrics(self, missing_class: int):
        """Print computed metrics."""
        for i in range(len(self.info["objects"])):
            if len(self.metrics[i]['re']) == 0:
                continue
            print(f"Object {self.info['objects'][i]}")
            print(f"re: {np.mean(self.metrics[i]['re']):.2f}, te: {np.mean(self.metrics[i]['te']):.2f}, add: {np.mean(self.metrics[i]['add'])* 100 :.2f}")
            print("\n\n")
            print(f"mAP : {self.metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='greedy')['mAP'] * (self.number_class/(self.number_class-missing_class))}")

    def extract_obj_id_n(self, data: Dict, n: int) -> Dict:
        """Extract data for specific object ID."""
        extracted_data = {}
        for key in data:
            filtered_objects = [entry for entry in data[key] if entry["obj_id"] == n]
            if filtered_objects:
                extracted_data[key] = filtered_objects
        return extracted_data

    def run(self):
        """Main execution function with GPU acceleration."""
        print(f"Running GPU-accelerated metrics computation on {self.device}")

        with open(self.predictions_path, 'r') as f:
            print(self.predictions_path)
            tot_predictions = json.load(f)
        with open(self.ground_truth_path, 'r') as f:
            print(self.ground_truth_path)
            tot_poses = json.load(f)

        missing_classes = 0
        start_time = time.time()

        for i in range(len(self.info['objects'])):
            predictions = self.extract_obj_id_n(tot_predictions, i)
            if not predictions:
                missing_classes += 1
                continue

            poses = self.extract_obj_id_n(tot_poses, i)

            # Process all images for this object class
            for num in predictions:
                pose = poses[num]
                prediction = predictions[num]
                self.compute_metrics_batch_gpu(prediction, pose, class_id=i)

        end_time = time.time()
        print(f"GPU computation time: {end_time - start_time:.2f} seconds")

        self.print_metrics(missing_classes)

        # Print memory usage statistics
        if torch.cuda.is_available() and self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"Peak GPU memory usage: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default='/home/elena/repos/6d-eval/config/metrics_cfg.yml')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                       help='Device to use for computation')
    parser.add_argument('--iou_thres', default=np.arange(0.5, 1.0, 0.05))
    args = parser.parse_args()

    try:
        metric_computation = GPUMetrics(conf_path=args.conf_path, device=args.device)
        metric_computation.run()
    except Exception as e:
        print(f"Error during GPU metrics computation: {e}")
        print("Consider installing PyTorch with CUDA support for GPU acceleration:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        raise