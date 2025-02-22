# 6D pose metrics 
Types of metrics:
- mAP (mean Average Precision) on bounding boxes (iou_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
- re (rotation error)
- te (translation error)
- add(-s) (average distance to the corresponding (-closest) point)
 
### Environment
```
git clone https://github.com/EleGo9/6d-eval.git
cd 6d-eval
conda env create -f 6deval_env.yml
```
### Compute Metrics
- The code to compute metrics is the following:
```
python metrics.py --conf_path /path/to/conf/cfg.yml
```
To properly run this code, you need the following files: 

cfg.yml with:
- predictions: path/to/json/file/predictions
- ground_truth: path/to/json/file/ground_truth
- info_path: path/to/info.json
where:
  - predictions and ground truth json files are as follows:
  {'num_img': [
  {cam_R_m2c':[
  rot matrix], 'cam_t_m2c':[translation vector], 'obj_bb':[xyhw], 'obj_id':int}
  {...},...,{...}], ...}
  - info file is as follows:{
    "objects": [],
    "obj2id": {
    },
    "id2obj" : {
    },
    "model_dir": "path/to/models",
    "vertex_scale": 0.001,
    "sym_obj": [],
    "cam": [
    ]
}

An example of predictions/ground truth files is 'scene_gt.json'
An example of info.json model is also provided as 'info.json'. 


Remember that you need a 'models' directory that must contain objects' ply CAD models and a model_info.json file.
Alert!: ply CAD models are named as their obj_id+1

models_info.json is structured as follows:
{
    "obj_id": {
        "diameter": ...,
        "min_x": ...,
        "min_y": ...,
        "min_z": ...,
        "size_x": ...,
        "size_y": ...,
        "size_z": ...,
        "symmetries_discrete" *: [{},...,{}],
        "symmetries_continuous" *: [{},...,{}]
    },

*optional, only if the object has symmetries

    
