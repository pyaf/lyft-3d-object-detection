# Lyft 3D object detection


## Approaches and Experiments:

### To learn and implement:

1. Run PointRCNN on nuscenes-mini, converted to kiiti.
2. Run SECOND on nuscenes-mini: DONE
3. Run SECOND on lyft. DONE
4. Run frustum-convnet on kitti-formatted lyft.
5.

[SOTA for 3D obj det](https://paperswithcode.com/sota/3d-object-detection-on-kitti-cars-moderate)




## Notes:


## TODO:

[x] apex support: it was useless warning, suppressed it. Now I can train at bs:8
[x] mAP metric computation in eval.
[x] prediction with sweeps
[] with pretrained models? hmm, looks like it's working after 100 iters
[x] understanding the evaluation metric
[x] filter too-close predictions
[x] good train/val split



some issues:
https://github.com/traveller59/second.pytorch/issues/226

## things to understand:

[x] the nuscenes/lyft sdk
[x] the nuscenes/lyft dataset scema
[x] the KITTI dataset scema
[x] the annotations, understanding them
[x] post process second inference
[x] z-centers for anchor_ranges
[] multi-sweep predictions

## Ideas:

* What about a kaggle kernel for Boxes only?
* A kernel for evaluation metric?
* For rare classes like animal and other_vehicle, what about training only on a subset of the dataset?


## Questions?

* .get_boxes says sample_data can be keyframe or intermediate sample_data, so key_frame=True sample_data are those which are in first sample_data
 For sample_data with is_key_frame=True, the time-stamps should be very close to the sample it points to. For non key-frames the sample_data points to the sample that follows closest in time. *still not able to understand*
Okay, so for key_frame=True, sample_data timestamps is very close to sample timestamp, not the same for non keyframe sample_data point.
*ALRIGHT*, so keyframes are those which pretty much match to annotations, but we haven'


* height, width and length, which axes do these correspond to?, : x, y and z
* How to decide point cloud range in second

## Some inferences about the dataset:

* not all samples have 3 LIDARS, all have LIDAR_TOP though
* example timestamp: 1546450430301986.8 is a UNix timestamp
* a quaternion
* each sample data has it's very own specific, caliberated_sensor and ego_pose records:
    * caliberated sensor has coord params w.r.t car's frame
    * ego_pose is car's coord params wrt to global cood system.
    * As the data is recorded w.r.t sensor's frame, we need to transform annotations to car's frame using caliberated sensor parameters, then to global frame using ego_pose paramters.
* The lidar pointcloud is defined in the sensor's reference frame.
* Boxes are in global frame.




## second.pytorch

* sudo apt install libsparsehash-dev
* install spconv, follow each instructions download boost headers

*NOTE* pass full path for root_path etc, it's being saved in the pickle
* python create_data.py nuscenes_data_prep --root_path=../../data/nuscenes/v1.0-mini  --version="v1.0-mini" --dataset_name="NuScenesDataset" --max_sweeps=10
* python create_data.py nuscenes_data_prep --root_path=/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/train/  --version="v1.0-trainval" --dataset_name="NuScenesDataset" --max_sweeps=10
* python create_data.py nuscenes_data_prep --root_path=/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/test/  --version="v1.0-test" --dataset_name="NuScenesDataset" --max_sweeps=10

* there's a concept of velocity in nuscenes, but not in lyft.

python -W ignore ./pytorch/train.py train --config_path=./configs/nuscenes/all.fhd.config --model_dir=/home/ags/second_test/all_fhd/
python -W ignore ./pytorch/train.py train --config_path=./configs/nuscenes/all.fhd.config --model_dir=/home/ags/second_test/test/

spconv issue on compute-01
https://github.com/traveller59/spconv/issues/78

* Installing nuscenes devkit 1.0.1
* quaternion, to be watched: https://www.youtube.com/watch?v=q-ESzg03mQc


### distribution of train set:

animal n=  186, width= 0.36±0.12, len= 0.73±0.19, height= 0.51±0.16, lw_aspect= 2.16±0.56
bicycle n=20928, width= 0.63±0.24, len= 1.76±0.29, height= 1.44±0.37, lw_aspect= 3.20±1.17
bus n= 8729, width= 2.96±0.24, len=12.34±3.41, height= 3.44±0.31, lw_aspect= 4.17±1.10
car n=534911, width= 1.93±0.16, len= 4.76±0.53, height= 1.72±0.24, lw_aspect= 2.47±0.22
emergency_vehicle n= 132, width= 2.45±0.43, len= 6.52±1.44, height= 2.39±0.59, lw_aspect= 2.66±0.28
motorcycle n=818, width= 0.96±0.20, len= 2.35±0.22, height= 1.59±0.16, lw_aspect= 2.53±0.50
other_vehicle n=33376, width= 2.79±0.30, len= 8.20±1.71, height= 3.23±0.50, lw_aspect= 2.93±0.53
pedestrian n=24935, width= 0.77±0.14, len= 0.81±0.17, height= 1.78±0.16, lw_aspect= 1.06±0.20
truck n=14164, width= 2.84±0.32, len=10.24±4.09, height= 3.44±0.62, lw_aspect= 3.56±1.25

animal   n=186
bicycle    n=20928
bus     n=8729
car    n=534911
emergency_vehicle   n=132
motorcycle      n=818
other_vehicle   n=33376
pedestrian  n=24935
truck   n=14164


*question* what's the point of such high nms iou threshold?

* `all.fhd.config.2`:
    * after 57000 iters, mean_ap: 0.28, LB: 0.061, Overall mAP: 0.02829285500610297 why??
    *  "mean_dist_aps": {
    "car": 0.8747493298150273,
    "pedestrian": 0.6251794869256213,
    "animal": 0.0,
    "other_vehicle": 0.5018276735532871,
    "bus": 0.07829943320585948,
    "motorcycle": 0.0,
    "truck": 0.1466217750538199,
    "emergency_vehicle": 0.0,
    "bicycle": 0.3598213471800674
  },
  "mean_ap": 0.28738878285929803,

    * zero mAP for animal (186), motorcycle(818), emergency_vehicle (132)
    * poor perfomance for bus [0.07], truck [0.14]
    *emergency_vehicle gt/pred: 0 0*
    *try with low nms iou th*


## 4 oct
`all.pp.lowa.config4`: 140K iters
val stats:
class: car: 0.26481168914758046
class: bicycle: 0.026110721574923158
class: animal: 0.0
class: bus: 0.08244517085140785
class: emergency_vehicle: 0.0
class: other_vehicle: 0.19598733527638573
class: motorcycle: 0.00706896551724138
class: pedestrian: 0.01484931969844282
class: truck: 0.11083860627446822

Overall mAP: 0.07801242314893884

train set:

car gt/pred: 441547 335601
bicycle gt/pred: 17979 12382
animal gt/pred: 114 0
bus gt/pred: 6596 4650
emergency_vehicle gt/pred: 132 0
other_vehicle gt/pred: 26460 13306
motorcycle gt/pred: 673 416
pedestrian gt/pred: 21059 17171
truck gt/pred: 12091 10068

class: car: 0.2728504228652352
class: bicycle: 0.02887044836895019
class: animal: 0.0
class: bus: 0.20561702808209564
class: emergency_vehicle: 0.0
class: other_vehicle: 0.21951157916481806
class: motorcycle: 0.01706045714854739
class: pedestrian: 0.013856075303431978
class: truck: 0.1901815495173459

Overall mAP: 0.10532750671671381
evan in train set model is completely failing to detect any animal or emergency vehicle.


*NOTE* One thing to note is that so far the eval code is using offcial nuScenes evaluation metric, which is based on different distance thresholds. which has little sense for this competition which uses iou thresholds.

*question* how's results.pkl different from results_nusc.json?: results.pkl: raw detections, results_nusc.json: boxes in global frame.

results.pkl is raw detections (net(examples) generated during eval process, while results_nusc.json contains same detections in global frame.
So, how are we gonna add lyft mAP in here?, simple we have raw detections, we already have the pipeline to get them to global frame, save them in lyft mAP required json format, for ground truth we have val_info.pkl, save it as required by lyft mAP algo. NO: val_info.pkl does contain gt_boxes but in lidar's frame

TODO:
* create a desent train/val split. DONE
* add lyft mAP evaluation code. DONE
* understand the epoch-end log during training.
* multi-sweep inference: DONE
* try submitting with scr th: 0.5: DONE 0.3 gives low score.


190k: CV: 0.704, LB: 0.09

after 190k, added all classes in data sampler.
remember: for given num of TPs, low FPs-> high precision -> high LB


### 4 Oct
http://www.cvlibs.net/datasets/kitti/eval_tracking.php : KITTI Benchmark, many open source code can be found here

*IDEA*:

What about pretraining on the full Nuscenes dataset?
I'm downloading lidar-only dataset from nuscenes, will account for atleast 130 GB.

Nuscenes already has 6 out of 9 classes of lyft. [doesn't have explicit "other_vehicle", emergency_vehicle and animal]
We can put its trailer and construction vehicle into other_vehicle category. As far as emergency vehicle and animal is concerned, there are mere 100 instances in lyft.
I'll download the dataset tonight, extract it. Free some space from internal disk. Transfer zip files to external drive.
Will be leaving for home tommorow, dussehra vacations. Will be back on wednesday.


### 12 Oct (Sat)

Thurs and Friday: no significant work.

I'm converting all of lyft dataset to KITTI format, with boxes in all directions. [code](https://github.com/stalkermustang/nuscenes-devkit)


python export_kitti.py nuscenes_gt_to_kitti --lyft_dataroot /media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/train/ --table_folder /media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/train/data/ --parallel_n_jobs 10 --get_all_detections True --store_dir /media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/KITTI/lyft/training/ --samples_count 10



### 13 Oct

Gotta train pp multi-head in background, meanwhile I'll read the point pillars paper today.


NOTE: target_assigner.classes are the order in which we have defined the class anchor props in config file.

python -W ignore ./pytorch/train.py train --config_path=./configs/nuscenes/all.pp.lowa.config.17 --model_dir=/home/ags/second_test/all.pp.lowa.config.4














# Revelations

* The `results.pkl` generated during training are more or less same as predictions made by this kernel
* my pred boxes to Box conversion code is perfect :D, because gt_boxes matches with lyft.get_boxes()

 What next?
* analyze your submission script for val set, -> there's some bug when translating the boxes to global frame, their lidar frame looks good.

the results_nusc.json contains val set predictions for all the sample_tokens, with boxes in global FoR, but this file is deleted (data/nuscene_dataset.py, L407) after evaluation, because of its large size (~300 MB)

* We use such a high IoU threshold in NMS because we are not 100% perfect in predicting boxes for a nearby object we may have a prediction having an overlap with the another prediction.










## New files/folders

* data/nusc_kitti/mini_train : kitti format of data/nuscene/v1.0-mini/ created using export_kitti.py
* data/nusc_kitti/mini_val : kitti format of data/nuscene/v1.0-mini/ created using export_kitti.py
* data/lyft/train/v1.0-trainval: a soft link to data/lyft/train/data, for nuscenes devkit.



  `python export_kitti.py nuscenes_gt_to_kitti --nusc_data_root data/nuscenes/v1.0-mini --nusc_kitti_dir data/nusc_kitti`
  `python export_kitti.py nuscenes_gt_to_kitti --nusc_data_root data/lyft/v1.0-mini --nusc_kitti_dir data/nusc_kitti --split mini_val`
`python export_kitti.py nuscenes_gt_to_kitti --nusc_data_root data/lyft/lyft_trainval --nusc_kitti_dir data/nusc_kitti --nusc_version v1.0-trainval`

python ./pytorch/train.py evaluate --config_path=./configs/nuscenes/all.fhd.config --model_dir=/home/ags/second_test/all_fhd_2/ --measure_time=True --batch_size=2



https://github.com/kuixu/kitti_object_vis


python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --ind 1 -d ../data/nusc_kitti/mini/KITTI/object



# Imp links
https://www.reddit.com/r/india/comments/d7b02t/finally_got_rid_of_dandruff_after_freaking_10/

https://github.com/traveller59/second.pytorch



# Dataset:

### NuScenes:
https://github.com/nutonomy/nuscenes-devkit

In March 2019, we released the full nuScenes dataset with all 1,000 scenes. The full dataset includes approximately 1.4M camera images, 390k LIDAR sweeps, 1.4M RADAR sweeps and 1.4M object bounding boxes in 40k keyframes. Additional features (map layers, raw sensor data, etc.) will follow soon. We are also organizing the nuScenes 3D detection challenge as part of the Workshop on Autonomous Driving at CVPR 2019.

* Concept of sweeps: the readme says sweeps have no annotations, which doesn't make sense to me as it looks like they indeed have it.


### KITTI:
    * In z-direction, the object coordinate system is located at the bottom of the object (contact point with the supporting surface).
    * x-length-front, y-width-left, z-height-up.


### nuscenes
    * lidar's frame: x-right, y-front, z-up
    * 90 degrees clockwise to kitti lidar frame, that's why



### Translations:

ego_pose:
    * position and rotation of ego vehicle in global coordinate system.
    * schema:
        "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0.
        "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
    * the transformation you need to make in order to bring a point in global coordinate to ego vehicle's frame of reference.

*question*, is ego_pose_token for all sensors on a ego vehicle same? technically it should be.
sensor_calibration:
    * schema
       "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z.
       "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
       "camera_intrinsic":        <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not

    * the tranformation you need to make in order to bring a point in ego vehicle's frame of reference to sensor's frame of reference.
    *
sample_annotation

   "translation":             <float> [3] -- Bounding box location in meters as center_x, center_y, center_z.
   "size":                    <float> [3] -- Bounding box size in meters as width, length, height.
   "rotation":                <float> [4] -- Bounding box orientation as quaternion: w, x, y, z.


## Extras

* gd in vim: go to definition
* pdb `s` to step through code




## PointRCNN notes:

KittiDataset takes in root_dir, split
root_dir is dir in which KITTI folder is present
data/
    KITTT/
        ImageSets/
            train.txt
            val.txt
        objects
            training
                image_2/
                label_2/
                calib_2/
                velodyne/
            testing


split can be train/val/
{split}.txt is selected
* so we should have all labelled data in `training`, prepare a val.txt
* 100000 - 120000 points in lidar cloud in org kitti dataset.: NUM_POINTS: 17000: ~20%
* ~34700 in nuscenes mini
* 67409 in lyft -> 10000
* np.fromfile, arg dtype is v.v. imp
*
Currently, the two stages of PointRCNN are trained separately.
Firstly, to use the ground truth sampling data augmentation for training, we should generate the ground truth database as follows:
* python generate_gt_database.py --class_name 'car' --split train
gt_database/train_gt_database_3level_car.pkl file generated


* To train the first proposal generation stage of PointRCNN with a single GPU, run the following command:

python train_rcnn.py --cfg_file cfgs/nuscene_mini.yaml --batch_size 16 --train_mode rpn --epochs 200


* What's the number of classes in org KITTI dataset? not much, the kitti benchmark is seperate for each class so people don't care about overall mAP here


python train_rcnn.py --cfg_file cfgs/nuscene_mini.yaml --batch_size 4 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/nuscene_mini/ckpt/checkpoint_epoch_200.pth --train_with_eval True

python eval_rcnn.py --cfg_file cfgs/nuscene_mini.yaml --ckpt  ../output/rcnn/nuscene_mini/ckpt/checkpoint_epoch_50.pth --batch_size 1 --eval_mode rcnn --set RPN.LOC_XZ_FINE False

python eval_rcnn.py --cfg_file cfgs/nuscene_mini.yaml --ckpt  ../output/rcnn/nuscene_mini/ckpt/checkpoint_epoch_50.pth --batch_size 1 --eval_mode rcnn



https://medium.com/the-artificial-impostor/use-nvidia-apex-for-easy-mixed-precision-training-in-pytorch-46841c6eed8c

python train_rcnn.py --cfg_file cfgs/lyft_13_test.yaml --batch_size 16 --train_mode rpn --epochs 200

## 13 oct

`lyft_13_test.yaml`: converted lyft-> kitti using stalker's script. Training on num_points 7000



python eval.py --det_file /home/ags/second_test/all.pp.lowa.config.4/detections/voxelnet-313529_train.pkl  --phase train




kittiviewer:

NuScenesDataset
/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/test/infos_test.pkl
/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/test/


/home/ags/second_test/all_fhd/results/step_29325/result.pkl
/home/ags/second_test/all_fhd/voxelnet-29369.tckpt

/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/second.pytorch/second/configs/nuscenes/all.fhd.config


## Experts speack

[here](https://github.com/traveller59/second.pytorch/issues/146#issuecomment-482415409) some important parameters: detection range (especially z range), nms parameters. The major problem of nuscenes AP is too much false negatives. increase det range and modify nms param can decrease false negatives and greatly increase AP performance.

 [here](https://github.com/traveller59/second.pytorch/issues/197) I have tested the simple-inference.ipynb, you need to use dataset instance to get correct point cloud (with 10 sweeps) and increase score threshold to get reasonable result on bev image. don't need to modify other parts of simple-inference.ipynb.




## Revelations:
* We gotta maintain max_voxel ratio for eval also.
*




Cuda error in file '/home/ags/sw/spconv/src/cuhash/hash_table.cpp' in line 123 : an illegal memory access was encountered.


## Creating lyft mini dataset

* NuScenes mini:
    Loading NuScenes tables for version v1.0-mini...
    23 category,
    8 attribute,
    4 visibility,
    911 instance,
    12 sensor,
    120 calibrated_sensor,
    31206 ego_pose,
    8 log,
    10 scene,
    404 sample,
    31206 sample_data,
    18538 sample_annotation,
    4 map,

* lyft overall:
    9 category,
    18 attribute,
    4 visibility,
    18421 instance,
    10 sensor,
    148 calibrated_sensor,
    177789 ego_pose,
    180 log,
    180 scene,
    22680 sample,
    189504 sample_data,
    638179 sample_annotation,
    1 map,

* expected lyft mini

    9 category,


fuck this shit.



## Questions:

1. a Box is defined by center (translation) and size (wlh), then where does the quaternion (rotation) come into the picture, only during plotting?
2.






## Data directory:

/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data

data
---- KITTI
    ---- KITTI
    ---- lyft
        ---- ImageSets
            ---- trainval.txt
            ---- train.txt
            ---- val.txt
        ---- object
            ---- calib
            ---- label_2
            ---- image_2
    ---- nuscenes
    ---- test
