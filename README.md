# Lyft 3D object detection


## Approaches and Experiments:

1. Run PointRCNN on nuscenes-mini, converted to kiiti.
2. Run SECOND on nuscenes-mini,

## Notes:





## things to understand:

[x] the nuscenes/lyft sdk
[x] the nuscenes/lyft dataset scema
[] the KITTI dataset scema
[] the annotations, understanding them

## Ideas:

* What about a kaggle kernel for Boxes only?
* A kernel for evaluation metric?


## Questions?

* .get_boxes says sample_data can be keyframe or intermediate sample_data, so key_frame=True sample_data are those which are in first sample_data
 For sample_data with is_key_frame=True, the time-stamps should be very close to the sample it points to. For non key-frames the sample_data points to the sample that follows closest in time. *still not able to understand*
Okay, so for key_frame=True, sample_data timestamps is very close to sample timestamp, not the same for non keyframe sample_data point.
*ALRIGHT*, so keyframes are those which pretty much match to annotations, but we haven'


* what are translation, size, rotation in a annotation? like how far its location  from the sample_data's sensor?
* height, width and length, which axes do these correspond to?

## Some inferences about the dataset:

* not all samples have 3 LIDARS, all have LIDAR_TOP though
* example timestamp: 1546450430301986.8 is a UNix timestamp
* a quaternion
* each sample data has it's very own specific, caliberated_sensor and ego_pose records:
    * caliberated sensor has coord params w.r.t car's frame
    * ego_pose is car's coord params wrt to global cood system.
    * As the data is recorded w.r.t sensor's frame, we need to transform annotations to car's frame using caliberated sensor parameters, then to global frame using ego_pose paramters.
* The lidar pointcloud is defined in the sensor's reference frame.
*


## second.pytorch
* `secondenv` for second code. python 3.7
* sudo apt install libsparsehash-dev
* install spconv, follow each instructions download boost headers
* Add numba paths and second directory to pythonpath in bashrc
* python create_data.py nuscenes_data_prep --root_path=../../data/nuscenes/v1.0-mini  --version="v1.0-mini" --dataset_name="NuScenesDataset" --max_sweeps=10
* python create_data.py nuscenes_data_prep --root_path=../../data/lyft/train/  --version="v1.0-trainval" --dataset_name="NuScenesDataset" --max_sweeps=10
* there's a concept of velocity in nuscenes, but not  in lyft.

python ./pytorch/train.py train --config_path=./configs/nuscenes/all.fhd.config --model_dir=/home/ags/second_test/all_fhd/

* Installing nuscenes devkit 1.0.1
* create_data logs for lyft:
    total scene num: 180
    exist scene num: 180
    train scene: 147, val scene: 32
    [100.0%][===================>][87.66it/s][04:37>00:00]
    train sample: 18522, val sample: 4158
    [100.0%][===================>][1.87it/s][01:30:40>00:00]
    load 434347 car database infos
    load 11411 truck database infos
    load 21476 pedestrian database infos
    load 18061 bicycle database infos
    load 652 motor`

first log:
WORKER 3 seed: 1569564604
runtime.step=50, runtime.steptime=1.525, runtime.voxel_gene_time=0.03066, runtime.prep_time=0.3198, loss.cls_loss=1.223e+03, loss.cls_loss_rt=362.1, loss.loc_loss=4.409, loss.loc_loss_rt=2.026, loss.loc_elem=[0.1201, 0.07527, 0.3091, 0.1961, 0.1621, 0.07491, 0.07517], loss.cls_pos_rt=323.7, loss.cls_neg_rt=38.32, loss.dir_rt=0.7188, rpn_acc=0.542,  pr.prec@10=0.0006639, pr.rec@10=1.0, pr.prec@30=0.0006639, pr.rec@30=0.9999, pr.prec@50=0.0005329, pr.rec@50=0.3673, pr.prec@70=0.008208, pr.rec@70=0.02797, pr.prec@80=0.009776, pr.rec@80=0.02017, pr.prec@90=0.01083, pr.rec@90=0.01346, pr.prec@95=0.0115, pr.rec@95=0.00999, misc.num_vox=114696, misc.num_pos=175, misc.num_neg=199594, misc.num_anchors=199888, misc.lr=0.0003, mis`


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

#how render_annotation works:

* take ann_token, get ann record
* get the corresponding sample record
* make sure LIDAR_TOP is present
* get the camera token in which the annotation is present
* get LIDAR_TOP token
* use get_sample_data to get lidar data path, boxes, and something called camera_intrinsic (?)
* use LidarPointCloud to render lidar
* An annotation can have multiple boxes? no, shouldn't have for loop for the single element present in the `boxes` list
* box.render to render the box (gotta take a look at the Box class)
* `view_points` function?, get the corners








# Nuscenes dataset:

https://github.com/nutonomy/nuscenes-devkit

In March 2019, we released the full nuScenes dataset with all 1,000 scenes. The full dataset includes approximately 1.4M camera images, 390k LIDAR sweeps, 1.4M RADAR sweeps and 1.4M object bounding boxes in 40k keyframes. Additional features (map layers, raw sensor data, etc.) will follow soon. We are also organizing the nuScenes 3D detection challenge as part of the Workshop on Autonomous Driving at CVPR 2019.


* Concept of sweeps: the readme says sweeps have no annotations, which doesn't make sense to me as it looks like they indeed have it.




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
* 67409 in lyft
* np.fromfile, arg dtype is v.v. imp
*
Currently, the two stages of PointRCNN are trained separately.
Firstly, to use the ground truth sampling data augmentation for training, we should generate the ground truth database as follows:
* python generate_gt_database.py --class_name 'car' --split train
gt_database/train_gt_database_3level_car.pkl file generated


* To train the first proposal generation stage of PointRCNN with a single GPU, run the following command:

python train_rcnn.py --cfg_file cfgs/nuscene_mini.yaml --batch_size 16 --train_mode rpn --epochs 200


* What's the number of classes in org KITTI dataset?


python train_rcnn.py --cfg_file cfgs/nuscene_mini.yaml --batch_size 4 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/nuscene_mini/ckpt/checkpoint_epoch_200.pth --train_with_eval True

python eval_rcnn.py --cfg_file cfgs/nuscene_mini.yaml --ckpt  ../output/rcnn/nuscene_mini/ckpt/checkpoint_epoch_50.pth --batch_size 1 --eval_mode rcnn --set RPN.LOC_XZ_FINE False

python eval_rcnn.py --cfg_file cfgs/nuscene_mini.yaml --ckpt  ../output/rcnn/nuscene_mini/ckpt/checkpoint_epoch_50.pth --batch_size 1 --eval_mode rcnn



https://medium.com/the-artificial-impostor/use-nvidia-apex-for-easy-mixed-precision-training-in-pytorch-46841c6eed8c


* KITTI:
    * In z-direction, the object coordinate system is located at the bottom of the object (contact point with the supporting surface).
    * x-length-front, y-width-left, z-height-up.
    *














## Translations:

ego_pose:
    * position and rotation of ego vehicle in global coordinate system.
    * schema:
        "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0.
        "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
    * the transformation you need to make in order to bring a point in global coordinate to ego vehicle's frame of reference.

sensor_calibration:
    * schema
       "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z.
       "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
       "camera_intrinsic":        <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not

    * the tranformation you need to make in order to bring a point in ego vehicle's frame of reference to sensor's frame of reference.
