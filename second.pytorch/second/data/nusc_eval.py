import fire
from pathlib import Path
import numpy as np
from numba import jit
from multiprocessing import Process

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from lyft_dataset_sdk.eval.detection.mAP_evaluation import *



#@jit(nopython=True)
def get_ap(gt, predictions, class_names, iou_threshold, output_dir):
    ap = get_average_precisions(gt, predictions, class_names, iou_threshold)
    metric = {c:ap[idx] for idx, c in enumerate(class_names)}
    summary_path = output_dir / f'metric_summary_{iou_threshold}.json'
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f)


def get_metric_overall_ap(iou_th_range, output_dir, class_names):
    metric = {}
    overall_ap = np.zeros(len(class_names))
    for iou_threshold in iou_th_range:
        summary_path = output_dir / f'metric_summary_{iou_threshold}.json'
        #import pdb; pdb.set_trace()
        with open(str(summary_path), 'r') as f:
            data = json.load(f) # type(data): dict
            metric[iou_threshold] = data
            overall_ap += np.array([data[c] for c in class_names])
        summary_path.unlink() # delete this file
    overall_ap /= len(iou_th_range)
    return metric, overall_ap


def eval_main(gt_file_path, pred_file_path, output_dir):
    gt_path = Path(gt_file_path)
    pred_path = Path(pred_file_path)
    output_dir = Path(output_dir)

    with open(str(pred_path)) as f:
        predictions = json.load(f)

    with open(str(gt_path)) as f:
        gt = json.load(f)


    class_names = get_class_names(gt)
    #print("Class_names = ", class_names)
    iou_th_range = np.linspace(0.5, 0.95, 10) # 0.5, 0.55, ..., 0.90, 0.95

    metric = {}

    processes = []
    for iou_threshold in iou_th_range:
        process = Process(target=get_ap, args=(gt, predictions, class_names, iou_threshold, output_dir))
        process.start()
        #process.join()
        processes.append(process)

    for process in processes:
        process.join()

    metric, overall_ap = get_metric_overall_ap(iou_th_range, output_dir, class_names)

    #print('Overall average precisions')
    #for name, AP in sorted(list(zip(class_names, overall_ap.flatten().tolist()))):
    #    print(name, AP)

    mAP = np.mean(overall_ap)
    metric['overall'] = {'mAP': mAP}
    #print("Average per class mean average precision = ", mAP)

    summary_path = Path(output_dir) / 'metric_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f, indent=4)



def eval_main_old(root_path, version, eval_version, res_path, eval_set, output_dir):
    #import pdb; pdb.set_trace()
    nusc = NuScenes(version=version, dataroot=str(root_path), verbose=False)

    cfg = config_factory(eval_version)
    nusc_eval = NuScenesEval(nusc, config=cfg, result_path=res_path, eval_set=eval_set,
                            output_dir=output_dir,
                            verbose=False)
    nusc_eval.main(render_curves=False)

if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    fire.Fire(eval_main)


'''
example command

python /media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/second.pytorch/second/data/nusc_eval.py --root_path="/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/train" --version=v1.0-trainval --eval_version=cvpr_2019 --res_path="/home/ags/second_test/all_fhd/results/step_5865/results_nusc.json" --eval_set=val --output_dir="/home/ags/second_test/all_fhd/results/step_5865"

python /media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/second.pytorch/second/data/nusc_eval.py --gt_file_path="root_path="/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/train/gt_data_val.json" --pred_file_path="/home/ags/second_test/all_fhd/results/step_5865/pred_data_val.json" --eval_set=val --output_dir="/home/ags/second_test/all_fhd/results/step_5865"


'''
