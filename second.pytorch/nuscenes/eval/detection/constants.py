# nuScenes dev-kit.
# Code written by Oscar Beijbom and Varun Bankiti, 2019.
# Licensed under the Creative Commons [see licence.txt]
DETECTION_NAMES = ['car', 'pedestrian', 'animal', 'other_vehicle', 'bus', 'motorcycle', 'truck',
        'emergency_vehicle', 'bicycle']

PRETTY_DETECTION_NAMES = {'car': 'car',
                          'pedestrian': 'pedestrian',
                          'animal': 'animal',
                          'other_vehicle': 'other_vehicle',
                          'bus': 'bus',
                          'motorcycle': 'motorcycle',
                          'truck': 'truck',
                          'emergency_vehicle': 'emergency_vehicle',
                          'bicycle': 'bicycle',
                          }

DETECTION_COLORS = {'car': 'C0',
                    'pedestrian': 'C1',
                    'animal': 'C2',
                    'other_vehicle': 'C3',
                    'bus': 'C4',
                    'motorcycle': 'C5',
                    'truck': 'C6',
                    'emergency_vehicle': 'C7',
                    'bicycle': 'C8',
                    }

ATTRIBUTE_NAMES = [
         'object_action_lane_change_right',
         'object_action_running',
         'object_action_lane_change_left',
         'object_action_parked',
         'object_action_standing',
         'object_action_right_turn',
         'object_action_gliding_on_wheels',
         'object_action_loss_of_control',
         'object_action_u_turn',
         'object_action_sitting',
         'object_action_walking',
         'object_action_stopped',
         'object_action_left_turn',
         'object_action_reversing',
         'is_stationary',
         'object_action_driving_straight_forward',
         'object_action_abnormal_or_traffic_violation',
         'object_action_other_motion'
 ]
PRETTY_ATTRIBUTE_NAMES = {x:x for x in ATTRIBUTE_NAMES}

'''
DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                   'traffic_cone', 'barrier']

PRETTY_DETECTION_NAMES = {'car': 'Car',
                          'truck': 'Truck',
                          'bus': 'Bus',
                          'trailer': 'Trailer',
                          'construction_vehicle': 'Constr. Veh.',
                          'pedestrian': 'Pedestrian',
                          'motorcycle': 'Motorcycle',
                          'bicycle': 'Bicycle',
                          'traffic_cone': 'Traffic Cone',
                          'barrier': 'Barrier'}

DETECTION_COLORS = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'construction_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    'traffic_cone': 'C8',
                    'barrier': 'C9'}

ATTRIBUTE_NAMES = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing', 'cycle.with_rider',
                   'cycle.without_rider', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']

PRETTY_ATTRIBUTE_NAMES = {'pedestrian.moving': 'Ped. Moving',
                          'pedestrian.sitting_lying_down': 'Ped. Sitting',
                          'pedestrian.standing': 'Ped. Standing',
                          'cycle.with_rider': 'Cycle w/ Rider',
                          'cycle.without_rider': 'Cycle w/o Rider',
                          'vehicle.moving': 'Veh. Moving',
                          'vehicle.parked': 'Veh. Parked',
                          'vehicle.stopped': 'Veh. Stopped'}
'''
TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

PRETTY_TP_METRICS = {'trans_err': 'Trans.', 'scale_err': 'Scale', 'orient_err': 'Orient.', 'vel_err': 'Vel.',
                     'attr_err': 'Attr.'}

TP_METRICS_UNITS = {'trans_err': 'm',
                    'scale_err': '1-IOU',
                    'orient_err': 'rad.',
                    'vel_err': 'm/s',
                    'attr_err': '1-acc.'}
