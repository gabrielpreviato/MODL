import cv2
import csv
import numpy as np
import lib.EvaluationUtils as EvaluationUtils
from config import get_config
import os
from glob import glob

from lib.Classes import Classes, Nothing
from lib.Evaluators import JMOD2Stats

import sklearn.metrics
import matplotlib.pyplot as plt

# python evaluate_on_soccerfield.py --data_set_dir /data --data_train_dirs 09_D --data_test_dirs 09_D --is_train False --dataset Soccer --is_deploy False --weights_path weights/nt-180-0.02.hdf5 --resume_training True
from lib.SampleType import DepthObstacles_SingleFrame_Multiclass_4


def preprocess_data_sqpr(rgb, w=256, h=160):
    rgb = cv2.resize(rgb, (w, h), cv2.INTER_LINEAR)

    rgb = np.asarray(rgb, dtype=np.float32) / 255.

    rgb = np.expand_dims(rgb, 0)

    return rgb


def preprocess_data(rgb, gt, seg, w=256, h=160, crop_w=0, crop_h=0, resize_only_rgb = False):
    crop_top = np.floor((rgb.shape[0] - crop_h) / 2).astype(np.uint8)
    crop_bottom = rgb.shape[0] - np.floor((rgb.shape[0] - crop_h) / 2).astype(np.uint8)
    crop_left = np.floor((rgb.shape[1] - crop_w) / 2).astype(np.uint8)
    crop_right = rgb.shape[1] - np.floor((rgb.shape[1] - crop_w) / 2).astype(np.uint8)

    rgb = np.asarray(rgb, dtype=np.float32) / 255.
    rgb = cv2.resize(rgb, (w, h), cv2.INTER_LINEAR)
    rgb = np.expand_dims(rgb, 0)
    gt = np.asarray(gt, dtype=np.float32)

    if not resize_only_rgb:
        gt = cv2.resize(gt, (w, h), cv2.INTER_NEAREST)
    gt = EvaluationUtils.depth_to_meters_airsim(gt)
    if not resize_only_rgb:
        seg = cv2.resize(seg, (w, h), cv2.INTER_NEAREST)
    return rgb, gt, seg

def read_labels_gt_viewer(obstacles_gt):
    with open(obstacles_gt, 'r') as f:
        obstacles = f.readlines()
    obstacles = [x.strip() for x in obstacles]

    labels = []

    for obs in obstacles:
        parsed_str_obs = obs.split(" ")
        parsed_obs = np.zeros(shape=(8))
        i = 0
        for n in parsed_str_obs:
            if i < 2:
                parsed_obs[i] = int(n)
            else:
                parsed_obs[i] = float(n)
            i += 1

        x = int(parsed_obs[0]*32 + parsed_obs[2]*32)
        y = int(parsed_obs[1]*32 + parsed_obs[3]*32)
        w = int(parsed_obs[4]*256)
        h = int(parsed_obs[5]*160)

        object = [[x - w/2, y - h/2, w, h],
                  [parsed_obs[6], parsed_obs[7]]
                  ]
        labels.append(object)


    return labels


def read_labels_gt_viewer_multiclass(obstacles_gt, number_classes):
    with open(obstacles_gt, 'r') as f:
        obstacles = f.readlines()
    obstacles = [x.strip() for x in obstacles]

    labels = []

    for obs in obstacles:
        parsed_str_obs = obs.split(" ")
        parsed_obs = np.zeros(shape=9)
        i = 0
        for n in parsed_str_obs:
            if i < 2:
                parsed_obs[i] = int(n)
            elif i == 8:
                if (number_classes == 2 or number_classes == 3) and (n == 'robot_team' or n == 'robot_opponent'):
                    n = 'robot'
                elif number_classes == 2 and n == 'goal':
                    n = 'nothing'

                parsed_obs[i] = Classes.str_to_class_enum(n)
            else:
                parsed_obs[i] = float(n)
            i += 1

        x = int(parsed_obs[0]*32 + parsed_obs[2]*32)
        y = int(parsed_obs[1]*32 + parsed_obs[3]*32)
        w = int(parsed_obs[4]*256)
        h = int(parsed_obs[5]*160)

        object = [[x - w/2, y - h/2, w, h],
                  [parsed_obs[6], parsed_obs[7]],
                  parsed_obs[8]
                  ]

        # Object with last value equal to -1 is a dispensable object
        if object[-1] != -1:
            labels.append(object)

    return labels


def read_labels_gt_viewer_multiclass_2(obstacles_gt):
    with open(obstacles_gt, 'r') as f:
        obstacles = f.readlines()
    obstacles = [x.strip() for x in obstacles]

    labels = []

    for obs in obstacles:
        parsed_str_obs = obs.split(" ")
        parsed_obs = np.zeros(shape=9)
        i = 0
        for n in parsed_str_obs:
            if i < 2:
                parsed_obs[i] = int(n)
            elif i == 8:
                if n == 'robot_team' or n == 'robot_opponent':
                    n = 'robot'

                parsed_obs[i] = Classes.str_to_class_enum(n)
            else:
                parsed_obs[i] = float(n)
            i += 1

        if parsed_obs[8] == 2:
            continue

        x = int(parsed_obs[0]*32 + parsed_obs[2]*32)
        y = int(parsed_obs[1]*32 + parsed_obs[3]*32)
        w = int(parsed_obs[4]*256)
        h = int(parsed_obs[5]*160)

        object = [[x - w/2, y - h/2, w, h],
                  [parsed_obs[6], parsed_obs[7]],
                  parsed_obs[8]
                  ]
        labels.append(object)


    return labels


def labels_from_file_multiclass_4(obstacles_gt):
    with open(obstacles_gt, 'r') as f:
        obstacles = f.readlines()
    obstacles = [x.strip() for x in obstacles]

    obstacles_label = np.zeros(shape=(5, 8, 10))

    for obs_ in obstacles:
        parsed_str_obs = obs_.split(" ")
        parsed_obs = np.zeros(shape=9)
        i_ = 0
        for n in parsed_str_obs:
            if i_ < 2:
                parsed_obs[i_] = int(n)
            elif i_ == 8:
                parsed_obs[i_] = Classes.str_to_class_enum(n)
            else:
                parsed_obs[i_] = float(n)
            i_ += 1

        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 0] = 1.0 if parsed_obs[8] == 3 else 0.0  # class 3
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 1] = 1.0 if parsed_obs[8] == 4 else 0.0  # class 4
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 2] = 1.0 if parsed_obs[8] == 1 else 0.0  # class 1
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 3] = 1.0 if parsed_obs[8] == 2 else 0.0  # class 2
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 4] = parsed_obs[2]  # x
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 5] = parsed_obs[3]  # y
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 6] = parsed_obs[4]  # w
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 7] = parsed_obs[5]  # h
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 8] = parsed_obs[6] * 0.1  # m
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 9] = parsed_obs[7] * 0.1  # v
    labels = np.reshape(obstacles_label, (40, 10))

    return labels


def labels_from_file_multiclass_3(obstacles_gt):
    with open(obstacles_gt, 'r') as f:
        obstacles = f.readlines()
    obstacles = [x.strip() for x in obstacles]

    obstacles_label = np.zeros(shape=(5, 8, 9))

    for obs_ in obstacles:
        parsed_str_obs = obs_.split(" ")
        parsed_obs = np.zeros(shape=9)
        i_ = 0
        for n in parsed_str_obs:
            if i_ < 2:
                parsed_obs[i_] = int(n)
            elif i_ == 8:
                if n == 'robot_team' or n == 'robot_opponent':
                    n = 'robot'
                parsed_obs[i_] = Classes.str_to_class_enum(n)
            else:
                parsed_obs[i_] = float(n)
            i_ += 1

        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 0] = 1.0 if parsed_obs[8] == 0 else 0.0  # class 0
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 1] = 1.0 if parsed_obs[8] == 1 else 0.0  # class 1
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 2] = 1.0 if parsed_obs[8] == 2 else 0.0  # class 2
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 3] = parsed_obs[2]  # x
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 4] = parsed_obs[3]  # y
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 5] = parsed_obs[4]  # w
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 6] = parsed_obs[5]  # h
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 7] = parsed_obs[6] * 0.1  # m
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 8] = parsed_obs[7] * 0.1  # v
    labels = np.reshape(obstacles_label, (40, 9))

    return labels

def labels_from_file_multiclass_2(obstacles_gt):
    with open(obstacles_gt, 'r') as f:
        obstacles = f.readlines()
    obstacles = [x.strip() for x in obstacles]

    obstacles_label = np.zeros(shape=(5, 8, 8))

    for obs_ in obstacles:
        parsed_str_obs = obs_.split(" ")
        parsed_obs = np.zeros(shape=9)
        i_ = 0
        for n in parsed_str_obs:
            if i_ < 2:
                parsed_obs[i_] = int(n)
            elif i_ == 8:
                if n == 'robot_team' or n == 'robot_opponent':
                    n = 'robot'
                elif n == 'goal':
                    n = 'nothing'
                parsed_obs[i_] = Classes.str_to_class_enum(n)
            else:
                parsed_obs[i_] = float(n)
            i_ += 1

        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 0] = 1.0 if parsed_obs[8] == 0 else 0.0  # class 0
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 1] = 1.0 if parsed_obs[8] == 1 else 0.0  # class 1
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 2] = parsed_obs[2]  # x
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 3] = parsed_obs[3]  # y
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 4] = parsed_obs[4]  # w
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 5] = parsed_obs[5]  # h
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 6] = parsed_obs[6] * 0.1  # m
        obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 7] = parsed_obs[7] * 0.1  # v
    labels = np.reshape(obstacles_label, (40, 8))

    return labels

#edit config.py as required
config, unparsed = get_config()

#Edit model_name to choose model between ['jmod2','cadena','detector','depth','eigen']
model_name = 'modl'
number_classes = config.number_classes

#model, detector_only = EvaluationUtils.load_model(model_name, config, number_classes)

showImages = True

dataset_main_dir = config.data_set_dir
test_dirs = config.data_test_dirs

#compute_depth_branch_stats_on_obs is set to False when evaluating detector-only models
#jmod2_stats = JMOD2Stats(model_name, compute_depth_branch_stats_on_obs=not detector_only)

i = 0

confMatrix = True
true_obs = []
pred_obs = []
conf_mat = np.zeros((number_classes + 1, number_classes + 1), dtype=int)

annotations = {}

with open("/home/previato/Dropbox/IC/dataset/SPQR_Dataset/obstacles_10m/annotations.txt", "r") as annotations_file:
    csv_file = csv.reader(annotations_file, delimiter=" ")

    for line in csv_file:
        if line[0] in annotations.keys():
            annotations[line[0]].append(line[1:])
        else:
            annotations[line[0]] = [line[1:]]


def spqr_str_to_class(string):
    if string == 'nao':
        return 0
    elif string == 'ball':
        return 1
    elif string == 'goal':
        return 2
    else:
        return -1

SPQR_WIDTH = 640
SPQR_HEIGHT = 480


for test_dir in test_dirs:
    # depth_gt_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'depth', '*' + '.png')))
    rgb_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'rgb', '*' + '.png')))
    # seg_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'segmentation', '*' + '.png')))
    obs_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'obstacles_10m', '*' + '.txt')))

    print("Total images:", len(rgb_paths))

    for rgb_path, obs_path in zip(rgb_paths, obs_paths):
        if i % (int(len(rgb_paths) / 100)) == 0:
            print("Progress:", str(int(i / (int(len(rgb_paths) / 100)))) + '/100')
        annotation = annotations[rgb_path.split("/")[-1]]

        for obj in annotation:
            obj[0] = int(obj[0])
            obj[1] = int(obj[1])
            obj[2] = int(obj[2])
            obj[3] = int(obj[3])

        gt_obj_str = [obj[4] for obj in annotation]
        gt_obj_classes = [spqr_str_to_class(obj[4]) for obj in annotation]
        gt_obj_coords = [[int(obj[0]/SPQR_WIDTH*config.input_width), int(obj[1]/SPQR_HEIGHT*config.input_height),
                          int((obj[2] - obj[0])/SPQR_WIDTH*config.input_width), int((obj[3] - obj[1])/SPQR_HEIGHT*config.input_height)]for obj in annotation]
        gt_obj_depth = [[0, 0] for obj in annotation]

        obs = [list(a) for a in zip(gt_obj_coords, gt_obj_depth, gt_obj_classes)]
        if model_name == 'modl':
            for ob in obs:
                ob[2] = Classes.generate_class(ob[2])

            gt_obs = EvaluationUtils.get_obstacles_from_list_multiclass(obs)

        rgb_raw = cv2.imread(rgb_path)

        # obs_path = ''.join(rgb_path.split('.')[0:-1]) + '.txt'
        with open(obs_path, 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL, delimiter=' ')

            # for ob_str in gt_obj_str:
            #     class_str = 'robot' if ob_str == 'nao' else ob_str
            #     writer.writerow([0, 0, 0, 0, 0, 0, 0, 0, class_str])

            for j in range(len(gt_obj_str)):
                class_str = 'robot' if gt_obj_str[j] == 'nao' else gt_obj_str[j]
                w = gt_obj_coords[j][2]
                h = gt_obj_coords[j][3]
                x_c = gt_obj_coords[j][0] + w/2
                y_c = gt_obj_coords[j][1] + h/2
                writer.writerow([int(x_c / 32), int(y_c / 32), (x_c % 32) / 32, (y_c % 32) / 32, w/config.input_width, h/config.input_height, 0, 0, class_str])

        i += 1
