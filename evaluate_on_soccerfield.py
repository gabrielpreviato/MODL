import cv2
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

#Edit model_name to choose model between ['jmod2', 'modl']
model_name = 'modl'
number_classes = config.number_classes

model, detector_only = EvaluationUtils.load_model(model_name, config, number_classes)

showImages = True

dataset_main_dir = config.data_set_dir
test_dirs = config.data_test_dirs

#compute_depth_branch_stats_on_obs is set to False when evaluating detector-only models
jmod2_stats = JMOD2Stats(model_name, compute_depth_branch_stats_on_obs=not detector_only)

i = 0
total_images = 0

confMatrix = True
true_obs = []
pred_obs = []
conf_mat = np.zeros((number_classes + 1, number_classes + 1), dtype=int)

for test_dir in test_dirs:
    depth_gt_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'depth', '*' + '.png')))
    rgb_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'rgb', '*' + '.png')))
    seg_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'segmentation', '*' + '.png')))
    obs_paths = sorted(glob(os.path.join(dataset_main_dir, test_dir, 'obstacles_10m', '*' + '.txt')))

    for gt_path, rgb_path, seg_path, obs_path in zip(depth_gt_paths, rgb_paths, seg_paths, obs_paths):

        rgb_raw = cv2.imread(rgb_path)
        gt = cv2.imread(gt_path, 0)
        seg = cv2.imread(seg_path, 0)
        obs = []
        if model_name == 'modl':
            obs = read_labels_gt_viewer_multiclass(obs_path, number_classes)
        else:
            obs = read_labels_gt_viewer(obs_path)

        if len([a for a in obs if a[2] == 0]) < 1:
            total_images += 1
            continue

        #Normalize input between 0 and 1, resize if required

        rgb, gt, seg = preprocess_data(rgb_raw, gt, seg, w=config.input_width, h=config.input_height, resize_only_rgb = True)

        #Forward pass to the net
        results = model.run(rgb)

        if results[0] is not None:
            depth_raw = results[0].copy()
            pred_depth_ = results[0][0, :, :, 0].copy()
            depth_gt = gt.copy()
            #evaluate only on valid predictions (some methods like Cadena's may return zero or negative values)
            depth_gt[np.nonzero(pred_depth_ <= 0)] = 0.0
        else:
            depth_gt = None
        #Corrected depth
        if results[2] is not None:
            corr_depth = results[2][0, :, :, 0].copy()
            corr_depth[np.nonzero(depth_gt <= 0)] = 0.0
            results[2] = corr_depth

        #Get obstacles from GT segmentation and depth
        #gt_obs = EvaluationUtils.get_obstacles_from_seg_and_depth(gt, seg, segm_thr=-1)
        if model_name == 'modl':
            for ob in obs:
                ob[2] = Classes.generate_class(ob[2])

            gt_obs = EvaluationUtils.get_obstacles_from_list_multiclass(obs)
        else:
            gt_obs = EvaluationUtils.get_obstacles_from_list(obs)

        if confMatrix:
            obs_labels = []
            if number_classes == 2:
                obs_labels = labels_from_file_multiclass_2(obs_path)
            elif number_classes == 3:
                obs_labels = labels_from_file_multiclass_3(obs_path)
            elif number_classes == 4:
                obs_labels = labels_from_file_multiclass_4(obs_path)

            conf_list_pred = []
            conf_list_true = []
            local_conf_mat = None

            if number_classes == 2:
                conf_list_pred, conf_list_true = EvaluationUtils.confusion_list_multiclass_2(results[3], obs_labels)
                local_conf_mat = sklearn.metrics.confusion_matrix(conf_list_true, conf_list_pred,
                                                                  labels=["nothing", "ball", "robot"])
            elif number_classes == 3:
                conf_list_pred, conf_list_true = EvaluationUtils.confusion_list_multiclass_3(results[3], obs_labels)
                local_conf_mat = sklearn.metrics.confusion_matrix(conf_list_true, conf_list_pred,
                                                                  labels=["nothing", "goal", "ball", "robot"])
            elif number_classes == 4:
                conf_list_pred, conf_list_true = EvaluationUtils.confusion_list_multiclass_4(results[3], obs_labels)
                local_conf_mat = sklearn.metrics.confusion_matrix(conf_list_true, conf_list_pred,
                                                                  labels=["nothing", "goal", "ball", "robot_team", "robot_opponent"])

            conf_mat = np.add(conf_mat, local_conf_mat)

        if showImages:
            if results[1] is not None:
                if model_name == 'modl':
                    EvaluationUtils.show_detections_multiclass(rgb_raw, results[1], gt_obs, save=True, save_dir="save_"+str(number_classes)+"_real_1rob",
                                                    file_name="sav_" + str(i) + ".png", sleep_for=10, multiclass=number_classes)
                else:
                    EvaluationUtils.show_detections(rgb_raw, results[1], gt_obs, save=True, save_dir="save", file_name="sav_"+ str(i) +".png", sleep_for=10)
            if results[0] is not None:
                EvaluationUtils.show_depth(rgb_raw, depth_raw, gt, save=True, save_dir="save_"+str(number_classes)+"_real_1rob", file_name="sav_"+ str(i) +".png", sleep_for=10)

        jmod2_stats.run(results, [depth_gt, gt_obs])

        i += 1

# cv2.destroyWindow('Detections(RED:predictions,GREEN: GT')
# cv2.destroyWindow('Predicted Depth')
# cv2.destroyWindow('GT Depth')

results = jmod2_stats.return_results()
print("Total images:", total_images)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if confMatrix:
    # Normalize nothingXnothing in confusion matrix
    conf_mat[0][0] = conf_mat[0][0] / 40

    plot_confusion_matrix(conf_mat, classes=["nothing", "ball", "robot", "goal"],
                          normalize=False, title=' ')

    plt.savefig('confusion_matrix_' + str(number_classes) + '_real_1rob.png', bbox_inches='tight')
