from keras.models import Model
from keras.layers import Reshape, Convolution2D
from keras.utils import plot_model
# from keras.metrics import accuracy

from lib.SampleType import DepthObstacles_SingleFrame_Multiclass_2, DepthObstacles_SingleFrame_Multiclass_3, \
    DepthObstacles_SingleFrame_Multiclass_4

from lib.DepthObjectives import log_normals_loss
from lib.ObstacleDetectionObjectives import yolo_v1_loss_multiclass_2, iou_metric_multiclass_2, recall_multiclass_2, \
    precision_multiclass_2, \
    mean_metric_multiclass_2, variance_metric_multiclass_2, yolo_v1_loss_multiclass_3, recall_multiclass_3, \
    mean_metric_multiclass_3, precision_multiclass_3, iou_metric_multiclass_3, variance_metric_multiclass_3, \
    yolo_v1_loss_multiclass_4, recall_multiclass_4, iou_metric_multiclass_4, precision_multiclass_4, \
    variance_metric_multiclass_4, mean_metric_multiclass_4
from lib.DepthMetrics import rmse_metric, logrmse_metric, sc_inv_logrmse_metric

import numpy as np

from .DepthFCNModel import DepthFCNModel

from lib.Dataset import SoccerFieldDatasetDepthSupervised
from lib.DataGenerationStrategy import SingleFrameGenerationStrategy

from keras.optimizers import Adam

from lib.EvaluationUtils import get_detected_obstacles_from_detector_multiclass_2, \
    get_detected_obstacles_from_detector_multiclass_3, get_detected_obstacles_from_detector_multiclass_4

import matplotlib.pyplot as plt


class MODL(DepthFCNModel):
    def __init__(self, config):
        self.number_classes = config.number_classes

        super(MODL, self).__init__(config)

    def load_dataset(self):
        if self.config.dataset == 'Soccer':
            if self.number_classes == 2:
                dataset = SoccerFieldDatasetDepthSupervised(self.config, SingleFrameGenerationStrategy(
                    sample_type=DepthObstacles_SingleFrame_Multiclass_2,
                    get_obstacles=True), read_obstacles=True)
            elif self.number_classes == 3:
                dataset = SoccerFieldDatasetDepthSupervised(self.config, SingleFrameGenerationStrategy(
                    sample_type=DepthObstacles_SingleFrame_Multiclass_3,
                    get_obstacles=True), read_obstacles=True)
            elif self.number_classes == 4:
                dataset = SoccerFieldDatasetDepthSupervised(self.config, SingleFrameGenerationStrategy(
                    sample_type=DepthObstacles_SingleFrame_Multiclass_4,
                    get_obstacles=True), read_obstacles=True)
            else:
                raise Exception("ODL not implemented with number of classes " + str(self.number_classes))

            dataset.data_generation_strategy.mean = dataset.mean
            dataset_name = 'Soccer'
            return dataset, dataset_name
        else:
            raise Exception("ODL not implemented with this type of dataset")

    def prepare_data_for_model(self, features, label):
        features = np.asarray(features)
        features = features.astype('float32')

        features /= 255.0

        labels_depth = np.zeros(shape=(features.shape[0], features.shape[1], features.shape[2], 1), dtype=np.float32)
        if self.number_classes == 2:
            labels_obs = np.zeros(shape=(features.shape[0], 16*10, 8), dtype=np.float32)
        elif self.number_classes == 3:
            labels_obs = np.zeros(shape=(features.shape[0], 40, 9), dtype=np.float32)
        elif self.number_classes == 4:
            labels_obs = np.zeros(shape=(features.shape[0], 40, 10), dtype=np.float32)
        else:
            raise Exception("ODL not implemented with number of classes " + str(self.number_classes))
        i = 0
        for elem in label:
            elem["depth"] = np.asarray(elem["depth"]).astype(np.float32)
            elem["depth"] = np.clip(elem["depth"], 0, 255)

            elem["depth"] = -4.586e-09 * (elem["depth"] ** 4) + 3.382e-06 * (elem["depth"] ** 3) - 0.000105 * (
                    elem["depth"] ** 2) + 0.04239 * elem["depth"] + 0.04072
            elem["depth"] /= 19.75

            labels_depth[i, :, :, :] = elem["depth"]
            labels_obs[i, :, :] = np.asarray(elem["obstacles"]).astype(np.float32)
            i += 1

        return features, [labels_depth, labels_obs]

    def build_model(self):
        depth_model = self.define_architecture()
        # Detection section
        output = depth_model.layers[-12].output

        print(output)

        # x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv1')(output)
        # x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv2')(x)
        # x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv3')(x)
        # x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv4')(x)
        # x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv5')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='det_conv1')(output)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='det_conv2')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='det_conv3')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='det_conv4')(x)
        x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='det_conv5')(x)

        print(x)

        if self.number_classes == 2:
            x = Convolution2D(320, (3, 3), activation='relu', padding='same', name='det_conv6')(x)
            print(x)
            x = Reshape((160, 8, 160))(x)
        elif self.number_classes == 3:
            x = Convolution2D(360, (3, 3), activation='relu', padding='same', name='det_conv6')(x)
            x = Reshape((40, 9, 160))(x)
        elif self.number_classes == 4:
            x = Convolution2D(400, (3, 3), activation='relu', padding='same', name='det_conv6')(x)
            x = Reshape((40, 10, 160))(x)
        else:
            raise Exception("ODL not implemented with number of classes " + str(self.number_classes))

        x = Convolution2D(160, (3, 3), activation='relu', padding='same', name='det_conv7')(x)
        x = Convolution2D(40, (3, 3), activation='relu', padding='same', name='det_conv8')(x)
        x = Convolution2D(1, (3, 3), activation='linear', padding='same', name='det_conv9')(x)

        if self.number_classes == 2:
            out_detection = Reshape((160, 8), name='detection_output')(x)
        elif self.number_classes == 3:
            out_detection = Reshape((40, 9), name='detection_output')(x)
        elif self.number_classes == 4:
            out_detection = Reshape((40, 10), name='detection_output')(x)
        else:
            raise Exception("ODL not implemented with number of classes " + str(self.number_classes))

        model = Model(inputs=depth_model.inputs[0], outputs=[depth_model.outputs[0], out_detection])
        plot_model(model, to_file='model.png', show_shapes=True)

        opt = Adam(lr=self.config.learning_rate, clipnorm=1.)

        if self.number_classes == 2:
            model.compile(loss={'depth_output': log_normals_loss, 'detection_output': yolo_v1_loss_multiclass_2},
                          optimizer=opt,
                          metrics={'depth_output': [rmse_metric, logrmse_metric, sc_inv_logrmse_metric],
                                   'detection_output': [iou_metric_multiclass_2, recall_multiclass_2,
                                                        precision_multiclass_2, mean_metric_multiclass_2,
                                                        variance_metric_multiclass_2]},
                          loss_weights=[1.0, 1.0])
        elif self.number_classes == 3:
            model.compile(loss={'depth_output': log_normals_loss, 'detection_output': yolo_v1_loss_multiclass_3},
                          optimizer=opt,
                          metrics={'depth_output': [rmse_metric, logrmse_metric, sc_inv_logrmse_metric],
                                   'detection_output': [iou_metric_multiclass_3, recall_multiclass_3,
                                                        precision_multiclass_3, mean_metric_multiclass_3,
                                                        variance_metric_multiclass_3]},
                          loss_weights=[1.0, 1.0])
        elif self.number_classes == 4:
            model.compile(loss={'depth_output': log_normals_loss, 'detection_output': yolo_v1_loss_multiclass_4},
                          optimizer=opt,
                          metrics={'depth_output': [rmse_metric, logrmse_metric, sc_inv_logrmse_metric],
                                   'detection_output': [iou_metric_multiclass_4, recall_multiclass_4,
                                                        precision_multiclass_4, mean_metric_multiclass_4,
                                                        variance_metric_multiclass_4]},
                          loss_weights=[1.0, 1.0])
        else:
            raise Exception("ODL not implemented with number of classes " + str(self.number_classes))

        model.summary()
        return model

    def plot_graphs(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['detection_output_precision'])
        plt.plot(self.history.history['val_detection_output_precision'])
        plt.title('Model detection precision')
        plt.ylabel('Detection Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.config.graphs_dir + "/detection_precision_" + self.config.exp_name + ".png")

        plt.close()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.config.graphs_dir + "/model_loss_" + self.config.exp_name + ".png")

        plt.close()

        plt.plot(self.history.history['depth_output_loss'])
        plt.plot(self.history.history['val_depth_output_loss'])
        plt.title('Depth loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.config.graphs_dir + "/depth_loss_" + self.config.exp_name + ".png")

        plt.close()

        plt.plot(self.history.history['detection_output_loss'])
        plt.plot(self.history.history['val_detection_output_loss'])
        plt.title('Detection loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.config.graphs_dir + "/detec_loss_" + self.config.exp_name + ".png")

        plt.close()

        plt.plot(self.history.history['depth_output_rmse_metric'])
        plt.plot(self.history.history['val_depth_output_rmse_metric'])
        plt.title('Depth RMSE metric')
        plt.ylabel('RMSE metric')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.config.graphs_dir + "/depth_rmse_" + self.config.exp_name + ".png")

        plt.close()

        plt.plot(self.history.history['depth_output_logrmse_metric'])
        plt.plot(self.history.history['val_depth_output_logrmse_metric'])
        plt.title('Depth Log RMSE metric')
        plt.ylabel('Log RMSE metric')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.config.graphs_dir + "/depth_log_rmse_" + self.config.exp_name + ".png")

        plt.close()

        plt.plot(self.history.history['detection_output_mean_metric'])
        plt.plot(self.history.history['val_detection_output_mean_metric'])
        plt.title('Detection Mean metric')
        plt.ylabel('Mean metric')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.config.graphs_dir + "/detec_mean_" + self.config.exp_name + ".png")

        plt.close()

        plt.plot(self.history.history['detection_output_variance_metric'])
        plt.plot(self.history.history['val_detection_output_variance_metric'])
        plt.title('Detection Variance metric')
        plt.ylabel('Variance metric')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.config.graphs_dir + "/detec_variance_" + self.config.exp_name + ".png")

        plt.close()

    def run(self, input_img):
        # import time
        mean = np.load('Unreal_RGB_mean.npy')

        if len(input_img.shape) == 2 or input_img.shape[2] == 1:
            tmp = np.zeros(shape=(input_img.shape[0], input_img.shape[1], 3))
            tmp[:, :, 0] = input_img
            tmp[:, :, 1] = input_img
            tmp[:, :, 2] = input_img

            input_img = tmp

        if len(input_img.shape) == 3:
            input_img = np.expand_dims(input_img - mean / 255., 0)
        else:
            input_img[0, :, :, :] -= mean / 255.

        # t0 = time.time()

        net_output = self.model.predict(input_img)

        # print ("Elapsed time: {}").format(time.time() - t0)

        pred_depth = net_output[0] * 19.75
        pred_detection = net_output[1]

        if self.number_classes == 2:
            pred_obstacles, rgb_with_detection = get_detected_obstacles_from_detector_multiclass_2(pred_detection,
                                                                                                   self.config.detector_confidence_thr)
        elif self.number_classes == 3:
            pred_obstacles, rgb_with_detection = get_detected_obstacles_from_detector_multiclass_3(pred_detection,
                                                                                                   self.config.detector_confidence_thr)
        elif self.number_classes == 4:
            pred_obstacles, rgb_with_detection = get_detected_obstacles_from_detector_multiclass_4(pred_detection,
                                                                                                   self.config.detector_confidence_thr)
        else:
            raise Exception("ODL not implemented with number of classes " + str(self.number_classes))

        correction_factor = self.compute_correction_factor(pred_depth, pred_obstacles)

        corrected_depth = np.array(pred_depth) * correction_factor

        return [pred_depth, pred_obstacles, corrected_depth, pred_detection]

    def compute_correction_factor(self, depth, obstacles):

        mean_corr = 0
        it = 0

        for obstacle in obstacles:
            depth_roi = depth[0, np.max((obstacle.y, 0)):np.min((obstacle.y + obstacle.h, depth.shape[1] - 1)),
                              np.max((obstacle.x, 0)):np.min((obstacle.x + obstacle.w, depth.shape[2] - 1)), 0]

            if len(depth_roi) > 0:
                mean_est = np.mean(depth_roi)
                it += 1
                mean_corr += obstacle.depth_mean / mean_est

        if it > 0:
            mean_corr /= it
        else:
            mean_corr = 1
        return mean_corr
