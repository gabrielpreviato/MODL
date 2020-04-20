import numpy as np
import os
import cv2
import scipy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

from lib.UE4Dataset import UE4Dataset
from lib.UE4DataGenerator import UE4DataGenerator

import lib.modl_metrics as modl_metrics


class YoloV1Error(losses.Loss):
    def yolo_conf_loss(self, y_true, y_pred, t):
        real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
        pobj = K.sigmoid(y_pred)
        lo = K.square(real_y_true - pobj)
        value_if_true = 5.0 * lo
        value_if_false = 0.05 * lo
        loss1 = tf.where(t, value_if_true, value_if_false)

        loss = K.sum(loss1)
        return loss
    
    def yoloxyloss(self, y_true, y_pred, t):
        # real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
        lo = K.square(y_true - y_pred) + 0.05 * K.square(0.5 - y_pred)
        value_if_true = lo
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.where(t, value_if_true, value_if_false)
        objsum = K.sum(y_true)
        return K.sum(loss1) / (objsum + 0.0000001)
    
    def yolo_wh_loss(self, y_true, y_pred, t):
        # real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
        lo = K.square(y_true - y_pred) + 0.05 * K.square(0.5 - y_pred)
        # lo = K.square(y_true - y_pred) + 0.3 * K.square(0.5 - y_pred)
        value_if_true = lo
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.where(t, value_if_true, value_if_false)
        objsum = K.sum(y_true)
        return K.sum(loss1) / (objsum + 0.0000001)


    def yolo_regressor_loss(self, y_true, y_pred, t):
        # real_y_true = tf.where(t, y_true, K.zeros_like(y_true))
        lo = K.square(y_true - y_pred)  # + 0.15 * K.square(0.5 - y_pred)
        # lo = K.square(y_true - y_pred) + 0.3 * K.square(0.5 - y_pred)
        value_if_true = lo
        value_if_false = K.zeros_like(y_true)
        loss1 = tf.where(t, value_if_true, value_if_false)

        objsum = K.sum(y_true)
        return K.sum(loss1) / (objsum + 0.0000001)

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        truth_conf_tensor = K.expand_dims(y_true[:, :, 0], 2)  # tf.slice(y_true, [0, 0, 0], [-1,-1, 0])
        truth_xy_tensor = y_true[:, :, 1:3]  # tf.slice(y_true, [0, 0, 1], [-1,-1, 2])
        truth_wh_tensor = y_true[:, :, 3:5]  # tf.slice(y_true, [0, 0, 3], [-1, -1, 4])
        truth_m_tensor = K.expand_dims(y_true[:, :, 5], 2)  # tf.slice(y_true, [0, 0, 5], [-1, -1, 5])
        truth_v_tensor = K.expand_dims(y_true[:, :, 6], 2)  # tf.slice(y_true, [0, 0, 6], [-1, -1, 6])

        pred_conf_tensor = K.expand_dims(y_pred[:, :, 0], 2)  # tf.slice(y_pred, [0, 0, 0], [-1, -1, 0])
        # pred_conf_tensor = K.tanh(pred_conf_tensor)
        pred_xy_tensor = y_pred[:, :, 1:3]  # tf.slice(y_pred, [0, 0, 1], [-1, -1, 2])
        pred_wh_tensor = y_pred[:, :, 3:5]  # tf.slice(y_pred, [0, 0, 3], [-1, -1, 4])
        pred_m_tensor = K.expand_dims(y_pred[:, :, 5], 2)  # tf.slice(y_pred, [0, 0, 5], [-1, -1, 5])
        pred_v_tensor = K.expand_dims(y_pred[:, :, 6], 2)  # tf.slice(y_pred, [0, 0, 6], [-1, -1, 6])

        # truth_xy_tensor = tf.Print(truth_xy_tensor, [truth_xy_tensor[:, 14:20, 0]], message='truth_xy', summarize=30)
        # pred_xy_tensor = tf.Print(pred_xy_tensor, [pred_xy_tensor[:, 14:20, 0]], message='pred_xy', summarize=30)

        tens = K.greater(K.sigmoid(truth_conf_tensor), 0.5)
        tens_2d = K.concatenate([tens, tens], axis=-1)

        conf_loss = self.yolo_conf_loss(truth_conf_tensor, pred_conf_tensor, tens)
        xy_loss = self.yoloxyloss(truth_xy_tensor, pred_xy_tensor, tens_2d)
        wh_loss = self.yolo_wh_loss(truth_wh_tensor, pred_wh_tensor, tens_2d)
        m_loss = self.yolo_regressor_loss(truth_m_tensor, pred_m_tensor, tens)
        v_loss = self.yolo_regressor_loss(truth_v_tensor, pred_v_tensor, tens)

        loss = 1.0 * conf_loss + 0.25 * xy_loss + 0.25 * wh_loss #+ 1.5 * m_loss + 1.25 * v_loss  # loss v3
        # loss = 2.0 * conf_loss + 0.25 * xy_loss + 0.25 * wh_loss + 1.5 * m_loss + 1.25 * v_loss  # loss v1
        # loss = 2.0 * conf_loss + 0.1 * xy_loss + 1.0 * wh_loss + 5.0 * m_loss + 2.5 * v_loss  # loss v2

        return loss

class RootMeanSquaredError(losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.sqrt(K.mean(math_ops.square(y_pred - y_true), axis=-1)) + 1e-5


class MeanSquaredError(losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.square(y_pred - y_true), axis=-1) + 1e-5


class NormalError(losses.Loss):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        super().__init__()

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        # first_log = K.log(y_pred + 1.)
        # second_log = K.log(y_true + 1.)

        # log_term = K.sqrt(K.mean(K.square(first_log - second_log), axis=-1) + 0.00001)
        # sc_inv_term = K.square(K.mean((first_log - second_log), axis=-1))
        
        # loss = log_term - (0.5 * sc_inv_term)
        # return loss
        y_true_clipped = y_true
        y_pred_clipped = y_pred

        w_x = np.array([[-1., 0., 1.],
                                [-1., 0., 1.],
                                [-1., 0., 1.]]).reshape(3, 3, 1, 1)

        w_y = np.array([[-1., -1., -1.],
                                [0., 0., 0.],
                                [1., 1., 1.]]).reshape(3, 3, 1, 1)

        #dzdx = K.conv2d(K.exp(y_true_clipped), w_x, padding='same')
        #dzdy = K.conv2d(K.exp(y_true_clipped), w_y, padding='same')
        dzdx = K.conv2d(y_true_clipped, w_x, padding='same')
        dzdy = K.conv2d(y_true_clipped, w_y, padding='same')

        dzdx_ = dzdx * -1.0#K.constant(-1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(-1.0, shape=K.int_shape(dzdx))
        dzdy_ = dzdy * -1.0#K.constant(-1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(-1.0, shape=K.int_shape(dzdy))

        mag_norm = K.pow(dzdx,2) + K.pow(dzdy,2) + 1.0#K.constant(1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(1.0, shape=K.int_shape(dzdx))

        mag_norm = K.sqrt(mag_norm)
        N3 = 1.0 / mag_norm #K.constant(1.0, shape=K.int_shape(dzdx)) / mag_norm
        N1 = dzdx_ / mag_norm
        N2 = dzdy_ / mag_norm

        normals = K.concatenate(tensors=[N1,N2,N3],axis=-1)

        #dzdx_pred = K.conv2d(K.exp(y_pred_clipped), w_x, padding='same')
        #dzdy_pred = K.conv2d(K.exp(y_pred_clipped), w_y, padding='same')
        dzdx_pred = K.conv2d(y_pred_clipped, w_x, padding='same')
        dzdy_pred = K.conv2d(y_pred_clipped, w_y, padding='same')

        mag_norm_pred_x = K.pow(dzdx_pred,2) + 1.0
        mag_norm_pred_x = K.sqrt(mag_norm_pred_x)
        mag_norm_pred_y = K.pow(dzdy_pred, 2) + 1.0
        mag_norm_pred_y = K.sqrt(mag_norm_pred_y)

        grad_x = K.concatenate(tensors=[K.constant(1.0, shape=[self.batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_x,
                                        K.constant(0.0, shape=[self.batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_x, dzdx_pred/ mag_norm_pred_x],axis=-1)
        grad_y = K.concatenate(tensors=[K.constant(0.0, shape=[self.batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_y,
                                        K.constant(1.0, shape=[self.batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_y, dzdy_pred/ mag_norm_pred_y],axis=-1)

        first_log = K.log(y_pred_clipped + 1.)
        second_log = K.log(y_true_clipped + 1.)

        log_term = K.sqrt(K.mean(K.square(first_log - second_log), axis=-1) + 0.00001)

        dot_term_x = K.sum(normals[:,:,:,:] * grad_x[:,:,:,:], axis=-1, keepdims=True)
        dot_term_y = K.sum(normals[:,:,:,:] * grad_y[:,:,:,:], axis=-1, keepdims=True)
        #dot_term_x = K.mean(K.sum(normals[:, :, :, :] * grad_x[:, :, :, :], axis=-1, keepdims=True), axis=-1)
        #dot_term_y = K.mean(K.sum(normals[:, :, :, :] * grad_y[:, :, :, :], axis=-1, keepdims=True), axis=-1)


        # dot_term_x = tf.Print(dot_term_x, [dot_term_x], message='dot_term_x', summarize=30)
        # dot_term_y = tf.Print(dot_term_y, [dot_term_y], message='dot_term_y', summarize=30)

        #commentare per vecchia versione
        sc_inv_term = K.square(K.mean((first_log - second_log), axis=-1))
        norm_term = K.mean(K.square(dot_term_x), axis=-1) + K.mean(K.square(dot_term_y), axis=-1)

        diff_x = dzdx_pred - dzdx
        diff_y = dzdy_pred - dzdy
        grad_loss = K.mean(K.square(diff_x) + K.square(diff_y), axis=-1)

        loss = log_term - (0.5 * sc_inv_term) + norm_term + grad_loss
        #loss = log_term + K.square(dot_term_x) + K.square(dot_term_y)

        return loss


class MODL2():
    def __init__(self, config, shuffle_data=True, is_test=False, test_size=1):
        self.config = config
        self.model = None
        self.shuffle = shuffle_data

        self.training_set_gen = None
        self.validation_set_gen = None
        self.test_set_gen = None

        self.input_dim = (self.config.input_height, self.config.input_width, self.config.input_channel)
        self.depth_dim = (self.config.input_height, self.config.input_width, 1)

        self.rng = np.random.default_rng()
        self.tb_images_indexes = None
        self.tb_images_X = None
        self.tb_images_y = None
        self.tb_images_len = 1
        self.batches_per_test = 100

        self.file_writer_depth = tf.summary.create_file_writer(os.path.join(self.config.tensorboard_dir, 'depth'))
    
    def log_depth_images(self, batch, logs):
        if batch % self.batches_per_test != 0:
            return
    
        # Use the model to predict the values from the validation dataset.
        with self.file_writer_depth.as_default():
            for i, rgb in enumerate(self.tb_images_X):
                test_pred = self.model.predict(rgb)[0][0]
                tf.summary.image("depth_pred_%i" % i, test_pred, step=batch)
                tf.summary.image("depth_true_%i" % i, self.tb_images_y[i], step=batch)
    
    def load_dataset(self):
        self.train_dataset = UE4Dataset(self.config, self.config.data_set_dir, self.config.data_train_dirs, 'png', 'pfm', 'json')
        X_train, y_train = self.train_dataset.get_train_dataset()
        self.training_set_gen = UE4DataGenerator(self.config, X_train, y_train, self.config.batch_size, self.input_dim, self.depth_dim)
        
        self.test_dataset = UE4Dataset(self.config, self.config.data_set_dir, self.config.data_test_dirs, 'png', 'pfm', 'json')
        X_test, y_test = self.test_dataset.get_test_dataset()
        self.validation_set_gen = UE4DataGenerator(self.config, X_test, y_test, self.config.batch_size, self.input_dim, self.depth_dim)

        self.tb_set_gen = UE4DataGenerator(self.config, X_test, y_test, 1, self.input_dim, self.depth_dim)
        self.tb_images_indexes = self.rng.integers(len(X_test), size=1)[0]

        self.tb_images_X = np.expand_dims(self.tb_set_gen[self.tb_images_indexes][0], 0)
        self.tb_images_y = np.expand_dims(self.tb_set_gen[self.tb_images_indexes][1][0], 0)
    
    def define_base_architecture(self):
        input = keras.Input(shape=(self.config.input_height,
                                   self.config.input_width,
                                   self.config.input_channel),
                            name='input')
        
        self.base_model = keras.applications.VGG19(include_top=False,
                                              weights='imagenet',
                                              input_tensor=input,
                                              input_shape=(self.config.input_height,
                                                           self.config.input_width,
                                                           self.config.input_channel)
                                              )
        
        self.base_model.summary()
    
    def define_depth_architecture(self):
        base_output = self.base_model.layers[-2].output
        x = layers.Conv2DTranspose(256, (4, 4), padding="same", strides=(2, 2), name='depth_convtransp1')(base_output)
        x = layers.PReLU(name='depth_prelu1')(x)
        x = layers.Conv2DTranspose(128, (4, 4), padding="same", strides=(2, 2), name='depth_convtransp2')(x)
        x = layers.PReLU(name='depth_prelu2')(x)
        x = layers.Conv2DTranspose(64, (4, 4), padding="same", strides=(2, 2), name='depth_convtransp3')(x)
        x = layers.PReLU(name='depth_prelu3')(x)
        # x = layers.Conv2DTranspose(32, (4, 4), padding="same", strides=(2, 2))(x)
        # x = layers.PReLU()(x)
        x = layers.Conv2DTranspose(16, (4, 4), padding="same", strides=(2, 2), name='depth_convtransp4')(x)
        x = layers.PReLU(name='depth_prelu4')(x)
        # out = layers.Convolution2D(1, (5, 5), padding="same", activation="tanh", name="depth_output")(x)
        out = layers.Convolution2D(1, (5, 5), padding="same", activation="relu", name="depth_output")(x)

        self.depth_model = keras.Model(inputs=self.base_model.inputs[0], outputs=out, name='modl2_depth_model')

        self.depth_model.summary()
    
    def define_obst_architecture(self):
        base_output = self.base_model.layers[-2].output

        x_grid_count = self.config.input_width / self.config.cell_size
        y_grid_count = self.config.input_height / self.config.cell_size
        grid_count = int(x_grid_count * y_grid_count)

        # x = layers.MaxPooling2D(name='det_pool1')(base_output)
        x = layers.Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv1')(base_output)
        x = layers.Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv2')(x)
        x = layers.Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv3')(x)
        x = layers.Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv4')(x)
        x = layers.Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv5')(x)

        x = layers.Convolution2D(280, (3, 3), activation='relu', padding='same', name='det_conv6')(x)
        # x = layers.Convolution2D(280, (3, 3), activation='relu', padding='same', name='det_conv7')(x)
        x = layers.Reshape((grid_count, 7, 160), name='det_reshape1')(x)

        x = layers.Convolution2D(160, (3, 3), activation='relu', padding='same', name='det_conv8')(x)
        x = layers.Convolution2D(40, (3, 3), activation='relu', padding='same', name='det_conv9')(x)

        x = layers.Convolution2D(1, (3, 3), activation='linear', padding='same', name='det_conv10')(x)

        out_detection = layers.Reshape((grid_count, 7), name='detection_output')(x)

        self.obs_model = keras.Model(inputs=self.base_model.inputs[0], outputs=out_detection, name='modl2_obs_model')
        self.obs_model.summary()
    
    def define_architecture(self):
        self.define_base_architecture()
        self.define_depth_architecture()
        self.define_obst_architecture()

        self.model = keras.Model(inputs=self.base_model.inputs[0], outputs=[self.depth_model.outputs[0], self.obs_model.outputs[0]], name='modl2_model')
        self.model.summary()
    
    def define_optimizer(self):
        self.optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    
    def build_model(self):
        self.define_architecture()
        self.define_optimizer()

        self.model.compile(loss={'depth_output': NormalError(self.config.batch_size), 'detection_output': YoloV1Error()},
                            optimizer=self.optimizer,
                            metrics={'depth_output': [keras.metrics.MeanSquaredError()], 'detection_output': [modl_metrics.recall, modl_metrics.precision, modl_metrics.iou_metric]},
                            loss_weights=[1.0, 1.0])
    
    def test(self):
        tf.config.set_visible_devices([], 'GPU')

        self.load_dataset()

        self.build_model()

        self.model.load_weights("cp-0060.ckpt")

        # X_test, y_test = self.test_dataset.get_test_dataset()
        # self.test_set_gen = UE4DataGenerator(self.config, X_test, y_test, self.config.batch_size, self.input_dim, self.depth_dim, is_test=True, test_size=4)

        for i in range(30):
            X, y_true = self.validation_set_gen[i]
            y_pred = self.model.predict(x=X)

            # print(X.shape)
            # print(y_pred[0].shape)
            # print(y_pred[1].shape)
            
            depth = y_pred[0][0]
            print(i, depth.shape, depth.mean())
            cv2.imwrite('test/test_100_%i.png' % i, depth * 100)
            cv2.imwrite('test/test_255_%i.png' % i, depth * 255)

            cp_img = np.copy(X[0])
            obs = y_pred[1]
            obs_true = y_true[1]

            # print(i, ':', indexes)
            print(i, ':', obs_true[0][obs_true[0][:, 6] > 0.5])

            x_grid_count = self.config.input_width / self.config.cell_size
            y_grid_count = self.config.input_height / self.config.cell_size
            grid_count = int(x_grid_count * y_grid_count)

            for j in range(grid_count):
                if scipy.special.expit(obs[0][j][0]) >= 0.5:
                    x, y, w, h = obs[0][j][1:5]
                    x_min = int((j%x_grid_count)*self.config.cell_size + x*self.config.cell_size - w*self.config.input_width/2)
                    x_max = int((j%x_grid_count)*self.config.cell_size + x*self.config.cell_size + w*self.config.input_width/2)
                    
                    y_min = int((j//x_grid_count)*self.config.cell_size + y*self.config.cell_size - h*self.config.input_height/2)
                    y_max = int((j//x_grid_count)*self.config.cell_size + y*self.config.cell_size + h*self.config.input_height/2)

                    # print(x_min)
                    # print(x_max)
                    # print(y_min)
                    # print(y_max)

                    cv2.rectangle(cp_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                if obs_true[0][j][0] > 0.75:
                    x, y, w, h = obs_true[0][j][1:5]
                    x_min = int((j%x_grid_count)*self.config.cell_size + x*self.config.cell_size - w*self.config.input_width/2)
                    x_max = int((j%x_grid_count)*self.config.cell_size + x*self.config.cell_size + w*self.config.input_width/2)
                    
                    y_min = int((j//x_grid_count)*self.config.cell_size + y*self.config.cell_size - h*self.config.input_height/2)
                    y_max = int((j//x_grid_count)*self.config.cell_size + y*self.config.cell_size + h*self.config.input_height/2)

                    print('######')
                    print(j%x_grid_count)
                    print(j//x_grid_count)
                    print((j%x_grid_count)*self.config.cell_size + x*self.config.cell_size)
                    print((j//x_grid_count)*self.config.cell_size + y*self.config.cell_size)
                    print(w*self.config.input_width)
                    print(h*self.config.input_height)

                    cv2.rectangle(cp_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            cv2.imwrite('test/test_rgb_%i.png' % i, cp_img)

    def train(self):
        self.load_dataset()
        
        self.build_model()

        if self.config.resume_training:
            print("Resuming")
            self.model.load_weights(self.config.weights_path)

        for layer in self.model.layers:
            layer.trainable = True

        # tf.keras.utils.plot_model(self.model, show_shapes=True, to_file=os.path.join(self.config.model_dir, 'model_structure.png'))
        checkpoint_path = self.config.tensorboard_dir + '/cp-{epoch:04d}.ckpt'        
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.config.tensorboard_dir, update_freq=4, write_images=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1),
            keras.callbacks.LambdaCallback(on_batch_end=self.log_depth_images)
        ]
        history = self.model.fit(
                            x=self.training_set_gen,
                            validation_data=self.validation_set_gen,
                            callbacks=callbacks,
                            epochs=self.config.num_epochs)
        
        self.model.save(self.config.model_dir)
        self.model.save_weights('my_checkpoint')
        tf.print(history.history)

    def resume_training(self):
        return

# python train.py --number_classes=2 --is_deploy=False --is_train=True --data_set_dir='E:\\Dropbox\\IC\\dataset' --input_height=320 --input_width=512 --cell_size=32 --obs_extension='json' --batch_size=8 --gpu_memory_fraction=0.9 --data_test_dirs=test5 --data_train_dirs=test6 --num_epochs=10
# python test.py --number_classes=2 --is_deploy=False --is_train=True --data_set_dir='E:\\Dropbox\\IC\\dataset' --input_height=320 --input_width=512 --cell_size=32 --obs_extension='json' --batch_size=1 --gpu_memory_fraction=0.8 --data_test_dirs=test5 --data_train_dirs=test6 --num_epochs=10
# python train.py --number_classes=2 --is_deploy=False --is_train=True --data_set_dir='E:\\Dropbox\\IC\\dataset' --input_height=320 --input_width=512 --cell_size=32 --obs_extension='json' --batch_size=8 --gpu_memory_fraction=0.9 --data_test_dirs=test5 --data_train_dirs=test6 --num_epochs=10 --exp-name="5vs0.05lr1-5adam-sum"
# python train.py --number_classes=2 --is_deploy=False --is_train=True --data_set_dir='E:\\Dropbox\\IC\\dataset' --input_height=320 --input_width=512 --cell_size=32 --obs_extension='json' --batch_size=8 --gpu_memory_fraction=0.9 --data_test_dirs=test5 --data_train_dirs=test6 --num_epochs=50 --exp_name="5vs0.05lr1-5adam-sum(10~50)" --resume_training=True --weights_path="C:\Users\Previato\MODL\logs\5vs0.05-loss3-lr5-5adam-sum__320_512_test_dirs_test5_2020-04-16_10-50-06\tensorboard\cp-0010.ckpt"
# python train.py --number_classes=2 --is_deploy=False --is_train=True --data_set_dir='/home/previato/dataset' --input_height=320 --input_width=512 --cell_size=32 --obs_extension='json' --batch_size=16 --gpu_memory_fraction=0.9 --data_test_dirs=test6 --data_train_dirs=test6 --num_epochs=50 --exp_name="5vs0.05lr1-5adam-sum-metrics"