import numpy as np
import math
import cv2
import json

import tensorflow as tf
from tensorflow import keras
from .utils import load_pfm

class UE4DataGenerator(keras.utils.Sequence):
    def __init__(self, config, x_set, y_set, batch_size, input_dim, depth_dim, shuffle=True, is_test=False, test_size=1):
        self.config = config

        x_grid_count = self.config.input_width / self.config.cell_size
        y_grid_count = self.config.input_height / self.config.cell_size
        self.grid_count = int(x_grid_count * y_grid_count)
        print(self.grid_count)

        self.x, self.y = x_set, y_set
        print(len(x_set))
        print(len(y_set))
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.depth_dim = depth_dim

        self.shuffle = shuffle
        self.is_test = is_test
        self.test_size = test_size

        self.indexes = np.arange(len(self.x))

        self.label_dict = {'goal': 1.0, 'ball': 1.0}

    def __len__(self):
        if self.is_test:
            return self.test_size

        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # print(indexes)
        x_temp = [self.x[k] for k in indexes]
        y_temp = [self.y[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(x_temp, y_temp)

        return X, Y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'   
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        print(self.indexes)

    def load_obstacle(self, filename):
        with open(filename, 'r') as f:
            json_obs = json.loads(f.read())
            image_width = json_obs['image_width']
            image_height = json_obs['image_height']
            cell_size = json_obs['cell_size']

            x_grid_count = int(image_width / cell_size)
            y_grid_count = int(image_height / cell_size)

            grid_count = x_grid_count * y_grid_count

            obs = np.zeros(shape=(x_grid_count, y_grid_count, 7), dtype=np.float32)

            for obstacle in json_obs['objects']:
                obs[obstacle['x_cell'], obstacle['y_cell'], 0] = self.label_dict[obstacle['label']]
                obs[obstacle['x_cell'], obstacle['y_cell'], 1] = obstacle['x_cell_position']
                obs[obstacle['x_cell'], obstacle['y_cell'], 2] = obstacle['y_cell_position']
                obs[obstacle['x_cell'], obstacle['y_cell'], 3] = obstacle['width'] / image_width
                obs[obstacle['x_cell'], obstacle['y_cell'], 4] = obstacle['height'] / image_height
                obs[obstacle['x_cell'], obstacle['y_cell'], 5] = obstacle['mean'] / 255.0
                obs[obstacle['x_cell'], obstacle['y_cell'], 6] = obstacle['std'] / 255.0
                
        
        obs = obs.reshape((grid_count, 7), order='F')
        return obs          

    def __data_generation(self, x_temp, y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.input_dim))
        Y_depth = np.empty((self.batch_size, *self.depth_dim), dtype=np.float32)

        Y_obs = np.zeros(shape=(self.batch_size, self.grid_count, 7), dtype=np.float32)

        for i, filename in enumerate(zip(x_temp, y_temp)):
            # Store sample
            X[i,] = cv2.imread(filename[0]) / 255.0
            # print(X[i].mean())

            # Store class
            Y_depth[i] = np.expand_dims(np.clip(load_pfm(filename[1][0]), 0, 255.0), axis=-1) / 255.0
            # print(i, Y[i].shape, Y[i].mean(), Y[i].std())
            Y_obs[i] = self.load_obstacle(filename[1][1])
        
        return X, [Y_depth, Y_obs]
