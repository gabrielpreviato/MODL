from glob import glob
import os
import numpy as np
import sklearn.model_selection


class UE4Dataset():
    def __init__(self, config, dataset_dir, sequence_dir, image_extension, depth_extension, obstacle_extension, split=False):
        self.config = config
        self.dataset_dir = dataset_dir
        self.sequence_dir = sequence_dir
        self.image_extension = image_extension
        self.depth_extension = depth_extension
        self.obstacle_extension = obstacle_extension

        self.image_paths = []
        self.depth_paths = []
        self.obstacle_paths = []
        self.load_dataset()

        self.split = split
        self.indexes = np.arange(len(self.image_paths))
        if self.split:
            i_train, i_test = sklearn.model_selection.train_test_split(self.indexes, test_size=config.validation_split, shuffle=True)

            # self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.image_paths, [self.depth_paths, self.obstacle_paths], test_size=config.validation_split)
            self.X_train = [self.image_paths[i] for i in i_train]
            self.X_test = [self.image_paths[i] for i in i_test]
            self.y_train = [[self.depth_paths[i], self.obstacle_paths[i]] for i in i_train]
            self.y_test = [[self.depth_paths[i], self.obstacle_paths[i]] for i in i_test]
        else:
            self.X = self.image_paths
            self.y = [[self.depth_paths[i], self.obstacle_paths[i]] for i in range(len(self.depth_paths))]
    
    def load_dataset(self):
        print("load_dataset:", self.dataset_dir, self.sequence_dir)
        self.image_paths = sorted(glob(os.path.join(self.dataset_dir, self.sequence_dir, 'rgb', '*' + '.' + self.image_extension)))
        self.depth_paths = sorted(glob(os.path.join(self.dataset_dir, self.sequence_dir, 'depth', '*' + '.' + self.depth_extension)))
        self.obstacle_paths = sorted(glob(os.path.join(self.dataset_dir, self.sequence_dir, 'obstacles', '*' + '.' + self.obstacle_extension)))

    def get_train_dataset(self):
        if self.split:
            return self.X_train, self.y_train
        else:
            return self.X, self.y
    
    def get_test_dataset(self):
        if self.split:
            return self.X_test, self.y_test
        else:
            return self.X, self.y
