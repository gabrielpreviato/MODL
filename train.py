import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from models.MODL import MODL
from models.DetectorModel import Detector
#import whatever model you need to train here
from lib.trainer import Trainer
from config import get_config
from lib.utils import prepare_dirs

config = None

def main(_):

  prepare_dirs(config)

  rng = RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  model = MODL(config)
  trainer = Trainer(config, model, rng)

  if config.is_train:
     if config.resume_training:
       trainer.resume_training()
     else:
       trainer.train()
  else:
     trainer.test(showFigure=True)

if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# python train.py --number_classes=2 --is_deploy=False --is_train=True --data_set_dir='E:\\Dropbox\\IC\\dataset' --data_train_dirs=test3 --data_test_dirs=test3 --input_height=640 --input_width=1024 --obs_extension='json' --batch_size=16 --gpu_memory_fraction=0.8