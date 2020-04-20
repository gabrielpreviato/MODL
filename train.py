import sys

from models.MODL2 import MODL2
from config import get_config
from lib.utils import prepare_dirs

config = None

def main():

  prepare_dirs(config)
  model = MODL2(config)

  if config.is_train:
    model.train()
  else:
    model.test()

if __name__ == "__main__":
  config, unparsed = get_config()
  main()


# python train.py --number_classes=2 --is_deploy=False --is_train=True --data_set_dir='E:\\Dropbox\\IC\\dataset' --data_train_dirs=test3 --data_test_dirs=test3 --input_height=640 --input_width=1024 --obs_extension='json' --batch_size=16 --gpu_memory_fraction=0.8