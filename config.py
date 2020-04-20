#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')

# Data
data_arg = add_argument_group('Data')

data_arg.add_argument('--data_set_dir', type=str, default='dataset')
data_arg.add_argument('--dataset', type=str, default='Soccer', choices=['UnrealDataset', 'Satellite', 'Soccer'])
data_arg.add_argument('--use_subsampled', type=str2bool, default=False)
data_arg.add_argument('--compute_mean', type=str2bool, default=False)

data_arg.add_argument('--data_main_dir', type=str, default='')
# data_arg.add_argument('--data_train_dirs', type=str, nargs='+', default=['ssnda_01', 'ssnda_02', 'ssnda_03', 'ssnda_04'])
# data_arg.add_argument('--data_test_dirs', type=str, nargs='+', default=['ssnda_05'])
data_arg.add_argument('--data_train_dirs', type=str, default='ssnda_01')
data_arg.add_argument('--data_test_dirs', type=str, default='ssnda_05')

data_arg.add_argument('--input_height', type=int, default=160)
data_arg.add_argument('--input_width', type=int, default=256)
data_arg.add_argument('--cell_size', type=int, default=32)

data_arg.add_argument('--input_channel', type=int, default=3)
data_arg.add_argument('--img_extension', type=str, default="png")
data_arg.add_argument('--obs_extension', type=str, default="txt")

#JMOD2 param
jmod2_arg = add_argument_group('JMOD2')
jmod2_arg.add_argument('--detector_confidence_thr', type=float, default=0.65)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=False, help='') #Set True for training
train_arg.add_argument('--exp_name', type=str, default='NAME_OF_EXPERIMENT')
train_arg.add_argument('--preload_ram',type=str2bool, default=False)
train_arg.add_argument('--use_augmentation', type=str2bool, default=True, help='')
train_arg.add_argument('--validation_split', type=float, default=0.2, help='')
train_arg.add_argument('--max_step', type=int, default=10000, help='')
train_arg.add_argument('--batch_size', type=int, default=32, help='')
train_arg.add_argument('--buffer_size', type=int, default=25600, help='')
train_arg.add_argument('--num_epochs', type=int, default=60, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--learning_rate', type=float, default=1e-5, help='')

train_arg.add_argument('--weights_path', type=str, default="") #Used for finetuning or to resume training
train_arg.add_argument('--resume_training',type=str2bool, default=False)
train_arg.add_argument('--resumed_epoch', type=int, default=0, help='')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_deploy', type=str2bool, default=True, help='')
misc_arg.add_argument('--log_step', type=int, default=20, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs') #DIRECTORY WHERE TO SAVE MODEL CHECKPOINTS
misc_arg.add_argument('--debug', type=str2bool, default=True)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=0.5)
misc_arg.add_argument('--max_image_summary',type=int, default=4)

misc_arg.add_argument('--graphs_dir', type=str, default='graphs')
misc_arg.add_argument('--number_classes', type=int, default=4)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
