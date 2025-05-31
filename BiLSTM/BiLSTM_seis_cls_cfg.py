import torch    
import argparse
import random
import logging
import sys
import os
import traceback
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger('__main__')

class Options(object):
    def __init__(self):
        fix_seed = 2023
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        self.parser = argparse.ArgumentParser(description='BiLSTM Seismic Classification')
        
        # Basic configuration
        self.parser.add_argument('--config', dest='config_filepath',
                                help='Configuration .json file (optional)')
        self.parser.add_argument('--output_dir', default='/home/dell/disk1/Jinlong/Time-Series-Library-main/output', 
                                help='Root output directory')
        
        # Experiment settings
        self.parser.add_argument('--dataset', type=str, default='nz', choices=['f3', 'nz'])
        self.parser.add_argument('--is_training', type=int, default=1, help='training status')
        self.parser.add_argument('--is_testing', type=int, default=0, help='testing status')
        self.parser.add_argument('--task_name', type=str, default='classification')
        self.parser.add_argument('--model', type=str, default='BiLSTM', help='model name')
        
        # Data paths
        self.parser.add_argument('--root_path', type=str, default='/home/dell/disk1/Jinlong/faciesdata', 
                                help='root path of the data file')
        self.parser.add_argument('--data_path', type=str, default='data_train.npz', help='data file')
        self.parser.add_argument('--label_path', type=str, default='labels_train.npz', help='label file')
        self.parser.add_argument('--mask_path', type=str, default='labels_train.npz', help='mask file')
        
        # VMD settings
        self.parser.add_argument('--vmd_data_path', type=str, default='full_F3_vmd.npy', help='vmd data file') 
        self.parser.add_argument('--is_vmd', type=bool, default=False, help='whether using VMD')
        
        # Data proportions
        self.parser.add_argument('--train_proportion', type=float, default=0.01, help='seismic data for train')
        self.parser.add_argument('--test_proportion', type=float, default=1, help='seismic data for test')
        self.parser.add_argument('--val_proportion', type=float, default=0.001, help='seismic data for validation')
        
        # Model checkpoints
        self.parser.add_argument('--checkpoints', type=str, default='./checkpoints', 
                                help='location of model checkpoints')
        self.parser.add_argument('--checkpoints_test_only', type=str, 
                                default='/home/dell/disk1/Jinlong/Time-Series-Library-main/checkpoints/...',
                                help='location of model checkpoints for testing only')
        
        # Training parameters
        self.parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
        self.parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
        self.parser.add_argument('--itr', type=int, default=1, help='experiments times')
        
        # BiLSTM specific parameters
        self.parser.add_argument('--seq_len', type=int, default=255, help='input sequence length')
        self.parser.add_argument('--hidden_size', type=int, default=64, help='BiLSTM hidden size')
        self.parser.add_argument('--num_layers', type=int, default=2, help='number of BiLSTM layers')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--bidirectional', type=bool, default=True, help='bidirectional LSTM')
        
        # Input/Output dimensions
        self.parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
        self.parser.add_argument('--num_class', type=int, default=6, help='number of classes')
        self.parser.add_argument('--num_class2', type=int, default=6, help='number of classes (duplicate)')
        
        # Processing parameters
        self.parser.add_argument('--mask_rate', type=float, default=0.125, help='mask rate')
        self.parser.add_argument('--embedding_flag', type=bool, default=True, help='embedding flag')
        
        # GPU settings
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        self.parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        self.parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        self.parser.add_argument('--devices', type=str, default='0', help='device ids')
        self.parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
        
        # Experiment tracking
        self.parser.add_argument('--model_id', type=str, default='train', help='model id')
        self.parser.add_argument('--exp_name', type=str, default='BiLSTM_Classification', help='experiment name')
        self.parser.add_argument('--des', type=str, default='BiLSTM_seismic_classification', help='exp description')

    def parse(self):
        args = self.parser.parse_args()
        
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        
        print('BiLSTM Classification Args:')
        print(args)
        
        return args

def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['exp_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (len(config['exp_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config

