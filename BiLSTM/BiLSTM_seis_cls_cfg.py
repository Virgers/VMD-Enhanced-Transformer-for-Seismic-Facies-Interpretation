import argparse
import json
import logging
import os
import random
import string
import sys
import traceback
from datetime import datetime

import numpy as np
import torch

logger = logging.getLogger('__main__')

class Options(object):
    def __init__(self):
        fix_seed = 2023
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        self.data = 'custom'
        self.parser = argparse.ArgumentParser(
            description='Run BiLSTM seismic classification pipeline. Optionally, a JSON configuration file can be used to overwrite command-line arguments.')
        
        # Basic configuration
        self.parser.add_argument('--config', dest='config_filepath',
                                help='Configuration .json file (optional). Overwrites existing command-line args!')
        self.parser.add_argument('--output_dir', default=r'output', 
                                help='Root output directory. Must exist. Time-stamped directories will be created inside.')
        
        # Experiment settings
        self.parser.add_argument('--dataset', type=str, default='f3', choices=['f3', 'nz'])
        self.parser.add_argument('--is_training', type=int, default=1, help='training status')
        self.parser.add_argument('--is_testing', type=int, default=0, help='testing status')
        self.parser.add_argument('--task_name', type=str, default='classification')
        self.parser.add_argument('--model', type=str, default='BiLSTM', 
                                help='model name, options: [BiLSTM, iTransformer, iInformer, etc.]')
        
        # Training parameters
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
        self.parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
        self.parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
        self.parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
        self.parser.add_argument('--itr', type=int, default=1, help='experiments times')
        
        # Data proportions
        self.parser.add_argument('--val_proportion', type=float, default=0.001, help='seismic data for validation')
        self.parser.add_argument('--train_proportion', type=float, default=0.01, help='seismic data for train')
        self.parser.add_argument('--test_proportion', type=float, default=0.01, help='seismic data for test')

        # F3 dataset configuration   
        self.parser.add_argument('--root_path', type=str, default=r'/home/dell/disk1/Jinlong/faciesdata', 
                                help='root path of the data file')
        self.parser.add_argument('--data_path', type=str, default='train_seismic.npy', help='data npy file')
        self.parser.add_argument('--label_path', type=str, default='train_labels.npy', help='label npy file')
        self.parser.add_argument('--mask_path', type=str, default='train_labels.npy', help='mask npy file')

        # VMD settings
        self.parser.add_argument('--vmd_data_path', type=str, default='full_F3_vmd.npy', help='vmd npy data file') 
        self.parser.add_argument('--is_vmd', type=bool, default=False, help='whether using VMD')
        self.parser.add_argument('--embedding_flag', type=bool, default=True, help='embedding flag')
        self.parser.add_argument('--mask_rate', type=float, default=0.125, help='mask rate below is masked')
        
        # GPU settings
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        self.parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        self.parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        self.parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')

        # Model checkpoints
        self.parser.add_argument('--checkpoints', type=str, default=r'./checkpoints', 
                                help='location of model checkpoints')
        self.parser.add_argument('--checkpoints_test_only', type=str, 
                                default=r'./checkpoints/best_model.pth',
                                help='location of model checkpoints for testing only')

        # Experiment tracking
        self.parser.add_argument('--model_id', type=str, default='train', help='model id')
        self.parser.add_argument('--exp_name', type=str, default='BiLSTM_Classification', help='experiment name')
        self.parser.add_argument('--des', type=str, default='BiLSTM_seismic_classification', help='exp description')
        
        # Data loader
        self.parser.add_argument('--data', type=str, default='TSF', help='dataset type')
        self.parser.add_argument('--features', type=str, default='S',
                                help='forecasting task, options:[M, S, MS]')
        self.parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        self.parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')

        # BiLSTM specific parameters
        self.parser.add_argument('--seq_len', type=int, default=255, help='input sequence length')
        self.parser.add_argument('--hidden_size', type=int, default=64, help='BiLSTM hidden size')
        self.parser.add_argument('--num_layers', type=int, default=2, help='number of BiLSTM layers')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--bidirectional', type=bool, default=True, help='bidirectional LSTM')
        
        # Input/Output dimensions
        self.parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
        self.parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
        self.parser.add_argument('--c_out', type=int, default=6, help='output size')
        self.parser.add_argument('--num_class', type=int, default=6, help='number of classes')
        self.parser.add_argument('--num_class2', type=int, default=6, help='number of classes (duplicate)')
        
        # Additional parameters for compatibility
        self.parser.add_argument('--d_model', type=int, default=64, help='dimension of model (for compatibility)')
        self.parser.add_argument('--embed_size', type=int, default=64, help='embedding dimension')
        self.parser.add_argument('--max_features', type=int, default=255, help='vocabulary size')
        
        # Optimization
        self.parser.add_argument('--loss', type=str, default='CrossEntropy', help='loss function')
        self.parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        self.parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        
        # Additional experiment settings
        self.parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
        self.parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')

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
    """Load configuration from JSON file"""
    with open(config_filepath) as cnfg:
        config = json.load(cnfg)
    return config

def create_dirs(dirs):
    """Create directories if they don't exist"""
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def setup(args):
    """Prepare training session: read configuration from file, create directories"""
    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:
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