import configparser
import torch
import os


class ParamConfigurator:
    """Parameter configurator class for deep learning pipeline."""

    def __init__(self):
        """# TODO: Docstring"""
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Global
        self.seed = config['global'].getint('seed')
        self.train = config['global']['train']
        self.device = config['global']['device']

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Data
        self.train_dir = config['data']['train_directory']
        self.test_dir = config['data']['test_directory']
        self.artifact_dir = config['data']['artifact_directory']

        if not os.path.exists(self.artifact_dir):
            os.mkdir(self.artifact_dir)

        # Architecture
        self.model_name = config['architecture']['model_name']
        self.f_maps = config['architecture'].getint('f_maps')
        self.levels = config['architecture'].getint('levels')
        self.residual_block = config['architecture'].getboolean('residual_block')
        self.se_block = config['architecture']['se_block']
        self.attention = config['architecture'].getboolean('attention')
        self.MHTSA_heads = config['architecture'].getint('MHTSA_heads')
        self.MHGSA_heads = config['architecture'].getint('MHGSA_heads')
        self.trilinear = config['architecture'].getboolean('trilinear')
        self.MSSC = config['architecture']['MSSC']

        # Training
        self.train_val_ratio = config['training'].getfloat('train_val_ratio')
        self.batch_size = config['training'].getint('batch_size')
        self.epochs = config['training'].getint('epochs')
        self.learning_rate = config['training'].getfloat('learning_rate')
        self.ignore_index = config['training'].getint('CE_ignore_index')
        self.include_background = config['training'].getboolean('DICE_include_background')
        self.num_workers = config['training'].getint('num_workers')
        self.pin_memory = config['training'].getboolean('pin_memory')

        # Dataset
        self.crop = config['dataset']['crop']
        self.crop_margin = config['dataset'].getint('crop_margin')
        self.data_augmentation = config['dataset'].getboolean('data_augmentation')
        self.data_transforms = config['dataset']['data_transforms']
