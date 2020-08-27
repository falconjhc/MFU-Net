"""
Entry point for running an SDNet experiment.
"""
import argparse
import importlib
import json
import logging
import os
import matplotlib
matplotlib.use('Agg')  # environment for non-interactive environments

from easydict import EasyDict
from numpy.random import seed
from tensorflow import set_random_seed
import shutil
# harric added to disable ignoring futurewarning messages
import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=FutureWarning)
#     import tensorflow as tf
#     from tensorflow import keras
#     from tensorflow.keras.preprocessing.text import Tokenizer
seed(1)
set_random_seed(1)

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


class Experiment(object):
    """
    Experiment class reads the configuration parameters (stored under configuration folder) and execute the experiment.
    Required command line arguments are:
        --config    the configuration file name.
        --split     split number for cross validation, e.g. 0, 1, ...

    Optional command line arguments are:
        --test          only test a model defined by the configuration file
        --l_mix         float [0, 1]. Sets the amount of labelled data.
        --augment       Use data augmentation
        --modality      Set the modality to load. Used in multimodal datasets.
    """
    def __init__(self):
        self.log = None

    def init_logging(self, config):
        if not os.path.exists(config.test_folder):
            os.makedirs(config.test_folder)
        logging.basicConfig(filename=config.test_folder + '/logfile.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())

        self.log = logging.getLogger()
        self.log.debug(config.items())
        self.log.info('---- Setting up experiment at ' + config.folder + '----')

    def get_config(self, args):
        """
        Read a config file and convert it into an object for easy processing.
        :param split: the cross-validation split id
        :param args:  the command arguments
        :return: config object(namespace)
        """
        config_script = args.config


        # harric modified to incorporate with the segmentation_option argument
        config_dict = importlib.import_module('configuration.' + config_script).get(args.testmode)
        test_folder = args.testdir + config_dict['folder'] + '_' + args.testmode

        if 'split' in args.expdir:
            split_id = args.expdir.split('split')[-1]
            test_folder = test_folder + '_split' + split_id

        config_dict.update({'test_folder': test_folder})
        config = EasyDict(config_dict)
        #config.segmentation_option = args.segmentation_option
        # config.loss_type = args.loss_type



        config.test_folder = config.test_folder.replace('experiment', 'challenge')
        config.folder = args.expdir
        config.filters = config.filters if args.filters == None else args.filters



        # harric added to include segmentation option info
        # in the created experimental directory
        #config.folder += ('_segopt%s' % config.segmentation_option)
        #config.folder+=('_losstype_%s' % config.loss_type)

        # if 'rohan' in config_script and config.infarction_weight!=1:
        #     config.folder += ('_infwgh%d' % config.infarction_weight)
        #config.folder += '_split%s' % split
        config.folder = config.folder.replace('.', '')

        # if args.augment:
        #     config.augment = args.augment
        config.load_pretrain = False
        self.save_config(config)
        return config

    def save_config(self, config):
        if os.path.exists(config.test_folder):
            shutil.rmtree(config.test_folder)
        os.makedirs(config.test_folder)
        with open(config.test_folder + '/experiment_configuration.json', 'w') as outfile:
            json.dump(dict(config.items()), outfile)

    def run(self):



        args = Experiment.read_console_parameters()
        expdir = args.expdir
        expname = expdir.split('/')
        if expdir[-1]=='/':
            expname = expname[-2]
        else:
            expname = expname[-1]

        if 'unet' in expname:
            args.config = 'unet_multi_modal_cardiac'
        elif 'psdnet' in expname:
            args.config = 'psdnet_multi_modal_cardiac'

        testmode = ''
        if 'feature-concat' in expname:
            testmode = 'feature-concat'
        elif 'pixel-concat' in expname:
            testmode = 'pixel-concat'

        if 'attention' in expname:
            testmode = testmode + '-' + 'attention'

        if 'merge-encoder' in expname:
            testmode = testmode + '-' + 'merge-endcoder'

        if 'maxfuse' in expname:
            if 'maxfuseall' in expname:
                testmode = testmode + '-' + 'maxfuseall'
            else:
                testmode = testmode + '-' + 'maxfuse'
            if 'keeporg' in expname:
                testmode = testmode + '-' + 'keeporg'

        if 'sideconv' in expname:
            testmode = testmode + '-' + 'sideconv'

        if 'psdnet' in expname:
            if 'noround' in expname:
                testmode = testmode + '-' + 'noround'
            elif 'simperound' in expname:
                testmode = testmode + '-' + 'simperound'
            elif 'onehotround' in expname:
                testmode = testmode + '-' + 'onehotround'

        args.testmode = testmode

        configuration = self.get_config(args)
        self.init_logging(configuration)
        self.run_experiment(configuration)


    def run_experiment(self, configuration):
        executor = self.get_executor(configuration)

        executor.challenge()

        # if test:
        #     executor.test()
        #     executor.test(best_mark=True)
        # else:
        #     executor.train()
        #     with open(configuration.folder + '/experiment_configuration.json', 'w') as outfile:
        #        json.dump(vars(configuration), outfile)
        #     executor.test()
        #     executor.test(best_mark=True)

    @staticmethod
    def read_console_parameters():
        parser = argparse.ArgumentParser(description='')
        # parser.add_argument('--config', default='', help='The experiment configuration file', required=True)
        # parser.add_argument('--test', help='Evaluate the model on test data', type=bool)
        # parser.add_argument('--split', help='Data split to run.', required=True)
        # parser.add_argument('--l_mix', help='Percentage of labelled data')
        # parser.add_argument('--augment', help='Augment training data', type=bool)
        # parser.add_argument('--modality', help='Modality to load', choices=['MR', 'CT', 'all', 'cine', 'BOLD'])
        # parser.add_argument('--testmode', help='Option to testmode')
        parser.add_argument('--expdir', default='', help='The experiment configuration file', required=True)
        parser.add_argument('--testdir', default='', help='The test configuration file', required=True)
        parser.add_argument('--filters', help='filters', type=int)



        # harric added to input the segmentation_option argument when performing the training


        # parser.add_argument('--infarction_weight', help='infarction section weight during training',
        #                     default=1)
        # # harric added to input the infarction_weight argument when performing the training

        # parser.add_argument('--loss_type',help='Which loss is going to be utilized', default='agis')
        # harric added to input the loss_type argument when performing the training

        return parser.parse_args()

    def get_executor(self, config):
        # Initialise model
        module_name = config.model.split('.')[0]
        model_name = config.model.split('.')[1]
        model = getattr(importlib.import_module('models.' + module_name), model_name)(config)

        if 'split' in config.folder:
            public_or_split = 1
        else:
            public_or_split = -1


        if 'concat' in config.testmode:
            if 'unet' in config.model:
                model.build_concat(public_or_split=public_or_split)
            else:
                model.build(public_or_split=public_or_split)
        elif 'cascaded' in config.testmode:
            model.build_cascaded_unet()
        # model.compile()

        # if config.l_mix == 0.015 and config.split == 1:
        #     config.seed = 10

        # Initialise executor
        module_name = config.executor.split('.')[0]
        model_name = config.executor.split('.')[1]
        executor = getattr(importlib.import_module('model_executors.' + module_name), model_name)(config, model)
        return executor


if __name__ == '__main__':
    exp = Experiment()
    exp.run()
