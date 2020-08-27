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
        if not os.path.exists(config.folder):
            os.makedirs(config.folder)
        logging.basicConfig(filename=config.folder + '/logfile.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
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
        config = EasyDict(config_dict)
        #config.segmentation_option = args.segmentation_option
        # config.loss_type = args.loss_type
        config.folder += '_' + config.testmode + '_pretrain'


        config.lr = config.lr if args.lr == None else  args.lr
        config.epochs = config.epochs if args.epochs == None else args.epochs
        config.patience = config.patience if args.patience == None else args.patience
        config.public_or_split = 0
        config.load_pretrain = False
        config.filters = config.filters if args.filters == None else args.filters



        # harric added to include segmentation option info
        # in the created experimental directory
        #config.folder += ('_segopt%s' % config.segmentation_option)
        #config.folder+=('_losstype_%s' % config.loss_type)

        # if 'rohan' in config_script and config.infarction_weight!=1:
        #     config.folder += ('_infwgh%d' % config.infarction_weight)
        config.folder = config.folder.replace('.', '')



        self.save_config(config)
        return config

    def save_config(self, config):
        if not os.path.exists(config.folder):
            os.makedirs(config.folder)
        with open(config.folder + '/experiment_configuration.json', 'w') as outfile:
            json.dump(dict(config.items()), outfile)

    def run(self):
        args = Experiment.read_console_parameters()

        configuration = self.get_config(args)
        self.init_logging(configuration)
        self.run_experiment(configuration, args.test)

    def run_experiment(self, configuration, test):
        executor = self.get_executor(configuration)

        executor.train()
        with open(configuration.folder + '/experiment_configuration.json', 'w') as outfile:
            json.dump(vars(configuration), outfile)

    @staticmethod
    def read_console_parameters():
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--config', default='', help='The experiment configuration file', required=True)
        parser.add_argument('--test', help='Evaluate the model on test data', type=bool)
        parser.add_argument('--testmode', help='Option to testmode')
        parser.add_argument('--lr', help='learning rate',type=float)
        parser.add_argument('--epochs', help='epochs', type=int)
        parser.add_argument('--patience', help='patience', type=int)
        parser.add_argument('--public_or_split', help='public_or_split', type=int, default=0)
        parser.add_argument('--filters', help='filters', type=int)
        # parser.add_argument('--load_pretrain', help='load_pretrain', type=bool, default=False)


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

        if 'concat' in config.testmode:
            if 'unet' in config.model:
                model.build_concat(public_or_split=config.public_or_split)
            else:
                model.build(public_or_split=config.public_or_split)
        elif 'cascaded' in config.testmode:
            model.build_cascaded_unet()
        model.compile()



        # Initialise executor
        module_name = config.executor.split('.')[0]
        model_name = config.executor.split('.')[1]
        executor = getattr(importlib.import_module('model_executors.' + module_name), model_name)(config, model)
        return executor




if __name__ == '__main__':
    exp = Experiment()
    exp.run()


def running():
    exp = Experiment()
    exp.run()