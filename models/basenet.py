import importlib
import os
from abc import abstractmethod
from keras import Input, Model

from keras.callbacks import EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

from callbacks.image_callback import SaveImage
from costs import make_dice_loss_fnc, weighted_cross_entropy_loss, \
    make_mse_loss_func, make_mae_loss_func, make_mse_loss_func_distributed, make_tversky_loss_func, make_focal_loss_func, ypred
import logging

from loaders import loader_factory

log = logging.getLogger('basenet')
eps = 1e-12 # harric added to engage the smooth factor


class BaseNet(object):
    """
    Base model for segmentation neural networks
    """
    def __init__(self, conf):
        self.model = None
        self.conf = conf
        self.loader = None
        if hasattr(self.conf, 'dataset_name') and len(self.conf.dataset_name) > 0:
            self.loader = loader_factory.init_loader(self.conf.dataset_name)

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def load_models(self,postfix='', public_or_split=0):
        if public_or_split == 0:
            if os.path.exists(os.path.join(self.conf.folder, 'model')):
                data_dir = self.conf.folder
            else:
                data_dir = self.conf.folder.split('split')[0][:-1]
        elif public_or_split == -1:
            if 'split' in self.conf.folder:
                data_dir = self.conf.folder.split('split')[0][:-1]
            else:
                data_dir = self.conf.folder
            assert os.path.exists(data_dir), "Not Existing dir: %s" % data_dir
        elif public_or_split == 1:
            data_dir = self.conf.folder
            assert 'split' in data_dir, "It is a public dir: %s" % data_dir
            assert os.path.exists(data_dir), "Not Existing dir: %s" % data_dir



        if postfix == '' and os.path.exists(data_dir + '/model'):
            log.info("load from trained model")
            log.info(data_dir)
            self.model.load_weights(data_dir + '/model')
        elif postfix != '' and os.path.exists(data_dir + '/model'+ '_' + postfix):
            log.info("load from trained model" + ':' + postfix)
            log.info(data_dir)
            self.model.load_weights(data_dir + '/model' + '_' + postfix)


    @abstractmethod
    def save_models(self, postfix='', public=False):
        if public:
            data_dir = self.conf.folder.split('split')[0][:-1]
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            log.info("Saving to the public")
        else:
            data_dir = self.conf.folder


        log.debug('Saving trained model')
        if postfix == '':
            log.info(data_dir)
            self.model.save_weights(data_dir + '/model')
        else:
            self.model.save_weights(data_dir + '/model' + '_' + postfix)
            log.info(data_dir)




    def compile(self):
        assert self.model is not None, 'Model has not been built'

        focal_anato = make_focal_loss_func(self.conf.num_anato_masks+1)
        dice_anato = make_tversky_loss_func(self.conf.num_anato_masks, beta=0.7)
        focal_patho = make_focal_loss_func(2)
        dice_patho = make_tversky_loss_func(1, beta=0.7)
        ce_anato = weighted_cross_entropy_loss()
        ce_patho = weighted_cross_entropy_loss()

        if 'attention' in self.conf.testmode:
            self.model.compile(optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay),
                               loss=[dice_anato, focal_anato,
                                     dice_patho, focal_patho, dice_patho, focal_patho,
                                     ypred, ypred,
                                     ce_anato, ce_anato,
                                     ce_patho, ce_patho, ce_patho, ce_patho],
                               loss_weights=[self.conf.anato_penalty,self.conf.anato_penalty*self.conf.ce_weight,
                                             self.conf.patho1_penalty,self.conf.patho1_penalty*self.conf.ce_weight,
                                             self.conf.patho2_penalty,self.conf.patho2_penalty*self.conf.ce_weight,
                                             self.conf.oot_penalty, self.conf.oot_penalty*self.conf.patho2_penalty/(self.conf.patho1_penalty+eps),
                                             self.conf.attention_map_penalty,self.conf.attention_map_penalty,
                                             self.conf.attention_map_penalty,self.conf.attention_map_penalty,
                                             self.conf.attention_map_penalty,self.conf.attention_map_penalty])
        else:
            self.model.compile(optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay),
                               loss=[dice_anato, focal_anato,
                                     dice_patho, focal_patho, dice_patho, focal_patho,
                                     ypred, ypred],
                               loss_weights=[self.conf.anato_penalty,self.conf.anato_penalty*self.conf.ce_weight,
                                             self.conf.patho1_penalty,self.conf.patho1_penalty*self.conf.ce_weight,
                                             self.conf.patho2_penalty,self.conf.patho2_penalty*self.conf.ce_weight,
                                             self.conf.oot_penalty, self.conf.oot_penalty*self.conf.patho2_penalty/self.conf.patho1_penalty])
    def load_data(self):
        train_data = self.loader.load_data(self.conf.split, 'training')
        valid_data = self.loader.load_data(self.conf.split, 'validation')

        # num_l = int(train_data.num_volumes * self.conf.l_mix)
        num_l = num_l if num_l <= self.conf.data_len else self.conf.data_len
        print('Using %d labelled volumes.' % (num_l))
        train_data.sample(num_l)
        return train_data, valid_data

    # def fit(self):
    #     train_data, valid_data = self.load_data()
    #
    #     es = EarlyStopping(min_delta=0.001, patience=20)
    #     si = SaveImage(os.path.join(self.conf.folder, 'training_results'), train_data.images, train_data.masks)
    #     cl = CSVLogger(os.path.join(self.conf.folder, 'training_results') + '/training.csv')
    #
    #     if not os.path.exists(os.path.join(self.conf.folder, 'training_results')):
    #         os.mkdir(os.path.join(self.conf.folder, 'training_results'))
    #
    #     if self.conf.augment:
    #         datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    #         self.model.fit_generator(datagen.flow(train_data.images, train_data.masks, batch_size=self.conf.batch_size),
    #                                  steps_per_epoch=2 * len(train_data.images) / self.conf.batch_size, epochs=self.conf.epochs,
    #                                  validation_data=(valid_data.images, valid_data.masks))
    #     else:
    #         self.model.train(train_data.images, train_data.masks, validation_data=(valid_data.images, valid_data.masks),
    #                          epochs=self.conf.epochs, callbacks=[es, si, cl], batch_size=self.conf.batch_size)

    @abstractmethod
    def get_segmentor(self):
        """
        Create a model for segmentation
        :return: a keras model
        """
        inp = Input(self.conf.input_shape)
        pred = self.model(inp)
        pred_anatomy = pred[0]
        pred_pathology1 = pred[2]
        pred_pathology2 = pred[4]
        return Model(inputs=inp, outputs=[pred_anatomy,pred_pathology1,pred_pathology2])
