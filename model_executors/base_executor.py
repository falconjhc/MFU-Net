import logging
import numpy as np
import os
from abc import abstractmethod
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from callbacks.image_callback import SaveEpochImages
from loaders import loader_factory
from utils.image_utils import save_segmentation, generate_attentions
from utils.image_utils import image_show
from costs import dice, calculate_false_negative # harric added regression2segmentation to incorporate with segmentation_option=4 case
from utils.image_utils import regression2segmentation
log = logging.getLogger('executor')
from keras.utils import Progbar
from imageio import imwrite as imsave # harric modified
from scipy.ndimage.morphology import binary_fill_holes
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from datetime import datetime
import random
import re
import math
elastic_alpha=1
elastic_sigma=0.25
elastic_alpha_affine=0.025
# challenge_full_size = 384
upf288=128-32+15
downf288=352+32+15
leftf288=131-32-15
rightf288=355+32-15

upf224=128
downf224=352
leftf224=131
rightf224=355
import nibabel as nib
import shutil
import utils.data_utils
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage.measure import label

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from parameters import conf as dataset_path
from model_executors.dynamic_sampling import dynamic_sample_implementation as dynamic_sampling


class Executor(object):
    """
    Base class for executor objects.
    """
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.loader = loader_factory.init_loader(self.conf.dataset_name)
        self.epoch = 0
        self.models_folder = self.conf.folder + '/models'
        self.train_data = None
        self.valid_data = None
        self.train_folder = None
        self.lr_schedule_coef = -math.log(0.1) / self.conf.epochs
        # l_mix = self.conf.l_mix
        # self.conf.l_mix = float(l_mix.split('-')[0])
        # self.conf.pctg_per_volume = float(l_mix.split('-')[1])


    @abstractmethod
    def init_train_data(self):
        self.train_data = \
            self.loader.load_labelled_data(self.conf.split, 'training',
                                           downsample=self.conf.image_downsample)
        self.valid_data = \
            self.loader.load_labelled_data(self.conf.split, 'validation',
                                           downsample=self.conf.image_downsample)

        # harric added modality and segmentation_option auguments
        #self.train_data.select_masks(self.conf.num_masks)
        #self.valid_data.select_masks(self.conf.num_masks)

        # self.train_data.sample_per_volume(-1, self.conf.pctg_per_volume, seed=self.conf.seed)
        # self.train_data.sample_by_volume(int(self.conf.l_mix * self.train_data.num_volumes), seed=self.conf.seed)

        self.conf.data_len = self.train_data.size()
        self.conf.batches = int(np.ceil(self.conf.data_len / self.conf.batch_size))

    @abstractmethod
    def get_loss_names(self):
        pass


    @abstractmethod
    def init_data_generators(self, train_images, train_anato_masks, train_patho_masks, valid_images):
        supervised_anatomy_pathology_augmentation_dict, \
        supervised_pathology_augmentation_dict, \
        unsupervised_reconstruction_augmentation_dict = self.get_datagen_params()
        supervised_anatomy_pathology_data_gen = ImageDataGenerator(**supervised_anatomy_pathology_augmentation_dict)
        supervised_pathology_data_gen = ImageDataGenerator(**supervised_pathology_augmentation_dict)
        unsupervised_reconstruction_data_gen = ImageDataGenerator(**unsupervised_reconstruction_augmentation_dict)

        x = train_images
        x_and_y1 = np.concatenate([x, train_anato_masks, train_patho_masks], axis=-1)
        supervised_anatomy_pathology_gen_iterator \
            = supervised_anatomy_pathology_data_gen.flow(x=x_and_y1,
                                                         batch_size=self.conf.batch_size,
                                                         seed=self.conf.seed)

        x_and_y2 = np.concatenate([x, train_patho_masks], axis=-1)
        supervised_pathology_gen_iterator \
            = supervised_pathology_data_gen.flow(x=x_and_y2,
                                                 batch_size=self.conf.batch_size,
                                                 seed=self.conf.seed)

        unsupervised_reconstruction_gen_iterator \
            = unsupervised_reconstruction_data_gen.flow(x=np.concatenate([train_images,valid_images], axis=0),
                                                        batch_size=self.conf.batch_size,
                                                        seed=self.conf.seed)

        return supervised_anatomy_pathology_gen_iterator, supervised_pathology_gen_iterator, unsupervised_reconstruction_gen_iterator

    @abstractmethod
    def init_data_single_supervising_generators(self, train_images, train_anato_masks, train_patho_masks, valid_images):
        _, \
        supervised_augmentation_dict, \
        unsupervised_reconstruction_augmentation_dict = self.get_datagen_params()
        supervised_data_gen = ImageDataGenerator(**supervised_augmentation_dict)
        unsupervised_reconstruction_data_gen = ImageDataGenerator(**unsupervised_reconstruction_augmentation_dict)

        x = train_images
        x_and_y1 = np.concatenate([x, train_anato_masks, train_patho_masks], axis=-1)
        supervised_gen_iterator \
            = supervised_data_gen.flow(x=x_and_y1,
                                       batch_size=self.conf.batch_size,
                                       seed=self.conf.seed)

        # x_and_y2 = np.concatenate([x, train_patho_masks], axis=-1)
        # supervised_pathology_gen_iterator \
        #     = supervised_pathology_data_gen.flow(x=x_and_y2,
        #                                          batch_size=self.conf.batch_size,
        #                                          seed=self.conf.seed)

        unsupervised_reconstruction_gen_iterator \
            = unsupervised_reconstruction_data_gen.flow(x=np.concatenate([train_images, valid_images], axis=0),
                                                        batch_size=self.conf.batch_size,
                                                        seed=self.conf.seed)

        return supervised_gen_iterator, unsupervised_reconstruction_gen_iterator


    def init_data_generators_basenet(self, train_images, train_anato_masks, train_patho_masks):
        _, \
        _, \
        augment_dict = self.get_datagen_params()
        data_generator = ImageDataGenerator(**augment_dict)
        # supervised_pathology_data_gen = ImageDataGenerator(**supervised_pathology_augmentation_dict)
        # unsupervised_reconstruction_data_gen = ImageDataGenerator(**unsupervised_reconstruction_augmentation_dict)

        x_and_y = np.concatenate([train_images, train_anato_masks, train_patho_masks], axis=-1)
        data_generator_iterator \
            = data_generator.flow(x=x_and_y,
                                  batch_size=self.conf.batch_size,
                                  seed=self.conf.seed)

        # x_and_y2 = np.concatenate([x, train_patho_masks], axis=-1)
        # supervised_pathology_gen_iterator \
        #     = supervised_pathology_data_gen.flow(x=x_and_y2,
        #                                          batch_size=self.conf.batch_size,
        #                                          seed=self.conf.seed)
        #
        # unsupervised_reconstruction_gen_iterator \
        #     = unsupervised_reconstruction_data_gen.flow(x=np.concatenate([train_images,valid_images], axis=0),
        #                                                 batch_size=self.conf.batch_size,
        #                                                 seed=self.conf.seed)

        return data_generator_iterator


    @abstractmethod
    def init_validation_generators(self,images, anato_masks, patho_masks):
        anato_background = np.zeros_like(anato_masks[:,:,:,0:1])
        for ii in range(anato_masks.shape[-1]):
            anato_background = anato_background - anato_masks[:,:,:,ii:ii+1]
        anato_masks = np.concatenate([anato_masks, anato_background], axis=-1)

        patho1_masks = patho_masks[:, :, :, 0:1]
        patho2_masks = patho_masks[:, :, :, 1:2]
        patho1_background = np.zeros_like(patho1_masks)
        patho2_background = np.zeros_like(patho2_masks)
        patho1_background = patho1_background - patho1_masks
        patho2_background = patho2_background - patho2_masks
        patho1_masks = np.concatenate([patho1_masks, patho1_background], axis=-1)
        patho2_masks = np.concatenate([patho2_masks, patho2_background], axis=-1)

        supervised_anatomy_list = [anato_masks,anato_masks]
        supervised_pathology_list = [patho1_masks, patho1_masks,
                                     patho2_masks, patho2_masks,
                                     np.zeros_like(patho1_masks),
                                     np.zeros_like(patho2_masks)]

        # unsupervised_reconstructor_list = [images[:,:,:,0:1],images[:,:,:,1:2],images[:,:,:,2:3]]
        attention_list = []
        if 'attention' in self.conf.testmode:
            attention_list =  [anato_masks,anato_masks] + [patho1_masks, patho1_masks, patho2_masks, patho2_masks]
        return [images, supervised_anatomy_list + supervised_pathology_list + attention_list]


    @abstractmethod
    def train(self):
        def _learning_rate_schedule(epoch):
            return self.conf.lr * math.exp(self.lr_schedule_coef * (-epoch - 1))

        def _elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
            """Elastic deformation of images as described in
            [Simard2003]_ (with modifications).
            .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
                 Convolutional Neural Networks applied to Visual Document Analysis", in
                 Proc. of the International Conference on Document Analysis and
                 Recognition, 2003.

             Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
            """
            if random_state is None:
                random_state = np.random.RandomState(random.seed(datetime.now()))

            shape = image.shape
            shape_size = shape[:2]

            # Random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32([center_square + square_size,
                               [center_square[0] + square_size,
                                center_square[1] - square_size],
                               center_square - square_size])
            pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dz = np.zeros_like(dx)

            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

            return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

        def _supervising_data_generator(gen_iterator):
            while True:
                batch_x_and_y = next(gen_iterator)
                batch_x = batch_x_and_y[:, :, :, 0:self.conf.input_shape[-1]]
                batch_y_raw = batch_x_and_y[:, :, :, self.conf.input_shape[-1]:]
                batch_x, batch_y_raw = dynamic_sampling(batch_x,batch_y_raw)

                batch_x_new = batch_x
                if 'elastic' in self.conf.testmode:
                    batch_x_new = []
                    for ii in range(batch_x.shape[0]):
                        this_pack = batch_x[ii, :, :, :]
                        this_pack_new = _elastic_transform(this_pack,
                                                           this_pack.shape[1] * elastic_alpha,
                                                           this_pack.shape[1] * elastic_sigma,
                                                           this_pack.shape[1] * elastic_alpha_affine)
                        batch_x_new.append(np.expand_dims(this_pack_new, axis=0))
                    batch_x_new = np.concatenate(batch_x_new, axis=0)

                
                batch_y = []

                anato_masks = np.round(batch_y_raw[:, :, :, :self.conf.num_anato_masks])
                background_anato = np.ones(shape=anato_masks.shape[:-1] + (1,))
                for ii in range(anato_masks.shape[3]):
                    background_anato = background_anato - np.expand_dims(anato_masks[:, :, :, ii], axis=-1)
                anato_masks = np.concatenate([anato_masks, background_anato], axis=-1)

                patho1_masks = np.round(batch_y_raw[:, :, :, self.conf.num_anato_masks])
                background_patho1 = np.ones_like(patho1_masks)
                background_patho1 = background_patho1 - patho1_masks
                patho1_masks = np.concatenate([np.expand_dims(patho1_masks, axis=-1),
                                               np.expand_dims(background_patho1, axis=-1)], axis=-1)

                patho2_masks = np.round(batch_y_raw[:, :, :, self.conf.num_anato_masks + 1])
                background_patho2 = np.ones_like(patho2_masks)
                background_patho2 = background_patho2 - patho2_masks
                patho2_masks = np.concatenate([np.expand_dims(patho2_masks, axis=-1),
                                               np.expand_dims(background_patho2, axis=-1)], axis=-1)

                # y for anatomy
                for ii in range(2):
                    batch_y.append(anato_masks)

                # y for pathology
                for ii in range(2):
                    batch_y.append(patho1_masks)
                for ii in range(2):
                    batch_y.append(patho2_masks)
                batch_y.append(np.zeros_like(patho1_masks[:, :, :, 0:1]))
                batch_y.append(np.zeros_like(patho2_masks[:, :, :, 0:1]))

                # for attention
                if 'attention' in self.conf.testmode:
                    for ii in range(2):
                        batch_y.append(anato_masks)
                    for ii in range(2):
                        batch_y.append(patho1_masks)
                    for ii in range(2):
                        batch_y.append(patho2_masks)

                # # y for reconstruction
                # batch_y.append(batch_x_new[:, :, :, 0:1])
                # batch_y.append(batch_x_new[:, :, :, 1:2])
                # batch_y.append(batch_x_new[:, :, :, 2:3])

                yield batch_x,batch_y


        log.info('Training Model')
        self.init_train_data()
        lr_callback = LearningRateScheduler(_learning_rate_schedule)

        self.train_folder = os.path.join(self.conf.folder, 'training_results')
        if not os.path.exists(self.train_folder):
            os.mkdir(self.train_folder)
        self.conf.batches = int(np.ceil(self.conf.data_len / self.conf.batch_size))
        callbacks = self.init_callbacks()
        callbacks.append(lr_callback)

        train_images = self.get_inputs(self.train_data)
        train_anato_labels = self.get_anato_labels(self.train_data)
        train_patho_labels = self.get_patho_labels(self.train_data)

        valid_images = self.get_inputs(self.valid_data)
        valid_anato_labels = self.get_anato_labels(self.valid_data)
        valid_patho_labels = self.get_patho_labels(self.valid_data)

        data_gen = self.init_data_generators_basenet(train_images, train_anato_labels, train_patho_labels)
        valid_supervised = self.init_validation_generators(valid_images, valid_anato_labels, valid_patho_labels)
        train_supervised = self.init_validation_generators(train_images, train_anato_labels, train_patho_labels)

        data_gen_iterator = _supervising_data_generator(data_gen)

        supervised = self.model.model.fit_generator(data_gen_iterator,
                                                    steps_per_epoch=len(train_images) / self.conf.batch_size,
                                                    epochs=self.conf.epochs, validation_data=valid_supervised, callbacks=callbacks)
        self.model.save_models(public=self.conf.public_or_split==-1)
        if not 'attention' in self.conf.testmode:
            self.model.save_pretrain_model_from_non_attention_split()




    def init_callbacks(self):
        _,_,datagen_dict = self.get_datagen_params()

        data_pack = np.concatenate([self.train_data.images,
                                    self.train_data.anato_masks,
                                    self.train_data.patho_masks], axis=-1)
        # image_channels = self.train_data.images.shape[-1]
        # anato_mask_channels = self.train_data.anato_masks.shape[-1]
        # patho_mask_channels = self.train_data.patho_masks.shape[-1]
        gen = ImageDataGenerator(**datagen_dict).flow(x=data_pack,
                                                      batch_size=self.conf.batch_size,
                                                      seed=self.conf.seed)

        es = EarlyStopping(min_delta=self.conf.min_delta, patience=self.conf.patience)
        if 'attention' in self.conf.testmode:
            attention_map = self.model.get_attention_maps()
            attention_out = self.model.get_attention_output()
        else:
            attention_map = attention_out = None

        if 'psdnet' in self.conf.model:
            si = SaveEpochImages(self.conf, self.model, gen,
                                 attention_map, attention_out, self.model.get_input_full(),
                                 self.model.get_segmentor(), self.model.Enc_Anatomy, self.model.Enc_Modality, self.model.Decoder)
        else:
            si = SaveEpochImages(self.conf, self.model, gen,
                                 attention_map, attention_out, self.model.get_input_full(),
                                 self.model.get_segmentor(), None,None,None)
        cl = CSVLogger(self.train_folder + '/training.csv')
        mc = ModelCheckpoint(self.conf.folder + '/model', monitor='val_loss', verbose=0, save_best_only=False,
                             save_weights_only=True, mode='min', period=1)
        mc_best = ModelCheckpoint(self.conf.folder + '/model_best', monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=True, mode='min', period=1)
        return [es, si, cl, mc, mc_best]

    def get_anato_labels(self, data):
        """
        :param data: the Data object used in training
        :return:     the network's target, usually the masks
        """
        return data.anato_masks


    def correct_lv(self,anato, lv_size_threshold):
        myo = anato[:,:,:,0]
        lv = anato[:,:,:,1]

        for batch_counter in range(myo.shape[0]):
            curt_myo = myo[batch_counter,:,:]
            curt_lv = lv[batch_counter,:,:]
            curt_myo_filled =np.int64(binary_fill_holes(curt_myo)) - curt_myo

            if np.sum(curt_lv) < lv_size_threshold:
                anato[batch_counter,:,:,1] = curt_myo_filled

        return anato

    def dense_crf(self,img, prob, iters):
        full_list = []
        for batch_counter in range(img.shape[0]):
            curt_img = img[batch_counter, :, :]
            curt_img = np.tile(np.expand_dims(curt_img, axis=0), [3, 1, 1])
            curt_img = np.swapaxes(curt_img, 0, 2)
            mask_list = []
            for mask_counter in range(prob.shape[-1]):

                curt_prob = prob[batch_counter,:,:,mask_counter]
                curt_prob = np.expand_dims(curt_prob, axis=0)
                curt_prob = np.swapaxes(curt_prob, 1, 2)  # shape: (1, width, height)

                num_classes = 2
                curt_probs = np.tile(curt_prob, (num_classes, 1, 1))  # shape: (2, width, height)
                curt_probs[0] = np.subtract(1, curt_prob)  # class 0 is background
                curt_probs[1] = curt_prob  # class 1 is car

                U = unary_from_softmax(curt_probs)
                d = dcrf.DenseCRF2D(curt_img.shape[0], curt_img.shape[1], num_classes)
                d.setUnaryEnergy(U)

                pairwise_energy = create_pairwise_bilateral(sdims=(3, 3), schan=(0.01,), img=curt_img, chdim=2)
                d.addPairwiseEnergy(pairwise_energy, compat=3)  # `compat` is the "strength" of this potential.


                feats = create_pairwise_gaussian(sdims=(3, 3), shape=curt_img.shape[:2])
                d.addPairwiseEnergy(feats, compat=3)




                Q, tmp1, tmp2 = d.startInference()
                for _ in range(iters[mask_counter]):
                    d.stepInference(Q, tmp1, tmp2)
                map_soln = np.argmax(Q, axis=0).reshape((curt_img.shape[0], curt_img.shape[1]))
                mask_list.append(np.expand_dims(map_soln,axis=-1))
                del U, d
            curt_mask_crf = np.concatenate(mask_list,axis=-1)
            full_list.append(np.expand_dims(curt_mask_crf,axis=0))
        crf_mask = np.concatenate(full_list,axis=0)
        return crf_mask

    def get_patho_labels(self, data):
        """
        :param data: the Data object used in training
        :return:     the network's target, usually the masks
        """
        return data.patho_masks

    def get_inputs(self, data):
        """
        :param data: the Data object used in training
        :return:     the network's input, usually the images
        """
        return data.images


    def challenge(self, best_mark=False):

        def _linear_scale(img):
            img = img - np.min(img)
            img = img / np.max(img)
            img = img * 255
            return np.float64(img)/255


        org_data_path = dataset_path['multimodalcardiac_challenge_source']

        org_data_list = [ii for ii in os.listdir(org_data_path) if not ii.startswith('.')]
        org_data_list.sort()
        # if best_mark:
        #     self.model.load_models('best')
        #     log.info('Evaluating the best model on test data')
        #     # test_folder = os.path.join(self.conf.folder, 'best_test_results_%s' % self.conf.test_dataset)
        #     # train_folder = os.path.join(self.conf.folder, 'best_training_results_%s' % self.conf.test_dataset)
        # else:
        #     self.model.load_models()
        #     log.info('Evaluating model on test data')
        #     # test_folder = os.path.join(self.conf.folder, 'test_results_%s' % self.conf.test_dataset)
        #     # train_folder = os.path.join(self.conf.folder, 'training_results_%s' % self.conf.test_dataset)

        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_data = test_loader.load_challenge_data(downsample=self.conf.image_downsample)
        segmentor = self.model.get_segmentor()


        for counter, vol_i in enumerate(test_data.volumes()):

            print("Counter:%d/%d" % (counter+1, len(test_data.volumes())))
            test_image = test_data.get_images(vol_i)
            patient = test_data.get_patient(vol_i)
            assert len(np.unique(patient))==1, "Incorret patient indices"
            patient = np.unique(patient)[0]
            related_img_path_list = [ii for ii in org_data_list if patient in ii]
            assert len(np.unique(related_img_path_list)) == 3, "Incorret modality number"
            bssfp_img_path = [ii for ii in related_img_path_list if 'C0' in ii][0]
            lge_img_path = [ii for ii in related_img_path_list if 'DE' in ii][0]
            t2_img_path = [ii for ii in related_img_path_list if 'T2' in ii][0]
            bssfp_img = _linear_scale(nib.load(os.path.join(org_data_path, bssfp_img_path)).get_data())
            lge_img = _linear_scale(nib.load(os.path.join(org_data_path, lge_img_path)).get_data())
            t2_img = _linear_scale(nib.load(os.path.join(org_data_path, t2_img_path)).get_data())
            challenge_full_size1 = bssfp_img.shape[0]
            challenge_full_size2 = bssfp_img.shape[1]

            bssfp_img_cropped = bssfp_img[upf288:downf288,leftf288:rightf288,:]
            lge_img_cropped = lge_img[upf288:downf288, leftf288:rightf288, :]
            t2_img_cropped = t2_img[upf288:downf288, leftf288:rightf288, :]

            p5 = np.percentile(bssfp_img_cropped.flatten(), 5)
            p95 = np.percentile(bssfp_img_cropped.flatten(), 95)
            bssfp_img_cropped = np.clip(bssfp_img_cropped, p5, p95)

            lge_img_cropped = utils.data_utils.normalise(lge_img_cropped, -1, 1)
            t2_img_cropped = utils.data_utils.normalise(t2_img_cropped, -1, 1)
            bssfp_img_cropped = utils.data_utils.normalise(bssfp_img_cropped, -1, 1)

            lge_img_cropped = np.moveaxis(lge_img_cropped, 2, 0)
            t2_img_cropped = np.moveaxis(t2_img_cropped, 2, 0)
            bssfp_img_cropped = np.moveaxis(bssfp_img_cropped, 2, 0)

            test_data_new = np.concatenate([np.expand_dims(lge_img_cropped,axis=-1),
                                            np.expand_dims(t2_img_cropped,axis=-1),
                                            np.expand_dims(bssfp_img_cropped,axis=-1)],axis=-1)

            pred = segmentor.predict(test_data_new)
            # pred = segmentor.predict(test_image)

            pred_anato = np.round(pred[0][:, :, :, :self.conf.num_anato_masks])
            pred_infarct = np.round(pred[1][:, :, :, 0])
            pred_edema = np.round(pred[2][:, :, :, 0])

            pred_patho = np.concatenate([np.expand_dims(pred_infarct,axis=-1),
                                         np.expand_dims(pred_edema,axis=-1)], axis=-1)

            pred_anato = self.correct_with_largest_connection(pred_anato)
            pred_patho = self.correct_with_largest_connection(pred_patho)

            pred_infarct = pred_patho[:,:,:,0]
            pred_edema = pred_patho[:, :, :, 1]


            pred_myo = pred_anato[:,:,:,0]
            pred_lv = pred_anato[:, :, :, 1]
            pred_rv = pred_anato[:, :, :, 2]

            pred_infarct = pred_infarct * pred_myo
            pred_edema = pred_edema * pred_myo
            union_pathology = pred_edema + pred_infarct
            union_pathology[np.where(union_pathology > 1)] = 1
            pred_normal_myo = pred_myo - union_pathology

            full_list = []
            full_row = []
            for batch_counter in range(pred_normal_myo.shape[0]):
                curt_normal_myo = pred_normal_myo[batch_counter,:,:]
                curt_full_myo = pred_myo[batch_counter,:,:]
                curt_lv = pred_lv[batch_counter,:,:]
                curt_rv = pred_rv[batch_counter,:,:]
                curt_inf = pred_infarct[batch_counter,:,:]
                curt_ede = pred_edema[batch_counter,:,:]

                curt_normal_myo_padded = np.zeros_like(bssfp_img[:,:,batch_counter])
                curt_lv_padded = np.zeros_like(bssfp_img[:,:,batch_counter])
                curt_rv_padded = np.zeros_like(bssfp_img[:,:,batch_counter])
                curt_inf_padded = np.zeros_like(bssfp_img[:,:,batch_counter])
                curt_ede_padded = np.zeros_like(bssfp_img[:,:,batch_counter])
                curt_full_myo_padded = np.zeros_like(bssfp_img[:,:,batch_counter])

                curt_normal_myo_padded[upf288:downf288, leftf288:rightf288]=curt_normal_myo
                curt_lv_padded[upf288:downf288, leftf288:rightf288] = curt_lv
                curt_rv_padded[upf288:downf288, leftf288:rightf288] = curt_rv
                curt_inf_padded[upf288:downf288, leftf288:rightf288] = curt_inf
                curt_ede_padded[upf288:downf288, leftf288:rightf288] = curt_ede
                curt_full_myo_padded[upf288:downf288, leftf288:rightf288] = curt_full_myo

                current_row = np.concatenate([lge_img[:,:,batch_counter],
                                              t2_img[:,:,batch_counter],
                                              bssfp_img[:,:,batch_counter],
                                              curt_full_myo_padded,
                                              curt_lv_padded,
                                              curt_rv_padded,
                                              curt_inf_padded,
                                              curt_ede_padded], axis=1)

                curt_full_size = np.zeros(shape=(curt_normal_myo_padded.shape+(1,)),dtype='i2')
                curt_full_size[np.where(curt_normal_myo_padded==1)]=200
                curt_full_size[np.where(curt_lv_padded == 1)] = 500
                curt_full_size[np.where(curt_rv_padded == 1)] = 600
                curt_full_size[np.where(curt_ede_padded == 1)] = 1220
                curt_full_size[np.where(curt_inf_padded == 1)] = 2221
                # curt_full = np.zeros(shape=(challenge_full_size1,challenge_full_size2)+(1,),dtype='i2')
                # curt_full[upf288:downf288, leftf288:rightf288, :] = curt_crop
                full_list.append(curt_full_size)
                full_row.append(current_row)

            full_mask = np.concatenate(full_list,axis=-1)
            affine = nib.load(os.path.join(org_data_path, bssfp_img_path)).affine
            write_img = nib.Nifti1Image(full_mask,affine)
            nib.save(write_img,os.path.join(self.conf.test_folder,'myops_test_%s_seg.nii.gz' % patient))
            shutil.copyfile(os.path.join(org_data_path, bssfp_img_path), os.path.join(self.conf.test_folder, bssfp_img_path))
            shutil.copyfile(os.path.join(org_data_path, lge_img_path), os.path.join(self.conf.test_folder, lge_img_path))
            shutil.copyfile(os.path.join(org_data_path, t2_img_path),os.path.join(self.conf.test_folder, t2_img_path))
            full_row = np.concatenate(full_row,axis=0)
            imsave(os.path.join(self.conf.test_folder, "Seg_%s" % patient + '.png'), full_row)

    @abstractmethod
    def test(self, best_mark=False, public_or_split=1):
        """
        Evaluate a model on the test data.
        """

        # evaluate on test set
        if best_mark:
            self.model.load_models('best', public_or_split=1)
            log.info('Evaluating the best model on test data')
            test_folder = os.path.join(self.conf.folder, 'best_test_results_%s' % self.conf.test_dataset)
            train_folder = os.path.join(self.conf.folder, 'best_training_results_%s' % self.conf.test_dataset)
        else:
            self.model.load_models(public_or_split=public_or_split)
            log.info('Evaluating model on test data')
            test_folder = os.path.join(self.conf.folder, 'test_results_%s' % self.conf.test_dataset)
            train_folder = os.path.join(self.conf.folder, 'training_results_%s' % self.conf.test_dataset)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        if 'psdnet' in self.conf.model:
            self.test_pseudo_health(test_folder, 'test')
            self.test_reconstruction(test_folder, 'test')
        self.test_modality(test_folder, 'test')
        self.test_attention(test_folder, 'test')
        #self.test_fullimg(test_folder, 'test')


        # evaluate on train set
        log.info('Evaluating model on training data')
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        # if 'psdnet' in self.conf.model:
        #     self.test_reconstruction(train_folder, 'training')
        #     self.test_pseudo_health(train_folder, 'training')
        self.test_modality(train_folder, 'training')
        # self.test_attention(train_folder, 'training')
        #self.test_fullimg(folder, 'training')

        # # evaluate on the validation set
        # log.info('Evaluating model on validation data')
        # folder = os.path.join(self.conf.folder, 'validation_results_%s' % self.conf.test_dataset)
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # if 'psdnet' in self.conf.model:
        #     self.test_reconstruction(folder, 'validation')
        #     self.test_pseudo_health(folder, 'validation')
        # self.test_modality(folder, 'validation')
        # self.test_attention(folder, 'validation')
        # self.test_fullimg(folder, 'validation')

    def test_fullimg(self, folder,group):
        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_data = test_loader.load_labelled_data(self.conf.split, group,
                                                   downsample=self.conf.image_downsample)
        # harric added modality and segmentation_option arguments
        full_input = self.model.get_input_full()

        for counter, vol_i in enumerate(test_data.volumes()):
            vol_image = test_data.get_images(vol_i)
            vol_anato_mask = test_data.get_anato_masks(vol_i)
            vol_patho_mask = test_data.get_patho_masks(vol_i)
            # vol_slice = test_data.get_slice(vol_i)
            assert vol_image.shape[0] > 0 and vol_image.shape[:-1] == vol_anato_mask.shape[:-1] == vol_patho_mask.shape[
                                                                                                   :-1]
            # input_full = np.concatenate([(vol_image + 1) / 2, full_input.predict(vol_image)], axis=-1)
            input_full = full_input.predict(vol_image)
            input_full_save = []
            for ii in range(input_full.shape[0]):
                current_input_full = input_full[ii, :, :, :]
                row = []
                for jj in range(current_input_full.shape[-1]):
                    row.append(current_input_full[:, :, jj:jj + 1])
                row = np.concatenate(row, axis=1)
                input_full_save.append(row)
            input_full_save = np.concatenate(input_full_save, axis=0)
            imsave(os.path.join(folder, "FullInput_Vol%s" % (str(vol_i)) + '.png'), input_full_save)

    def test_pseudo_health(self,folder,group):
        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_data = test_loader.load_labelled_data(self.conf.split, group,
                                                   downsample=self.conf.image_downsample)

        modality_encoder = self.model.Enc_Modality
        spatial_encoder = self.model.Enc_Anatomy
        reconstructor = self.model.Decoder


        for counter, vol_i in enumerate(test_data.volumes()):
            vol_image = test_data.get_images(vol_i)
            spatial_out = spatial_encoder.predict(vol_image)
            if 'attention' in self.conf.testmode:
                spatial_factor, pred_p1, pred_p2, _, _ = spatial_out
            else:
                spatial_factor, pred_p1, pred_p2 = spatial_out

            modality_factor = modality_encoder.predict([spatial_factor, vol_image])
            rec_lge = reconstructor.predict([spatial_factor, modality_factor[0]])
            rec_t2 = reconstructor.predict([spatial_factor, modality_factor[1]])
            rec_bssfp = reconstructor.predict([spatial_factor, modality_factor[2]])

            org_lge = np.expand_dims(vol_image[:,:,:,0],axis=-1)
            org_t2 = np.expand_dims(vol_image[:, :, :, 1],axis=-1)
            org_bssfp = np.expand_dims(vol_image[:, :, :, 2],axis=-1)

            patho_inf = np.expand_dims(spatial_factor[:,:,:,-2],axis=-1)
            patho_ede = np.expand_dims(spatial_factor[:, :, :, -1],axis=-1)
            patho_null = np.expand_dims(np.zeros_like(spatial_factor[:, :, :, -1]),axis=-1)

            spatial_factor_null_inf = np.copy(spatial_factor)
            spatial_factor_null_ede = np.copy(spatial_factor)
            spatial_factor_null_all = np.copy(spatial_factor)
            spatial_factor_null_inf[:,:,:,-2] = np.zeros_like(spatial_factor_null_inf[:,:,:,-2])
            spatial_factor_null_ede[:, :, :, -1] = np.zeros_like(spatial_factor_null_ede[:, :, :, -1] )
            spatial_factor_null_all[:, :, :, -2:] = np.zeros_like(spatial_factor_null_all[:, :, :, -2:])


            rec_lge_ph1 = reconstructor.predict([spatial_factor_null_inf, modality_factor[0]])
            rec_lge_ph2 = reconstructor.predict([spatial_factor_null_ede, modality_factor[0]])
            rec_lge_ph = reconstructor.predict([spatial_factor_null_all, modality_factor[0]])
            rec_t2_ph1 = reconstructor.predict([spatial_factor_null_inf, modality_factor[1]])
            rec_t2_ph2 = reconstructor.predict([spatial_factor_null_ede, modality_factor[1]])
            rec_t2_ph = reconstructor.predict([spatial_factor_null_all, modality_factor[1]])
            rec_bssfp_ph1 = reconstructor.predict([spatial_factor_null_inf, modality_factor[2]])
            rec_bssfp_ph2 = reconstructor.predict([spatial_factor_null_ede, modality_factor[2]])
            rec_bssfp_ph = reconstructor.predict([spatial_factor_null_all, modality_factor[2]])

            # rec_lge_pseudo_health = reconstructor.predict([spatial_factor_null_inf, modality_factor[0]])
            # rec_t2_pseudo_health = reconstructor.predict([spatial_factor_null_ede, modality_factor[1]])
            # rec_lge_pseudo_health_all_null = reconstructor.predict([spatial_factor_null, modality_factor[0]])
            # rec_t2_pseudo_health_all_null = reconstructor.predict([spatial_factor_null, modality_factor[1]])
            # rec_bssfp_pseudo_health = reconstructor.predict([spatial_factor_null, modality_factor[2]])

            lge_row = np.concatenate([org_lge, rec_lge, (patho_inf-0.5)*2, rec_lge_ph1, rec_lge_ph2,rec_lge_ph], axis=2)
            t2_row = np.concatenate([org_t2, rec_t2, (patho_ede-0.5)*2, rec_t2_ph1, rec_t2_ph2,rec_t2_ph], axis=2)
            bssfp_row = np.concatenate([org_bssfp, rec_bssfp, patho_null, rec_bssfp_ph1, rec_bssfp_ph2, rec_bssfp_ph], axis=2)
            row = np.concatenate([lge_row, t2_row, bssfp_row], axis=1)

            rows = np.reshape(row, (row.shape[0]*row.shape[1], row.shape[2]))

            imsave(os.path.join(folder, "PseudoHealthComparison_Vol%s" % (str(vol_i)) + '.png'), rows)





    def test_reconstruction(self,folder,group):
        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_data = test_loader.load_labelled_data(self.conf.split, group,
                                                   downsample=self.conf.image_downsample)

        modality_encoder = self.model.Enc_Modality
        spatial_encoder = self.model.Enc_Anatomy
        reconstructor = self.model.Decoder

        for counter, vol_i in enumerate(test_data.volumes()):
            vol_image = test_data.get_images(vol_i)

            spatial_out = spatial_encoder.predict(vol_image)
            if 'attention' in self.conf.testmode:
                spatial_factor, pred_p1, pred_p2, _, _ = spatial_out
            else:
                spatial_factor, pred_p1, pred_p2 = spatial_out
            # pred_p1 = pred_p1[:, :, :, 0:1]
            # pred_p2 = pred_p2[:, :, :, 0:1]

            # spatial_factor_concat = np.concatenate([spatial_factor, pred_p1, pred_p2], axis=-1)
            full_channel_factor = []
            for ii in range(vol_image.shape[3]):
                full_channel_factor.append((vol_image[:,:,:,ii]+1)/2)
            full_channel_factor = np.concatenate(full_channel_factor,axis=-1)


            # full_channel_factor = np.reshape(vol_image,(vol_image.shape[0],vol_image.shape[1], vol_image.shape[2]*vol_image.shape[3]))
            for channel in range(spatial_factor.shape[3]):
                current_factor = spatial_factor[:,:,:,channel]
                full_channel_factor = np.concatenate([full_channel_factor,current_factor],axis=-1)

            full_channel_factor = np.reshape(full_channel_factor,(full_channel_factor.shape[0] * full_channel_factor.shape[1], full_channel_factor.shape[2]))
            imsave(os.path.join(folder, "SpatialFactor_Vol%s" % (str(vol_i)) + '.png'), full_channel_factor)


            # if reconstructor is not None:
            #     modality_factor = modality_encoder.predict([spatial_factor, vol_image])
            #     for ii in range(len(modality_factor)):
            #         if ii ==0:
            #             rec_x0 = reconstructor.predict([spatial_factor, modality_factor[ii]])
            #         elif ii == 1:
            #             rec_x1 = reconstructor.predict([spatial_factor, modality_factor[ii]])
            #         elif ii ==2:
            #             rec_x2 = reconstructor.predict([spatial_factor, modality_factor[ii]])
            #     rec_x0 = np.concatenate([vol_image[:,:,:,0], np.squeeze(rec_x0)], axis=2)
            #     rec_x1 = np.concatenate([vol_image[:,:,:,1], np.squeeze(rec_x1)], axis=2)
            #     rec_x2 = np.concatenate([vol_image[:,:,:,2], np.squeeze(rec_x2)], axis=2)
            #
            #     for channel in range(spatial_factor.shape[3]):
            #         current_spatial_factor_concat = np.copy(spatial_factor)
            #         current_spatial_factor_concat[:,:,:,channel:channel+1] = np.zeros_like(current_spatial_factor_concat[:,:,:,channel:channel+1])
            #         current_rec_x0 = reconstructor.predict([current_spatial_factor_concat,modality_factor[0]])
            #         current_rec_x1 = reconstructor.predict([current_spatial_factor_concat, modality_factor[1]])
            #         current_rec_x2 = reconstructor.predict([current_spatial_factor_concat, modality_factor[2]])
            #         rec_x0 = np.concatenate([rec_x0, np.squeeze(current_rec_x0)], axis=2)
            #         rec_x1 = np.concatenate([rec_x1, np.squeeze(current_rec_x1)], axis=2)
            #         rec_x2 = np.concatenate([rec_x2, np.squeeze(current_rec_x2)], axis=2)
            #         # # current_rec_x0 = current_rec_x[:, :, :, 0]
            #         # # current_rec_x1 = current_rec_x[:, :, :, 1]
            #         # # current_rec_x2 = current_rec_x[:, :, :, 2]
            #         #
            #         # if channel==0:
            #         #
            #         # elif channel==1:
            #         #     rec_x1 = np.concatenate([rec_x1, np.squeeze(current_rec_x)], axis=2)
            #         # elif channel==2:
            #         #     rec_x2 = np.concatenate([rec_x2, np.squeeze(current_rec_x)], axis=2)
            #
            #     plot_rec_x0 = np.reshape(rec_x0, (rec_x0.shape[0] * rec_x0.shape[1], rec_x0.shape[2]))
            #     plot_rec_x1 = np.reshape(rec_x1, (rec_x1.shape[0] * rec_x1.shape[1], rec_x1.shape[2]))
            #     plot_rec_x2 = np.reshape(rec_x2, (rec_x2.shape[0] * rec_x2.shape[1], rec_x2.shape[2]))
            #
            #     plot_rec_x = np.concatenate([plot_rec_x0,plot_rec_x1,plot_rec_x2], axis=0)
            #     imsave(os.path.join(folder, "Reconstruction_Vol%s" % (str(vol_i)) + '.png'), plot_rec_x)



    def test_attention(self, folder,group):
        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_data = test_loader.load_labelled_data(self.conf.split, group,
                                                   downsample=self.conf.image_downsample)

        # attention_maps = None
        if 'attention' in self.conf.testmode:
            # attention_maps = self.model.get_attention_maps()
            attention_output = self.model.get_attention_output()

            for counter, vol_i in enumerate(test_data.volumes()):
                vol_image = test_data.get_images(vol_i)
                vol_anato_mask = test_data.get_anato_masks(vol_i)
                vol_patho_mask = test_data.get_patho_masks(vol_i)
                # vol_slice = test_data.get_slice(vol_i)
                assert vol_image.shape[0] > 0 and vol_image.shape[:-1] == vol_anato_mask.shape[:-1] == vol_patho_mask.shape[:-1]
                #if 'attention' in self.conf.testmode:
                attention_output_list = attention_output.predict(vol_image)

                # for i in range(vol_image.shape[0]):
                # segmentor = self.model.get_segmentor()
                # pred = segmentor.predict(vol_image)
                # pred_anato = pred[0][:, :, :, :vol_anato_mask.shape[3]]
                # pred_patho1 = pred[1][:, :, :, 0:1]
                # pred_patho2 = pred[2][:, :, :, 0:1]

                # pred_anato = np.round(pred_anato)
                # pred_patho1 = np.round(pred_patho1)
                # pred_patho2 = np.round(pred_patho2)
                # anato = np.concatenate([vol_anato_mask, pred_anato], axis=-1)
                # patho1 = np.concatenate([vol_patho_mask[:, :, :, 0:1], pred_patho1], axis=-1)
                # patho2 = np.concatenate([vol_patho_mask[:, :, :, 1:], pred_patho2], axis=-1)

                # if np.any(attention_maps):
                #     current_attention_map = attention_map_list[self.conf.downsample]
                #     rows = generate_attentions(anato, [patho1, patho2], current_attention_map)
                #     im_Attention_Map = np.concatenate([img_rows, np.concatenate(rows, axis=0)], axis=1)
                #     imsave(os.path.join(folder, "AttentionMap_Vol%s" % (str(vol_i)) + '.png'), im_Attention_Map)
                #
                # if 'attention' in self.conf.testmode:
                img_rows = []
                for ii in range(vol_image.shape[0]):
                    img_row = []
                    for jj in range(vol_image.shape[-1]):
                        current_vol_img = vol_image[ii, :, :, jj]
                        img_row.append(current_vol_img)
                    img_row = np.concatenate(img_row, axis=1)
                    img_rows.append(img_row)
                    img_rows.append(img_row)
                img_rows = np.concatenate(img_rows, axis=0)

                spatial_attention_output = attention_output_list[0]
                channel_attention_output = attention_output_list[1]
                attention_rows = []
                for ii in range(spatial_attention_output.shape[0]):
                    attention_row = []
                    for jj in range(spatial_attention_output.shape[-1]):
                        current_spatial_attention_output = spatial_attention_output[ii, :, :, jj]
                        current_channel_attention_output = channel_attention_output[ii, :, :, jj]
                        current_spatial_attention_output = current_spatial_attention_output - np.min(
                            current_spatial_attention_output)
                        current_channel_attention_output = current_channel_attention_output - np.min(
                            current_channel_attention_output)
                        current_spatial_attention_output = current_spatial_attention_output / np.max(
                            current_spatial_attention_output)
                        current_channel_attention_output = current_channel_attention_output / np.max(
                            current_channel_attention_output)

                        current_attention_output = np.concatenate(
                            [current_spatial_attention_output, current_channel_attention_output], axis=0)
                        attention_row.append(current_attention_output)
                    attention_row = np.concatenate(attention_row, axis=1)
                    attention_rows.append(attention_row)
                attention_rows = np.concatenate(attention_rows, axis=0)
                im_attention_outputs = np.concatenate([img_rows, attention_rows], axis=1)
                imsave(os.path.join(folder, "AttentionOutput_Vol%s" % (str(vol_i)) + '.png'), im_attention_outputs)

    def correct_with_largest_connection(self,segmentation):
        mask_corrected_list = []
        for mask_counter in range(segmentation.shape[-1]):
            current_segmentation = segmentation[:,:,:,mask_counter]
            mask_list = []
            for batch_counter in range(current_segmentation.shape[0]):
                current_mask = current_segmentation[batch_counter,:,:]
                labels = label(current_mask)
                if labels.max() == 0:
                    continue
                current_largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                current_largestCC = np.int32(current_largestCC)
                mask_list.append(np.expand_dims(current_largestCC,axis=0))
            if len(mask_list)==0:
                mask_corrected_list.append(np.expand_dims(current_segmentation,axis=-1))
                continue
            mask_union=np.sum(np.concatenate(mask_list,axis=0),axis=0)
            mask_union[np.where(mask_union>0)]=1
            current_mask_corrected = mask_union * current_segmentation
            mask_corrected_list.append(np.expand_dims(current_mask_corrected,axis=-1))
        mask_corrected = np.concatenate(mask_corrected_list,axis=-1)
        return mask_corrected

    def fillholes(self, input_mask, SizeThreshold):
        def _implementation(im_in_rgb):

            if np.sum(im_in_rgb)==0:
                return im_in_rgb

            gray_mark = False
            if im_in_rgb.shape[-1] != 3:
                im_in_rgb = gray2rgb(np.squeeze(im_in_rgb)).astype(np.uint64)
                gray_mark=True

            im_in_lbl = im_in_rgb[:, :, 0] + (im_in_rgb[:, :, 1] << 8) + (im_in_rgb[:, :, 2] << 16)
            colors, im_in_lbl_new = np.unique(im_in_lbl, return_inverse=True)
            im_in_lbl_new = np.reshape(im_in_lbl_new, im_in_lbl.shape)
            colorize = np.empty((len(colors), 3), np.uint8)
            colorize[:, 0] = (colors & 0x0000FF)
            colorize[:, 1] = (colors & 0x00FF00) >> 8
            colorize[:, 2] = (colors & 0xFF0000) >> 16

            im_result = np.zeros((len(colors),) + im_in_lbl_new.shape, np.uint8)

            im_th = np.zeros(im_in_lbl_new.shape, np.uint8)

            for i in range(len(colors)):
                for j in range(im_th.shape[0]):
                    for k in range(im_th.shape[1]):
                        if im_in_lbl_new[j][k] == i:
                            im_th[j][k] = 255
                        else:
                            im_th[j][k] = 0

                im_floodfill = im_th.copy()

                h, w = im_th.shape[:2]
                mask = np.zeros((h + 2, w + 2), np.uint8)

                isbreak = False
                for m in range(im_floodfill.shape[0]):
                    for n in range(im_floodfill.shape[1]):
                        if (im_floodfill[m][n] == 0):
                            seedPoint = (m, n)
                            isbreak = True
                            break
                    if (isbreak):
                        break
                cv2.floodFill(im_floodfill, mask, seedPoint, 255, 4)
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                im_floodfill_inv_copy = im_floodfill_inv.copy()
                contours, hierarchy = cv2.findContours(im_floodfill_inv_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                for num in range(len(contours)):
                    if (cv2.contourArea(contours[num]) >= SizeThreshold):
                        cv2.fillConvexPoly(im_floodfill_inv, contours[num], 0)

                im_out = im_th | im_floodfill_inv
                im_result[i] = im_out

            im_fillhole = np.zeros((im_in_lbl_new.shape[0], im_in_lbl_new.shape[1], 3), np.uint8)

            for i in range(im_result.shape[1]):
                for j in range(im_result.shape[2]):
                    for k in range(im_result.shape[0]):
                        if (im_result[k][i][j] == 255):
                            im_fillhole[i][j] = colorize[k]
                            break

            if gray_mark:
                return im_fillhole[:,:,0:1]
            else:
                return im_fillhole

        input_mask = input_mask.astype(np.int64)
        filled_mask_list = []
        for batch_counter in range(input_mask.shape[0]):
            # print(batch_counter)
            current_filled_mask = _implementation(input_mask[batch_counter,:,:,:])
            filled_mask_list.append(np.expand_dims(current_filled_mask,axis=0))
        return np.concatenate(filled_mask_list,axis=0)


    def test_modality(self, folder, group):

        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_data = test_loader.load_labelled_data(self.conf.split, group,
                                                   downsample=self.conf.image_downsample)

        # harric added modality and segmentation_option arguments
        segmentor = self.model.get_segmentor()

        synth = []
        im_dice_anato, im_false_negative_anato = {}, {}
        im_dice_patho, im_false_negative_patho = {}, {}

        sep_dice_list_anato, sep_false_negative_list_anato = [], []
        sep_dice_list_patho, sep_false_negative_list_patho = [], []
        anato_mask_num = len(test_data.anato_mask_names)
        patho_mask_num = len(test_data.patho_mask_names)
        for ii in range(anato_mask_num):
            sep_dice_list_anato.append([])
            sep_false_negative_list_anato.append([])
        for ii in range(patho_mask_num):
            sep_dice_list_patho.append([])
            sep_false_negative_list_patho.append([])

        if folder is not '':
            f = open(os.path.join(folder, 'results.csv'), 'w')
            f_slice = open(os.path.join(folder, 'results_details.csv'), 'w')
            f_slice.writelines('ID,Patho,Inf,Ede,Anato,Myo,LV,RV\n')
        full_counter = 0
        for counter, vol_i in enumerate(test_data.volumes()):
            vol_image = test_data.get_images(vol_i)
            vol_anato_mask = test_data.get_anato_masks(vol_i)
            vol_patho_mask = test_data.get_patho_masks(vol_i)
            # vol_slice = test_data.get_slice(vol_i)
            assert vol_image.shape[0] > 0 and vol_image.shape[:-1] == vol_anato_mask.shape[:-1] == vol_patho_mask.shape[:-1]
            pred = segmentor.predict(vol_image)

            bssfp = vol_image[:,:,:,2]
            # lge = vol_image[:, :, :, 0]
            # t2 = vol_image[:, :, :, 1]
            anato_masks = pred[0][:,:,:,:vol_anato_mask.shape[3]]
            pred_patho1 = pred[1][:, :, :, 0:1]
            pred_patho2 = pred[2][:, :, :, 0:1]

            # correct with conditional random field
            # anato_binarized_crf = self.dense_crf(bssfp, anato_masks,[3,3,3])
            anato_binarized = np.round(anato_masks)
            patho1_binarized = np.round(pred_patho1)
            patho2_binarized = np.round(pred_patho2)



            patho_binarized = np.concatenate([patho1_binarized, patho2_binarized], axis=-1)

            # # correct with volume largest connected area union
            # anato_cc = self.correct_with_largest_connection(anato_binarized)
            # patho_cc = self.correct_with_largest_connection(patho_binarized)
            #
            #
            # # # fill holes
            # anato_filled = self.fillholes(anato_cc,25)
            # patho1_filled = self.fillholes(patho_cc[:,:,:,0:1], 10)
            # patho2_filled = self.fillholes(patho_cc[:, :, :, 1:], 10)
            # patho_filled = np.concatenate([patho1_filled, patho2_filled], axis=-1)
            #
            # # correct lv
            # anato_corrected = self.correct_lv(anato_filled, 50)
            anato_corrected = self.correct_lv(anato_binarized, 50)

            final_corrected_anato = anato_corrected
            final_corrected_patho = patho_binarized

            # calculate performance
            synth.append(pred)

            im_dice_patho[vol_i], sep_dice_patho \
                = dice(vol_patho_mask, final_corrected_patho)
            im_false_negative_patho[vol_i], sep_false_negative_patho \
                = calculate_false_negative(vol_patho_mask, final_corrected_patho)

            im_dice_anato[vol_i], sep_dice_anato \
                = dice(vol_anato_mask, final_corrected_anato)
            im_false_negative_anato[vol_i], sep_false_negative_anato \
                = calculate_false_negative(vol_anato_mask, final_corrected_anato)

            if folder is not '':
                for ii in range(final_corrected_patho.shape[0]):
                    slice_id = 'Vol%d-Slice%s,' % (vol_i, re.search('\d+', test_data.slice[full_counter]).group())
                    slice_patho_dice, slice_patho_dice_sep = dice(vol_patho_mask[ii:ii + 1, :, :, :],
                                                                  final_corrected_patho[ii:ii + 1, :, :, :])
                    slice_anato_dice, slice_anato_dice_sep = dice(vol_anato_mask[ii:ii + 1, :, :, :],
                                                                  final_corrected_anato[ii:ii + 1, :, :, :])

                    patho_info = "%.3f,%.3f,%.3f," % (
                        slice_patho_dice, slice_patho_dice_sep[0], slice_patho_dice_sep[1])
                    anato_info = "%.3f,%.3f,%.3f,%.3f\n" % (
                        slice_anato_dice, slice_anato_dice_sep[0], slice_anato_dice_sep[1], slice_anato_dice_sep[2])

                    write_line = slice_id + patho_info + anato_info
                    f_slice.writelines(write_line)

                    full_counter += 1

                slice_id = 'Vol%d,' % (vol_i)
                patho_info = "%.3f,%.3f,%.3f," % (im_dice_patho[vol_i], sep_dice_patho[0], sep_dice_patho[1])
                anato_info = "%.3f,%.3f,%.3f,%.3f\n" % (
                    im_dice_anato[vol_i], sep_dice_anato[0], sep_dice_anato[1], sep_dice_anato[2])
                write_line = slice_id + patho_info + anato_info
                f_slice.writelines(write_line)

            # harric added to specify dice scores across different masks
            #assert anato_mask_num == len(sep_dice_anato), 'Incorrect mask num !'
            assert patho_mask_num == len(sep_dice_patho), 'Incorrect mask num !'
            for ii in range(anato_mask_num):
                sep_dice_list_anato[ii].append(sep_dice_anato[ii])
                sep_false_negative_list_anato[ii].append(sep_false_negative_anato[ii])
            for ii in range(patho_mask_num):
                sep_dice_list_patho[ii].append(sep_dice_patho[ii])
                sep_false_negative_list_patho[ii].append(sep_false_negative_patho[ii])

            # harric added to specify dice scores across different masks
            if folder is not '':
                s = 'Volume:%s, ' \
                    + 'Pathology:%.3f,%.3f, ' \
                    + ', '.join(['%s, %.3f, %.3f'] * len(test_data.patho_mask_names)) \
                    + ', Anatomy:%.3f,%.3f, ' \
                    + ', '.join(['%s, %.3f, %.3f'] * len(test_data.anato_mask_names)) \
                    + '\n'
                d = (str(vol_i), im_dice_patho[vol_i], im_false_negative_patho[vol_i])
                for info_travesal in range(patho_mask_num):
                    d += (test_data.patho_mask_names[info_travesal],
                          sep_dice_patho[info_travesal],
                          sep_false_negative_patho[info_travesal])
                d += (im_dice_anato[vol_i], im_false_negative_anato[vol_i])
                for info_travesal in range(anato_mask_num):
                    d += (test_data.anato_mask_names[info_travesal],
                          sep_dice_anato[info_travesal],
                          sep_false_negative_anato[info_travesal])
                f.writelines(s % d)

            if not folder =='':
                anato = np.concatenate([vol_anato_mask, final_corrected_anato], axis=-1)
                patho1 = np.concatenate([vol_patho_mask[:, :, :, 0:1], final_corrected_patho[:,:,:,0:1]], axis=-1)
                patho2 = np.concatenate([vol_patho_mask[:, :, :, 1:], final_corrected_patho[:,:,:,1:]], axis=-1)
                im_Seg = save_segmentation(vol_image, anato, [patho1, patho2])
                imsave(os.path.join(folder, "Seg_Vol%s" % (str(vol_i)) + '.png'), im_Seg)

        # harric added to specify dice scores across different masks
        print_info_anato = group + ', Anato:%.3f,%.3f,' % \
                           (np.mean(list(im_dice_anato.values())),
                            np.mean(list(im_false_negative_anato.values())))
        for ii in range(anato_mask_num):
            print_info_anato += '%s,%.3f,%.3f' % \
                                (test_data.anato_mask_names[ii],
                                 np.mean(sep_dice_list_anato[ii]),
                                 np.mean(sep_false_negative_list_anato[ii]))
            if not ii==anato_mask_num-1:
                print_info_anato=print_info_anato+','
        log.info(print_info_anato)
        if folder is not '':
            f.write(print_info_anato)

        print_info_patho = ', Patho:%.3f,%.3f,' % \
                           (np.mean(list(im_dice_patho.values())),
                            np.mean(list(im_false_negative_patho.values())))
        for ii in range(patho_mask_num):
            print_info_patho += '%s,%.3f,%.3f' % \
                                (test_data.patho_mask_names[ii],
                                 np.mean(sep_dice_list_patho[ii]),
                                 np.mean(sep_false_negative_list_patho[ii]))
            if not ii==patho_mask_num-1:
                print_info_patho=print_info_patho+','
        log.info(print_info_patho[2:])
        if folder is not '':
            f.write(print_info_patho)
            f.close()
            f_slice.close()
        return print_info_anato + print_info_patho

    def stop_criterion(self, es, logs):
        es.on_epoch_end(self.epoch, logs)
        if es.stopped_epoch > 0:
            return True

    def get_datagen_params(self):
        """
        Construct a dictionary of augmentations.
        :param augment_spatial:
        :param augment_intensity:
        :return: a dictionary of augmentation parameters to use with a keras image processor
        """
        supervised_anatomy_pathology_augmentation_dict = {}
        supervised_pathology_augmentation_dict = {}
        unsupervised_reconstruction_augmentation_dict = {}

        if self.conf.augment:
            supervised_anatomy_pathology_augmentation_dict['rotation_range'] = 0.
            supervised_anatomy_pathology_augmentation_dict['horizontal_flip'] = False
            supervised_anatomy_pathology_augmentation_dict['vertical_flip'] = False
            supervised_anatomy_pathology_augmentation_dict['width_shift_range'] = 0.
            supervised_anatomy_pathology_augmentation_dict['height_shift_range'] = 0.
            # supervised_anatomy_augmentation_dict['rotation_range'] = 90.
            # supervised_anatomy_augmentation_dict['horizontal_flip'] = True
            # supervised_anatomy_augmentation_dict['vertical_flip'] = True
            # supervised_anatomy_augmentation_dict['width_shift_range'] = 0.15
            # supervised_anatomy_augmentation_dict['height_shift_range'] = 0.15

        if self.conf.augment:
            supervised_pathology_augmentation_dict['rotation_range'] = 90.
            supervised_pathology_augmentation_dict['horizontal_flip'] = True
            supervised_pathology_augmentation_dict['vertical_flip'] = True
            supervised_pathology_augmentation_dict['width_shift_range'] = 0.15
            supervised_pathology_augmentation_dict['height_shift_range'] = 0.15

        if self.conf.augment:
            unsupervised_reconstruction_augmentation_dict['rotation_range'] = 90.
            unsupervised_reconstruction_augmentation_dict['horizontal_flip'] = True
            unsupervised_reconstruction_augmentation_dict['vertical_flip'] = True
            unsupervised_reconstruction_augmentation_dict['width_shift_range'] = 0.15
            unsupervised_reconstruction_augmentation_dict['height_shift_range'] = 0.15

        return supervised_anatomy_pathology_augmentation_dict, \
               supervised_pathology_augmentation_dict, \
               unsupervised_reconstruction_augmentation_dict

    def align_batches(self, array_list):
        """
        Align the arrays of the input list, based on batch size.
        :param array_list: list of 4-d arrays to align
        """
        mn = np.min([x.shape[0] for x in array_list])
        new_list = [x[0:mn] for x in array_list]
        return new_list

    def get_fake(self, pred, fake_pool, sample_size=-1):
        sample_size = self.conf.batch_size if sample_size == -1 else sample_size

        if pred.shape[0] > 0:
            fake_pool.extend(pred)

        fake_pool = fake_pool[-self.conf.pool_size:]
        sel = np.random.choice(len(fake_pool), size=(sample_size,), replace=False)
        fake_A = np.array([fake_pool[ind] for ind in sel])
        return fake_pool, fake_A



