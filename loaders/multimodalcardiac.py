
import os
import scipy.io as sio
import nibabel as nib
import numpy as np
from skimage import transform
from PIL import Image

import utils.data_utils
from loaders.base_loader import Loader

from loaders.data import Data
from parameters import conf
import logging
GRAY_SCALE = 255
from scipy import misc
from utils.image_utils import image_show


class MultiModalCardiacLoader(Loader):

    def __init__(self):
        super(MultiModalCardiacLoader, self).__init__()
        self.num_anato_masks = 3
        self.num_patho_masks = 2
        self.input_shape = (None, None, 3)
        self.data_folder = conf['multimodalcardiac']
        self.log = logging.getLogger('multimodalcardiac')
        self.challenge_folder = conf['multimodalcardiac_challenge']


    def splits(self):
        """
        :return: an array of splits into validation, test and train indices
        """
        # valid_volume = [vol for vol in os.listdir(self.data_folder)
        #                 if (not vol[0]=='.'
        #                     and os.path.isdir(os.path.join(self.data_folder,
        #                                                    os.path.join(vol,'LGE'))))]
        # total_vol_num = len(valid_volume)
        # split_train_num_0 = 80
        # train_num_0 = np.float(split_train_num_0) / 100.0 * total_vol_num

        splits = [
            {'validation': [20, 21, 22, 23, 24],
             'test': [20, 21, 22, 23, 24],
             'training': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
             },


            {'validation': [0, 1, 2, 3, 4],
             'test': [0, 1, 2, 3, 4],
             'training': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
             },

            {'validation': [5, 6, 7, 8, 9],
             'test': [5, 6, 7, 8, 9],
             'training': [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
             },

            {'validation': [10, 11, 12, 13, 14],
             'test': [10, 11, 12, 13, 14],
             'training': [0,1,2,3,4,5,6,7,8,9,15,16,17,18,19,20,21,22,23,24]
             },

            {'validation': [15, 16, 17, 18, 19],
             'test': [15, 16, 17, 18, 19],
             'training': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,20,21,22,23,24]
             },

            {'validation': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
             'test': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
             'training': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
             },
        ]

        return splits

    def load_challenge_data(self,  normalise=True, value_crop=True, downsample=1, datafolder='default'):
        """
        Load labelled data, and return a Data object. In ACDC there are ES and ED annotations. Preprocessed data
        are saved in .npz files. If they don't exist, load the original images and preprocess.

        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :param downsample:      Downsample data to smaller size. Only used for testing.
        :return:                a Data object
        """
        # if segmentation_option == 0:
        #     input("Segmentation 0")
        if datafolder=='default':
            datafolder=self.challenge_folder

        npz_prefix = 'norm_' if normalise else 'unnorm_'

        # If numpy arrays are not saved, load and process raw data
        # images, ids = \
        #     self.load_raw_challenge_data(normalise, value_crop)
        if not os.path.exists(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_lges.npz')):
            images,  ids = \
                self.load_raw_challenge_data(normalise, value_crop)

            lges, t2s, bssfps = images
            # infs, edes = pathologies
            # myos, left_vens,right_vens=anatomies
            patient_index, index, slice = ids

            # save numpy arrays
            np.savez_compressed(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_lges'), lges)
            np.savez_compressed(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_t2s'), t2s)
            np.savez_compressed(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_bssfps'), bssfps)
            np.savez_compressed(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_patienet_index'), patient_index)
            np.savez_compressed(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_index'), index)
            np.savez_compressed(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_slice'), slice)
            slice = np.load(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_slice.npz'))['arr_0']
        # Load data from saved numpy arrays
        else:
            lges = np.load(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_lges.npz'))['arr_0']
            t2s = np.load(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_t2s.npz'))['arr_0']
            bssfps = np.load(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_bssfps.npz'))['arr_0']
            patient_index = np.load(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_patienet_index.npz'))['arr_0']
            index = np.load(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_index.npz'))['arr_0']
            slice = np.load(os.path.join(datafolder, npz_prefix + 'multimodalcardiac_slice.npz'))['arr_0']

        assert lges is not None \
               and t2s is not None \
               and bssfps is not None \
               and index is not None \
               and patient_index is not None \
               and slice is not None, \
            'Could not find saved data'

        assert lges.max() == 1 and lges.min() == -1,  'Images max=%.3f, min=%.3f' % (lges.max(), lges.min())
        assert t2s.max() == 1 and t2s.min() == -1,  'Images max=%.3f, min=%.3f' % (t2s.max(), t2s.min())
        assert bssfps.max() == 1 and bssfps.min() == -1,  'Images max=%.3f, min=%.3f' % (bssfps.max(), bssfps.min())

        self.log.debug('Loaded compressed multimodalcardiac data of shape: ' + str(lges.shape) + ' ' + str(index.shape))

        images = np.concatenate([lges,t2s,bssfps], axis=-1)
        anato_mask_names = ['myocardium','left_ventricle','right_ventricle']
        patho_mask_names = ['infarction','edema']


        # scanner = np.array([modality] * index.shape[0])

        # Select images belonging to the volumes of the split_type (training, validation, test)
        # volumes = self.splits()[split][split_type]
        # images = np.concatenate([images[index == v] for v in volumes])
        # anato_masks = np.concatenate([anato_masks[index == v] for v in volumes])
        # patho_masks = np.concatenate([patho_masks[index == v] for v in volumes])
        # # create a volume index
        # slice = np.concatenate([slice[index == v] for v in volumes])
        # index = np.concatenate([index[index == v] for v in volumes])


        # scanner = np.array([modality] * index.shape[0])
        assert images.shape[0] == index.shape[0]
        # assert anato_masks.max() == 1 and anato_masks.min() == 0, \
        #     'Masks max=%.3f, min=%.3f' % (anato_masks.max(), anato_masks.min())
        # assert patho_masks.max() == 1 and patho_masks.min() == 0, \
        #     'Masks max=%.3f, min=%.3f' % (patho_masks.max(), patho_masks.min())
        # assert images.shape[0] == anato_masks.shape[0] == patho_masks.shape[0], "Num of Images inconsistent"

        self.log.debug('challenge set: ' + str(images.shape))


        tmp_anatomy_masks = np.zeros(shape=images.shape[:-1]+(len(anato_mask_names),))
        tmp_pathology_masks = np.zeros(shape=images.shape[:-1] + (len(patho_mask_names),))

        return Data(images, [tmp_anatomy_masks, tmp_pathology_masks], [anato_mask_names, patho_mask_names], index, slice, downsample,patient_index)



    def load_labelled_data(self, split, split_type,normalise=True, value_crop=True, downsample=1):
        """
        Load labelled data, and return a Data object. In ACDC there are ES and ED annotations. Preprocessed data
        are saved in .npz files. If they don't exist, load the original images and preprocess.

        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :param downsample:      Downsample data to smaller size. Only used for testing.
        :return:                a Data object
        """
        # if segmentation_option == 0:
        #     input("Segmentation 0")

        if split < -1 or split > 5:
            raise ValueError('Invalid value for split: %d. Allowed values are 0, 1, 2.' % split)
        if split_type not in ['training', 'validation', 'test', 'all']:
            raise ValueError('Invalid value for split_type: %s. Allowed values are training, validation, test, all'
                             % split_type)

        npz_prefix = 'norm_' if normalise else 'unnorm_'

        # If numpy arrays are not saved, load and process raw data
        if not os.path.exists(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_lges.npz')):
            images, pathologies,anatomies, ids = \
                self.load_raw_labelled_data(normalise, value_crop)

            lges, t2s, bssfps = images
            infs, edes = pathologies
            myos, left_vens,right_vens=anatomies
            patient_index, index, slice = ids

            # save numpy arrays
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_lges'), lges)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_t2s'), t2s)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_bssfps'), bssfps)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_infs'), infs)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_edes'), edes)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_myos'), myos)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_left_vens'), left_vens)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_right_vens'), right_vens)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_patienet_index'), patient_index)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_index'), index)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_slice'), slice)
            slice = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_slice.npz'))['arr_0']
        # Load data from saved numpy arrays
        else:
            lges = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_lges.npz'))['arr_0']
            t2s = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_t2s.npz'))['arr_0']
            bssfps = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_bssfps.npz'))['arr_0']
            infs = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_infs.npz'))['arr_0']
            edes = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_edes.npz'))['arr_0']
            myos = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_myos.npz'))['arr_0']
            left_vens = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_left_vens.npz'))['arr_0']
            right_vens = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_right_vens.npz'))['arr_0']
            patient_index = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_patienet_index.npz'))['arr_0']
            index = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_index.npz'))['arr_0']
            slice = np.load(os.path.join(self.data_folder, npz_prefix + 'multimodalcardiac_slice.npz'))['arr_0']

        assert lges is not None \
               and t2s is not None \
               and bssfps is not None \
               and infs is not None \
               and edes is not None \
               and myos is not None \
               and left_vens is not None \
               and right_vens is not None \
               and index is not None \
               and patient_index is not None \
               and slice is not None, \
            'Could not find saved data'

        assert lges.max() == 1 and lges.min() == -1,  'Images max=%.3f, min=%.3f' % (lges.max(), lges.min())
        assert t2s.max() == 1 and t2s.min() == -1,  'Images max=%.3f, min=%.3f' % (t2s.max(), t2s.min())
        assert bssfps.max() == 1 and bssfps.min() == -1,  'Images max=%.3f, min=%.3f' % (bssfps.max(), bssfps.min())

        self.log.debug('Loaded compressed multimodalcardiac data of shape: ' + str(lges.shape) + ' ' + str(index.shape))

        images = np.concatenate([lges,t2s,bssfps], axis=-1)
        anato_masks = np.concatenate([myos,left_vens,right_vens], axis=-1)
        patho_masks = np.concatenate([infs,edes],axis=-1)
        anato_mask_names = ['myocardium','left_ventricle','right_ventricle']
        patho_mask_names = ['infarction','edema']


        # scanner = np.array([modality] * index.shape[0])

        # Select images belonging to the volumes of the split_type (training, validation, test)
        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        anato_masks = np.concatenate([anato_masks[index == v] for v in volumes])
        patho_masks = np.concatenate([patho_masks[index == v] for v in volumes])
        # create a volume index
        slice = np.concatenate([slice[index == v] for v in volumes])
        index = np.concatenate([index[index == v] for v in volumes])


        # scanner = np.array([modality] * index.shape[0])
        assert images.shape[0] == index.shape[0]
        assert anato_masks.max() == 1 and anato_masks.min() == 0, \
            'Masks max=%.3f, min=%.3f' % (anato_masks.max(), anato_masks.min())
        assert patho_masks.max() == 1 and patho_masks.min() == 0, \
            'Masks max=%.3f, min=%.3f' % (patho_masks.max(), patho_masks.min())
        assert images.shape[0] == anato_masks.shape[0] == patho_masks.shape[0], "Num of Images inconsistent"

        self.log.debug(split_type + ' set: ' + str(images.shape))
        return Data(images, [anato_masks, patho_masks], [anato_mask_names, patho_mask_names], index, slice, downsample)

    def load_unlabelled_data(self, split, split_type, modality='LGE', normalise=True, value_crop=True):
        """
        Load unlabelled data. In ACDC, this contains images from the cardiac phases between ES and ED.
        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :return:                a Data object
        """
        images, index = self.load_unlabelled_images('multimodalcardiac', split, split_type, False, normalise, value_crop,modality=modality)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))
        scanner = np.array([modality] * index.shape[0])
        return Data(images, masks, '-1', index, scanner)

    def load_all_data(self, split, split_type, modality='MR', normalise=True, value_crop=True):
        """
        Load all images, unlabelled and labelled, meaning all images from all cardiac phases.
        :param split:           Cross validation split: can be 0, 1, 2.
        :param split_type:      Cross validation type: can be training, validation, test, all
        :param modality:        Data modality. Unused here.
        :param normalise:       Use normalised data: can be True/False
        :param value_crop:      Crop extreme values: can be True/False
        :return:                a Data object
        """
        images, index = self.load_unlabelled_images('multimodalcardiac', split, split_type, True, normalise, value_crop,modality=modality)
        masks = np.zeros(shape=(images.shape[:-1]) + (1,))
        scanner = np.array([modality] * index.shape[0])
        return Data(images, masks, '-1', index, scanner)

    def load_raw_challenge_data(self, normalise=True, value_crop=True):
        """
        Load labelled data iterating through the ACDC folder structure.
        :param normalise:   normalise data between -1, 1
        :param value_crop:  crop between 5 and 95 percentile
        :return:            a tuple of the image and mask arrays
        """
        self.log.debug('Loading multimodalcardiac data from original location')
        lges, t2s, bssfps, patient_index, index, slice = [], [], [], [], [], []
        existed_directories = [vol for vol in os.listdir(self.challenge_folder)
                               if (not vol.startswith('.'))
                               and os.path.isdir(os.path.join(self.challenge_folder,vol))]
        existed_directories.sort()
        self.volumes = list(range(len(existed_directories)))


        # assert len(existed_directories) == len(self.volumes), 'Incorrect Volume Num !'

        for patient_i in self.volumes:

            patient_lge, patient_t2, patient_bssfp = [], [], []
            # if not os.path.isdir(os.path.join(os.path.join(self.data_folder,modality),existed_directories[patient_i-1])):
            #     continue
            patient = existed_directories[patient_i-1]
            print('Extracting Labeled Patient: %s @ %d / %d' % (patient, patient_i, len(self.volumes)))


            patient_folder = os.path.join(self.challenge_folder,patient)
            lge_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('lge')==-1)]
            t2_file_list = [file for file in os.listdir(patient_folder)
                            if (not file.startswith('.')) and (not file.find('t2') == -1)]
            bssfp_file_list = [file for file in os.listdir(patient_folder)
                               if (not file.startswith('.')) and (not file.find('bssfp') == -1)]
            # inf_file_list = [file for file in os.listdir(patient_folder)
            #                  if (not file.startswith('.')) and (not file.find('infarct') == -1)]
            # ede_file_list = [file for file in os.listdir(patient_folder)
            #                  if (not file.startswith('.')) and (not file.find('edema') == -1)]
            # myo_file_list = [file for file in os.listdir(patient_folder)
            #                  if (not file.startswith('.')) and (not file.find('myo') == -1)]
            # left_ven_file_list = [file for file in os.listdir(patient_folder)
            #                       if (not file.startswith('.')) and (not file.find('left_ven') == -1)]
            # right_ven_file_list = [file for file in os.listdir(patient_folder)
            #                        if (not file.startswith('.')) and (not file.find('right_ven') == -1)]
            lge_file_list.sort()
            t2_file_list.sort()
            bssfp_file_list.sort()
            # inf_file_list.sort()
            # ede_file_list.sort()
            # myo_file_list.sort()
            # left_ven_file_list.sort()
            # right_ven_file_list.sort()
            slices_num = len(lge_file_list)

            for v in range(slices_num):
                current_lge_name = lge_file_list[v]
                current_t2_name = t2_file_list[v]
                current_bssfp_name = bssfp_file_list[v]
                # current_inf_name = inf_file_list[v]
                # current_ede_name = ede_file_list[v]
                # current_myo_name = myo_file_list[v]
                # current_left_ven_name = left_ven_file_list[v]
                # current_right_ven_name = right_ven_file_list[v]
                v_id_from_lge = current_lge_name.split('_')[0]
                v_id_from_t2 = current_t2_name.split('_')[0]
                v_id_from_bssfp = current_bssfp_name.split('_')[0]
                # v_id_from_inf = current_inf_name.split('_')[0]
                # v_id_from_ede = current_ede_name.split('_')[0]
                # v_id_from_myo = current_myo_name.split('_')[0]
                # v_id_from_left_ven = current_left_ven_name.split('_')[0]
                # v_id_from_right_ven = current_right_ven_name.split('_')[0]

                assert v_id_from_lge \
                       == v_id_from_t2 \
                       == v_id_from_bssfp, 'Mis-Alignment !'
                slice.append(v_id_from_lge)

            # for patient index (patient names)
            for ii in range(slices_num):
                patient_index.append(patient)
                index.append(patient_i)
                lge = np.array(Image.open(os.path.join(patient_folder, lge_file_list[ii])))
                t2 = np.array(Image.open(os.path.join(patient_folder, t2_file_list[ii])))
                bssfp = np.array(Image.open(os.path.join(patient_folder, bssfp_file_list[ii])))

                lge = np.expand_dims(lge / GRAY_SCALE,axis=-1)
                t2 = np.expand_dims(t2 / GRAY_SCALE,axis=-1)
                bssfp = np.expand_dims(bssfp / GRAY_SCALE,axis=-1)



                patient_lge.append(lge)
                patient_t2.append(t2)
                patient_bssfp.append(bssfp)

            patient_lge = np.concatenate(patient_lge,axis=-1)
            patient_t2 = np.concatenate(patient_t2, axis=-1)
            patient_bssfp = np.concatenate(patient_bssfp, axis=-1)


            # crop to 5-95 percentile
            if value_crop:
                p5 = np.percentile(patient_bssfp.flatten(), 5)
                p95 = np.percentile(patient_bssfp.flatten(), 95)
                patient_bssfp = np.clip(patient_bssfp, p5, p95)

            # normalise to -1, 1
            if normalise:
                patient_lge = utils.data_utils.normalise(patient_lge, -1, 1)
                patient_t2 = utils.data_utils.normalise(patient_t2, -1, 1)
                patient_bssfp = utils.data_utils.normalise(patient_bssfp, -1, 1)
            lges.append(np.expand_dims(patient_lge,axis=-1))
            t2s.append(np.expand_dims(patient_t2, axis=-1))
            bssfps.append(np.expand_dims(patient_bssfp, axis=-1))




        # move slice axis to the first position
        lges = [np.moveaxis(im, 2, 0) for im in lges]
        t2s = [np.moveaxis(im, 2, 0) for im in t2s]
        bssfps = [np.moveaxis(im, 2, 0) for im in bssfps]


        # crop images and masks to the same pixel dimensions and concatenate all data
        lge_cropped, t2_cropped = utils.data_utils.crop_same(lges, t2s, (self.input_shape[0], self.input_shape[1]))
        _, bssfp_cropped = utils.data_utils.crop_same(lges, bssfps, (self.input_shape[0], self.input_shape[1]))

        lge_cropped = np.concatenate(lge_cropped, axis=0)
        t2_cropped = np.concatenate(t2_cropped, axis=0)
        bssfp_cropped = np.concatenate(bssfp_cropped, axis=0)


        patient_index = np.array(patient_index)
        index = np.array(index)

        return [lge_cropped,t2_cropped,bssfp_cropped],  [patient_index, index, slice]

    def load_raw_labelled_data(self, normalise=True, value_crop=True):
        """
        Load labelled data iterating through the ACDC folder structure.
        :param normalise:   normalise data between -1, 1
        :param value_crop:  crop between 5 and 95 percentile
        :return:            a tuple of the image and mask arrays
        """
        self.log.debug('Loading multimodalcardiac data from original location')
        lges, t2s, bssfps, infs, edes, myos, left_vens, right_vens, patient_index, index, slice = [], [], [], [], [], [], [], [], [], [], []
        existed_directories = [vol for vol in os.listdir(self.data_folder)
                               if (not vol.startswith('.'))
                               and os.path.isdir(os.path.join(self.data_folder,vol))]
        existed_directories.sort()
        assert len(existed_directories) == len(self.volumes), 'Incorrect Volume Num !'

        for patient_i in self.volumes:

            patient_lge, patient_t2, patient_bssfp, patient_inf, patient_ede, patient_myo, patient_left_ven, patient_right_ven = [], [], [], [], [], [], [], []
            # if not os.path.isdir(os.path.join(os.path.join(self.data_folder,modality),existed_directories[patient_i-1])):
            #     continue
            patient = existed_directories[patient_i-1]
            print('Extracting Labeled Patient: %s @ %d / %d' % (patient, patient_i, len(self.volumes)))


            patient_folder = os.path.join(self.data_folder,patient)
            lge_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('lge')==-1)]
            t2_file_list = [file for file in os.listdir(patient_folder)
                            if (not file.startswith('.')) and (not file.find('t2') == -1)]
            bssfp_file_list = [file for file in os.listdir(patient_folder)
                               if (not file.startswith('.')) and (not file.find('bssfp') == -1)]
            inf_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('infarct') == -1)]
            ede_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('edema') == -1)]
            myo_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('myo') == -1)]
            left_ven_file_list = [file for file in os.listdir(patient_folder)
                                  if (not file.startswith('.')) and (not file.find('left_ven') == -1)]
            right_ven_file_list = [file for file in os.listdir(patient_folder)
                                   if (not file.startswith('.')) and (not file.find('right_ven') == -1)]
            lge_file_list.sort()
            t2_file_list.sort()
            bssfp_file_list.sort()
            inf_file_list.sort()
            ede_file_list.sort()
            myo_file_list.sort()
            left_ven_file_list.sort()
            right_ven_file_list.sort()
            slices_num = len(lge_file_list)

            for v in range(slices_num):
                current_lge_name = lge_file_list[v]
                current_t2_name = t2_file_list[v]
                current_bssfp_name = bssfp_file_list[v]
                current_inf_name = inf_file_list[v]
                current_ede_name = ede_file_list[v]
                current_myo_name = myo_file_list[v]
                current_left_ven_name = left_ven_file_list[v]
                current_right_ven_name = right_ven_file_list[v]
                v_id_from_lge = current_lge_name.split('_')[0]
                v_id_from_t2 = current_t2_name.split('_')[0]
                v_id_from_bssfp = current_bssfp_name.split('_')[0]
                v_id_from_inf = current_inf_name.split('_')[0]
                v_id_from_ede = current_ede_name.split('_')[0]
                v_id_from_myo = current_myo_name.split('_')[0]
                v_id_from_left_ven = current_left_ven_name.split('_')[0]
                v_id_from_right_ven = current_right_ven_name.split('_')[0]

                assert v_id_from_lge \
                       == v_id_from_t2 \
                       == v_id_from_bssfp \
                       ==v_id_from_bssfp \
                       == v_id_from_inf \
                       == v_id_from_ede \
                       == v_id_from_myo \
                       == v_id_from_left_ven \
                       == v_id_from_right_ven, 'Mis-Alignment !'
                slice.append(v_id_from_lge)

            # for patient index (patient names)
            for ii in range(slices_num):
                patient_index.append(patient)
                index.append(patient_i)
                lge = np.array(Image.open(os.path.join(patient_folder, lge_file_list[ii])))
                t2 = np.array(Image.open(os.path.join(patient_folder, t2_file_list[ii])))
                bssfp = np.array(Image.open(os.path.join(patient_folder, bssfp_file_list[ii])))
                inf = np.array(Image.open(os.path.join(patient_folder, inf_file_list[ii])))
                ede = np.array(Image.open(os.path.join(patient_folder, ede_file_list[ii])))
                myo = np.array(Image.open(os.path.join(patient_folder, myo_file_list[ii])))
                left_ven = np.array(Image.open(os.path.join(patient_folder, left_ven_file_list[ii])))
                right_ven = np.array(Image.open(os.path.join(patient_folder, right_ven_file_list[ii])))

                lge = np.expand_dims(lge / GRAY_SCALE,axis=-1)
                t2 = np.expand_dims(t2 / GRAY_SCALE,axis=-1)
                bssfp = np.expand_dims(bssfp / GRAY_SCALE,axis=-1)
                inf = np.expand_dims(inf / GRAY_SCALE,axis=-1)
                ede = np.expand_dims(ede / GRAY_SCALE,axis=-1)
                myo = np.expand_dims(myo / GRAY_SCALE,axis=-1)
                left_ven = np.expand_dims(left_ven / GRAY_SCALE,axis=-1)
                right_ven = np.expand_dims(right_ven / GRAY_SCALE,axis=-1)


                patient_lge.append(lge)
                patient_t2.append(t2)
                patient_bssfp.append(bssfp)
                patient_inf.append(inf)
                patient_ede.append(ede)
                patient_myo.append(myo)
                patient_left_ven.append(left_ven)
                patient_right_ven.append(right_ven)
            patient_lge = np.concatenate(patient_lge,axis=-1)
            patient_t2 = np.concatenate(patient_t2, axis=-1)
            patient_bssfp = np.concatenate(patient_bssfp, axis=-1)
            patient_inf = np.concatenate(patient_inf, axis=-1)
            patient_ede = np.concatenate(patient_ede, axis=-1)
            patient_myo = np.concatenate(patient_myo, axis=-1)
            patient_left_ven = np.concatenate(patient_left_ven, axis=-1)
            patient_right_ven = np.concatenate(patient_right_ven, axis=-1)

            # crop to 5-95 percentile
            if value_crop:
                p5 = np.percentile(patient_bssfp.flatten(), 5)
                p95 = np.percentile(patient_bssfp.flatten(), 95)
                patient_bssfp = np.clip(patient_bssfp, p5, p95)

            # normalise to -1, 1
            if normalise:
                patient_lge = utils.data_utils.normalise(patient_lge, -1, 1)
                patient_t2 = utils.data_utils.normalise(patient_t2, -1, 1)
                patient_bssfp = utils.data_utils.normalise(patient_bssfp, -1, 1)
            lges.append(np.expand_dims(patient_lge,axis=-1))
            t2s.append(np.expand_dims(patient_t2, axis=-1))
            bssfps.append(np.expand_dims(patient_bssfp, axis=-1))

            infs.append(np.expand_dims(patient_inf,axis=-1))
            edes.append(np.expand_dims(patient_ede, axis=-1))

            myos.append(np.expand_dims(patient_myo, axis=-1))
            left_vens.append(np.expand_dims(patient_left_ven, axis=-1))
            right_vens.append(np.expand_dims(patient_right_ven, axis=-1))


        # move slice axis to the first position
        lges = [np.moveaxis(im, 2, 0) for im in lges]
        t2s = [np.moveaxis(im, 2, 0) for im in t2s]
        bssfps = [np.moveaxis(im, 2, 0) for im in bssfps]
        infs = [np.moveaxis(m, 2, 0) for m in infs]
        edes = [np.moveaxis(m, 2, 0) for m in edes]
        myos = [np.moveaxis(m, 2, 0) for m in myos]
        left_vens = [np.moveaxis(m, 2, 0) for m in left_vens]
        right_vens = [np.moveaxis(m, 2, 0) for m in right_vens]

        # crop images and masks to the same pixel dimensions and concatenate all data
        lge_cropped, t2_cropped = utils.data_utils.crop_same(lges, t2s, (self.input_shape[0], self.input_shape[1]))
        _, bssfp_cropped = utils.data_utils.crop_same(lges, bssfps, (self.input_shape[0], self.input_shape[1]))
        _, inf_cropped = utils.data_utils.crop_same(lges, infs, (self.input_shape[0], self.input_shape[1]))
        _, ede_cropped = utils.data_utils.crop_same(lges, edes, (self.input_shape[0], self.input_shape[1]))
        _, myo_cropped = utils.data_utils.crop_same(lges, myos, (self.input_shape[0], self.input_shape[1]))
        _, left_ven_cropped = utils.data_utils.crop_same(lges, left_vens, (self.input_shape[0], self.input_shape[1]))
        _, right_ven_cropped = utils.data_utils.crop_same(lges, right_vens, (self.input_shape[0], self.input_shape[1]))

        lge_cropped = np.concatenate(lge_cropped, axis=0)
        t2_cropped = np.concatenate(t2_cropped, axis=0)
        bssfp_cropped = np.concatenate(bssfp_cropped, axis=0)
        inf_cropped = np.concatenate(inf_cropped, axis=0)
        ede_cropped = np.concatenate(ede_cropped, axis=0)
        myo_cropped = np.concatenate(myo_cropped, axis=0)
        left_ven_cropped = np.concatenate(left_ven_cropped, axis=0)
        right_ven_cropped = np.concatenate(right_ven_cropped, axis=0)

        patient_index = np.array(patient_index)
        index = np.array(index)

        return [lge_cropped,t2_cropped,bssfp_cropped], [inf_cropped,ede_cropped], [myo_cropped,left_ven_cropped,right_ven_cropped], [patient_index, index, slice]

    def resample_raw_image(self, mask_fname, patient_folder, binary=True):
        """
        Load raw data (image/mask) and resample to fixed resolution.
        :param mask_fname:     filename of mask
        :param patient_folder: folder containing patient data
        :param binary:         boolean to define binary masks or not
        :return:               the resampled image
        """
        m_nii_fname = os.path.join(patient_folder, mask_fname)
        new_res = (1.37, 1.37)
        print('Resampling %s at resolution %s to file %s' % (m_nii_fname, str(new_res), new_res))
        im_nii = nib.load(m_nii_fname)
        im_data = im_nii.get_data()
        voxel_size = im_nii.header.get_zooms()

        scale_vector = [voxel_size[i] / new_res[i] for i in range(len(new_res))]
        order = 0 if binary else 1

        result = []
        for i in range(im_data.shape[-1]):
            im = im_data[..., i]
            rescaled = transform.rescale(im, scale_vector, order=order, preserve_range=True, mode='constant')
            result.append(np.expand_dims(rescaled, axis=-1))
        return np.concatenate(result, axis=-1)

    def process_raw_image(self, im_fname, patient_folder, value_crop, normalise):
        """
        Rescale between -1 and 1 and crop extreme values of an image
        :param im_fname:        filename of the image
        :param patient_folder:  folder of patient data
        :param value_crop:      True/False to crop values between 5/95 percentiles
        :param normalise:       True/False normalise images
        :return:                a processed image
        """
        im = self.resample_raw_image(im_fname, patient_folder, binary=False)

        # crop to 5-95 percentile
        if value_crop:
            p5 = np.percentile(im.flatten(), 5)
            p95 = np.percentile(im.flatten(), 95)
            im = np.clip(im, p5, p95)

        # normalise to -1, 1
        if normalise:
            im = utils.data_utils.normalise(im, -1, 1)

        return im

    def load_raw_unlabelled_data(self, include_labelled=True, normalise=True, value_crop=True, modality='LGE'):
        """
        Load unlabelled data iterating through the ACDC folder structure.
        :param include_labelled:    include images from ES, ED phases that are labelled. Can be True/False
        :param normalise:           normalise data between -1, 1
        :param value_crop:          crop between 5 and 95 percentile
        :return:                    an image array
        """
        self.log.debug('Loading unlabelled multimodalcardiac data from original location')
        images, patient_index, index = [], [], []
        # existed_directories = [vol for vol in os.listdir(self.data_folder)
        #                        if (not vol.startswith('.')) and os.path.isdir(os.path.join(self.data_folder, vol))]
        existed_directories = [vol for vol in os.listdir(os.path.join(self.data_folder,modality))
                               if (not vol.startswith('.'))
                               and os.path.isdir(os.path.join(os.path.join(self.data_folder, modality),vol))]
        existed_directories.sort()
        assert len(existed_directories) == len(self.volumes),'Incorrect Volume Num !'

        for patient_i in self.volumes:
            patient_images = []
            # if not os.path.isdir(os.path.join(os.path.join(self.data_folder,modality),existed_directories[patient_i-1])):
            #     continue
            patient = existed_directories[patient_i-1]
            print('Extracting UnLabeled Patient: %s @ %d / %d' % (patient, patient_i, len(self.volumes)))

            patient_folder = os.path.join(os.path.join(self.data_folder, modality), patient)
            img_file_list = [file for file in os.listdir(patient_folder)
                             if (not file.startswith('.')) and (not file.find('OrgImg') == -1)]
            # mi_file_list = [file for file in os.listdir(patient_folder)
            #                 if (not file.startswith('.')) and (not file.find('SegMi') == -1)]
            # my_file_list = [file for file in os.listdir(patient_folder)
            #                 if (not file.startswith('.')) and (not file.find('SegMy') == -1)]
            img_file_list.sort()
            # mi_file_list.sort()
            # my_file_list.sort()
            slices_num = len(img_file_list)

            # for patient index (patient names)
            for ii in range(slices_num):
                patient_index.append(patient)
                index.append(patient_i)

            # for original images
            for org_img_path in img_file_list:
                im = np.array(Image.open(os.path.join(patient_folder, org_img_path)))
                # im = im / np.max(im - np.min(im))
                im = im[:, :, 0]
                patient_images.append(np.expand_dims(im, axis=-1))
            patient_images = np.concatenate(patient_images, axis=-1)

            # crop to 5-95 percentile
            if value_crop:
                p5 = np.percentile(patient_images.flatten(), 5)
                p95 = np.percentile(patient_images.flatten(), 95)
                patient_images = np.clip(patient_images, p5, p95)

            # normalise to -1, 1
            if normalise:
                patient_images = utils.data_utils.normalise(patient_images, -1, 1)
            images.append(np.expand_dims(patient_images, axis=-1))


        images = [np.moveaxis(im, 2, 0) for im in images]
        zeros = [np.zeros(im.shape) for im in images]
        images_cropped, _ = utils.data_utils.crop_same(images, zeros,
                                                       (self.input_shape[0], self.input_shape[1]))
        images_cropped = np.concatenate(images_cropped, axis=0)[..., 0]
        index = np.array(index)

        return images_cropped, patient_index, index

    def load_unlabelled_images(self, dataset, split, split_type, include_labelled, normalise, value_crop, modality):
        """
        Load only images.
        :param dataset:
        :param split:
        :param split_type:
        :param include_labelled:
        :param normalise:
        :param value_crop:
        :return:
        """
        npz_prefix_type = 'ul_' if not include_labelled else 'all_'
        npz_prefix = npz_prefix_type + 'norm_' if normalise else npz_prefix_type + 'unnorm_'

        # Load saved numpy array
        if os.path.exists(os.path.join(os.path.join(self.data_folder,modality), npz_prefix + dataset + 'multimodalcardiac_images.npz')):
            images = \
                np.load(os.path.join(os.path.join(self.data_folder,modality),
                                     npz_prefix + dataset + 'multimodalcardiac_images.npz'))['arr_0']
            index  = \
                np.load(os.path.join(os.path.join(self.data_folder,modality),
                                     npz_prefix + dataset + 'multimodalcardiac_index.npz'))['arr_0']
            patient_index = \
                np.load(os.path.join(os.path.join(self.data_folder,modality),
                                     npz_prefix + dataset + 'multimodalcardiac_patient_index.npz'))['arr_0']
            self.log.debug('Loaded compressed ' + dataset + ' unlabelled data of shape ' + str(images.shape))
        # Load from source
        else:
            images, patient_index, index = \
                self.load_raw_unlabelled_data(include_labelled, normalise, value_crop, modality=modality)
            images = np.expand_dims(images, axis=3)
            np.savez_compressed(os.path.join(os.path.join(self.data_folder,modality),
                                             npz_prefix + dataset + 'multimodalcardiac_images'), images)
            np.savez_compressed(os.path.join(os.path.join(self.data_folder,modality),
                                             npz_prefix + dataset + 'multimodalcardiac_index'), index)
            np.savez_compressed(os.path.join(os.path.join(self.data_folder,modality),
                                             npz_prefix + dataset + 'multimodalcardiac_patient_index'), patient_index)
        assert split_type in ['training', 'validation', 'test', 'all'], 'Unknown split_type: ' + split_type

        if split_type == 'all':
            return images, index

        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        index  = np.concatenate([index[index==v] for v in volumes])
        return images, index