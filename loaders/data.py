from skimage.measure import block_reduce
import numpy as np

import utils.image_utils, utils.data_utils
import logging
log = logging.getLogger('data')


class Data(object):

    def __init__(self, images, masks, masknames, index, slice, downsample=1, patient_index=-1):
        # harric added the masknames argument to print different mask dice scorers
        # in the testing phase to specify dices scores across different masks
        """
        Data constructor.
        :param images:      a 4-D numpy array of images. Expected shape: (N, H, W, 1)
        :param masks:       a 4-D numpy array of myocardium segmentation masks. Expected shape: (N, H, W, 1)
        :param index:       a 1-D numpy array indicating the volume each image/mask belongs to. Used for data selection.
        """

        anatomy_masks, patho_masks = masks
        anatomy_mask_names, patho_masks_names = masknames
        if images is None:
            raise ValueError('Images cannot be None.')
        if anatomy_masks is None:
            raise ValueError('Anatomy Masks cannot be None.')
        if patho_masks is None:
            raise ValueError('Pathology Masks cannot be None.')
        if index is None:
            raise ValueError('Index cannot be None.')
        if images.shape[:-1] != anatomy_masks.shape[:-1]:
            raise ValueError('Image shape=%s different from Anatomy Mask shape=%s' % (str(images.shape), str(anatomy_masks.shape)))
        if images.shape[:-1] != patho_masks.shape[:-1]:
            raise ValueError('Image shape=%s different from Pathology Mask shape=%s' % (str(images.shape), str(patho_masks.shape)))
        if images.shape[0] != index.shape[0]:
            raise ValueError('Different number of images and indices: %d vs %d' % (images.shape[0], index.shape[0]))

        self.patient_index = patient_index
        self.images = images
        self.anato_masks  = anatomy_masks
        self.patho_masks = patho_masks
        self.anato_mask_names = anatomy_mask_names # harric added
        self.patho_mask_names = patho_masks_names  # harric added
        self.index  = index
        self.num_volumes = len(self.volumes())
        self.slice = slice

        self.downsample(downsample)

        log.info('Creating Data object with images of shape %s and %d volumes' % (str(images.shape), self.num_volumes))
        log.info('Images value range [%.1f, %.1f]' % (images.min(), images.max()))
        log.info('Anatomy Masks value range [%.1f, %.1f]' % (anatomy_masks.min(), anatomy_masks.max()))
        log.info('Pathology Masks value range [%.1f, %.1f]' % (patho_masks.min(), patho_masks.max()))

    def copy(self):
        return Data(np.copy(self.images), np.copy(self.masks), np.copy(self.index), np.copy(self.scanner))

    def merge(self, other):
        assert self.images.shape[1:] == other.images.shape[1:], str(self.images.shape) + ' vs ' + str(other.images.shape)
        assert self.masks.shape[1:] == other.masks.shape[1:], str(self.masks.shape) + ' vs ' + str(other.masks.shape)

        self.images = np.concatenate([self.images, other.images], axis=0)
        self.masks  = np.concatenate([self.masks, other.masks], axis=0)
        self.index  = np.concatenate([self.index, other.index], axis=0)
        self.scanner= np.concatenate([self.scanner, other.scanner], axis=0)
        self.num_volumes = len(self.volumes())
        log.info('Merged Data object of %d to this Data object of size %d' % (other.size(), self.size()))

    def select_masks(self, num_masks):
        log.info('Selecting the first %d masks out of %d.' % (num_masks, self.masks.shape[-1]))
        self.masks = self.masks[..., 0:num_masks]

    def crop(self, shape):
        log.debug('Cropping images and masks to shape ' + str(shape))
        [images], [anatomy_masks] = utils.data_utils.crop_same([self.images], [self.anato_masks],
                                                       size=shape, pad_mode='constant')
        _, [pathology_masks] = utils.data_utils.crop_same([self.images], [self.patho_masks],
                                                               size=shape, pad_mode='constant')
        self.images = images
        self.anato_masks  = anatomy_masks
        self.patho_masks = pathology_masks
        assert self.images.shape[1:-1] == self.anato_masks.shape[1:-1] == self.patho_masks.shape[1:-1], \
            'Invalid shapes: ' \
            + str(self.images.shape[1:-1]) + ' ' \
            + str(self.anato_masks.shape[1:-1])+ ' ' \
            + str(self.patho_masks.shape[1:-1])

    def volumes(self):
        return sorted(set(self.index))

    def get_images(self, vol):
        return self.images[self.index == vol]



    def get_anato_masks(self, vol):
        return self.anato_masks[self.index == vol]

    def get_patho_masks(self, vol):
        return self.patho_masks[self.index == vol]

    def get_slice(self, vol):
        return self.slice[self.index == vol]
    def get_scanner(self, vol):
        return self.scanner[self.index == vol]
    def get_patient(self,vol):
        return self.patient_index[self.index == vol]

    def filter_by_scanner(self, scanner):
        assert scanner in self.scanner, '%s is not a valid scanner type' % str(scanner)
        self.images  = self.images[self.scanner == scanner]
        self.masks   = self.masks[self.scanner == scanner]
        self.index   = self.index[self.scanner == scanner]
        self.scanner = self.scanner[self.scanner == scanner]
        self.num_volumes = len(self.volumes())
        log.debug('Selected %d volumes acquired with scanner %s' % (self.num_volumes, str(scanner)))

    def size(self):
        return len(self.images)

    def sample_per_volume(self, num=-1, pctg=1, seed=-1):
        log.info('Sampling %d from each volume' % num)
        if seed > -1:
            np.random.seed(seed)

        new_images, new_anato_masks, new_patho_masks, new_scanner, new_index, new_slice = [], [], [], [], [], []
        for vol in self.volumes():
            images = self.get_images(vol)
            anato_masks = self.get_anato_masks(vol)
            patho_masks = self.get_patho_masks(vol)
            # scanner = self.get_scanner(vol)
            slice = self.slice.copy()
            self.index = self.index.copy()
            if vol == 29:
                a = 1
            if num == -1:
                num_actual = int(pctg * images.shape[0])
            else:
                num_actual = num
            if images.shape[0] < num:
                log.debug('Volume %d contains less images: %d < %d. Sampling %d images.' %
                          (vol, images.shape[0], num_actual, images.shape[0]))
                num_actual = images.shape[0]

            idx = np.random.choice(images.shape[0], size=num_actual, replace=False)
            images = np.array([images[i] for i in idx])
            anato_masks = np.array([anato_masks[i] for i in idx])
            patho_masks = np.array([patho_masks[i] for i in idx])
            # scanner = np.array([scanner[i] for i in idx])
            slice = np.array([slice[i] for i in idx])
            index = np.array([vol] * num_actual)

            new_images.append(images)
            new_anato_masks.append(anato_masks)
            new_patho_masks.append(patho_masks)
            # new_scanner.append(scanner)
            new_index.append(index)
            new_slice.append(slice)

        self.images = np.concatenate(new_images, axis=0)
        self.anato_masks = np.concatenate(new_anato_masks, axis=0)
        self.patho_masks = np.concatenate(new_patho_masks, axis=0)
        # self.scanner = np.concatenate(new_scanner, axis=0)
        self.slice = np.concatenate(new_slice, axis=0)
        self.index = np.concatenate(new_index, axis=0)

        log.info('Sampled %d images.' % len(self.images))


    def sample_by_volume(self, num, seed=-1):
        log.info('Sampling %d volumes out of total %d' % (num, self.num_volumes))
        if seed > -1:
            np.random.seed(seed)

        if num == self.num_volumes:
            return

        volumes = np.random.choice(self.volumes(), size=num, replace=False)
        if num == 0 or len(volumes) == 0:
            self.images = np.zeros(shape=(0,) + self.images.shape[1:])
            self.masks = np.zeros(shape=(0,) + self.masks.shape[1:])
            # self.scanner = np.zeros(shape=(0,) + self.scanner.shape[1:])
            self.index = np.zeros(shape=(0,) + self.index.shape[1:])
            self.slice = np.zeros(shape=(0,) + self.slice.shape[1:])
            self.num_volumes = 0
            return

        self.images = np.concatenate([self.get_images(v) for v in volumes], axis=0)
        self.anato_masks = np.concatenate([self.get_anato_masks(v) for v in volumes], axis=0)
        self.patho_masks = np.concatenate([self.get_patho_masks(v) for v in volumes], axis=0)
        #self.scanner = np.concatenate([self.get_scanner(v) for v in volumes], axis=0)
        self.slice = np.concatenate([self.slice.copy()[self.index == v] for v in volumes], axis=0)
        self.index = np.concatenate([self.index.copy()[self.index == v] for v in volumes], axis=0)
        self.num_volumes = len(volumes)

        log.info('Sampled volumes: %s of total %d images' % (str(volumes), self.size()))

    def sample_images(self, num, seed=-1):
        log.info('Sampling %d images out of total %d' % (num, self.size()))
        if seed > -1:
            np.random.seed(seed)

        idx = np.random.choice(self.size(), size=num, replace=False)
        self.images  = np.array([self.images[i] for i in idx])
        self.masks   = np.array([self.masks[i] for i in idx])  # self.masks[:num]
        self.scanner = np.array([self.scanner[i] for i in idx])
        self.index   = np.array([self.index[i] for i in idx])

    def sample(self, num, seed=-1):
        log.info('Sampling %d volumes out of total %d' % (num, self.num_volumes))
        if seed > -1:
            np.random.seed(seed)

        if num == self.num_volumes:
            return

        volumes = np.random.choice(self.volumes(), size=num, replace=False)
        if num == 0 or len(volumes) == 0:
            self.images  = np.zeros(shape=(0,) + self.images.shape[1:])
            self.masks   = np.zeros(shape=(0,) + self.masks.shape[1:])
            self.scanner = np.zeros(shape=(0,) + self.scanner.shape[1:])
            self.index   = np.zeros(shape=(0,) + self.index.shape[1:])
            self.slice   = np.zeros(shape=(0,) + self.slice.shape[1:])
            self.num_volumes = 0
            return

        self.images  = np.concatenate([self.get_images(v) for v in volumes], axis=0)
        self.anato_masks   = np.concatenate([self.get_anato_masks(v) for v in volumes], axis=0)
        self.patho_masks = np.concatenate([self.get_patho_masks(v) for v in volumes], axis=0)
        self.scanner = np.concatenate([self.get_scanner(v) for v in volumes], axis=0)
        self.slice   = np.concatenate([self.slice.copy()[self.index == v] for v in volumes], axis=0)
        self.index = np.concatenate([self.index.copy()[self.index == v] for v in volumes], axis=0)
        self.num_volumes = len(volumes)

        log.info('Sampled volumes: %s of total %d images' % (str(volumes), self.size()))


    def shape(self):
        return self.images.shape

    def downsample(self, ratio=2):
        if ratio == 1: return

        self.images = block_reduce(self.images, block_size=(1, ratio, ratio, 1), func=np.mean)
        if self.masks is not None:
            self.masks  = block_reduce(self.masks, block_size=(1, ratio, ratio, 1), func=np.mean)

        log.info('Downsampled data by %d to shape %s' % (ratio, str(self.images.shape)))

    def get_lvv(self, slice_thickness, pixel_resolution):
        lv = self.masks[..., 1:2]
        return lv.sum(axis=(1, 2, 3)) * slice_thickness * pixel_resolution

    def get_lvv_per_slice(self, vol_i, slice_thickness, pixel_resolution):
        masks = self.get_masks(vol_i)
        lv = masks[..., 1:2]
        return lv.sum(axis=(1, 2, 3)) * slice_thickness * pixel_resolution

    def get_lvv_per_volume(self, vol_i, slice_thickness, pixel_resolution):
        masks = self.get_masks(vol_i)
        lv = masks[..., 1:2]
        return np.sum(lv.sum(axis=(1, 2, 3)) * slice_thickness * pixel_resolution)
