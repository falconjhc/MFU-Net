import logging
import os
from abc import abstractmethod
from utils.image_utils import image_show
import numpy as np
from keras.callbacks import Callback
from imageio import imwrite as imsave # harric modified

from costs import dice
from utils.image_utils import save_segmentation, save_multiimage_segmentation, generate_attentions

log = logging.getLogger('BaseSaveImage')


class BaseSaveImage(Callback):
    """
    Abstract base class for saving training images
    """
    def __init__(self, folder, model):
        super(BaseSaveImage, self).__init__()
        self.folder = os.path.join(folder, 'training_images')
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.model = model

    @abstractmethod
    def on_epoch_end(self, epoch=None, logs=None):
        pass


class SaveImage(Callback):
    """
    Simple callback that saves segmentation masks and dice error.
    """
    def __init__(self, folder, test_data, test_masks=None, input_len=None):
        super(SaveImage, self).__init__()
        self.folder = folder
        self.test_data = test_data  # this can be a list of images of different spatial dimensions
        self.test_masks = test_masks
        self.input_len = input_len

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        all_dice = []
        for i in range(len(self.test_data)):
            d, m = self.test_data[i], self.test_masks[i]
            s = save_segmentation(self.folder, self.model, d, m, 'slc_%d' % i)
            all_dice.append(-dice(self.test_masks[i:i+1], s))

        f = open(os.path.join(self.folder, 'test_error.txt'), 'a+')
        f.writelines("%d, %.3f\n" % (epoch, np.mean(all_dice)))
        f.close()


class SaveEpochImages(Callback):
    def __init__(self, conf, model, gen,
                 attention_maps, attention_output_list, input_full, segmentor, spatial_encoder, modality_encoder, reconstructor):
        super(SaveEpochImages, self).__init__()
        self.folder = conf.folder + '/training'
        self.conf = conf
        self.model = model
        self.gen = gen
        self.image_channels = int(self.model.input_full.shape[-1])
        self.anato_mask_channels = self.conf.num_anato_masks
        self.patho_mask_channels = self.conf.num_patho_masks
        self.testmode = self.conf.testmode
        self.downsample = self.conf.downsample
        self.attention_maps = attention_maps
        self.attention_output_list = attention_output_list
        self.input_full = input_full
        self.segmentor = segmentor
        self.spatial_encoder = spatial_encoder
        self.modality_encoder = modality_encoder
        self.reconstructor = reconstructor
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def on_epoch_end(self, epoch, logs=None):

        self.callback_pseudo_health(epoch, logs)
        self.callback_reconstruction(epoch, logs)
        if not self.segmentor==None: self.callback_save_segmentation(epoch,logs)
        # self.callback_full_images(epoch,logs)
        self.callback_attention(epoch, logs)


    def callback_pseudo_health(self,epoch, logs=None):
        batch_data_pack = next(self.gen)
        x = batch_data_pack[:, :, :, :self.image_channels]

        if self.spatial_encoder is not None:
            spatial_out = self.spatial_encoder.predict(x)
            if 'attention' in self.testmode:
                spatial_factor, pred_p1, pred_p2, _, _ = spatial_out
            else:
                spatial_factor, pred_p1, pred_p2 = spatial_out
            modality_factor = self.modality_encoder.predict([spatial_factor, x])

            rec_lge = self.reconstructor.predict([spatial_factor, modality_factor[0]])
            rec_t2 = self.reconstructor.predict([spatial_factor, modality_factor[1]])
            rec_bssfp = self.reconstructor.predict([spatial_factor, modality_factor[2]])

            org_lge = np.expand_dims(x[:, :, :, 0], axis=-1)
            org_t2 = np.expand_dims(x[:, :, :, 1], axis=-1)
            org_bssfp = np.expand_dims(x[:, :, :, 2], axis=-1)

            patho_inf = np.expand_dims(spatial_factor[:, :, :, -2], axis=-1)
            patho_ede = np.expand_dims(spatial_factor[:, :, :, -1], axis=-1)
            patho_null = np.expand_dims(np.zeros_like(spatial_factor[:, :, :, -1]), axis=-1)

            spatial_factor_null_inf = np.copy(spatial_factor)
            spatial_factor_null_ede = np.copy(spatial_factor)
            spatial_factor_null_all = np.copy(spatial_factor)
            spatial_factor_null_inf[:, :, :, -2] = np.zeros_like(spatial_factor_null_inf[:, :, :, -2])
            spatial_factor_null_ede[:, :, :, -1] = np.zeros_like(spatial_factor_null_ede[:, :, :, -1])
            spatial_factor_null_all[:, :, :, -2:] = np.zeros_like(spatial_factor_null_all[:, :, :, -2:])

            rec_lge_ph1 = self.reconstructor.predict([spatial_factor_null_inf, modality_factor[0]])
            rec_lge_ph2 = self.reconstructor.predict([spatial_factor_null_ede, modality_factor[0]])
            rec_lge_ph = self.reconstructor.predict([spatial_factor_null_all, modality_factor[0]])
            rec_t2_ph1 = self.reconstructor.predict([spatial_factor_null_inf, modality_factor[1]])
            rec_t2_ph2 = self.reconstructor.predict([spatial_factor_null_ede, modality_factor[1]])
            rec_t2_ph = self.reconstructor.predict([spatial_factor_null_all, modality_factor[1]])
            rec_bssfp_ph1 = self.reconstructor.predict([spatial_factor_null_inf, modality_factor[2]])
            rec_bssfp_ph2 = self.reconstructor.predict([spatial_factor_null_ede, modality_factor[2]])
            rec_bssfp_ph = self.reconstructor.predict([spatial_factor_null_all, modality_factor[2]])

            lge_row = np.concatenate([org_lge, rec_lge, (patho_inf - 0.5) * 2, rec_lge_ph1, rec_lge_ph2, rec_lge_ph],axis=2)
            t2_row = np.concatenate([org_t2, rec_t2, (patho_ede - 0.5) * 2, rec_t2_ph1, rec_t2_ph2, rec_t2_ph], axis=2)
            bssfp_row = np.concatenate([org_bssfp, rec_bssfp, patho_null, rec_bssfp_ph1, rec_bssfp_ph2, rec_bssfp_ph], axis=2)
            row = np.concatenate([lge_row, t2_row, bssfp_row], axis=1)

            rows = np.reshape(row, (row.shape[0] * row.shape[1], row.shape[2]))
            imsave(os.path.join(self.folder, "PseudoHealthComparison_%d.png" % (epoch)), rows)




    def callback_reconstruction(self,epoch, logs=None):
        batch_data_pack = next(self.gen)
        x = batch_data_pack[:, :, :, :self.image_channels]

        if self.spatial_encoder is not None:
            spatial_out = self.spatial_encoder.predict(x)
            if 'attention' in self.testmode:
                spatial_factor, pred_p1, pred_p2, _, _ = spatial_out
            else:
                spatial_factor, pred_p1, pred_p2 = spatial_out

            full_channel_factor = []
            for ii in range(x.shape[3]):
                full_channel_factor.append((x[:, :, :, ii]+1)/2)
            full_channel_factor = np.concatenate(full_channel_factor, axis=-1)
            for channel in range(spatial_factor.shape[3]):
                current_factor = spatial_factor[:, :, :, channel]
                full_channel_factor = np.concatenate([full_channel_factor, current_factor], axis=-1)
            full_channel_factor = np.reshape(full_channel_factor, (full_channel_factor.shape[0] * full_channel_factor.shape[1],
                                                                   full_channel_factor.shape[2]))
            imsave(os.path.join(self.folder, "SpatialFactor_%d.png" % (epoch)), full_channel_factor)


            # if self.reconstructor is not None:
            #     modality_factor = self.modality_encoder.predict([spatial_factor, x])
            #     # rec_x = self.reconstructor.predict([spatial_factor, modality_factor])
            #
            #     for ii in range(len(modality_factor)):
            #         if ii ==0:
            #             rec_x0 = self.reconstructor.predict([spatial_factor, modality_factor[ii]])
            #         elif ii == 1:
            #             rec_x1 = self.reconstructor.predict([spatial_factor, modality_factor[ii]])
            #         elif ii ==2:
            #             rec_x2 = self.reconstructor.predict([spatial_factor, modality_factor[ii]])
            #     if rec_x0.shape[0]==1:
            #         rec_x0 = np.expand_dims(np.squeeze(rec_x0),axis=0)
            #         rec_x1 = np.expand_dims(np.squeeze(rec_x1), axis=0)
            #         rec_x2 = np.expand_dims(np.squeeze(rec_x2), axis=0)
            #     else:
            #         rec_x0 = np.squeeze(rec_x0)
            #         rec_x1 = np.squeeze(rec_x1)
            #         rec_x2 = np.squeeze(rec_x2)
            #     rec_x0 = np.concatenate([x[:,:,:,0], rec_x0], axis=2)
            #     rec_x1 = np.concatenate([x[:,:,:,1], rec_x1], axis=2)
            #     rec_x2 = np.concatenate([x[:,:,:,2], rec_x2], axis=2)
            #
            #     # rec_x0 = np.concatenate([x[:, :, :, 0], rec_x[:, :, :, 0]], axis=2)
            #     # rec_x1 = np.concatenate([x[:, :, :, 1], rec_x[:, :, :, 1]], axis=2)
            #     # rec_x2 = np.concatenate([x[:, :, :, 2], rec_x[:, :, :, 2]], axis=2)
            #
            #     for channel in range(spatial_factor.shape[3]):
            #         current_spatial_factor_concat = np.copy(spatial_factor)
            #         current_spatial_factor_concat[:, :, :, channel:channel + 1] = np.zeros_like(
            #             current_spatial_factor_concat[:, :, :, channel:channel + 1])
            #         current_rec_x0 = self.reconstructor.predict([current_spatial_factor_concat, modality_factor[0]])
            #         current_rec_x1 = self.reconstructor.predict([current_spatial_factor_concat, modality_factor[1]])
            #         current_rec_x2 = self.reconstructor.predict([current_spatial_factor_concat, modality_factor[2]])
            #
            #         if rec_x0.shape[0] == 1:
            #             current_rec_x0 = np.expand_dims(np.squeeze(current_rec_x0), axis=0)
            #             current_rec_x1 = np.expand_dims(np.squeeze(current_rec_x1), axis=0)
            #             current_rec_x2 = np.expand_dims(np.squeeze(current_rec_x2), axis=0)
            #         else:
            #             current_rec_x0 = np.squeeze(current_rec_x0)
            #             current_rec_x1 = np.squeeze(current_rec_x1)
            #             current_rec_x2 = np.squeeze(current_rec_x2)
            #
            #         rec_x0 = np.concatenate([rec_x0, current_rec_x0], axis=2)
            #         rec_x1 = np.concatenate([rec_x1, current_rec_x1], axis=2)
            #         rec_x2 = np.concatenate([rec_x2, current_rec_x2], axis=2)
            #
            #         # rec_x0 = np.concatenate([rec_x0, current_rec_x0], axis=2)
            #         # rec_x1 = np.concatenate([rec_x1, current_rec_x1], axis=2)
            #         # rec_x2 = np.concatenate([rec_x2, current_rec_x2], axis=2)
            #
            #     plot_rec_x0 = np.reshape(rec_x0, (rec_x0.shape[0] * rec_x0.shape[1], rec_x0.shape[2]))
            #     plot_rec_x1 = np.reshape(rec_x1, (rec_x1.shape[0] * rec_x1.shape[1], rec_x1.shape[2]))
            #     plot_rec_x2 = np.reshape(rec_x2, (rec_x2.shape[0] * rec_x2.shape[1], rec_x2.shape[2]))
            #
            #     plot_rec_x = np.concatenate([plot_rec_x0, plot_rec_x1, plot_rec_x2], axis=0)
            #     imsave(os.path.join(self.folder, "Reconstruction_%d.png" % (epoch)), plot_rec_x)

    def callback_full_images(self, epoch, logs):
        batch_data_pack = next(self.gen)
        x = batch_data_pack[:, :, :, :self.image_channels]

        input_full = self.input_full.predict(x)
        input_full_save = []
        for ii in range(input_full.shape[0]):
            current_input_full = input_full[ii, :, :, :]
            row = []
            for jj in range(current_input_full.shape[-1]):
                row.append(current_input_full[:, :, jj:jj + 1])
            row = np.concatenate(row, axis=1)
            input_full_save.append(row)
        input_full_save = np.concatenate(input_full_save, axis=0)
        imsave(self.folder + '/full_input_%d.png' % (epoch), input_full_save)

    def callback_attention(self, epoch, logs):
        batch_data_pack = next(self.gen)
        x = batch_data_pack[:, :, :, :self.image_channels]
        # real_anato = batch_data_pack[:, :, :, self.image_channels:self.image_channels + self.anato_mask_channels]
        # real_patho1 = batch_data_pack[:, :, :,self.image_channels + self.anato_mask_channels:self.image_channels + self.anato_mask_channels + 1]
        # real_patho2 = batch_data_pack[:, :, :, self.image_channels + self.anato_mask_channels + 1:]

        # y = self.segmentor.predict(x)
        # pred_anato = y[0]
        # pred_patho1 = y[1]
        # pred_patho2 = y[2]

        # anato = np.concatenate([real_anato, pred_anato[:, :, :, 0:self.anato_mask_channels]], axis=-1)
        # patho1 = np.concatenate([real_patho1, pred_patho1[:, :, :, 0:1]], axis=-1)
        # patho2 = np.concatenate([real_patho2, pred_patho2[:, :, :, 0:1]], axis=-1)

        if 'attention' in self.testmode:
            #attention_map_list = self.attention_maps.predict(x)
            #current_attention_map = attention_map_list[self.conf.downsample]

            attention_output_list = self.attention_output_list.predict(x)
            spatial_attention_output = attention_output_list[0]
            channel_attention_output = attention_output_list[1]

            # rows = generate_attentions(anato, [patho1, patho2], current_attention_map)
            # rows = np.concatenate(rows, axis=0)
            x = (x + 1) / 2
            batch_size = x.shape[0]
            x_rows = []
            for ii in range(batch_size):
                row1, row2 = [], []
                current_x = x[ii, :, :, :]
                for jj in range(current_x.shape[2]):
                    row1.append(current_x[:, :, jj])
                    row2.append(current_x[:, :, jj])
                x_rows.append(np.concatenate(row1, axis=-1))
                x_rows.append(np.concatenate(row2, axis=-1))
            # im_plot_attention = np.concatenate([np.concatenate(x_rows, axis=0), rows], axis=1)
            # imsave(self.folder + '/attention_map_epoch_%d.png' % (epoch), im_plot_attention)

            attention_rows = []
            for ii in range(spatial_attention_output.shape[0]):
                attention_row = []
                for jj in range(spatial_attention_output.shape[-1]):
                    current_spatial_attention_output = spatial_attention_output[ii, :, :, jj]
                    current_channel_attention_output = channel_attention_output[ii, :, :, jj]
                    current_spatial_attention_output = current_spatial_attention_output \
                                                       - np.min(current_spatial_attention_output)
                    current_channel_attention_output = current_channel_attention_output \
                                                       - np.min(current_channel_attention_output)
                    current_spatial_attention_output = current_spatial_attention_output \
                                                       / np.max(current_spatial_attention_output)
                    current_channel_attention_output = current_channel_attention_output \
                                                       / np.max(current_channel_attention_output)

                    current_attention_output = np.concatenate([current_spatial_attention_output,
                                                               current_channel_attention_output], axis=0)
                    attention_row.append(current_attention_output)
                attention_row = np.concatenate(attention_row, axis=1)
                attention_rows.append(attention_row)
            attention_rows = np.concatenate(attention_rows, axis=0)
            im_attention_outputs = np.concatenate([np.concatenate(x_rows, axis=0), attention_rows], axis=1)
            imsave(self.folder + '/attention_output_epoch_%d.png' % (epoch), im_attention_outputs)

    def callback_save_segmentation(self, epoch, logs=None):
        batch_data_pack = next(self.gen)
        x = batch_data_pack[:, :, :, :self.image_channels]
        real_anato = batch_data_pack[:, :, :, self.image_channels:self.image_channels + self.anato_mask_channels]
        real_patho1 = batch_data_pack[:, :, :,self.image_channels + self.anato_mask_channels:self.image_channels + self.anato_mask_channels + 1]
        real_patho2 = batch_data_pack[:, :, :, self.image_channels + self.anato_mask_channels + 1:]

        y = self.segmentor.predict(x)
        pred_anato = y[0]
        pred_patho1 = y[1]
        pred_patho2 = y[2]

        anato = np.concatenate([real_anato, pred_anato[:, :, :, 0:self.anato_mask_channels]], axis=-1)
        patho1 = np.concatenate([real_patho1, pred_patho1[:, :, :, 0:1]], axis=-1)
        patho2 = np.concatenate([real_patho2, pred_patho2[:, :, :, 0:1]], axis=-1)

        im_plot = save_multiimage_segmentation(x, anato, [patho1, patho2], self.folder, epoch)
