from keras import Input, Model
from keras.layers import Concatenate, MaxPooling2D, LeakyReLU, Add, Activation, UpSampling2D, \
    BatchNormalization, Lambda, Dense, Flatten, Multiply, Subtract
from keras_contrib.layers import InstanceNormalization
from keras.backend import concatenate, ones_like,dot, transpose
from keras import backend as K

from models.basenet import BaseNet
import logging
log = logging.getLogger('unet')
from keras.layers import concatenate # harric added to incorporate the segementation correction when segmentation_option=1
from keras.backend import expand_dims # because mi only in my
from keras import regularizers
import os

from layers.Conv2D_Implementation import Conv2D_Implementation as Conv2D
from layers.Conv2D_Implementation import Conv2D_WithNorm_Implementation as Conv2D_Norm


class UNet(BaseNet):
    """
    UNet Implementation of 4 downsampling and 4 upsampling blocks.
    Each block has 2 convolutions, batch normalisation and relu.
    The number of filters for the 1st layer is 64 and at every block, this is doubled. Each upsampling block halves the
    number of filters.
    """
    def __init__(self, conf):
        """
        Constructor.
        :param conf: the configuration object
        """
        super(UNet, self).__init__(conf) # inherent from the BaseNet Class
        self.input_shape            = conf.input_shape
        self.residual               = conf.residual
        self.out_anato_channels     = conf.out_anato_channels
        self.out_patho_channels     = conf.out_patho_channels
        self.normalise              = conf.normalise
        self.f                      = conf.filters
        self.downsample             = conf.downsample
        self.regularizer            = conf.regularizer
        self.testmode               = conf.testmode
        assert self.downsample > 0, 'Unet downsample must be over 0.'
        self.channel_map_l1=self.channel_map_l2=self.channel_map_l3=None
        self.spatial_map_l1=self.spatial_map_l2=self.spatial_map_l3=None

    def build_cascaded_unet(self):
        """
        Build the model
        """
        self.input = Input(shape=self.input_shape)

        l = self.unet_downsample(self.input, self.normalise)
        self.unet_bottleneck(l, self.normalise)
        l = self.unet_upsample(self.bottleneck, self.normalise)

        # harric modified to incorporate with segmentation_option=2 case
        # when the mask prediction is performed in a channel-wised manner
        # possibly useless
        out_anatomy = self.normal_seg_out(l, out_channels=self.conf.out_anato_channels)
        out_anatomy_dice = Lambda(lambda x: x, name='DiceAna')(out_anatomy)
        out_anatomy_cross_entropy = Lambda(lambda x: x, name='CEAna')(out_anatomy)
        out_anatomy_of_interest = Lambda(lambda x: x[:,:,:,0:1])(out_anatomy)

        input2 = Multiply()([self.input, out_anatomy_of_interest])
        l = self.unet_downsample(input2, self.normalise)
        self.unet_bottleneck(l, self.normalise)
        l = self.unet_upsample(self.bottleneck, self.normalise)

        out_pathology_list = []
        out_pathology_list_oot = []
        full_backgrounds = Lambda(lambda x: ones_like(x))(Lambda(lambda x: x[:, :, :, 0:1])(out_anatomy))
        anatomy_of_interest_background = Subtract()([full_backgrounds, Lambda(lambda x: x[:, :, :, 0:1])(out_anatomy)])
        for ii in range(self.conf.num_patho_masks):
            out_pathology = self.normal_seg_out(l, out_channels=2)
            out_pathology_dice = Lambda(lambda x: x, name='DicePat_%d' % (ii + 1))(out_pathology)
            out_pathology_cross_entropy = Lambda(lambda x: x, name='CEPat_%d' % (ii + 1))(out_pathology)
            out_pathology_list.append(out_pathology_dice)
            out_pathology_list.append(out_pathology_cross_entropy)
            out_pathology_list_oot.append(Lambda(lambda x: x[:, :, :, 0:1])(out_pathology))
        out_pathology_oot_list = []
        for ii in range(self.conf.num_patho_masks):
            current_pat = Multiply()([out_pathology_list_oot[ii], anatomy_of_interest_background])
            current_pat = Lambda(lambda x: x, name='OOTPat_%d' % (ii + 1))(current_pat)
            out_pathology_oot_list.append(current_pat)

        self.model = Model(inputs=self.input, outputs=[out_anatomy_dice, out_anatomy_cross_entropy]
                                                      + out_pathology_list
                                                      + out_pathology_oot_list)
        self.model.summary(print_fn=log.info)

    def get_attention_maps(self):
        attention_map = Model(inputs=self.input,
                              outputs=[self.channel_map_l0,self.channel_map_l1,self.channel_map_l2,self.channel_map_l3]
                                      + [self.spatial_map_l0,self.spatial_map_l1,self.spatial_map_l2,self.spatial_map_l3])
        return attention_map

    def get_attention_output(self):
        attention_output = Model(inputs=self.input, outputs=[self.spatial_output, self.channel_output])
        return attention_output

    def get_input_full(self):
        input_full = Model(inputs=self.input,outputs=self.input_full)
        return input_full

    def build_concat(self, public_or_split=0):
        """
        Build the model
        """

        def _trainable_constrast_window(_x, filters, activ, upper_bound):
            l = Conv2D(filters, 1, strides=1, padding='same',
                       kernel_regularizer=regularizers.l2(self.regularizer),input_feature=_x,
                       side_connect='sideconv' in self.conf.testmode)

            def _relu_return(input_args):
                _x_ = input_args
                return K.minimum(K.maximum(_x_,0),upper_bound)

            def _sigmoid_return(input_args):
                _x_ = input_args
                return upper_bound * K.sigmoid(_x_)

            if activ =='sigmoid':
                return Lambda(lambda x:_sigmoid_return(x))(l)
            elif activ == 'relu':
                return Lambda(lambda x:_relu_return(x))(l)

        self.input = Input(shape=self.input_shape)
        input_list = [Lambda(lambda x: x[:, :, :, 0:1])(self.input),
                      Lambda(lambda x: x[:, :, :, 1:2])(self.input),
                      Lambda(lambda x: x[:, :, :, 2:3])(self.input)]

        if 'contrast-window' in self.conf.testmode:
            inp0 = _trainable_constrast_window(input_list[0], 2, 'sigmoid', 1)
            inp1 = _trainable_constrast_window(input_list[1], 2, 'sigmoid', 1)
            inp2 = _trainable_constrast_window(input_list[2], 4, 'sigmoid', 1)
            input_list[0] = Concatenate(axis=-1)([input_list[0], inp0])
            input_list[1] = Concatenate(axis=-1)([input_list[1], inp1])
            input_list[2] = Concatenate(axis=-1)([input_list[2], inp2])
        self.input_full = Concatenate(axis=-1)(input_list)

        self.alpha = K.variable(value=0.)
        self.beta = K.variable(value=0.)
        self.gamma = K.variable(value=0.)
        self.anato_label_input = Input(shape=self.input_shape[:2]+[self.conf.num_anato_masks])
        self.patho_label_input = []
        for ii in range(self.conf.num_patho_masks):
            self.patho_label_input.append(Input(shape=self.input_shape[:2]+[1]))

        # define the unet with pixel concat
        if self.testmode == 'pixel-concat':
            l = self.unet_downsample(input_list, self.normalise)
            self.unet_bottleneck(l, self.normalise)
            l = self.unet_upsample(self.bottleneck, self.normalise)
        elif 'feature-concat' in self.testmode: # define the unet with feature concat
            l = self.unet_downsample_distributed_v2(input_list, self.normalise)
            self.unet_bottleneck(l, self.normalise)
            l = self.unet_unsample_distributed(self.bottleneck, self.normalise)

        out_anatomy = self.normal_seg_out(l, out_channels=self.conf.num_anato_masks+1)
        out_anatomy_dice = Lambda(lambda x: x, name='DiceAna')(out_anatomy)
        out_anatomy_cross_entropy = Lambda(lambda x: x, name='CEAna')(out_anatomy)

        out_pathology_list = []
        out_pathology_list_oot = []
        full_backgrounds = Lambda(lambda x:ones_like(x))(Lambda(lambda x:x[:,:,:,0:1])(out_anatomy))
        anatomy_of_interest_background = Subtract()([full_backgrounds, Lambda(lambda x: x[:,:,:,0:1])(out_anatomy)])
        for ii in range(self.conf.num_patho_masks):
            out_pathology = self.normal_seg_out(l,out_channels=2)
            out_pathology_dice = Lambda(lambda x: x, name='DicePat_%d' % (ii+1))(out_pathology)
            out_pathology_cross_entropy = Lambda(lambda x: x, name='CEPat_%d' % (ii+1))(out_pathology)
            out_pathology_list.append(out_pathology_dice)
            out_pathology_list.append(out_pathology_cross_entropy)
            out_pathology_list_oot.append(Lambda(lambda x: x[:,:,:,0:1])(out_pathology))
        out_pathology_oot_list = []
        for ii in range(self.conf.num_patho_masks):
            current_pat = Multiply()([out_pathology_list_oot[ii], anatomy_of_interest_background])
            current_pat = Lambda(lambda x:x, name='OOTPat_%d' % (ii+1))(current_pat)
            out_pathology_oot_list.append(current_pat)

        attention_output_list = []
        if 'attention' in self.conf.testmode:
            out_anatomy_spatial_attention_output = Lambda(lambda x: x[:, :, :, :self.conf.num_anato_masks + 1], name='AnaSACE')(self.spatial_output)
            out_anatomy_channel_attention_output = Lambda(lambda x: x[:, :, :, :self.conf.num_anato_masks + 1], name='AnaCACE')(self.channel_output)
            attention_output_list = [out_anatomy_spatial_attention_output,out_anatomy_channel_attention_output]
            for ii in range(self.conf.num_patho_masks):
                current_spatial_patho = Lambda(lambda x: x[:, :, :, self.conf.num_anato_masks+1+ii*2:self.conf.num_anato_masks+1 + (ii + 1) * 2],
                                               name='Pat%dSACE' % (ii + 1))(self.spatial_output)
                current_channel_patho = Lambda(lambda x: x[:, :, :, self.conf.num_anato_masks+1+ii*2:self.conf.num_anato_masks+1 + (ii + 1) * 2],
                                               name='Pat%dSCCE' % (ii + 1))(self.channel_output)
                attention_output_list.append(current_spatial_patho)
                attention_output_list.append(current_channel_patho)

        self.model = Model(inputs=self.input, outputs=[out_anatomy_dice, out_anatomy_cross_entropy]
                                                      + out_pathology_list
                                                      + out_pathology_oot_list + attention_output_list)
        self.Enc_Anatomy_Pretrain = Model(inputs=self.input,
                                          outputs=self.bottleneck,
                                          name='Enc_Anatomy_Pretrain')
        self.model.summary(print_fn=log.info)

        # self.load_models()  # loaded already trained model
        # or pre-trained model
        # or the training is started in the middle stage
        if self.conf.load_pretrain==1:
            self.load_pretrain_model_from_public()
        elif self.conf.load_pretrain==2:
            self.load_pretrain_model_from_non_attention_split()
        self.load_models(public_or_split=public_or_split)

        return True

    def unet_downsample(self, inp, normalise):
        """
        Build downsampling path
        :param inp:         input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :return:            last layer of the downsampling path
        """
        inp = Concatenate(axis=-1)(inp)
        self.d_l0 = conv_block(inp, self.f, normalise, self.residual,
                               regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
        l = MaxPooling2D(pool_size=(2, 2))(self.d_l0)

        if self.downsample > 1:
            self.d_l1 = conv_block(l, self.f * 2, normalise, self.residual,
                                   regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l1)

        if self.downsample > 2:
            self.d_l2 = conv_block(l, self.f * 4, normalise, self.residual,
                                   regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l2)

        if self.downsample > 3:
            self.d_l3 = conv_block(l, self.f * 8, normalise,
                                   regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l3)
        return l

    def unet_downsample_distributed(self, inp, normalise):
        self.d_l0, self.d_l1, self.d_l2, self.d_l3, final_list = [],[],[],[], []
        for ii in range(len(inp)):
            d_l0 = conv_block(inp[ii], self.f, normalise,self.residual,
                              regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l = MaxPooling2D(pool_size=(2, 2))(d_l0)
            self.d_l0.append(d_l0)

            if self.downsample > 1:
                d_l1 = conv_block(l, self.f * 2, normalise, self.residual,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
                l = MaxPooling2D(pool_size=(2, 2))(d_l1)
                self.d_l1.append(d_l1)
                final_l = l

            if self.downsample > 2:
                d_l2 = conv_block(l, self.f * 4, normalise, self.residual,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
                l = MaxPooling2D(pool_size=(2, 2))(d_l2)
                self.d_l2.append(d_l2)
                final_l = l

            if self.downsample > 3:
                d_l3 = conv_block(l, self.f * 8, normalise,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
                l = MaxPooling2D(pool_size=(2, 2))(d_l3)
                self.d_l3.append(d_l3)
                final_l = l
            final_list.append(final_l)
        final_l = Concatenate()(final_list)
        return final_l


    def unet_downsample_distributed_v2(self, inp, normalise):

        def _merge_fusion(input_args):
            alpha = K.variable(value=0.)
            original_feature, attention_feature = input_args
            return attention_feature * alpha + original_feature

        def _max_operator(input_args):
            lge, t2, bssfp = input_args
            return_V = K.maximum(bssfp, K.maximum(lge, t2))
            return return_V

        def _fusion(lge, t2, bssfp, filters):
            d_fusion = Lambda(lambda x: _max_operator(x))([lge, t2, bssfp])
            lge = Add()([lge, d_fusion])
            t2 = Add()([t2, d_fusion])
            bssfp = Add()([bssfp, d_fusion])
            return Activation('relu')(lge), Activation('relu')(t2),Activation('relu')(bssfp)

        def _share_encoder_layer(input_feature_shape, filters, layer_name):
            input_feature = Input([int(input_feature_shape[0]),
                                   int(input_feature_shape[1]),
                                   int(input_feature_shape[2])])
            l_skip = conv_block(input_feature, filters, normalise, self.residual,
                                regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l = MaxPooling2D(pool_size=(2, 2))(l_skip)
            encoder_layer = Model(inputs=input_feature, outputs=[l, l_skip], name=layer_name)
            return encoder_layer

        d_l0_lge = conv_block(inp[0], self.f, normalise, self.residual,
                              regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
        l_lge = MaxPooling2D(pool_size=(2, 2))(d_l0_lge)
        d_l0_t2 = conv_block(inp[1], self.f, normalise, self.residual,
                             regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
        l_t2 = MaxPooling2D(pool_size=(2, 2))(d_l0_t2)
        d_l0_bssfp = conv_block(inp[2], self.f, normalise, self.residual,
                                regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
        l_bssfp = MaxPooling2D(pool_size=(2, 2))(d_l0_bssfp)
        self.d_l0 = [d_l0_lge, d_l0_t2, d_l0_bssfp]

        if self.downsample > 1:

            d_l1_lge = conv_block(l_lge, self.f * 2, normalise, self.residual,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            d_l1_t2 = conv_block(l_t2, self.f * 2, normalise, self.residual,
                                 regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            d_l1_bssfp = conv_block(l_bssfp, self.f * 2, normalise, self.residual,
                                    regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

            l_lge = MaxPooling2D(pool_size=(2, 2))(d_l1_lge)
            l_t2 = MaxPooling2D(pool_size=(2, 2))(d_l1_t2)
            l_bssfp = MaxPooling2D(pool_size=(2, 2))(d_l1_bssfp)

            final_l_lge = l_lge
            final_l_t2 = l_t2
            final_l_bssfp = l_bssfp
            self.d_l1 = [d_l1_lge, d_l1_t2, d_l1_bssfp]

        if self.downsample > 2:
            d_l2_lge = conv_block(l_lge, self.f * 4, normalise, self.residual,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            d_l2_t2 = conv_block(l_t2, self.f * 4, normalise, self.residual,
                                 regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            d_l2_bssfp = conv_block(l_bssfp, self.f * 4, normalise, self.residual,
                                    regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_lge = MaxPooling2D(pool_size=(2, 2))(d_l2_lge)
            l_t2 = MaxPooling2D(pool_size=(2, 2))(d_l2_t2)
            l_bssfp = MaxPooling2D(pool_size=(2, 2))(d_l2_bssfp)

            final_l_lge = l_lge
            final_l_t2 = l_t2
            final_l_bssfp = l_bssfp
            self.d_l2 = [d_l2_lge, d_l2_t2, d_l2_bssfp]

        if self.downsample > 3:
            d_l3_lge = conv_block(l_lge, self.f * 8, normalise,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_lge = MaxPooling2D(pool_size=(2, 2))(d_l3_lge)
            d_l3_t2 = conv_block(l_t2, self.f * 8, normalise,
                                 regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_t2 = MaxPooling2D(pool_size=(2, 2))(d_l3_t2)
            d_l3_bssfp = conv_block(l_bssfp, self.f * 8, normalise,
                                    regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_bssfp = MaxPooling2D(pool_size=(2, 2))(d_l3_bssfp)
            final_l_lge = l_lge
            final_l_t2 = l_t2
            final_l_bssfp = l_bssfp
            self.d_l3 = [d_l3_lge, d_l3_t2, d_l3_bssfp]

        final_list = [final_l_lge, final_l_t2, final_l_bssfp]
        final_l = Concatenate()(final_list)
        return final_l

    def unet_downsample_distributed_v3(self, inp, normalise):

        def _merge_fusion(input_args):
            alpha = K.variable(value=0.)
            original_feature, attention_feature = input_args
            return attention_feature * alpha + original_feature

        def _max_operator(input_args):
            lge, t2, bssfp = input_args
            return_V = K.maximum(bssfp,K.maximum(lge,t2))
            return return_V

        def _fusion(lge, t2, bssfp, filters):
            # fusion = Concatenate(axis=-1)([lge, t2, bssfp])
            d_fusion = Lambda(lambda x: _max_operator(x))([lge, t2, bssfp])
            # d_fusion = conv_block(fusion, filters, normalise, self.residual, regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            lge = Lambda(lambda x: _merge_fusion(x))([lge, d_fusion])
            t2 = Lambda(lambda x: _merge_fusion(x))([t2, d_fusion])
            bssfp = Lambda(lambda x: _merge_fusion(x))([bssfp, d_fusion])
            return lge, t2, bssfp


        def _share_encoder_layer(input_feature_shape, filters, layer_name):
            input_feature = Input([int(input_feature_shape[0]),
                                   int(input_feature_shape[1]),
                                   int(input_feature_shape[2])])
            l_skip = conv_block(input_feature, filters, normalise, self.residual,
                                regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l = MaxPooling2D(pool_size=(2, 2))(l_skip)
            encoder_layer = Model(inputs=input_feature,outputs=[l, l_skip], name=layer_name)
            return encoder_layer


        self.shared_layer0 = _share_encoder_layer(input_feature_shape=inp[0].shape[1:], filters=self.f, layer_name='shared_layer0')
        l_lge,d_l0_lge = self.shared_layer0(inp[0])
        l_t2,d_l0_t2 = self.shared_layer0(inp[1])
        l_bssfp,d_l0_bssfp = self.shared_layer0(inp[2])
        # d_l0_lge = conv_block(inp[0], self.f, normalise, self.residual, regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
        # l_lge = MaxPooling2D(pool_size=(2, 2))(d_l0_lge)
        # d_l0_t2 = conv_block(inp[1], self.f, normalise, self.residual, regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
        # l_t2 = MaxPooling2D(pool_size=(2, 2))(d_l0_t2)
        # d_l0_bssfp = conv_block(inp[2], self.f, normalise, self.residual, regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
        # l_bssfp = MaxPooling2D(pool_size=(2, 2))(d_l0_bssfp)
        self.d_l0 = [d_l0_lge, d_l0_t2, d_l0_bssfp]

        if self.downsample > 1:
            # shared_layer1 = _share_encoder_layer(input_feature_shape=l_lge.shape[1:], filters=self.f * 2, layer_name='shared_layer1')
            # l_lge,d_l1_lge = shared_layer1(l_lge)
            # l_t2,d_l1_t2 = shared_layer1(l_t2)
            # l_bssfp,d_l1_bssfp = shared_layer1(l_bssfp)
            d_l1_lge = conv_block(l_lge, self.f * 2, normalise, self.residual,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_lge = MaxPooling2D(pool_size=(2, 2))(d_l1_lge)
            d_l1_t2 = conv_block(l_t2, self.f * 2, normalise, self.residual,
                                 regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_t2 = MaxPooling2D(pool_size=(2, 2))(d_l1_t2)
            d_l1_bssfp = conv_block(l_bssfp, self.f * 2, normalise, self.residual,
                                    regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_bssfp = MaxPooling2D(pool_size=(2, 2))(d_l1_bssfp)

            final_l_lge = l_lge
            final_l_t2 = l_t2
            final_l_bssfp = l_bssfp
            self.d_l1 = [d_l1_lge, d_l1_t2, d_l1_bssfp]

        if self.downsample > 2:
            d_l2_lge = conv_block(l_lge, self.f * 4, normalise, self.residual,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_lge = MaxPooling2D(pool_size=(2, 2))(d_l2_lge)
            d_l2_t2 = conv_block(l_t2, self.f * 4, normalise, self.residual,
                                 regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_t2 = MaxPooling2D(pool_size=(2, 2))(d_l2_t2)
            d_l2_bssfp = conv_block(l_bssfp, self.f * 4, normalise, self.residual,
                                    regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_bssfp = MaxPooling2D(pool_size=(2, 2))(d_l2_bssfp)
            # if 'merge-encoder' in self.conf.testmode:
            #     l_lge, l_t2, l_bssfp = _fusion(l_lge, l_t2, l_bssfp, self.f * 4)
            final_l_lge = l_lge
            final_l_t2 = l_t2
            final_l_bssfp = l_bssfp
            self.d_l2 = [d_l2_lge, d_l2_t2, d_l2_bssfp]

        if self.downsample > 3:
            d_l3_lge = conv_block(l_lge, self.f * 8, normalise,
                                  regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_lge = MaxPooling2D(pool_size=(2, 2))(d_l3_lge)
            d_l3_t2 = conv_block(l_t2, self.f * 8, normalise,
                                 regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_t2 = MaxPooling2D(pool_size=(2, 2))(d_l3_t2)
            d_l3_bssfp = conv_block(l_bssfp, self.f * 8, normalise,
                                    regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)
            l_bssfp = MaxPooling2D(pool_size=(2, 2))(d_l3_bssfp)
            if 'merge-encoder' in self.conf.testmode:
                l_lge, l_t2, l_bssfp = _fusion(l_lge, l_t2, l_bssfp, self.f * 8)
            final_l_lge = l_lge
            final_l_t2 = l_t2
            final_l_bssfp = l_bssfp
            self.d_l3 = [d_l3_lge, d_l3_t2, d_l3_bssfp]

        final_l = Concatenate()([final_l_lge, final_l_t2, final_l_bssfp])
        return final_l

    def unet_bottleneck(self, l, normalise, name=''):
        """
        Build bottleneck layers
        :param inp:         input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :param name:        name of the layer
        """
        flt = self.f * 2
        if self.downsample > 1:
            flt *= 2
        if self.downsample > 2:
            flt *= 2
        if self.downsample > 3:
            flt *= 2
        self.bottleneck = conv_block(l, flt, normalise, self.residual, name,
                                     regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

    def unet_upsample(self, l, normalise):
        """
        Build upsampling path
        :param l:           the input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :return:            the last layer of the upsampling path
        """
        if self.downsample > 3:
            l = upsample_block(l, self.f * 8, normalise, activation='linear', regularizer=self.regularizer,
                               side_connect='sideconv' in self.conf.testmode)
            l = Concatenate()([l, self.d_l3])
            l = conv_block(l, self.f * 8, normalise, self.residual,
                           regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

        if self.downsample > 2:
            l = upsample_block(l, self.f * 4, normalise, activation='linear', regularizer=self.regularizer,
                               side_connect='sideconv' in self.conf.testmode)
            l = Concatenate()([l, self.d_l2])
            l = conv_block(l, self.f * 4, normalise, self.residual,
                           regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

        if self.downsample > 1:
            l = upsample_block(l, self.f * 2, normalise, activation='linear', regularizer=self.regularizer,
                               side_connect='sideconv' in self.conf.testmode)
            l = Concatenate()([l, self.d_l1])
            l = conv_block(l, self.f * 2, normalise, self.residual,
                           regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

        if self.downsample > 0:
            l = upsample_block(l, self.f, normalise, activation='linear', regularizer=self.regularizer,
                               side_connect='sideconv' in self.conf.testmode)
            l = Concatenate()([l, self.d_l0])
            l = conv_block(l, self.f, normalise, self.residual,
                           regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

        return l

    def unet_unsample_distributed(self, l, normalise):
        def _build_spatial_attention(input_args):
            query, key, value,in_features = input_args
            q_shape = [K.tf.shape(query)[k] for k in range(4)]
            v_shape = [K.tf.shape(value)[k] for k in range(4)]
            k_shape = [K.tf.shape(key)[k] for k in range(4)]

            query = K.tf.reshape(query, shape=[-1, q_shape[1]*q_shape[2], q_shape[3]])
            key = K.tf.reshape(key, shape=[-1, k_shape[1] * k_shape[2], k_shape[3]])
            spatial_attention_map = K.tf.nn.softmax(K.tf.matmul(query, key, transpose_b=True))
            spatial_output = K.tf.matmul(spatial_attention_map,
                                         K.tf.reshape(value, shape=[-1,
                                                                    v_shape[1] * v_shape[2],
                                                                    v_shape[3]]))
            spatial_output = K.tf.reshape(spatial_output, [-1, v_shape[1],
                                                           v_shape[2],
                                                           v_shape[3]])*self.alpha + in_features
            return [spatial_output, spatial_attention_map]

        def _build_channel_attention(input_args):
            query, key, value, in_features = input_args
            query = K.tf.reshape(query, shape=[-1, query.shape[1] * query.shape[2], query.shape[-1]])
            key = K.tf.reshape(key, shape=[-1, key.shape[1] * key.shape[2], key.shape[-1]])
            channel_attention_map = K.tf.nn.softmax(K.tf.matmul(query, key, transpose_a=True))
            channel_output = K.tf.matmul(K.tf.reshape(value, shape=[-1,
                                                                    value.shape[1] * value.shape[2],
                                                                    value.shape[-1]]), channel_attention_map)
            channel_output = K.tf.reshape(channel_output, [-1, value.shape[1],
                                                           value.shape[2],
                                                           value.shape[-1]])*self.beta + in_features
            return [channel_output, channel_attention_map]

        def _calculate_attention_feature(f, concat_features, last=False):

            def _merge_spatial_channel(input_args):
                spatial, channel = input_args
                return spatial*self.gamma + channel


            f_num = f # inter_channels
            f_attention = f // 2 # attention channels, inter_channels // 8
            if not last:
                f_num_final = f_num
            else:
                f_num_final = self.conf.num_anato_masks + self.conf.num_patho_masks + self.conf.num_patho_masks + 1


            # spatial brunch
            spatial_input = Conv2D(f_num, 3, padding='same',
                                   kernel_regularizer=regularizers.l2(self.regularizer),
                                   input_feature=concat_features)
            spatial_input = MaxPooling2D(pool_size=(2, 2))(spatial_input)
            spatial_input = Conv2D(f_num, 3, padding='same',
                                   kernel_regularizer=regularizers.l2(self.regularizer),
                                   input_feature=spatial_input)
            spatial_input = MaxPooling2D(pool_size=(2, 2))(spatial_input)
            query = Conv2D(f_attention, 1, padding='same',
                           kernel_regularizer=regularizers.l2(self.regularizer),
                           input_feature=spatial_input)
            key = Conv2D(f_attention, 1, padding='same',
                         kernel_regularizer=regularizers.l2(self.regularizer),
                         input_feature=spatial_input)
            value = Conv2D(int(spatial_input.shape[3]), 1,
                           padding='same',
                           kernel_regularizer=regularizers.l2(self.regularizer),
                           input_feature=spatial_input)
            # query = MaxPooling2D(pool_size=(attention_shrink, attention_shrink))(query)
            # key = MaxPooling2D(pool_size=(attention_shrink, attention_shrink))(key)
            # value = MaxPooling2D(pool_size=(attention_shrink, attention_shrink))(value)


            spatial_attention_pack \
                = Lambda(lambda x:_build_spatial_attention(x))([query,key,value,spatial_input])
            spatial_output, spatial_attention_map = spatial_attention_pack
            spatial_output = UpSampling2D(size=2)(spatial_output)
            spatial_output = Conv2D(f_num, 3, padding='same',
                                    kernel_regularizer=regularizers.l2(self.regularizer),
                                    input_feature=spatial_output)
            spatial_output = UpSampling2D(size=2)(spatial_output)
            spatial_output = Conv2D(f_num_final, 1, padding='same',
                                    kernel_regularizer=regularizers.l2(self.regularizer),
                                    input_feature=spatial_output)


            # # channel brunch
            # channel_input = Conv2D(f_num, 3, padding='same', kernel_regularizer=regularizers.l2(self.regularizer))(concat_features)
            # channel_input = MaxPooling2D(pool_size=(2, 2))(channel_input)
            # channel_input = Conv2D(f_num, 3, padding='same', kernel_regularizer=regularizers.l2(self.regularizer))(channel_input)
            # channel_input = MaxPooling2D(pool_size=(2, 2))(channel_input)
            # # query = Conv2D(int(concat_features.shape[3]), 1, padding='same', kernel_regularizer=regularizers.l2(self.regularizer))(channel_input)
            # # key = Conv2D(int(concat_features.shape[3]), 1, padding='same', kernel_regularizer=regularizers.l2(self.regularizer))(channel_input)
            # # value = Conv2D(int(concat_features.shape[3]), 1, padding='same', kernel_regularizer=regularizers.l2(self.regularizer))(channel_input)
            # channel_attention_pack\
            #     = Lambda(lambda x: _build_channel_attention(x))([channel_input, channel_input, channel_input, channel_input])
            # channel_output, channel_attention_map = channel_attention_pack
            # channel_output = UpSampling2D(size=2)(channel_output)
            # channel_output = Conv2D(f_num, 3, padding='same', kernel_regularizer=regularizers.l2(self.regularizer))(channel_output)
            # channel_output = UpSampling2D(size=2)(channel_output)
            # channel_output = Conv2D(f_num_final, 1, padding='same', kernel_regularizer=regularizers.l2(self.regularizer))(channel_output)
            #
            #
            # # combine spatial and channel
            # # attention_output = Conv2D(int((spatial_output + channel_output).shape[3]), 1, padding='same',
            # #                           kernel_regularizer=regularizers.l2(self.regularizer))((Add()([spatial_output, channel_output])))
            # # attention_output = Add()([spatial_output, channel_output])
            # attention_output = Lambda(lambda x:_merge_spatial_channel(x))([spatial_output,channel_output])
            # return attention_output, spatial_attention_map, channel_attention_map, spatial_output, channel_output

            attention_output = spatial_output
            channel_attention_map = spatial_attention_map
            channel_output = spatial_output
            return attention_output, spatial_attention_map, channel_attention_map, spatial_output, channel_output

        def _max_operator(input_args):
            lge, t2, bssfp = input_args
            return_V = K.maximum(bssfp, K.maximum(lge, t2))
            return return_V

        if self.downsample > 3:
            l = upsample_block(l, self.f * 8, normalise, activation='linear',
                               regularizer=self.regularizer, side_connect='sideconv' in self.conf.testmode)
            if 'maxfuseall' in self.conf.testmode:
                d_fusion = Lambda(lambda x: _max_operator(x))(self.d_l3)
                if 'keeporg' in self.conf.testmode:
                    l = Concatenate()([l, Concatenate()(self.d_l3), d_fusion])
                else:
                    l = Concatenate()([l, d_fusion])
            else:
                l = Concatenate()([l, Concatenate()(self.d_l3)])
            # l = Concatenate()([l, Concatenate()(self.d_l3)])
            l = conv_block(l, self.f * 8, normalise, self.residual,
                           regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

        if self.downsample > 2:
            l = upsample_block(l, self.f * 4, normalise, activation='linear',
                               regularizer=self.regularizer, side_connect='sideconv' in self.conf.testmode)
            if 'maxfuseall' in self.conf.testmode:
                d_fusion = Lambda(lambda x: _max_operator(x))(self.d_l2)
                if 'keeporg' in self.conf.testmode:
                    l = Concatenate()([l, Concatenate()(self.d_l2), d_fusion])
                else:
                    l = Concatenate()([l, d_fusion])
            else:
                l = Concatenate()([l, Concatenate()(self.d_l2)])
            # l = Concatenate()([l, Concatenate()(self.d_l2)])
            l = conv_block(l, self.f * 4, normalise, self.residual,
                           regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

        if self.downsample > 1:
            l = upsample_block(l, self.f * 2, normalise, activation='linear',
                               regularizer=self.regularizer, side_connect='sideconv' in self.conf.testmode)
            if 'maxfuseall' in self.conf.testmode:
                d_fusion = Lambda(lambda x: _max_operator(x))(self.d_l1)
                if 'keeporg' in self.conf.testmode:
                    l = Concatenate()([l, Concatenate()(self.d_l1), d_fusion])
                else:
                    l = Concatenate()([l, d_fusion])
            else:
                l = Concatenate()([l, Concatenate()(self.d_l1)])
            # l = Concatenate()([l, Concatenate()(self.d_l1)])
            l = conv_block(l, self.f * 2, normalise, self.residual,
                           regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)


        if self.downsample > 0:
            l = upsample_block(l, self.f, normalise, activation='linear',
                               regularizer=self.regularizer, side_connect='sideconv' in self.conf.testmode)
            if 'maxfuse' in self.conf.testmode:
                d_fusion = Lambda(lambda x: _max_operator(x))(self.d_l0)
                if 'keeporg' in self.conf.testmode:
                    l = Concatenate()([l, Concatenate()(self.d_l0),d_fusion])
                else:
                    l = Concatenate()([l,d_fusion])
            else:
                l = Concatenate()([l, Concatenate()(self.d_l0)])
            l = conv_block(l, self.f, normalise, self.residual,
                           regularizer=self.regularizer,side_connect='sideconv' in self.conf.testmode)

            self.EncDec_Anatomy_Pretrain = Model(inputs=self.input,
                                                 outputs=l,
                                                 name='EncDec_Anatomy_Pretrain')

            if 'attention' in self.testmode:
                l, self.spatial_map_l0, self.channel_map_l0, self.spatial_output, self.channel_output \
                    = _calculate_attention_feature(self.f, l, last=True)
                self.channel_map_l1 = self.channel_map_l2 = self.channel_map_l3 = self.channel_map_l0
                self.spatial_map_l1 = self.spatial_map_l2 = self.spatial_map_l3 = self.spatial_map_l0
        return l

    def normal_seg_out(self, l, out_activ=None,out_channels=-1):
        """
        Build ouput layer
        :param l: last layer from the upsampling path
        :return:  the final segmentation layer
        """
        if out_activ is None:
            out_activ = 'sigmoid' if out_channels == 1 else 'softmax'
        return Conv2D(out_channels, 1, activation=out_activ,
                      kernel_regularizer=regularizers.l2(self.regularizer),
                      input_feature=l,
                      side_connect='sideconv' in self.conf.testmode)

    def load_pretrain_model_from_public(self):
        data_dir = self.conf.folder.split('split')[0][:-1] + '_pretrain'
        log.info("Loading Pretrain Model" )
        log.info(data_dir)
        self.Enc_Anatomy_Pretrain.load_weights(data_dir + '/Enc_Anatomy_Pretrain' )
        
    def load_pretrain_model_from_non_attention_split(self):
        data_dir = self.conf.folder.replace('-attention','')
        log.info("Loading Pretrain ENCDEC BOTH Model" )
        log.info(data_dir)
        self.EncDec_Anatomy_Pretrain.load_weights(data_dir + '/EncDec_Anatomy_Pretrain' )

    def save_pretrain_model_from_non_attention_split(self):
        data_dir = self.conf.folder
        log.info("Saving Pretrain ENCDEC BOTH Model" )
        log.info(data_dir)
        self.EncDec_Anatomy_Pretrain.save_weights(data_dir + '/EncDec_Anatomy_Pretrain' )





def conv_block(l0, f, norm_name, residual=False, name='', regularizer=0, lastact=True, side_connect=False):
    """
    Convolutional block
    :param l0:        the input layer
    :param f:         number of feature maps
    :param residual:  True/False to define residual connections
    :return:          the last layer of the convolutional block
    """
    l = Conv2D_Norm(f, 3, norm_name=norm_name,strides=1, padding='same',
                    kernel_regularizer=regularizers.l2(regularizer),
                    input_feature=l0,
                    side_connect=side_connect)

    if not side_connect:
        l = normalise(norm_name)(l)
    l = Activation('relu')(l)
    l = Conv2D_Norm(f, 3, norm_name=norm_name,strides=1, padding='same',
                    kernel_regularizer=regularizers.l2(regularizer),
                    input_feature=l,
                    side_connect=side_connect)

    if not side_connect:
        l = normalise(norm_name)(l)
    if lastact:
        l = Activation('relu', name=name)(l)
    if residual:
        l = Concatenate([l0, l], axis=-1)
    return l


def upsample_block(l0, f, norm_name, activation='relu', regularizer=0, side_connect=False):
    """
    Upsampling block.
    :param l0:          input layer
    :param f:           number of feature maps
    :param activation:  activation name
    :return:            the last layer of the upsampling block
    """
    l = UpSampling2D(size=2)(l0)
    l = Conv2D_Norm(f, 3, norm_name=norm_name,padding='same',
                    kernel_regularizer=regularizers.l2(regularizer), input_feature=l, side_connect=side_connect)
    if not side_connect:
        l = normalise(norm_name)(l)

    if activation == 'leakyrelu':
        return LeakyReLU()(l)
    else:
        return Activation(activation)(l)


def normalise(norm=None, **kwargs):
    if norm == 'instance':
        return InstanceNormalization(**kwargs)
    elif norm == 'batch':
        return BatchNormalization()
    else:
        return Lambda(lambda x : x)