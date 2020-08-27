from keras.layers import Conv2D, Add, BatchNormalization, Lambda
from keras_contrib.layers import InstanceNormalization



def normalise(norm=None, **kwargs):
    if norm == 'instance':
        return InstanceNormalization(**kwargs)
    elif norm == 'batch':
        return BatchNormalization()
    else:
        return Lambda(lambda x : x)
def Conv2D_Implementation(filters,kernel,strides=1,
                          padding='same',
                          kernel_regularizer=-1,
                          input_feature=-1,
                          side_connect=False,
                          activation=None):
    if not side_connect or kernel==1:
        conv = Conv2D(filters, kernel_size=(kernel,kernel), strides=strides, padding=padding,
                      kernel_regularizer=kernel_regularizer,
                      activation=activation)(input_feature)
    elif side_connect and kernel>1:
        conv_main = Conv2D(filters, kernel_size=(kernel,kernel), strides=strides, padding=padding,
                           kernel_regularizer=kernel_regularizer,
                           activation=activation)(input_feature)
        conv_ver = Conv2D(filters, kernel_size=(kernel,1), strides=strides, padding=padding,
                          kernel_regularizer=kernel_regularizer,
                          activation=activation)(input_feature)
        conv_hor = Conv2D(filters, kernel_size=(1, kernel), strides=strides, padding=padding,
                          kernel_regularizer=kernel_regularizer,
                          activation=activation)(input_feature)
        conv = Add()([conv_main,conv_ver,conv_hor])
    return conv


def Conv2D_WithNorm_Implementation(filters,kernel,norm_name, strides=1,
                                   padding='same',
                                   kernel_regularizer=-1,
                                   input_feature=-1,
                                   side_connect=False,
                                   activation=None):
    if not side_connect or kernel==1:
        conv = Conv2D(filters, kernel_size=(kernel,kernel), strides=strides, padding=padding,
                      kernel_regularizer=kernel_regularizer,
                      activation=activation)(input_feature)
    elif side_connect and kernel>1:
        conv_main = Conv2D(filters, kernel_size=(kernel,kernel), strides=strides, padding=padding,
                           kernel_regularizer=kernel_regularizer,
                           activation=activation)(input_feature)
        conv_ver = Conv2D(filters, kernel_size=(kernel,1), strides=strides, padding=padding,
                          kernel_regularizer=kernel_regularizer,
                          activation=activation)(input_feature)
        conv_hor = Conv2D(filters, kernel_size=(1, kernel), strides=strides, padding=padding,
                          kernel_regularizer=kernel_regularizer,
                          activation=activation)(input_feature)

        conv_main = normalise(norm_name)(conv_main)
        conv_ver = normalise(norm_name)(conv_ver)
        conv_hor = normalise(norm_name)(conv_hor)
        conv = Add()([conv_main,conv_ver,conv_hor])
    return conv