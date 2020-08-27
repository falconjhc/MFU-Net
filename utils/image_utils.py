
import numpy as np
import os
import scipy
from PIL import Image, ImageDraw
from imageio import imwrite as imsave # harric modified
from scipy.ndimage.morphology import binary_fill_holes
import utils.data_utils
import matplotlib.pylab as plt
import imageio.core.util
import matplotlib.path as pth



def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning
# harric added to disable ignoring warning messages

# harric added for efficient image drawing
def image_show(img):
    img =np.squeeze(img)
    if len(img.shape)==2:
        img = np.expand_dims(img,axis=2)
        img = np.tile(img,[1,1,3])
    img = img - np.min(img)
    img = img / np.max(img)
    plt.imshow(img)
    plt.show()

# harric added to incorporate with segmentation_option=4 case
def regression2segmentation(input_regression):
    indices_mask_background = np.where(np.logical_and(input_regression >= 0.0, input_regression < 1.0 / 3.0))
    indices_mask_myocardium = np.where(np.logical_and(input_regression >= 1.0 / 3.0, input_regression <= 2.0 / 3.0))
    indices_mask_infarction = np.where(np.logical_and(input_regression > 2.0 / 3.0, input_regression <= 1.0))
    mask_background = np.zeros(shape=input_regression.shape, dtype=input_regression.dtype)
    mask_myocardium = np.zeros(shape=input_regression.shape, dtype=input_regression.dtype)
    mask_infarction = np.zeros(shape=input_regression.shape, dtype=input_regression.dtype)
    mask_background[indices_mask_background] = 1.
    mask_myocardium[indices_mask_myocardium] = 1.
    mask_infarction[indices_mask_infarction] = 1.
    output_segmentation = np.concatenate([mask_myocardium, mask_infarction, mask_background], axis=3)
    return output_segmentation



def save_multiimage_segmentation(x, anato, patho, folder, epoch):
    x = (x + 1) / 2
    batch_size = x.shape[0]
    rows = []
    for ii in range(batch_size):
        row1, row2 = [],[]
        current_x = x[ii,:,:,:]
        for jj in range(current_x.shape[2]):
            row1.append(current_x[:,:,jj])
            row2.append(current_x[:,:,jj])

        anato_mask_num = int(anato.shape[3]/2)
        current_anato_real = anato[ii,:,:,0:anato_mask_num]
        current_anato_pred = anato[ii,:,:,anato_mask_num:]
        for jj in range(anato_mask_num):
            row1.append(current_anato_real[:,:,jj])
            row2.append(current_anato_pred[:,:,jj])
        for jj in range(len(patho)):
            current_patho_real = patho[jj][ii,:,:,0]
            current_patho_pred = patho[jj][ii,:,:,1]
            row1.append(current_patho_real)
            row2.append(current_patho_pred)
        rows.append(np.concatenate(row1, axis=-1))
        rows.append(np.concatenate(row2, axis=-1))

    im_plot = np.concatenate(rows, axis=0)
    imsave(folder + '/segmentations_epoch_%d.png' % (epoch), im_plot)
    # harric modified
    return im_plot





def generate_attentions(anato, patho, input_attention_matrix):

    def _generate_attention_map_implementation(map, mask):
        output_list = []
        for ii in range(mask.shape[0]):
            non_zero_indices = np.where(mask[ii,:]==1)
            current_activations = np.sum(map[non_zero_indices[0], :], axis=0)
            current_activations = 1 / (1 + np.exp(-current_activations))
            output_list.append(np.reshape(current_activations, (current_activations.shape[0],-1)))
        return np.transpose(np.concatenate(output_list, axis=-1))

    patho1, patho2 = patho
    rows = []
    for ii in range(anato.shape[0]):
        current_attention_map_checking = input_attention_matrix[ii,:,:]
        current_anato = anato[ii, :, :, :]
        current_patho1 = patho1[ii, :, :, :]
        current_patho2 = patho2[ii, :, :, :]
        current_anato = np.moveaxis(current_anato, 2, 0)
        current_patho1 = np.moveaxis(current_patho1, 2, 0)
        current_patho2 = np.moveaxis(current_patho2, 2, 0)
        current_anato = np.reshape(current_anato, (current_anato.shape[0], current_anato.shape[1] * current_anato.shape[2]))
        current_patho1 = np.reshape(current_patho1, (current_patho1.shape[0], current_patho1.shape[1] * current_patho1.shape[2]))
        current_patho2 = np.reshape(current_patho2, (current_patho2.shape[0], current_patho2.shape[1] * current_patho2.shape[2]))

        attention_anato = _generate_attention_map_implementation(current_attention_map_checking, current_anato)
        attention_patho1 = _generate_attention_map_implementation(current_attention_map_checking, current_patho1)
        attention_patho2 = _generate_attention_map_implementation(current_attention_map_checking, current_patho2)


        attention_anato = np.reshape(attention_anato,
                                     (int(np.sqrt(attention_anato.shape[1])),
                                      int(np.sqrt(attention_anato.shape[1])),
                                      attention_anato.shape[0]))
        attention_patho1 = np.reshape(attention_patho1,
                                      (int(np.sqrt(attention_patho1.shape[1])),
                                       int(np.sqrt(attention_patho1.shape[1])),
                                       attention_patho1.shape[0]))
        attention_patho2 = np.reshape(attention_patho2,
                                      (int(np.sqrt(attention_patho2.shape[1])),
                                       int(np.sqrt(attention_patho2.shape[1])),
                                       attention_patho2.shape[0]))

        if not np.max(attention_anato) == 0:
            attention_anato = attention_anato / np.max(attention_anato)
        if not np.max(attention_patho1) == 0:
            attention_patho1 = attention_patho1 / np.max(attention_patho1)
        if not np.max(attention_patho2) == 0:
            attention_patho2 = attention_patho2 / np.max(attention_patho2)

        row1, row2 = [], []
        anato_mask_num = int(attention_anato.shape[2] / 2)
        current_anato_real = attention_anato[:, :, 0:anato_mask_num]
        current_anato_pred = attention_anato[:, :, anato_mask_num:]
        for jj in range(anato_mask_num):
            row1.append(current_anato_real[:, :, jj])
            row2.append(current_anato_pred[:, :, jj])
        current_patho_real = attention_patho1[:, :, 0]
        current_patho_pred = attention_patho1[:, :, 1]
        row1.append(current_patho_real)
        row2.append(current_patho_pred)
        current_patho_real = attention_patho2[:, :, 0]
        current_patho_pred = attention_patho2[:, :, 1]
        row1.append(current_patho_real)
        row2.append(current_patho_pred)
        rows.append(np.concatenate(row1, axis=-1))
        rows.append(np.concatenate(row2, axis=-1))

    return rows

def save_segmentation(x, anato, patho):
    x = (x + 1) / 2
    batch_size = x.shape[0]
    rows = []
    for ii in range(batch_size):
        row1, row2 = [],[]
        current_x = x[ii,:,:,:]
        for jj in range(current_x.shape[2]):
            row1.append(current_x[:,:,jj])
            row2.append(current_x[:,:,jj])

        anato_mask_num = int(anato.shape[3]/2)
        current_anato_real = anato[ii,:,:,0:anato_mask_num]
        current_anato_pred = anato[ii,:,:,anato_mask_num:]
        for jj in range(anato_mask_num):
            row1.append(current_anato_real[:,:,jj])
            row2.append(current_anato_pred[:,:,jj])
        for jj in range(len(patho)):
            current_patho_real = patho[jj][ii,:,:,0]
            current_patho_pred = patho[jj][ii,:,:,1]
            row1.append(current_patho_real)
            row2.append(current_patho_pred)
        rows.append(np.concatenate(row1, axis=-1))
        rows.append(np.concatenate(row2, axis=-1))

    im_plot = np.concatenate(rows, axis=0)
    return im_plot





def convert_myo_to_lv(mask):
    '''
    Create a LV mask from a MYO mask. This assumes that the MYO is connected.
    :param mask: a 4-dim myo mask
    :return:     a 4-dim array with the lv mask.
    '''
    assert len(mask.shape) == 4, mask.shape

    # If there is no myocardium, then there's also no LV.
    if mask.sum() == 0:
        return np.zeros(mask)

    assert mask.max() == 1 and mask.min() == 0

    mask_lv = []
    for slc in range(mask.shape[0]):
        myo = mask[slc, :, :, 0]
        myo_lv = binary_fill_holes(myo).astype(int)
        lv = myo_lv - myo
        mask_lv.append(np.expand_dims(np.expand_dims(lv, axis=0), axis=-1))
    return np.concatenate(mask_lv, axis=0)


def makeTextHeaderImage(col_widths, headings, padding=(5, 5)):
    im_width = len(headings) * col_widths
    im_height = padding[1] * 2 + 11

    img = Image.new('RGB', (im_width, im_height), (0, 0, 0))
    d = ImageDraw.Draw(img)

    for i, txt in enumerate(headings):

        while d.textsize(txt)[0] > col_widths - padding[0]:
            txt = txt[:-1]
        d.text((col_widths * i + padding[0], + padding[1]), txt, fill=(1, 0, 0))

    raw_img_data = np.asarray(img, dtype="int32")

    return raw_img_data[:, :, 0]


def get_roi_dims(mask_list, size_mult=16):
    # This assumes each element in the mask list has the same dimensions
    masks = np.concatenate(mask_list, axis=0)
    masks = np.squeeze(masks)
    assert len(masks.shape) == 3

    lx, hx, ly, hy = 0, 0, 0, 0
    for y in range(masks.shape[2] - 1, 0, -1):
        if masks[:, :, y].max() == 1:
            hy = y
            break
    for y in range(masks.shape[2]):
        if masks[:, :, y].max() == 1:
            ly = y
            break
    for x in range(masks.shape[1] - 1, 0, -1):
        if masks[:, x, :].max() == 1:
            hx = x
            break
    for x in range(masks.shape[1]):
        if masks[:, x, :].max() == 1:
            lx = x
            break

    l = np.max([np.min([lx, ly]) - 10, 0])
    r = np.min([np.max([hx, hy]) + 10, masks.shape[2]])

    l, r = greatest_common_divisor(l, r, size_mult)

    return l, r


def greatest_common_divisor(l, r, size_mult):
    if (r - l) % size_mult != 0:
        div = (r - l) / size_mult
        if div * size_mult < (div + 1) * size_mult:
            diff = (r - l) - div * size_mult
            l += diff / 2
            r -= diff - (diff / 2)
        else:
            diff = (div + 1) * size_mult - (r - l)
            l -= diff / 2
            r += diff - (diff / 2)
    return int(l), int(r)


def process_contour(input_img, endocardium, epicardium=None):
    '''
    in each pixel we sample these 8 points:
     _________________
    |    *        *   |
    |  *            * |
    |                 |
    |                 |
    |                 |
    |  *            * |
    |    *        *   |
     ------------------
    we say a pixel is in the contour if half or more of these 8 points fall within the contour line
    '''
    segm_mask = np.zeros(shape=input_img.shape,dtype=input_img.dtype)

    contour_endo = pth.Path(endocardium, closed=True)
    contour_epi = pth.Path(epicardium, closed=True) if epicardium is not None else None
    for x in range(segm_mask.shape[1]):
        for y in range(segm_mask.shape[0]):
            for (dx, dy) in [(-0.25, -0.375), (-0.375, -0.25), (-0.25, 0.375), (-0.375, 0.25), (0.25, 0.375),
                             (0.375, 0.25), (0.25, -0.375), (0.375, -0.25)]:

                point = (x + dx, y + dy)
                if contour_epi is None and contour_endo.contains_point(point):
                    segm_mask[y, x] += 1
                elif contour_epi is not None and \
                        contour_epi.contains_point(point) and not contour_endo.contains_point(point):
                    segm_mask[y, x] += 1

    segm_mask = (segm_mask >= 4) * 1.
    return segm_mask
