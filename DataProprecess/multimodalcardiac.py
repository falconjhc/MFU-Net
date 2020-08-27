import nibabel as nib
import os
import numpy as np
import random
import scipy.misc as misc

from utils.image_utils import image_show
# for train
data_path = '/Volumes/HarricUoE/ChallengeData/train25/'
mask_path = '/Volumes/HarricUoE/ChallengeData/train25_myops_gd/'
check_path = '/Volumes/HarricUoE/ChallengeData/Check288/'
save_path = '/Volumes/HarricUoE/ChallengeData/Data288/'

# for test
# data_path = '/Volumes/HarricUoE/ChallengeData/MultiModalData/test20/'
# mask_path = ''
# save_path = '/Volumes/HarricUoE/ChallengeData/MultiModalData/DataTest224/'
# check_path = '/Volumes/HarricUoE/ChallengeData/MultiModalData/CheckTest224/'


# crop_shape = 384
# reshape_size = 384
reshape_size = 288
background_threshold = 30

upf=128-32+15
downf=352+32+15
leftf=131-32-15
rightf=355+32-15


# upf=128
# downf=352
# leftf=131
# rightf=355


def linear_scale(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return np.int16(img)


def process_raw_data(data_list, mask_list):
    subject_id_list = []
    for data in data_list:
        subject_id = data.split('_')[2]
        subject_id_list.append(subject_id)
    subject_id_list = np.unique(subject_id_list)

    for id in subject_id_list:

        current_id_data_save_path = os.path.join(save_path,id)
        if not os.path.exists(current_id_data_save_path):
            os.makedirs(current_id_data_save_path)
        if not os.path.exists(check_path):
            os.makedirs(check_path)

        current_data_list = [ii for ii in data_list if id in ii]

        bssfp_img_file = [ii for ii in current_data_list if 'C0' in ii][0]
        lge_img_file = [ii for ii in current_data_list if 'DE' in ii][0]
        t2_img_file = [ii for ii in current_data_list if 'T2' in ii][0]
        bssfp_img = linear_scale(nib.load(os.path.join(data_path,bssfp_img_file)).get_data())
        lge_img = linear_scale(nib.load(os.path.join(data_path, lge_img_file)).get_data())
        t2_img = linear_scale(nib.load(os.path.join(data_path, t2_img_file)).get_data())



        assert bssfp_img.shape[2]==lge_img.shape[2]==t2_img.shape[2], 'Slice Number Inconsistent'

        bssfp_img = bssfp_img[upf:downf,leftf:rightf,:]
        lge_img = lge_img[upf:downf, leftf:rightf, :]
        t2_img = t2_img[upf:downf, leftf:rightf, :]



        if len(mask_list)==0:
            for slice in range(bssfp_img.shape[2]):
                bssfp = bssfp_img[:, :, slice]
                lge = lge_img[:, :, slice]
                t2 = t2_img[:, :, slice]

                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_bssfp.png' % (slice + 1)), bssfp)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_lge.png' % (slice + 1)), lge)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_t2.png' % (slice + 1)), t2)

                check = np.concatenate([lge,t2,bssfp], axis=1)
                misc.imsave(os.path.join(check_path, 'Subject%s_Slice%d.png' % (id, slice + 1)), check)

        else:
            current_mask_file = [ii for ii in mask_list if id in ii][0]
            mask = nib.load(os.path.join(mask_path, current_mask_file)).get_data()
            mask = mask[upf:downf, leftf:rightf, :]
            assert bssfp_img.shape[2]  == mask.shape[2], 'Slice Number Inconsistent'
            for slice in range(bssfp_img.shape[2]):
                bssfp = bssfp_img[:,:,slice]
                lge = lge_img[:,:,slice]
                t2 = t2_img[:,:,slice]
                m = mask[:,:,slice]
                m[np.where(m!=0)]=1


                m1 = np.zeros_like(m)
                m2 = np.zeros_like(m)
                m3 = np.zeros_like(m)
                m4 = np.zeros_like(m)
                m5 = np.zeros_like(m)
                m1[np.where(m==2221)]=1
                m2[np.where(m==1220)]=1
                m3[np.where(m == 600)] = 1
                m4[np.where(m == 500)] = 1
                m5[np.where(m == 200)] = 1


                inf = m1
                ede = m2
                right_ven = m3
                left_ven = m4
                normal_myo = m5
                myo = normal_myo + inf + ede
                myo[np.where(myo>0)]=1

                full_picture = np.concatenate([np.expand_dims(bssfp, axis=-1),
                                               np.expand_dims(lge, axis=-1),
                                               np.expand_dims(t2, axis=-1)], axis=-1)
                background = np.zeros_like(full_picture)
                background[np.where(full_picture > background_threshold)] = 1
                background = np.mean(background,axis=-1)
                background[np.where(background>0)]=1
                background = background - myo - right_ven - left_ven
                background[np.where(background<0)]=0
                background = np.tile(np.expand_dims(background, axis=-1),[1,1,3])
                background_check = np.copy(full_picture)
                background_check[np.where(background==1)]=255*0.5
                background_check = np.concatenate([full_picture,background_check], axis=1)


                misc.imsave(os.path.join(current_id_data_save_path,'slice%d_bssfp.png' % (slice+1)),bssfp)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_lge.png' % (slice+1)), lge)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_t2.png' % (slice+1)), t2)

                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_infarct.png' % (slice+1)), inf)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_edema.png' % (slice + 1)), ede)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_right_ven.png' % (slice + 1)), right_ven)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_left_ven.png' % (slice + 1)), left_ven)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_myo.png' % (slice + 1)), myo)

                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_background.png' % (slice+1)), background)
                misc.imsave(os.path.join(current_id_data_save_path, 'slice%d_mask_source.png' % (slice + 1)), m)



                bssfp_overlay_left_ven = np.copy(bssfp)
                bssfp_overlay_right_ven = np.copy(bssfp)
                bssfp_overlay_myo = np.copy(bssfp)
                lge_inf_overlay = np.copy(lge)
                lge_inf_overlay2 = np.copy(lge)
                lge_inf_overlay3 = np.copy(lge)
                t2_ede_overlay = np.copy(t2)
                t2_ede_overlay2 = np.copy(t2)
                t2_ede_overlay3 = np.copy(t2)

                bssfp_overlay_left_ven[np.where(left_ven==1)]=255*0.8
                bssfp_overlay_right_ven[np.where(right_ven == 1)] = 255*0.8
                bssfp_overlay_myo[np.where(myo == 1)] = 255*0.8
                bssfp_overlay_left_ven = np.expand_dims(bssfp_overlay_left_ven,axis=-1)
                bssfp_overlay_right_ven = np.expand_dims(bssfp_overlay_right_ven, axis=-1)
                bssfp_overlay_myo = np.expand_dims(bssfp_overlay_myo, axis=-1)
                bssfp_overlay = np.concatenate([bssfp_overlay_left_ven, bssfp_overlay_right_ven, bssfp_overlay_myo], axis=-1)

                lge_inf_overlay[np.where(inf==1)]=255*0.5
                lge_inf_overlay = np.expand_dims(lge_inf_overlay,axis=-1)
                lge_inf_overlay2 = np.expand_dims(lge_inf_overlay2, axis=-1)
                lge_inf_overlay3 = np.expand_dims(lge_inf_overlay3, axis=-1)
                lge_overlay = np.concatenate([lge_inf_overlay,lge_inf_overlay2,lge_inf_overlay3],axis=-1)

                t2_ede_overlay[np.where(ede == 1)] = 255*0.5
                t2_ede_overlay = np.expand_dims(t2_ede_overlay, axis=-1)
                t2_ede_overlay2 = np.expand_dims(t2_ede_overlay2, axis=-1)
                t2_ede_overlay3 = np.expand_dims(t2_ede_overlay3, axis=-1)
                t2_overlay = np.concatenate([t2_ede_overlay, t2_ede_overlay2, t2_ede_overlay3], axis=-1)

                bssfp = np.tile(np.expand_dims(bssfp, axis=-1), [1, 1, 3])
                lge = np.tile(np.expand_dims(lge, axis=-1), [1, 1, 3])
                t2 = np.tile(np.expand_dims(t2, axis=-1), [1, 1, 3])

                bssfp_check = np.concatenate([bssfp, bssfp_overlay],axis=1)
                lge_check = np.concatenate([lge, lge_overlay],axis=1)
                t2_check = np.concatenate([t2, t2_overlay], axis=1)

                check = np.concatenate([bssfp_check,lge_check,t2_check, background_check],axis=0)

                file_name = 'Subject%s_Slice%d' % (id, slice+1) + '.png'
                if not os.path.exists(check_path):
                    os.makedirs(check_path)
                check_save_path =os.path.join(check_path,file_name)
                misc.imsave(check_save_path,check)

data_list = [ii for ii in os.listdir(data_path) if not ii.startswith('.')]
if not mask_path == '':
    mask_list = [ii for ii in os.listdir(mask_path) if not ii.startswith('.')]
else:
    mask_list = []
data_list.sort()
mask_list.sort()
process_raw_data(data_list, mask_list)


print("Complete All !")