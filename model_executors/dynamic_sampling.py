import numpy as np
import random
from utils.image_utils import image_show
def dynamic_sample_implementation(batch_x, batch_y_raw):
    def extract_sub_range(vol_dim, subvol_dim_centre, subvol_dim_size,
                          limit_within_vol=True):
        assert (vol_dim >= subvol_dim_size)
        half_dim_size = subvol_dim_size // 2.0
        ind_min = subvol_dim_centre - half_dim_size

        if subvol_dim_size % 2 == 0:
            ind_max = subvol_dim_centre + half_dim_size - 1
        else:
            ind_max = subvol_dim_centre + half_dim_size

        if limit_within_vol:
            target_min_coord = 0
            target_max_coord = subvol_dim_size

            if ind_min < 0:
                vol_min_coord = 0
                vol_max_coord = subvol_dim_size

            elif ind_max + 1 > vol_dim:
                vol_max_coord = vol_dim
                vol_min_coord = vol_dim - subvol_dim_size

            else:
                vol_min_coord = ind_min
                vol_max_coord = ind_max + 1

        else:
            if ind_min < 0:
                # print('situlation1')
                target_min_coord = -ind_min
                target_max_coord = subvol_dim_size
                vol_min_coord = 0
                vol_max_coord = subvol_dim_size + ind_min

            elif ind_max + 1 > vol_dim:
                # print('situation2')
                target_min_coord = 0
                target_max_coord = vol_dim - ind_max - 1  # subvol_dim_size - (ind_max+1-vol_dim)
                vol_min_coord = -subvol_dim_size - vol_dim + ind_max + 1  # vol_dim - ind_max -1
                vol_max_coord = vol_dim

            else:
                # print('situation3')
                target_min_coord = 0
                target_max_coord = subvol_dim_size
                vol_min_coord = ind_min
                vol_max_coord = ind_max + 1

        return int(vol_min_coord), \
               int(vol_max_coord), \
               int(target_min_coord), \
               int(target_max_coord)

    def select_centres2D(label_vol, n_class=None):
        # background is 0
        if n_class is None:
            n_class = len(np.unique(label_vol))

        if n_class > 2:
            target_class = np.random.choice(
                np.arange(start=1, stop=n_class)
            )

        else:
            target_class = 1

        label_vol1 = np.sum((label_vol == target_class).astype(np.float), axis=-1)
        indices = np.argwhere(label_vol1 > 0)
        centers_ind = np.random.choice(len(indices))

        return indices[centers_ind]


    def extract_subvolumn(vol,
                          centre_coord,
                          subvolumn_size,
                          limit_within_vol=True):

        vol_minx, vol_maxx, target_minx, target_maxx = \
            extract_sub_range(vol.shape[0],
                              centre_coord[0],
                              subvolumn_size[0],
                              limit_within_vol=limit_within_vol)

        vol_miny, vol_maxy, target_miny, target_maxy = \
            extract_sub_range(vol.shape[1],
                              centre_coord[1],
                              subvolumn_size[1],
                              limit_within_vol=limit_within_vol)

        return vol[..., vol_minx:vol_maxx, vol_miny:vol_maxy, :]


    def obtain_centre_patch(cached_volume, cached_label, image_size, train_arb_p):

        if random.random() < train_arb_p:
            if np.sum(cached_label[...,-2:])==0:
                clabel = np.sum(np.round(cached_label[..., :2]), axis=-1, keepdims=True)
            else:
                clabel = np.sum(np.round(cached_label[..., -2:]), axis=-1, keepdims=True)
            sub_vol_centre = select_centres2D(
                clabel
            )

            sub_vol_data = extract_subvolumn(
                cached_volume, sub_vol_centre, image_size[:2]
            )

            sub_vol_label = extract_subvolumn(
                cached_label, sub_vol_centre, image_size[:2]
            )

        else:

            cached_size = np.array(cached_volume.shape)
            valid_range = cached_size[:2] - np.array(image_size)
            # print(valid_range)

            if valid_range[0] != 0:
                x_start = np.random.randint(valid_range[0])

            else:
                x_start = 0

            if valid_range[1] != 0:
                y_start = np.random.randint(valid_range[1])

            else:
                y_start = 0

            sub_vol_data = cached_volume[
                           x_start:x_start + image_size[0],
                           y_start:y_start + image_size[1],
                           ...
                           ]
            sub_vol_label = cached_label[
                            x_start:x_start + image_size[0],
                            y_start:y_start + image_size[1],
                            ...
                            ]
        return sub_vol_data, sub_vol_label



    ap = 0.9
    candi_sizes = np.arange(128, 289, 16)

    temp_size = np.random.choice(candi_sizes)
    temp_batch_size = (288 * 288 * batch_x.shape[0] )// (temp_size * temp_size)


    x_temp = []
    y_temp = []

    for i in range(temp_batch_size):
        n_image = np.random.choice(
            batch_x.shape[0]
        )

        x0, y0 = obtain_centre_patch(
            batch_x[n_image],
            batch_y_raw[n_image],
            [temp_size, temp_size],
            ap
        )

        x_temp += [x0]
        y_temp += [y0]

    batch_x = np.stack(x_temp)
    batch_y_raw = np.stack(y_temp)
    #print(batch_x.shape)
    return batch_x, batch_y_raw
