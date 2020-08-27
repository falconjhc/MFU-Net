
from loaders import multimodalcardiac
loader = multimodalcardiac


from scipy import misc

params = {
    'normalise': 'batch',
    'seed': 1,
    'folder': 'experiment_unet_multimodalcardiac',
    'epochs': 1,
    'batch_size': 4,
    'split': 0,
    'dataset_name': 'multimodalcardiac',
    'test_dataset': 'multimodalcardiac',
    'prefix': 'norm',  # Prefix used to load a dataset, e.g. norm_baseline for data_by_dog
    'augment': True,
    'model': 'unet.UNet',
    'executor': 'base_executor.Executor',
    'num_anato_masks': loader.MultiModalCardiacLoader().num_anato_masks,
    'num_patho_masks': loader.MultiModalCardiacLoader().num_patho_masks,
    'out_anato_channels': loader.MultiModalCardiacLoader().num_anato_masks + 1,
    'out_patho_channels': loader.MultiModalCardiacLoader().num_patho_masks + 1,
    'residual': False,
    'filters': 64,
    'downsample': 4,
    'input_shape': loader.MultiModalCardiacLoader().input_shape,  # harric modified
    'image_downsample': 1,
    'lr': 0.0001,
    'l_mix': '1.-1.',
    'decay': 0.,
    'regularizer': 0,
    'ce_weight': 1,
    'anato_penalty':1,
    'patho1_penalty':3,
    'patho2_penalty':7,
    'oot_penalty': 0.5,
    'attention_map_penalty':0.1,
    'patience': 35,
    'min_delta': 0.005,
}


def get(testmode):
    params['testmode']=testmode
    return params
