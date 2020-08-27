import h5py
MODEL_PATH = '../experiment_sdnet_isles_segopt6-All_losstype_agis_split0/G_trainer'


print("读取模型中...")
with h5py.File(MODEL_PATH, 'r') as f:

    basic_keys = list(f.keys())
    for k in basic_keys:
        current_sub = list(f[k])
        for var in current_sub:
            print(f[k][var].name + ': ' )
            print(list(f[k][var].values()))

            b=1
            c=1




a=1