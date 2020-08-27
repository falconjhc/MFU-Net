config_file=unet_multi_modal_cardiac_288
config_pretrain_file=psdnet_multi_modal_cardiac_pretrain_288
config=unet_multimodalcardiac
testmode=feature-concat-attention
testmode_pretrain=-onehotround
testmode_pretrain=$testmode$testmode_pretrain

connection=_
public_model_dir=experiment_
public_model_dir=$public_model_dir$config$connection$testmode
save_result_dir=challenge_
save_result_dir=$save_result_dir$config$connection$testmode
save_result_dir=./

cd ..
python pretrain.py --config $config_pretrain_file --testmode $testmode_pretrain --lr 0.0001  --epoch 7

python experiment.py --load_pretrain True --config $config_file  --testmode $testmode --split 1 --lr 0.0001  --epoch 7 --patience 25
split_info=_split1
python test.py --expdir $public_model_dir$split_info  --testdir $save_result_dir

# python experiment.py --config $config_file  --testmode $testmode --split 0 --lr 0.00008  --epoch 20 --patience 15
# split_info=_split0
# python test.py --expdir $public_model_dir$split_info  --testdir $save_result_dir

# python experiment.py --config $config_file  --testmode $testmode --split 2 --lr 0.00006  --epoch 15 --patience 10
# split_info=_split2
# python test.py --expdir $public_model_dir$split_info  --testdir $save_result_dir

# python experiment.py --config $config_file  --testmode $testmode --split 3 --lr 0.00005  --epoch 10 --patience 5
# split_info=_split3
# python test.py --expdir $public_model_dir$split_info  --testdir $save_result_dir

# python experiment.py --config $config_file  --testmode $testmode --split 4 --lr 0.00005  --epoch 10 --patience 5
# split_info=_split4
# python test.py --expdir $public_model_dir$split_info  --testdir $save_result_dir

# python experiment.py --config $config_file  --testmode $testmode --split 5 --lr 0.00003  --epoch 5 --patience 5
# split_info=_split5
# python test.py --expdir $public_model_dir$split_info  --testdir $save_result_dir


python test.py --expdir $public_model_dir  --testdir $save_result_dir