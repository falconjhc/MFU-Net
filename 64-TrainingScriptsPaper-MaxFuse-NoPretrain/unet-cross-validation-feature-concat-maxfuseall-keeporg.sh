config_file=unet_multi_modal_cardiac
config_pretrain_file=psdnet_multi_modal_cardiac_pretrain
config=unet_multimodalcardiac
testmode=feature-concat-maxfuseall-keeporg
testmode_pretrain=-onehotround
testmode_pretrain=$testmode$testmode_pretrain

connection=_
public_model_dir=experiment_
public_model_dir=$public_model_dir$config$connection$testmode
save_result_dir=challenge_
save_result_dir=$save_result_dir$config$connection$testmode
save_result_dir=./

cd ..

python experiment.py    --filters 64 --config $config_file  --testmode $testmode --split 0 --lr 0.0001  --epoch 150 --patience 55 --public_or_split 1
split_info=_split0
python test.py --filters 64 --expdir $public_model_dir$split_info  --testdir $save_result_dir

python experiment.py    --filters 64 --config $config_file  --testmode $testmode --split 1 --lr 0.0001  --epoch 150 --patience 55 --public_or_split 1
split_info=_split1
python test.py --filters 64 --expdir $public_model_dir$split_info  --testdir $save_result_dir

python experiment.py   --filters 64 --config $config_file  --testmode $testmode --split 2 --lr 0.0001  --epoch 150 --patience 55 --public_or_split 1
split_info=_split2
python test.py --filters 64 --expdir $public_model_dir$split_info  --testdir $save_result_dir

python experiment.py    --filters 64 --config $config_file  --testmode $testmode --split 3 --lr 0.0001  --epoch 150 --patience 55 --public_or_split 1
split_info=_split3
python test.py --filters 64 --expdir $public_model_dir$split_info  --testdir $save_result_dir

python experiment.py    --filters 64 --config $config_file  --testmode $testmode --split 4 --lr 0.0001  --epoch 150 --patience 55 --public_or_split 1
split_info=_split4
python test.py --filters 64 --expdir $public_model_dir$split_info  --testdir $save_result_dir
