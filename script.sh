python strat1_hr_train.py --exp_name strat1_hr_unfrozen_res152 --gpu_id 2,3 --backbone resnet152 --num_epochs 100 --lr 1e-4 --unfreeze --log_interval 10 --save_epoch_interval 10 --bsz 512&&
python strat1_hr_train.py --exp_name strat1_hr_unfrozen_dense201 --gpu_id 2,3 --backbone densenet201 --num_epochs 100 --lr 1e-4 --unfreeze --log_interval 10 --save_epoch_interval 10 --bsz 512&&
python strat1_hr_train.py --exp_name strat1_hr_res152 --gpu_id 2,3 --backbone resnet152 --num_epochs 100 --lr 1e-4 --log_interval 10 --save_epoch_interval 10 --bsz 512&&
python strat1_hr_train.py --exp_name strat1_hr_dense201 --gpu_id 2,3 --backbone densenet201 --num_epochs 100 --lr 1e-4 --log_interval 10 --save_epoch_interval 10 --bsz 512