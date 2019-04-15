#!/bin/bash
# Script for launching condor jobs invoking both condor_offline.py and condor_online.py scripts.
# Dependencies: condor_offline.py condor_online.py

# OVERVIEW OF PARAMETERS

# 0. dag_create_data / dag_train_and_evaluate
# --number_of_recorders
# --number_of_models
# --destination
# --dont_retry
# --copy_dataset

# 1. directory and logging
# --summary_dir
# --data_root
# --code_root
# --home
# --log_tag

# 2. tensorflow code
# --python_project q-learning/pilot
# --python_script main.py
# --python_environment sing

# 3. condor machine specifications
# --gpumem
# --rammem
# --diskmem
# --evaluate_after
# --wall_time
# --not_nice

# 4. others for offline training (see main.py) for online (see run_script.py)

#------------------------------------------------------------
#
# Direct condor_online job
#
#------------------------------------------------------------

# name="condor_one --wall_time $((60*30))"
# condor_args="--not_nice"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0.5 --buffer_size 1000 --tensorboard --discrete --max_episodes 4000 --loss CrossEntropy --il_weight 0.91"
# python condor_online.py -t $name $condor_args $script_args $pytorch_args


#------------------------------------------------------------
#
# DAG condor_online job
#
#------------------------------------------------------------

# Online learning notes:
# buffer_size of 100.000 requires 20G RAM
# max_episodes 100.000 with 1 gradient step ~ 100.000 gradient steps ~ 75h 
# max_episodes  10.000 with 1 gradient step ~ 7.5h
# max episodes 1300 and prefill ~ 1 round

# Collect data:
# name="collect_esatv3_2"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 -ds --number_of_runs 10 --no_training --evaluate_every -1 --final_evaluation_runs 0"
# pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --discrete --turn_speed 0.8 --speed 0.8"
# dag_args="--number_of_recorders 9 --destination esatv3_expert --val_len 0 --test_len 0"
# condor_args="--wall_time_rec $((2*60*60)) --rammem 7"
# python dag_create_data.py -t $name $script_args $pytorch_args $dag_args $condor_args


# name="offline_offpolicy_recovery"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 -r"
# pytorch_args="--pause_simulator --learning_rate 0.01 --export_buffer --buffer_size 30000 --batch_size 300 --gradient_steps 1 --online --alpha 1 --tensorboard --discrete --max_episodes 100 --prefill --loss CrossEntropy --il_weight 1 --turn_speed 0.8 --speed 0.8"
# dag_args="--number_of_models 1"
# condor_args="--wall_time $((20*60*60)) --rammem 20"
# python dag_train_online.py -t $name $dag_args $condor_args $script_args $pytorch_args


#------------------------------------------------------------
#
# DAG condor_offline job
#
#------------------------------------------------------------
# name="tinyv2condor"
# pytorch_args="--dataset esatv3_expert_1K --discrete --turn_speed 0.8 --speed 0.8 --discrete --load_data_in_ram --owr\
#  --continue_training --checkpoint_path tiny_net_scratch --tensorboard --max_episodes 10 --batch_size 20\
#  --learning_rate 0.01"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_rec $((60*60)) --rammem 15"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args


# for d in 'esatv3_expert_20K' 'esatv3_expert_50K' 'esatv3_expert_100K' 'esatv3_expert_200K'  ; do
# for d in 'esatv3_expert_200K'  ; do
#   name="test_tinyv2/$d"
#   pytorch_args="--dataset $d --turn_speed 0.8 --speed 0.8 --discrete --loss CrossEntropy \
#    --continue_training --checkpoint_path tiny_net_scratch --tensorboard --max_episodes 10 --batch_size 10\
#    --learning_rate 0.01"
#   dag_args="--number_of_models 1 --not_nice"
#   condor_args="--wall_time_train $((20*60)) --rammem 7 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done



#---------------------------------------------- CONTINUOUS VS DISCRETE
# for LR in 1 01 001 0001 00001; do
#   name="continous_discrete_stochastic/tiny_net/continuous/$LR"
#   pytorch_args="--network tiny_net --checkpoint_path tiny_net_cont_scratch --continue_training --dataset esatv3_expert_stochastic_200K --turn_speed 0.8 --speed 0.8\
#  --tensorboard --max_episodes 20000 --batch_size 32 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --clip 1.0"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args

#   name="continous_discrete_stochastic/tiny_net/discrete/$LR"
#   pytorch_args="--network tiny_net --checkpoint_path tiny_net_scratch --continue_training --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8\
#  --tensorboard --max_episodes 20000 --batch_size 32 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --clip 1.0"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done



#---------------------------------------------- LSTM TEST 



# 3d CNN
# for LR in 1 01 001 ; do
#   name="tiny_3d_net_1/$LR"
#   pytorch_args="--network tiny_3d_net --n_frames 1 --continue_training --checkpoint_path tiny_3d_net_1_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --loss CrossEntropy\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
# for LR in 1 01 001 ; do
#   name="tiny_3d_net_3/$LR"
#   pytorch_args="--network tiny_3d_net --n_frames 3 --continue_training --checkpoint_path tiny_3d_net_3_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --loss CrossEntropy\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# # # nfc
# for LR in 1 01 001 ; do
#   name="tiny_nfc_net_1/$LR"
#   pytorch_args="--network tiny_nfc_net --n_frames 1 --continue_training --checkpoint_path tiny_nfc_net_1_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --loss CrossEntropy\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
# for LR in 1 01 001 ; do
#   name="tiny_nfc_net_3/$LR"
#   pytorch_args="--network tiny_nfc_net --n_frames 3 --continue_training --checkpoint_path tiny_nfc_net_3_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --loss CrossEntropy\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# wwbptt
# for LR in 1 ; do
#   name="tiny_LSTM_1ss/wwbptt/2_$LR"
#   pytorch_args="--network tiny_LSTM_net --checkpoint_path tiny_LSTM_net_scratch --dataset esatv3_expert_10K --discrete --turn_speed 0.8 --speed 0.8\
#  --tensorboard --max_episodes 30000 --batch_size 4 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --subsample 1 --load_data_in_ram"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((300*5*60+2*3600)) --rammem 7 --gpumem 900"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# # # sbptt
# for LR in 1 01 ; do
#   name="tiny_LSTM_1ss/sbptt/$LR"
#   pytorch_args="--network tiny_LSTM_net --checkpoint_path tiny_LSTM_net_scratch --dataset esatv3_expert_10K --discrete --turn_speed 0.8 --speed 0.8\
#  --tensorboard --max_episodes 30000 --batch_size 4 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --sliding_tbptt --subsample 1 --sliding_step_size 5 --load_data_in_ram"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((300*1*60+2*3600)) --rammem 7 --gpumem 900"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# # # fbptt
# for LR in 1 01 ; do
#   name="tiny_LSTM_1ss/fbptt/$LR"
#   pytorch_args="--network tiny_LSTM_net --checkpoint_path tiny_LSTM_net_scratch --dataset esatv3_expert_10K --discrete --turn_speed 0.8 --speed 0.8\
#  --tensorboard --max_episodes 3000 --batch_size 4 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --time_length -1 --subsample 1 --load_data_in_ram"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((3000*10+3600)) --rammem 11 --gpumem 6000"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# --network tiny_LSTM_net --checkpoint_path tiny_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --max_episodes 100 --batch_size 4 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --subsample 1
# --network tiny_LSTM_net --checkpoint_path tiny_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --tensorboard --max_episodes 30000 --batch_size 4 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --sliding_tbptt --subsample 1 --sliding_step_size 5

# wwbptt
# for LR in 1 ; do
#   name="tiny_LSTM/wwbptt"
#   pytorch_args="--network tiny_LSTM_net --checkpoint_path tiny_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8\
#  --tensorboard --max_episodes 30000 --batch_size 4 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --subsample 1"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((300*10*60+2*3600)) --rammem 7 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# # sbptt
# for LR in 1 ; do
#   name="tiny_LSTM/sbptt"
#   pytorch_args="--network tiny_LSTM_net --checkpoint_path tiny_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8\
#  --tensorboard --max_episodes 30000 --batch_size 4 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --sliding_tbptt --subsample 1 --sliding_step_size 5"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((300*5*60+2*3600)) --rammem 7 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# # fbptt
# for LR in 1 ; do
#   name="tiny_LSTM/fbptt"
#   pytorch_args="--network tiny_LSTM_net --checkpoint_path tiny_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8\
#  --tensorboard --max_episodes 1000 --batch_size 4 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --time_length -1 --subsample 1"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((1000*1*60+3600)) --rammem 12 --gpumem 6000 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done


#---------------------------------------------- SMALLER DATASET
# for DS in 100K 50K 20K 10K 5K 1K; do 
# for DS in 200K ; do 
# 	for LR in 1 ; do
# 		name="tiny_net/esatv3_expert_$DS/$LR"
# 		pytorch_args="--network tiny_net --checkpoint_path tiny_net_scratch --dataset esatv3_expert_$DS --discrete --turn_speed 0.8 --speed 0.8\
# 	--tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
# 		dag_args="--number_of_models 1"
# 		condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
# 		python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# 	done
# done

# for DS in 100K 50K 20K 10K 5K 1K; do 
#   for LR in 1 01 001 ; do
# for DS in 5K; do 
#   for LR in 1 ; do
#     for WD in 0 00001 0001 001 01 1 ; do
#       name="overgeneralizing/tiny_net_$WD/$LR"
#       pytorch_args="--network tiny_net --checkpoint_path tiny_net_scratch --continue_training --dataset esatv3_expert_$DS --discrete --turn_speed 0.8 --speed 0.8\
#     --tensorboard --max_episodes 20000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --clip 0.5 --weight_decay 0.$WD"
#       dag_args="--number_of_models 1"
#       condor_args="--wall_time_train $((200*1*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#       python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#     done
#   done
# done

#---------------------------------------------- GET BACK TO REALITY
# for LR in 1 01 001 ; do
#   name="squeeze_net_normalized_output/esatv3_expert_200K/$LR"
#   pytorch_args="--network squeeze_net --dataset esatv3_expert_200K --normalized_output --discrete --turn_speed 0.8 --speed 0.8\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((300*3*60+2*3600)) --rammem 6 --gpumem 2500 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
#for OP in Adam Adadelta ; do
#for LR in 1 01 001 0001; do
#  name="squeeze_net/esatv3_expert_200K/$OP/$LR"
#  pytorch_args="--network squeeze_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 0.5\
#  --tensorboard --max_episodes 30000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer $OP"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((300*3*60+2*3600)) --rammem 6 --gpumem 2500 --copy_dataset"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#done
#done

#for LR in 1 ; do
#  name="squeeze_net_pretrained/esatv3_expert_200K/$LR"
#  pytorch_args="--network squeeze_net --pretrained --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 0.5\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((100*3*60+2*3600)) --rammem 6 --gpumem 2500 --copy_dataset"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#done

# for LR in 01 001 0001 ; do
#   name="tiny_net/esatv3_expert_200K/$LR"
#   pytorch_args="--network tiny_net --checkpoint_path tiny_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done


#---------------------------------------------- GOING DEEPER INCEPTION-DENSE-RES NET
# for LR in 1 01 001; do
#   name="inception_net_finetune/$LR"
#   pytorch_args="--network inception_net --pretrained --feature_extract --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 0.5\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*5*60+2*3600)) --rammem 6 --gpumem 5000 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
# for LR in 1 01 001; do
#   name="res18_net_finetune/$LR"
#   pytorch_args="--network res18_net --pretrained --feature_extract --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 0.5\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*4*60+2*3600)) --rammem 6 --gpumem 1900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
# for LR in 1 01 001; do
#   name="dense_net_finetune/$LR"
#   pytorch_args="--network dense_net --pretrained --feature_extract --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 0.5\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*3*60+2*3600)) --rammem 6 --gpumem 6000 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# for LR in 1 01 001; do
#  name="alex_net_finetune/$LR"
#  pytorch_args="--network alex_net --pretrained --feature_extract --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 0.5\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((100*3*60+2*3600)) --rammem 6 --gpumem 1900 --copy_dataset"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
# for LR in 1 01 001; do
#  name="vgg16_net_finetune/$LR"
#  pytorch_args="--network vgg16_net --pretrained --feature_extract --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 0.5\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((100*5*60+2*3600)) --rammem 6 --gpumem 6000 --copy_dataset"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# for LR in 1 01 001; do
#  name="squeeze_net_finetune/$LR"
#  pytorch_args="--network squeeze_net --pretrained --feature_extract --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 0.5\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((100*5*60+2*3600)) --rammem 6 --gpumem 6000 --copy_dataset"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done



#---------------------------------------------- VGG16 OPTIMIZERS PRETRAINED


# for LR in 1 00001 ; do
#  for OP in SGD Adadelta Adam ; do 
# for LR in 1 ; do
#  for OP in SGD Adam ; do 
#   name="vgg16_net_pretrained/esatv3_expert_200K/$OP/$LR"
#   pytorch_args="--network vgg16_net --pretrained --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 1.\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer $OP"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*5*60+2*3600)) --rammem 7 --gpumem 6000 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#  done
# done
# for LR in 1 001 00001 ; do
#   OP="Adadelta"
#   name="vgg16_net/esatv3_expert_200K/$OP/$LR"
#   pytorch_args="--network vgg16_net --checkpoint_path vgg16_net_scratch --continue_training --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --clip 1.\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer $OP"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*5*60+2*3600)) --rammem 7 --gpumem 6000 --copy_dataset --not_nice"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
#---------------------------------------------- DATA NORMALIZATION
# Wall time = number_of_episodes*(batch_size/10.)*60+7200

# for LR in 1 01 001 0001 ; do

#    name="alex_net/esatv3_expert_200K/ref/$LR"
#    pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8\
#     --continue_training --checkpoint_path alex_net_scratch --tensorboard --max_episodes 20000 --batch_size 100\
#     --loss CrossEntropy --learning_rate 0.$LR"
#    dag_args="--number_of_models 3"
#    condor_args="--wall_time_train $((5*200*60+3600*2)) -gpumem 1900 --rammem 7 --copy_dataset"
#    python dag_train.py -t $name $pytorch_args $dag_args $condor_args

#   name="alex_net/esatv3_expert_200K/shifted_input/$LR"
#   pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --discrete\
#    --continue_training --checkpoint_path alex_net_scratch --tensorboard --max_episodes 20000 --batch_size 100 --loss CrossEntropy\
#    --learning_rate 0.$LR --shifted_input"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((5*200*60+3600*2)) -gpumem 1900 --rammem 7 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args

#   name="alex_net/esatv3_expert_200K/scaled_input/$LR"
#   pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --discrete\
#    --continue_training --checkpoint_path alex_net_scratch --tensorboard --max_episodes 20000 --batch_size 100 --loss CrossEntropy\
#    --learning_rate 0.$LR --scaled_input"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((5*200*60+3600*2)) -gpumem 1900 --rammem 7 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args

#   name="alex_net/esatv3_expert_200K/normalized_output/$LR"
#   pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --discrete\
#    --continue_training --checkpoint_path alex_net_scratch --tensorboard --max_episodes 20000 --batch_size 100 --loss CrossEntropy\
#    --learning_rate 0.$LR --normalized_output"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((5*200*60+3600*2)) -gpumem 1900 --rammem 7 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

#--------------------------- COLLECT DATASET

# Collect data:
# name="collect_esatv3_stochastic"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 10 --no_training --evaluate_every -1 --final_evaluation_runs 0"
# pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8 --stochastic"
# dag_args="--number_of_recorders 12 --destination esatv3_expert_stochastic --val_len 1 --test_len 1 --min_rgb 2400 --max_rgb 2600"
# condor_args="--wall_time_rec $((10*10*60+3600)) --rammem 6"
# python dag_create_data.py -t $name $script_args $pytorch_args $dag_args $condor_args


watch condor_q
