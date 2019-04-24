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
# jobs
#
#------------------------------------------------------------



# LSTM

# reference ww subsampled init state calculations


# for S in 2 4 8 16 ; do 
#   for LR in 1 ; do
#     name="tinyv3_3D_LSTM_net/wbptt_$S/$LR"
#     pytorch_args="--network tiny_3d_LSTM_net --n_frames 2 --checkpoint_path tiny_3d_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#     --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram --init_state_subsample $S"
#     dag_args="--number_of_models 1"
#     condor_args="--wall_time_train $((30*3600)) --rammem 7 --gpumem 1900"
#     python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#   done
# done

# name="tinyv3_3D_LSTM_net/wbptt_init/1"
# pytorch_args="--network tiny_3d_LSTM_net --n_frames 2 --checkpoint_path tiny_3d_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss MSE --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram --only_init_state"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((30*3600)) --rammem 7 --gpumem 1900"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args



name="tinyv3_3D_LSTM_net/wbptt_init_1"
pytorch_args="--network tiny_3d_LSTM_net --n_frames 2 --checkpoint_path tiny_3d_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss MSE --shifted_input --optimizer SGD --time_length 1 --subsample 10 --load_data_in_ram --only_init_state"
dag_args="--number_of_models 1"
condor_args="--wall_time_train $((30*3600)) --rammem 7 --gpumem 1900"
python dag_train.py -t $name $pytorch_args $dag_args $condor_args




# for LR in 1 01 ; do
#   name="tinyv3_3D_LSTM_net/ref_3D/$LR"
#   pytorch_args="--network tinyv3_3d_net --n_frames 2 --checkpoint_path tinyv3_3d_net_2_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#    --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --subsample 10 --load_data_in_ram"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((30*3600)) --rammem 7 --gpumem 1900"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
#for LR in 1 01 ; do
#  name="tinyv3_3D_LSTM_net/fbptt/$LR"
#  pytorch_args="--network tiny_3d_LSTM_net --n_frames 2 --checkpoint_path tiny_3d_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#   --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --time_length -1 --subsample 10 --load_data_in_ram"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((30*3600)) --rammem 7 --gpumem 1900"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#done
#for LR in 1 01 ; do
#  name="tinyv3_3D_LSTM_net/sbptt/$LR"
#  pytorch_args="--network tiny_3d_LSTM_net --n_frames 2 --checkpoint_path tiny_3d_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#   --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram --sliding_tbptt"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((30*3600)) --rammem 7 --gpumem 1900"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#done
#for LR in 1 01 ; do
#  name="tinyv3_3D_LSTM_net/wbptt/$LR"
#  pytorch_args="--network tiny_3d_LSTM_net --n_frames 2 --checkpoint_path tiny_3d_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#   --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((30*3600)) --rammem 7 --gpumem 1900"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#done
# for LR in 1 01 ; do
#  name="tinyv3_3D_LSTM_net/sbptt_gradaccum/$LR"
#  pytorch_args="--network tiny_3d_LSTM_net  --n_frames 2 --checkpoint_path tiny_3d_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#   --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram --sliding_tbptt"
#  dag_args="--number_of_models 1"
#  condor_args="--wall_time_train $((30*3600)) --rammem 7 --gpumem 1900"
#  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
#
## REFERENCE
#name="tinyv3_LSTM_net_reference/fbptt"
#pytorch_args="--network tinyv3_LSTM_net --checkpoint_path tinyv3_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
# --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length -1 --subsample 10 --load_data_in_ram"
#dag_args="--number_of_models 1"
#condor_args="--wall_time_train $((24*3600)) --rammem 7 --gpumem 1800"
#python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# name="tinyv3_LSTM_net_reference/sbptt"
# pytorch_args="--network tinyv3_LSTM_net --checkpoint_path tinyv3_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram --sliding_tbptt"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((24*3600)) --rammem 7 --gpumem 1800"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#name="tinyv3_LSTM_net_reference/wbptt"
#pytorch_args="--network tinyv3_LSTM_net --checkpoint_path tinyv3_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
# --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram"
#dag_args="--number_of_models 1"
#condor_args="--wall_time_train $((24*3600)) --rammem 7 --gpumem 1800"
#python dag_train.py -t $name $pytorch_args $dag_args $condor_args


#--------------------------- REDO 


# Alexnet different seeds
# name="alex_net_seeds"
# pytorch_args="--weight_decay 0 --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 20000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD"
# dag_args="--number_of_models 3"
# condor_args="--wall_time_train $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args


# name="alex_net_255input"
# pytorch_args="--weight_decay 0 --skew_input --network alex_net --checkpoint_path alex_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 20000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args



# # Tinyv3 Discrete CE
# name="discrete_continuous/tinyv3_CE"
# pytorch_args="--weight_decay 0 --network tinyv3_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --checkpoint_path tinyv3_net_scratch --continue_training\
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --shifted_input"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((67200)) --rammem 7 --gpumem 800 --copy_dataset"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# # Tinyv3 Discrete MSE
# name="discrete_continuous/tinyv3_MSE"
# pytorch_args="--weight_decay 0 --network tinyv3_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --checkpoint_path tinyv3_net_scratch --continue_training\
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss MSE --optimizer SGD --shifted_input"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((67200)) --rammem 7 --gpumem 800 --copy_dataset"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# # Tinyv3 Continuous MSE
# name="discrete_continuous/tinyv3_continuous"
# pytorch_args="--weight_decay 0 --network tinyv3_net --dataset esatv3_expert_200K --turn_speed 0.8 --speed 0.8 --checkpoint_path tinyv3_net_continuous_scratch --continue_training\
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss MSE --optimizer SGD --shifted_input"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((67200)) --rammem 7 --gpumem 800 --copy_dataset"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args


# LSTM
# name="tinyv3_LSTM_net_nobias/fbptt"
# pytorch_args="--network tinyv3_LSTM_net --checkpoint_path tinyv3_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length -1 --subsample 10 --load_data_in_ram"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((24*3600)) --rammem 7 --gpumem 1800"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# name="tinyv3_LSTM_net_nobias/wwbptt"
# pytorch_args="--network tinyv3_LSTM_net --checkpoint_path tinyv3_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((24*3600)) --rammem 9 --gpumem 800"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# name="tinyv3_LSTM_net/sbptt"
# pytorch_args="--network tinyv3_LSTM_net --checkpoint_path tinyv3_LSTM_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram --sliding_tbptt"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((13200)) --rammem 7 --gpumem 800"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# name="tinyv3_LSTM_net_512/sbptt"
# pytorch_args="--network tiny_LSTM512_net --checkpoint_path tiny_LSTM512_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --time_length 20 --subsample 10 --load_data_in_ram --sliding_tbptt"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((13200)) --rammem 7 --gpumem 800"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# name="tinyv3_LSTM_net/reference"
# pytorch_args="--network tinyv3_net --checkpoint_path tinyv3_net_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --subsample 10 --load_data_in_ram"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((13200)) --rammem 7 --gpumem 800"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args


# name="tinyv3_LSTM_net/3d"
# pytorch_args="--network tinyv3_3d_net --n_frames 3 --checkpoint_path tinyv3_3d_net_3_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 30000 --batch_size 5 --learning_rate 0.1 --loss CrossEntropy --shifted_input --optimizer SGD --subsample 10 --load_data_in_ram"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((13200)) --rammem 7 --gpumem 800"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args


# 3d CNN
# for LR in 1 ; do
#   name="tinyv3_3d_net_1/$LR"
#   pytorch_args="--network tinyv3_3d_net --n_frames 1 --continue_training --checkpoint_path tinyv3_3d_net_1_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD --loss MSE"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done


# for NF in 1 2 3 5 8 16 ; do
#   LR=1
#   name="tinyv3_3d_net_$NF/$NF"
#   pytorch_args="--network tinyv3_3d_net --n_frames $NF --continue_training --checkpoint_path tinyv3_3d_net_${NF}_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD --loss MSE"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# nfc
# for LR in 1 ; do
#   name="tinyv3_nfc_net_1/$LR"
#   pytorch_args="--network tinyv3_nfc_net --n_frames 1 --continue_training --checkpoint_path tinyv3_nfc_net_1_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD --loss MSE"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
# for LR in 1 ; do
# LR=1
# name="tinyv3_nfc_net_3"
# pytorch_args="--network tinyv3_nfc_net --n_frames 3 --continue_training --checkpoint_path tinyv3_nfc_net_3_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD --loss MSE"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
# for LR in 1 ; do
#   name="tinyv3_nfc_net_5/$LR"
#   pytorch_args="--network tinyv3_nfc_net --n_frames 5 --continue_training --checkpoint_path tinyv3_nfc_net_5_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD --loss MSE"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done



# for LR in 1 01 001 0001 00001; do
#   name="continous_discrete/alex_net/continuous/$LR"
#   pytorch_args="--weight_decay 0 --network alex_net --checkpoint_path alex_net_cont_scratch --continue_training --dataset esatv3_expert_200K --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 20000 --batch_size 32 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --clip 1.0"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 1900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args

#   name="continous_discrete/alex_net/discrete/$LR"
#   pytorch_args="--weight_decay 0 --network alex_net --checkpoint_path alex_net_scratch --continue_training --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 20000 --batch_size 32 --learning_rate 0.$LR --loss MSE --shifted_input --optimizer SGD --clip 1.0"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 1900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done


# for DS in 200K 100K 50K 20K 10K 5K 1K; do 
#   for LR in 1 ; do
#     name="tiny_net/esatv3_expert_$DS"
#     pytorch_args="--network tiny_net --checkpoint_path tiny_net_scratch --dataset esatv3_expert_$DS --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --continue_training"
#     dag_args="--number_of_models 1"
#     condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#     python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#   done
# done


# for DS in 200K 100K 50K 20K 10K 5K 1K; do 
# # for DS in 100K ; do 
#   for LR in 1 ; do
#     name="tinyv3_net/esatv3_expert_$DS"
#     pytorch_args="--network tinyv3_net --checkpoint_path tinyv3_net_scratch --dataset esatv3_expert_$DS --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --continue_training"
#     dag_args="--number_of_models 1"
#     condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#     python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#   done
# done

# for DS in 200K 100K 50K 20K 10K 5K 1K; do 
# for DS in 200K ; do 
#   for LR in 1 ; do
#     name="tinyv4_net"
#     pytorch_args="--network tinyv4_net --checkpoint_path tinyv4_net_scratch --dataset esatv3_expert_$DS --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#   --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --loss CrossEntropy --shifted_input --optimizer SGD --continue_training"
#     dag_args="--number_of_models 1"
#     condor_args="--wall_time_train $((100*1*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#     python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#   done
# done

#--------------------------- REDO TINYNFC

# for LR in 1 01 001 ; do
#   name="tiny_nfc_net_1/$LR"
#   pytorch_args="--weight_decay 0 --network tiny_nfc_net --n_frames 1 --continue_training --checkpoint_path tiny_nfc_net_1_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --loss CrossEntropy\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done
# for LR in 1 01 001 ; do
#   name="tiny_nfc_net_3/$LR"
#   pytorch_args="--weight_decay 0 --network tiny_nfc_net --n_frames 3 --continue_training --checkpoint_path tiny_nfc_net_3_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --loss CrossEntropy\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

# for LR in 1 01 001 ; do
#   name="tiny_nfc_net/$LR"
#   pytorch_args="--weight_decay 0 --network tiny_nfc_net --n_frames 5 --continue_training --checkpoint_path tiny_nfc_net_3_scratch --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --loss CrossEntropy\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.$LR --shifted_input --optimizer SGD"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

#--------------------------- CONDOR ONLINE

# name="condor_one --wall_time $((60*30))"
# condor_args="--not_nice"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0.5 --buffer_size 1000 --tensorboard --discrete --max_episodes 4000 --loss CrossEntropy --il_weight 0.91"
# python condor_online.py -t $name $condor_args $script_args $pytorch_args

#--------------------------- DAG TRAIN OFFLINE


#   name="alex_net/esatv3_expert_200K/normalized_output/$LR"
#   pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --discrete\
#    --continue_training --checkpoint_path alex_net_scratch --tensorboard --max_episodes 20000 --batch_size 100 --loss CrossEntropy\
#    --learning_rate 0.$LR --normalized_output"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((5*200*60+3600*2)) -gpumem 1900 --rammem 7 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

#--------------------------- DAG COLLECT DATASET ONLINE

# Collect data:
# name="collect_esatv3_stochastic"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 10 --no_training --evaluate_every -1 --final_evaluation_runs 0"
# pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8 --stochastic"
# dag_args="--number_of_recorders 12 --destination esatv3_expert_stochastic --val_len 1 --test_len 1 --min_rgb 2400 --max_rgb 2600"
# condor_args="--wall_time_rec $((10*10*60+3600)) --rammem 6"
# python dag_create_data.py -t $name $script_args $pytorch_args $dag_args $condor_args


watch condor_q
