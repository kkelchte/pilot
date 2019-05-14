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

#--------------------------- CONDOR ONLINE
# for i in $(seq 1) ; do
#   name="test_condor_variance/$i"
#   model="log_neural_architectures/discrete_continuous/tinyv3_continuous/seed_0"
#   condor_args=" --wall_time $((60*60)) --gpumem 800 --rammem 7 --cpus 11 --not_nice"
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation --python_project pytorch_pilot/pilot"
#   pytorch_args=" --on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#   python condor_online.py -t $name $condor_args $script_args $pytorch_args
# done
#--------------------------- DAG EVALUATE ONLINE

###### DAGGER
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"
# condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13"

# for mod in 5K_concat 10K_concat dagger_model_1 dagger_model_2 dagger_model_3 ; do 
#   name="DAGGER/evaluate_$mod"
#   model="DAGGER/$mod"
#   pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
#   dag_args="--number_of_models 2"
#   condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13"
#   python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
# done 

###### RECOVERY

# python dag_evaluate.py -t recovery/evaluate_5K_concat --number_of_models 2 --wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13\
#   --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --on_policy --tensorboard --checkpoint_path recovery/5K_concat --load_config --continue_training
# python dag_evaluate.py -t recovery/evaluate_5K --number_of_models 2 --wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13\
#   --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --on_policy --tensorboard --checkpoint_path recovery/5K --load_config --continue_training

###### EPSILON 

# python dag_evaluate.py -t epsilon/evaluate_5K_concat --number_of_models 2 --wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13\
#   --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --on_policy --tensorboard --checkpoint_path epsilon/5K_concat --load_config --continue_training
# python dag_evaluate.py -t epsilon/evaluate_5K --number_of_models 2 --wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13\
#   --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --on_policy --tensorboard --checkpoint_path epsilon/5K --load_config --continue_training

###### Redo evaluation of concat
##TinyNet_Concat_2
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"
# condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13 --greenlist andromeda vladimir"
# for model in 200K_continuous  5K_discrete ; do
#   name="test_2concat/test_$model"
#   pytorch_args="--on_policy --tensorboard --checkpoint_path test_2concat/$model --load_config --continue_training"
#   python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
# done

###### REDO 5K_concat for different seeds out of mistrust
# Question: how much does it all depend on the final local minimum???
# name="validate_different_seeds_online"
# pytorch_args="--network tinyv3_3d_net --n_frames 2 --dataset esatv3_expert_5K --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --load_data_in_ram\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --shifted_input --optimizer SGD --loss MSE --weight_decay 0 --clip 1"
# dag_args="--number_of_models 3"
# condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 7 --gpumem 900"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation -pp pytorch_pilot/pilot"
# dag_args="--number_of_models 1 --use_greenlist"
# condor_args="--wall_time $((1*60*60)) --gpumem 700 --rammem 7 --cpus 13"
# for mod in 0 1 2; do 
#   name="redo_evaluate_different_seeds/$mod"
#   model="validate_different_seeds_online/seed_$mod"
#   pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#   python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
# done
#_________________________________________________________________________________
# Test dag train and evaluate
# for DS in '5K' '10K' '20K' '50K' '100K' '200K' ; do
#   name="datadependency_online_concat/$DS"
#   pytorch_args="--network tinyv3_3d_net --n_frames 2 --dataset esatv3_expert_${DS} --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#    --checkpoint_path tinyv3_3d_net_2_continuous_scratch --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --shifted_input\
#    --optimizer SGD --loss MSE --weight_decay 0 --clip 1."
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 4 --evaluation --python_project pytorch_pilot/pilot"
#   dag_args="--number_of_models 3"
#   condor_args="--wall_time_train $((6*60*60)) --wall_time_eva $((60*60)) --gpumem 700 --rammem 7 --cpus 13 --use_greenlist"
#   python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $dag_args $condor_args
# done

for DS in '5K' '10K' '20K' '50K' '100K' '200K' ; do
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation -pp pytorch_pilot/pilot"
  dag_args="--number_of_models 1"
  condor_args="--wall_time $((1*60*60)) --gpumem 1900 --rammem 7 --cpus 7"
  for mod in 0 1 2; do 
    name="datadependency_online_concat_evaluate/$DS/$mod"
    model="datadependency_online_concat/$DS/$mod"
    pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --use_greenlist"
    python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
  done
done
#_________________________________________________________________________________
# MAS on TinyNet 
# 10000 frames in one hour ==> 50000 in 5hours if 1.5fps 
# 3x slower with savefig

# Train without MAS and see how it 'forgets' along the different runs
# for LR in 01 001 0001 ; do
#   name="continual_learning/3/$LR"
#   pytorch_args="--online --dataset forest_trail_dataset --tensorboard --network tinyv3_net \
#    --buffer_size 100 --min_buffer_size 100 --learning_rate 0.$LR --gradient_steps 10 --clip 5.0 --load_data_in_ram\
#    --discrete --loss_window_mean_threshold 0.1 --loss_window_std_threshold 0.002 --weight_decay 0.0005"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((3*5*60*60+5*3600)) --rammem 7 --gpumem 3900 --copy_dataset"
#   python condor_offline.py -t $name $pytorch_args $dag_args $condor_args
# done


#_________________________________________________________________________________
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"
# condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13"

# ##TinyNet_Siamese
# name="online_NA_evaluation_redo/TinyNet_Siamese"
# model='log_neural_architectures/tinyv3_nfc_net_3/seed_0'
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##TinyNet_Concat_2
# name="online_NA_evaluation_redo/TinyNet_Concat_2"
# model='tinyv3_3d_net_2/2/seed_0'
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

##TinyNet_LSTM_FBPTT
#name="online_NA_evaluation/TinyNet_LSTM_FBPTT_2"
#model='log_neural_architectures/tinyv3_3D_LSTM_net/fbptt/1/seed_0'
#pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

##TinyNet_LSTM_SBPTT
#name="online_NA_evaluation/TinyNet_LSTM_SBPTT_2"
#model='log_neural_architectures/tinyv3_3D_LSTM_net/sbptt/1/seed_0'
#pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

##TinyNet_LSTM_WBPTT
#name="online_NA_evaluation/TinyNet_LSTM_WBPTT_2"
#model='log_neural_architectures/tinyv3_3D_LSTM_net/wbptt/1/seed_0'
#pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

## TinyNet_LSTM_WBPTT_init
# name="online_NA_evaluation/TinyNet_LSTM_WBPTT_init"
# model='log_neural_architectures/tinyv3_3D_LSTM_net/wbptt_init/1/seed_0'
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

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

# name="collect_esatv3_stochastic"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 10 --no_training --evaluate_every -1 --final_evaluation_runs 0"
# pytorch_args="--pause_simulator --on_policy --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8 --stochastic"
# dag_args="--number_of_recorders 12 --destination esatv3_expert_stochastic --val_len 1 --test_len 1 --min_rgb 2400 --max_rgb 2600"
# condor_args="--wall_time_rec $((10*10*60+3600)) --rammem 6"
# python dag_create_data.py -t $name $script_args $pytorch_args $dag_args $condor_args


watch condor_q
