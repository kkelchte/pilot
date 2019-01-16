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

name="standard_il_2"
dag_args="--number_of_models 3"
condor_args="--not_nice --wall_time $((12*60*60))"
script_args="--z_pos 1 -w esatv3 --random_seed 512"
pytorch_args="--pause_simulator --online --alpha 0.5 --buffer_size 1000 --tensorboard --discrete --max_episodes 10000 --loss CrossEntropy --il_weight 1"
python dag_train_online.py -t $name $dag_args $condor_args $script_args $pytorch_args

# name="standard_rl_2"
# dag_args="--number_of_models 3"
# condor_args="--rammem 18 --wall_time $((12*60*60))"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0. --buffer_size 100000 --tensorboard --discrete --max_episodes 80000 --loss CrossEntropy --il_weight 0"
# python dag_train_online.py -t $name $dag_args $condor_args $script_args $pytorch_args

# name="on_policy_il_2"
# dag_args="--number_of_models 3"
# condor_args="--wall_time $((12*60*60))"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0. --buffer_size 1000 --tensorboard --discrete --max_episodes 20000 --loss CrossEntropy --il_weight 1"
# python dag_train_online.py -t $name $dag_args $condor_args $script_args $pytorch_args

# name="off_policy_il_2"
# dag_args="--number_of_models 3"
# condor_args="--wall_time $((12*60*60))"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 1. --buffer_size 1000 --tensorboard --discrete --max_episodes 20000 --loss CrossEntropy --il_weight 1"
# python dag_train_online.py -t $name $dag_args $condor_args $script_args $pytorch_args

# name="online_il_2"
# dag_args="--number_of_models 3"
# condor_args="--wall_time $((12*60*60))"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0.5 --buffer_size 100 --tensorboard --discrete --max_episodes 20000 --loss CrossEntropy --il_weight 1"
# python dag_train_online.py -t $name $dag_args $condor_args $script_args $pytorch_args

# name="offline_il_2"
# dag_args="--number_of_models 3"
# condor_args=" --rammem 18 --wall_time $((12*60*60))"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0.5 --buffer_size 100000 --tensorboard --discrete --max_episodes 20000 --loss CrossEntropy --il_weight 1"
# python dag_train_online.py -t $name $dag_args $condor_args $script_args $pytorch_args


watch condor_q