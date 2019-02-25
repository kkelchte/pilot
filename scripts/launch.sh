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

# PRIMAL EXPERIMENT: Testing
# name="testing_condor"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -r"
# pytorch_args="--pause_simulator --learning_rate 0.01 --export_buffer --buffer_size 1000 --batch_size 100 --gradient_steps 1 --online --alpha 1 --tensorboard --discrete --max_episodes 10000 --prefill --loss CrossEntropy --il_weight 1 --turn_speed 0.8 --speed 0.8"
# dag_args="--number_of_models 3"
# condor_args="--wall_time $((20*60)) --rammem 10 --not_nice"
# python dag_train_online.py -t $name $dag_args $condor_args $script_args $pytorch_args

# Collect data:
# name="collect_esatv3"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 10 --no_training --evaluate_every -1 --final_evaluation_runs 0"
# pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --discrete --turn_speed 0.8 --speed 0.8"
# dag_args="--number_of_recorders 1 --destination esatv3_expert --val_len 1 --test_len 1"
# condor_args="--wall_time_rec $((2*60*60)) --rammem 7"
# python dag_create_data.py -t $name $script_args $pytorch_args $dag_args $condor_args


# name="offline_offpolicy_recovery"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -r"
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
# pytorch_args="--dataset esatv3_expert_1K --discrete --turn_speed 0.8 --speed 0.8 --discrete --load_in_ram --owr\
#  --continue_training --checkpoint_path tiny_net_scratch --tensorboard --max_episodes 10 --batch_size 20\
#  --learning_rate 0.01"
# dag_args="--number_of_models 1"
# condor_args="--wall_time_rec $((60*60)) --rammem 15"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args


for d in 'esatv3_expert' 'esatv3_expert_10K' 'esatv3_expert_5K' 'esatv3_expert_1K' 'esatv3_expert_500' ; do
  name="tinyv2/$d"
  pytorch_args="--dataset $d --turn_speed 0.8 --speed 0.8 --discrete --load_in_ram --owr --loss CrossEntropy \
   --continue_training --checkpoint_path tiny_net_scratch --tensorboard --max_episodes 100 --batch_size 64\
   --learning_rate 0.01"
  dag_args="--number_of_models 2"
  condor_args="--wall_time_rec $((200*60)) --rammem 15"
  python dag_train.py -t $name $pytorch_args $dag_args $condor_args
done

watch condor_q
