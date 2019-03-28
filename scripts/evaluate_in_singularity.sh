#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
# source .entrypoint_graph_debug
# source .entrypoint_xpra
# source .entrypoint_xpra_no_build
roscd simulation_supervised/python

#############
# COMMAND

# test data creation
name="test_collect_esatv3"
script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 10 --no_training --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8 --stochastic"
# dag_args="--number_of_recorders 1 --destination esatv3_expert --val_len 1 --test_len 1 --min_rgb 1000 --max_rgb 3000"
# condor_args="--wall_time_rec $((10*60*60)) --rammem 7"


# name="test_train_model"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr --evaluate_every -1 --final_evaluation_runs 5"
# pytorch_args="--pause_simulator --online --alpha 1 --prefill --turn_speed 0.8 --speed 0.8 --learning_rate 0.1 \
# --buffer_size 2000 --batch_size 2000 --gradient_steps 2001 --tensorboard --discrete --max_episodes 2000 --loss CrossEntropy --il_weight 1"


# EVALUATE MODEL
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 50 --gradient_steps 1 --online --alpha 1 --tensorboard --discrete --max_episodes 10000 --loss CrossEntropy --il_weight 1"

# TRAIN MODEL
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 50 --gradient_steps 1 --online --alpha 1 --tensorboard --discrete --max_episodes 10000 --loss CrossEntropy --il_weight 1"

# name="test_standard_rl"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0. --buffer_size 100000 --tensorboard --discrete --max_episodes 80000 --loss CrossEntropy --il_weight 0"

python run_script.py -t $name $script_args $pytorch_args
