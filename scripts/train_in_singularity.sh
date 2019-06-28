#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
# source .entrypoint_graph
# source .entrypoint_graph_debug
source .entrypoint_xpra
# source .entrypoint_xpra_no_build
# roscd simulation_supervised/python
roscd simulation_supervised/python

pwd
#############
# COMMAND

defaut_args="--network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input --batch_size -1 --alpha 0\
 --loss MSE --optimizer SGD --clip 1.0 --weight_decay 0 --normalized_output --on_policy --tensorboard --pause_simulator --learning_rate 0.01"
script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 100000 --final_evaluation_runs 10 --python_project pytorch_pilot_beta/pilot --evaluate_every 20"


# First model, no policy mixing, ou noise in yaw
# name="testing --owr"
# specific_args="--max_episodes 1000 --noise ou --sigma_yaw 0.1 --min_buffer_size 32 --buffer_size 1000 --gradient_steps 3 --batch_size 32"
# python run_script.py -t $name $script_args $defaut_args $specific_args

name="chapter_policy_learning/on_policy_training/large"
specific_args="--max_episodes 100000 --noise ou --sigma_yaw 0.1 --min_buffer_size 1000 --buffer_size 10000 --gradient_steps 1000 --batch_size 32 --train_every_N_steps 100"
python run_script.py -t $name $script_args $defaut_args $specific_args
