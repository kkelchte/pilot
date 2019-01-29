#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
# source .entrypoint_xpra
# source .entrypoint_xpra_no_build
roscd simulation_supervised/python

#############
# COMMAND

# TESTING BA with WAYPOINTS
name="testing_recovery_cameras"
script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -g -r"
pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 10000 --batch_size 10000 --gradient_steps 100 --online --alpha 1 --tensorboard --discrete --max_episodes 2 --prefill --loss CrossEntropy --il_weight 1 --turn_speed 0.8 --speed 0.8"


# TRAIN WITHOUT STOPPING
# name="no_stopping_2"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -g"
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 5000 --batch_size 5000 --gradient_steps 100 --online --alpha 1 --tensorboard --discrete --max_episodes 2 --prefill --loss CrossEntropy --il_weight 1 --turn_speed 0.8 --speed 0.8"

# EVALUATE MODEL
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 50 --gradient_steps 1 --online --alpha 1 --tensorboard --discrete --max_episodes 10000 --loss CrossEntropy --il_weight 1"

# TRAIN MODEL
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 50 --gradient_steps 1 --online --alpha 1 --tensorboard --discrete --max_episodes 10000 --loss CrossEntropy --il_weight 1"

# name="test_standard_rl"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0. --buffer_size 100000 --tensorboard --discrete --max_episodes 80000 --loss CrossEntropy --il_weight 0"

python run_script.py -t $name $script_args $pytorch_args
