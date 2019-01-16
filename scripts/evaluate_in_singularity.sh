#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
# source .entrypoint_graph
# source .entrypoint_xpra
source .entrypoint_xpra_no_build
roscd simulation_supervised/python

#############
# COMMAND
name="test_standard_rl"
script_args="--z_pos 1 -w esatv3 --random_seed 512"
pytorch_args="--pause_simulator --online --alpha 0. --buffer_size 100000 --tensorboard --discrete --max_episodes 80000 --loss CrossEntropy --il_weight 0"
python run_script.py -t $name $dag_args $condor_args $script_args $pytorch_args
