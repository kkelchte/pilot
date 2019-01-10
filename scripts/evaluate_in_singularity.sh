#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
# source .entrypoint_xpra_no_build
roscd simulation_supervised/python

#############
# COMMAND
script_args="-w esatv3 --random_seed 512 -g"
pytorch_args="--pause_simulator --online --alpha 0.5 --buffer_size 500 --discrete --max_episodes 20 --loss CrossEntropy --tensorboard"
python run_script.py -t test_esat --owr --z_pos 1 $script_args $pytorch_args


