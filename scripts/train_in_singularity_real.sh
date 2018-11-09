#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online

# ---------------------settings
graphics=false
# params:
## old params:
# PARAMS=train_params_old.yaml
## old params:
# PARAMS=LLL_train_params_old.yaml
## default params:
# PARAMS=train_params.yaml
## LLL params:
# PARAMS=LLL_train_params.yaml
## LLL params hard:
PARAMS=LLL_train_params_hard_replay.yaml
# LLL params:
# PARAMS=LLL_train_params_debug.yaml
# default with hard replay
# PARAMS=train_params_hard_replay.yaml


#----------------------code
cd /esat/opal/kkelchte/docker_home

source .entrypoint_graph
roscd simulation_supervised/python
export ROS_MASTER_URI=http://10.42.0.16:11311 && export ROS_HOSTNAME=10.42.0.16

python run_script.py -t online_yellow_barrel/real_pretrained -pe sing -pp pilot_online/pilot -w osb_yellow_barrel -p LLL_train_params_hard_replay.yaml -m -n 1 --robot turtle_real --fsm console_nn_db_turtle_fsm

# rostopic pub /go std_msgs/Empty
