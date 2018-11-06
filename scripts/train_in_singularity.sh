#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online

# ---------------------settings
graphics=false
# params:
## old params:
# PARAMS=train_params_old.yaml
## default params:
# PARAMS=train_params.yaml
## LLL params:
PARAMS=LLL_train_params.yaml
# LLL params:
# PARAMS=LLL_train_params_debug.yaml
# default with hard replay
# PARAMS=train_params_hard_replay.yaml



#----------------------code
cd /esat/opal/kkelchte/docker_home

if [ $graphics = true ] ; then
  source .entrypoint_graph
else
  source .entrypoint_xpra_no_build
fi

roscd simulation_supervised/python

if [ $graphics = true ] ; then
  # short test
  python run_script.py -t online_yellow_barrel/test -pe sing -pp pilot_online/pilot -w osb_yellow_barrel -p $PARAMS -n 30 --robot turtle_sim --fsm nn_turtle_fsm -g --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
else
  # long in background
  python run_script.py -t online_yellow_barrel/pretest_LLL -pe sing -pp pilot_online/pilot -w osb_yellow_barrel -p $PARAMS -n 300 --robot turtle_sim --fsm nn_turtle_fsm --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
fi
