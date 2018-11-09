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
# PARAMS=LLL_train_params_hard_replay.yaml
# LLL params:
# PARAMS=LLL_train_params_debug.yaml
# default with hard replay
PARAMS=train_params_hard_replay.yaml


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
  python run_script.py -t online_yellow_barrel/test -pe sing -pp pilot_online/pilot -w osb_yellow_barrel -p $PARAMS -n 3 --robot turtle_sim --fsm nn_turtle_fsm -g --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
  # python run_script.py -t online_yellow_barrel/test_driveback -pe sing -pp pilot_online/pilot -w osb_yellow_barrel -p LLL_train_params_debug.yaml -n 1 --robot turtle_sim --fsm console_nn_db_turtle_fsm -g --x_pos 0.45 --x_var 0 --yaw_var 0 --yaw_or 1.57 --owr
else
  # long in background
  python run_script.py -t online_yellow_barrel/noLL_hard/0 -pe sing -pp pilot_online/pilot -w osb_yellow_barrel -p $PARAMS -n 1000 --robot turtle_sim --fsm nn_turtle_fsm --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
fi
