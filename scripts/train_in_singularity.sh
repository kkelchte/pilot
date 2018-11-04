#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online

graphics=false

cd /esat/opal/kkelchte/docker_home

if [ $graphics = true ] ; then
  source .entrypoint_graph
else
  source .entrypoint_xpra_no_build
fi

roscd simulation_supervised/python



############## Test interactively

if [ $graphics = true ] ; then
  python run_script.py -t test_train_online -pe sing -pp pilot_online/pilot -w osb_yellow_barrel -p train_params_old.yaml -n 10 --robot turtle_sim --fsm nn_turtle_fsm -g --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
else
  python run_script.py -t online_lifelonglearning/osb_yellow_barrel/default -pe sing -pp pilot_online/pilot -w osb_yellow_barrel -p train_params_old.yaml -n 300 --robot turtle_sim --fsm nn_turtle_fsm --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
fi
