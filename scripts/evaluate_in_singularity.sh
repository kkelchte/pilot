#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
roscd simulation_supervised/python
python run_script.py -pe sing -pp pilot/pilot -m test_model -w canyon -p eva_params.yaml -n 2 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g