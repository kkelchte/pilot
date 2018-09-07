#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
# source .entrypoint_xpra_no_build
roscd simulation_supervised/python


# world=different_corridor
# world=corridor
# world=esatv1
# world=canyon
world=corridor

for model in first_model ; do
# for model in combined_corridor/alex_v4 combined_corridor/mobile ; do
# for model in combined_corridor/alex_v4 combined_corridor/mobile combined_corridor/squeeze_v1 all_factors/mobile ; do
  echo "$(date +%H:%M:%S) Evaluating model $model"
  python run_script.py -t ${model}_eva -pe sing -pp ensemble_v0/pilot -m $model -w $world --corridor_bends 0 --corridor_length 1 --extension_config passway_arc --corridor_type empty -p eva_params.yaml -n 1 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g
done
