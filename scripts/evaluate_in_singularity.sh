#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
# source .entrypoint_xpra_no_build
roscd simulation_supervised/python


# world=different_corridor
# world=esatv1
# world=radiator_left
world=osb_yellow_barrel


# for model in all_factors/tiny_pilot ; do
# for model in naive_ensemble/mobile_scratch/0 ; do
for model in lifelonglearning/domain_A/lr001/1 ; do
    echo "$(date +%H:%M:%S) Evaluating model $model in $world"
    python run_script.py -t testing -pe sing -pp pilot/pilot -m $model -w $world -p eva_params_slow.yaml -n 1 --robot turtle_sim --fsm nn_turtle_fsm -e -g --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
    # python run_script.py -t ${model}_eva -pe sing -pp pilot/pilot -m $model -w esatv1 --reuse_default_world -p eva_params.yaml -n 1 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g
    # python run_script.py -t ${model}_eva -pe sing -pp pilot/pilot -m $model -w esatv1 --reuse_default_world -p eva_params.yaml -n 1 --robot turtle_sim --fsm nn_turtle_fsm -e -g
    # python run_script.py -t ${model}_eva -pe sing -pp pilot/pilot -m $model -w corridor --corridor_bends 0 --corridor_length 1 --extension_config $world --corridor_type empty -p eva_params.yaml -n 2 --robot drone_sim --fsm oracle_nn_drone_fsm -e
  # done
done


