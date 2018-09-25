#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
# source .entrypoint_xpra_no_build
roscd simulation_supervised/python


# world=different_corridor
world=corridor
# world=radiator_left

# for model in all_factors/tiny_pilot ; do
for model in naive_ensemble/mobile_scratch/0 ; do
    echo "$(date +%H:%M:%S) Evaluating model $model in $world"
    python run_script.py -t testing -pe sing -pp ensemble_v2/pilot -m $model -w corridor --reuse_default_world -p eva_params.yaml -n 1 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g
    # python run_script.py -t ${model}_eva -pe sing -pp ensemble_v2/pilot -m $model -w esatv1 --reuse_default_world -p eva_params.yaml -n 1 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g
    # python run_script.py -t ${model}_eva -pe sing -pp ensemble_v2/pilot -m $model -w esatv1 --reuse_default_world -p eva_params.yaml -n 1 --robot turtle_sim --fsm nn_turtle_fsm -e -g
    # python run_script.py -t ${model}_eva -pe sing -pp ensemble_v2/pilot -m $model -w corridor --corridor_bends 0 --corridor_length 1 --extension_config $world --corridor_type empty -p eva_params.yaml -n 2 --robot drone_sim --fsm oracle_nn_drone_fsm -e
  # done
done
