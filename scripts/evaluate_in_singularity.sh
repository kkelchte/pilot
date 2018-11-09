#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
# source .entrypoint_graph
source .entrypoint_xpra_no_build
roscd simulation_supervised/python


# Redo online experiments on different models


for d in canyon_forest sandbox_forest sandbox_canyon canyon_sandbox  ; do
  for sd in noLL LL_1 LL_10 ; do 
    echo "-------$d/$sd"
    w1="$(echo $d | cut -d _ -f 1)"
    w2="$(echo $d | cut -d _ -f 2)"
    python run_script.py -t LLL_doshico_final/$d/$sd/0_eva2 -pe sing -pp pilot/pilot -m LLL_doshico_final/$d/$sd/0 -w $w1 -w $w2 -p eva_params.yaml -n 6 --robot drone_sim --fsm oracle_nn_drone_fsm -e
    sleep 60
  done
done




############## old
# world=osb_yellow_barrel
# model=lifelonglearning/domain_A
# python run_script.py -t testing -pe sing -pp pilot/pilot -m $model -w $world -p eva_params_slow.yaml -n 1 --robot turtle_sim --fsm nn_turtle_fsm -e -g --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 


# python run_script.py -t testing -pe sing -pp pilot/pilot -m LLL/domain_A -w osb_yellow_barrel -p eva_params_slow.yaml -n 1 --robot turtle_sim --fsm nn_turtle_fsm -e -g --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 

# python run_script.py -t testing -pe sing -pp pilot/pilot -m LLL_doshico/forest -w forest -p eva_params_no_break_and_turn.yaml -n 10 --robot drone_sim --fsm oracle_nn_drone_fsm -e
# python run_script.py -t testing -pe sing -pp pilot/pilot -m LLL_doshico/forest -w forest -p eva_params_no_break_and_turn.yaml -n 10 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g
# python run_script.py -t testing -pe sing -pp pilot/pilot -m LLL_doshico/forest_canyon_LL_2 -w canyon -w forest -p eva_params.yaml -n 6 --robot drone_sim --fsm oracle_nn_drone_fsm -e
# python run_script.py -t testing -pe sing -pp pilot/pilot -m LLL_doshico/forest_canyon_noLL -w forest -p eva_params.yaml -n 1 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g

# python run_script.py -t test -pe sing -pp pilot/pilot -m LLL_doshico_final/forest_sandbox_2/LL_10/0 -w forest -w sandbox -p eva_params.yaml -n 6 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g
# python run_script.py -t test -pe sing -pp pilot/pilot -m LLL_doshico_final/forest/0/2018-11-02_0542 -w sandbox -p eva_params.yaml -n 5 --robot drone_sim --fsm oracle_nn_drone_fsm -e


# python run_script.py -t LLL_doshico_final/forest_sandbox_2/LL_10/0_eva -pe sing -pp pilot/pilot -m LLL_doshico_final/forest_sandbox_2/LL_10/0 -w forest -w sandbox -p eva_params.yaml -n 6 --robot drone_sim --fsm oracle_nn_drone_fsm -e
# python run_script.py -t LLL_doshico_final/forest_sandbox_2/LL_10/1_eva -pe sing -pp pilot/pilot -m LLL_doshico_final/forest_sandbox_2/LL_10/1 -w forest -w sandbox -p eva_params.yaml -n 6 --robot drone_sim --fsm oracle_nn_drone_fsm -e


### create data
# python run_script.py -t testing -pe sing -pp pilot/pilot --number_of_runs 15 -w canyon --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e


# world=different_corridor
# world=esatv1
# world=radiator_left
# world=osb_yellow_barrel
# world=osb_yellow_barrel_blue


# for model in all_factors/tiny_pilot ; do
# for model in naive_ensemble/mobile_scratch/0 ; do
# for model in lifelonglearning/domain_C_actionnorm ; do
#     echo "$(date +%H:%M:%S) Evaluating model $model in $world"
#     python run_script.py -t testing -pe sing -pp pilot/pilot -m $model -w $world -p eva_params_slow.yaml -n 1 --robot turtle_sim --fsm nn_turtle_fsm -e -g --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
#     # python run_script.py -t ${model}_eva -pe sing -pp pilot/pilot -m $model -w esatv1 --reuse_default_world -p eva_params.yaml -n 1 --robot drone_sim --fsm oracle_nn_drone_fsm -e -g
#     # python run_script.py -t ${model}_eva -pe sing -pp pilot/pilot -m $model -w esatv1 --reuse_default_world -p eva_params.yaml -n 1 --robot turtle_sim --fsm nn_turtle_fsm -e -g
#     # python run_script.py -t ${model}_eva -pe sing -pp pilot/pilot -m $model -w corridor --corridor_bends 0 --corridor_length 1 --extension_config $world --corridor_type empty -p eva_params.yaml -n 2 --robot drone_sim --fsm oracle_nn_drone_fsm -e
#   # done
# done


