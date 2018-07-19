#!/bin/bash
# Script for launching condor jobs invoking both condor_offline.py and condor_online.py scripts.
# Dependencies: condor_offline.py condor_online.py

# OVERVIEW OF PARAMETERS

# 0. dag_create_data
# --number_of_recorders
# --destination

# 1. directory and logging
# --summary_dir
# --data_root
# --code_root
# --home
# --log_tag

# 2. tensorflow code
# --python_project q-learning/pilot
# --python_script main.py
# --python_environment sing

# 3. condor machine specifications
# --gpumem
# --rammem
# --diskmem
# --evaluate_after
# --wall_time
# --not_nice

# 4. others for offline training (see main.py) for online (see run_script.py)

# ------------OFFLINE----------
# python condor_offline.py -t test_offline --not_nice --wall_time $((10*60)) --evaluate_after --dataset canyon_ds50 --max_episodes 5 


# ------------ONLINE-------------
# create dataset
# for i in 10 11 12 ; do
# # for i in $(seq 0 9) ; do
# # for i in 0 ; do
# 	python condor_online.py -t rec_$i --wall_time $((5*60*60)) -w sandbox --robot drone_sim --fsm oracle_drone_fsm -n $((15)) --paramfile params.yaml -ds --save_only_success
# 	# python condor_online.py -t test_$i --wall_time $((2*24*60*60)) -w canyon -w forest -w sandbox --robot drone_sim --fsm oracle_drone_fsm -n $((2)) --paramfile params.yaml -ds -e --save_only_success
# done

# -----------DAG-----------------
# Create Dataset
# python dag_create_data.py -t rec_dd --destination doshico_drone --number_of_recorders 10 --number_of_runs $((3*15)) -w sandbox -w canyon -w forest --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3
python dag_create_data.py -t rec_dt --destination doshico_turtle --number_of_recorders 10 --number_of_runs $((3*15)) -w sandbox -w canyon -w forest --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success -e

# while [ true ] ; do clear; condor_q; sleep 2; done