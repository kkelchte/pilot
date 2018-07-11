#!/bin/bash
# Script for launching condor jobs invoking both condor_offline.py and condor_online.py scripts.
# Dependencies: condor_offline.py condor_online.py

# OVERVIEW OF PARAMETERS

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
for i in $(seq 10) ; do
	python condor_online.py -t rec_$i --wall_time $((3*200*60+60*15)) -w canyon -w forest -w sandbox --robot drone_sim --fsm oracle_drone_fsm -n $((3*200)) --paramfile params.yaml -ds
done



# while [ true ] ; do clear; condor_q; sleep 2; done