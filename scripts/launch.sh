#!/bin/bash
# Script for launching condor jobs invoking both condor_offline.py and condor_online.py scripts.
# Dependencies: condor_offline.py condor_online.py

# OVERVIEW OF PARAMETERS

# 0. dag_create_data / dag_train_and_evaluate
# --number_of_recorders
# --number_of_models
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

# ------------CREATE DATA-------

# Create Dataset
# python dag_create_data.py -t sandbox_drone --wall_time_rec $((4*60*60)) --destination sandbox_drone --number_of_recorders 10 --number_of_runs $((50)) -w sandbox --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3
# python dag_create_data.py -t canyon_drone --wall_time_rec $((4*60*60)) --destination canyon_drone --number_of_recorders 10 --number_of_runs $((50)) -w canyon --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3
# python dag_create_data.py -t forest_drone --wall_time_rec $((4*60*60)) --destination forest_drone --number_of_recorders 10 --number_of_runs $((50)) -w forest --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3
# python dag_create_data.py -t esat_drone --wall_time_rec $((4*60*60)) --destination esat_drone --number_of_recorders 1 --number_of_runs $((10)) -w esat_v1  -w esat_v2 --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3
# python dag_create_data.py -t test_drone --wall_time_rec $((4*60*60)) --destination test_drone --number_of_recorders 1 --number_of_runs $((10)) -w forest --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3
# python dag_create_data.py -t test_turtle --wall_time_rec $((3*15*10*60)) --destination test_turtle --number_of_recorders 1 --number_of_runs $((15)) -w sandbox -w canyon -w forest --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success -e


# ------------TRAIN_OFFLINE_AND_EVALUATE_ONLINE----------
# python condor_offline.py -t canyon_drone/0_scratch --not_nice --wall_time $((15*60*60)) --dataset canyon_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w canyon --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --scratch
# Sandbox fits into 25G of RAM

# python dag_train_and_evaluate.py -t canyon_drone_scratch --wall_time_train $((5*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 5 --learning_rate 0.01 --dataset canyon_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w canyon -w esat_v1 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --scratch --speed 1.3 --visualize_saliency_of_output --visualize_deep_dream_of_output
# python dag_train_and_evaluate.py -t forest_drone_scratch --wall_time_train $((5*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 5 --learning_rate 0.01 --dataset forest_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w forest -w esat_v1 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --scratch --speed 1.3 --visualize_saliency_of_output --visualize_deep_dream_of_output
# python dag_train_and_evaluate.py -t sandbox_drone_scratch --wall_time_train $((5*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 5 --learning_rate 0.01 --dataset sandbox_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w sandbox -w esat_v1 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --scratch --speed 1.3 --visualize_saliency_of_output --visualize_deep_dream_of_output

for world in canyon forest ; do
  for net in alex alex_v1 alex_v2 ; do
     python dag_train_and_evaluate.py -t ${world}_tiny_${net}_cont_ctrnrm --normalize_over_actions --wall_time_train $((3*60*60)) --wall_time_eva $((20*60)) --gpumem 3000 --number_of_models 2 --load_data_in_ram --learning_rate 0.0001 --dataset ${world}_drone_tiny --max_episodes 200 --paramfile eva_params.yaml --number_of_runs 2 -w ${world} --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network $net --scratch --speed 1.3 --visualize_deep_dream_of_output --visualize_saliency_of_output --histogram_of_weights --histogram_of_activations
 python dag_train_and_evaluate.py -t ${world}_tiny_${net}_disc_ctrnrm --normalize_over_actions --wall_time_train $((3*60*60)) --wall_time_eva $((20*60)) --gpumem 3000 --number_of_models 2 --load_data_in_ram --learning_rate 0.0001 --dataset ${world}_drone_tiny --max_episodes 200 --paramfile eva_params.yaml --number_of_runs 2 -w ${world} --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network $net --scratch --speed 1.3 --visualize_deep_dream_of_output --visualize_saliency_of_output --histogram_of_weights --histogram_of_activations --discrete
   done
   for net in mobile ; do
     python dag_train_and_evaluate.py -t ${world}_tiny_${net}_cont_ctrnrm --normalize_over_actions --wall_time_train $((24*60*60)) --wall_time_eva $((20*60)) --number_of_models 2 --load_data_in_ram --rammem 7 --learning_rate 0.001 --dataset ${world}_drone_tiny --max_episodes 300 --paramfile eva_params.yaml --number_of_runs 2 -w ${world} --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network $net --scratch --speed 1.3 --visualize_deep_dream_of_output --visualize_saliency_of_output --histogram_of_weights --histogram_of_activations
     python dag_train_and_evaluate.py -t ${world}_tiny_${net}_disc_ctrnrm --normalize_over_actions --wall_time_train $((24*60*60)) --wall_time_eva $((20*60)) --number_of_models 2 --load_data_in_ram --rammem 7 --learning_rate 0.001 --dataset ${world}_drone_tiny --max_episodes 300 --paramfile eva_params.yaml --number_of_runs 2 -w ${world} --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network $net --scratch --speed 1.3 --visualize_deep_dream_of_output --visualize_saliency_of_output --histogram_of_weights --histogram_of_activations --discrete
   done
done


# ------------ONLINE-------------
# create dataset
# for i in 10 11 12 ; do
# # for i in $(seq 0 9) ; do
# # for i in 0 ; do
# 	python condor_online.py -t rec_$i --wall_time $((5*60*60)) -w sandbox --robot drone_sim --fsm oracle_drone_fsm -n $((15)) --paramfile params.yaml -ds --save_only_success
# 	# python condor_online.py -t test_$i --wall_time $((2*24*60*60)) -w canyon -w forest -w sandbox --robot drone_sim --fsm oracle_drone_fsm -n $((2)) --paramfile params.yaml -ds -e --save_only_success
# done


# -----------TEST-----------------
#python dag_create_data.py -t test_turtle --wall_time_rec $((60*60)) --destination test_turtle --not_nice --number_of_recorders 2 --number_of_runs 2 -w sandbox --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success --evaluation
# python dag_train_and_evaluate.py -t test_train_eva --wall_time_train $((60*60)) --wall_time_eva $((60*60)) --not_nice --number_of_models 1 --dataset small --max_episodes 2 --paramfile eva_params.yaml --number_of_runs 2 -w esatv1 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --no_batchnorm_learning --speed 1.3
# python dag_train_and_evaluate.py -t test_train_eva --wall_time_train $((60*60)) --wall_time_eva $((60*60)) --not_nice --number_of_models 2 --dataset small --load_data_in_ram --max_episodes 10 --visualize_saliency_of_output --visualize_deep_dream_of_output --paramfile eva_params.yaml --number_of_runs 2 -w esatv1 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --no_batchnorm_learning --speed 1.3
# python dag_train_and_evaluate.py -t test_train_eva_discrete --wall_time_train $((60*60)) --wall_time_eva $((60*60)) --not_nice --number_of_models 2 --dataset small --load_data_in_ram --max_episodes 10 --visualize_saliency_of_output --visualize_deep_dream_of_output --paramfile eva_params.yaml --number_of_runs 5 -w sandbox --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --no_batchnorm_learning --discrete --speed 1.3


# train offline:
# python condor_offline.py --owr -t test_offline --not_nice --wall_time $((15*60)) --dataset small --max_episodes 10 --batch_size 3

# python dag_create_data.py -t rec_dt_san --wall_time_rec $((3*15*10*60)) --destination doshico_turtle --number_of_recorders 10 --number_of_runs $((5*15)) -w sandbox --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success --evaluation
# test:
# Train and evaluate model
# python dag_train_and_evaluate.py -t doshico_drone --wall_time_train $((15*60*60)) --wall_time_eva $((5*60*60)) --number_of_models 5 --dataset doshico_drone --max_episodes 100 --paramfile eva_params.yaml --number_of_runs 20 -w esat_v1 -w esat_v2 -w sandbox -w canyon -w forest --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --auxiliary_depth --network mobile_nfc --no_batchnorm_learning --speed 1.3
# python dag_train_and_evaluate.py -t doshico_turtle --wall_time_train $((15*60*60)) --wall_time_eva $((5*60*60)) --number_of_models 5 --dataset doshico_turtle --max_episodes 100 --paramfile eva_params.yaml --number_of_runs 20 -w esat_v1 -w esat_v2 -w sandbox -w canyon -w forest --robot turtle_sim --fsm nn_turtle_fsm --evaluation --network mobile_nfc --no_batchnorm_learning --speed 0.8



# python dag_train_and_evaluate.py -t canyon_drone_scratch --not_nice --wall_time_train $((25*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 2 --dataset canyon_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w canyon -w esat_v1 -w esat_v2 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --scratch --grad_mul_weight 1
# python dag_train_and_evaluate.py -t forest_drone_scratch --not_nice --wall_time_train $((25*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 2 --dataset forest_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w forest -w esat_v1 -w esat_v2 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --scratch --grad_mul_weight 1
# python dag_train_and_evaluate.py -t forest_drone --wall_time_train $((15*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 10 --dataset forest_drone --max_episodes 50 --paramfile eva_params.yaml --number_of_runs 10 -w forest --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --scratch


# while [ true ] ; do clear; condor_q; sleep 2; done
