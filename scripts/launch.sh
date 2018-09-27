#!/bin/bash
# Script for launching condor jobs invoking both condor_offline.py and condor_online.py scripts.
# Dependencies: condor_offline.py condor_online.py

# OVERVIEW OF PARAMETERS

# 0. dag_create_data / dag_train_and_evaluate
# --number_of_recorders
# --number_of_models
# --destination
# --dont_retry
# --copy_dataset

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

# PRIMAL EXPERIMENT:

# radiator
# # python dag_create_data.py -t radiator_left --wall_time_rec $((60*60)) --destination radiator_left --number_of_recorders 6 --number_of_runs 10 -w corridor --corridor_bends 0 --corridor_length 1 --extension_config radiator_left --corridor_type empty --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3 --val_len 3  --test_len 3 --x_pos '-0.2' --x_var 0.25 --y_pos 0.3 --y_var 1 --z_pos 1.5 --z_var 0.5 --yaw_or 2.09 --yaw_var 0.523
# python dag_create_data.py -t radiator_right --wall_time_rec $((60*60)) --destination radiator_right --number_of_recorders 6 --number_of_runs 10 -w corridor --corridor_bends 0 --corridor_length 1 --extension_config radiator_right --corridor_type empty --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3 --val_len 3  --test_len 3 --x_pos '+0.2' --x_var 0.25 --y_pos 0.3 --y_var 1 --z_pos 1.5 --z_var 0.5 --yaw_or 1.05 --yaw_var 0.523

# combined_corridor
# python dag_create_data.py -t combined_corridor --wall_time_rec $((2*60*60)) --destination combined_corridor --number_of_recorders 50 --number_of_runs 20 -w corridor --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3 --corridor_bends 5 --extension_config combined --x_var 0.5 --y_var 0.7 --z_pos 1.3 --z_var 0.3 --yaw_or 1.57 --yaw_var 0.5233

# esat
# python dag_create_data.py -t esatv1 --wall_time_rec $((3*60*60)) --destination esatv1 --number_of_recorders 10 --number_of_runs 10 -w esatv1 --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3 --val_len 6  --test_len 6 --x_pos 0 --x_var 0.5 --y_pos 0 --y_var 0.7 --z_pos 1.5 --z_var 0.5 --yaw_or 1.57 --yaw_var 0.523

# ------------TRAIN_OFFLINE_AND_EVALUATE_ONLINE----------
# python condor_offline.py -t canyon_drone/0_scratch --not_nice --wall_time $((15*60*60)) --dataset canyon_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w canyon --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --scratch
# Sandbox fits into 25G of RAM

# python dag_train_and_evaluate.py -t naive_ensemble/mobile_scratch --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network mobile --discrete --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
python dag_train_and_evaluate.py -t test_separate_ensemble/mobile_imgnet --wall_time_train $((7*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network mobile --discrete --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble/squeeze_v1 --gpumem 2600 --wall_time_train $((10*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network squeeze_v1 --discrete  --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble/squeeze_v3 --gpumem 2600 --wall_time_train $((10*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network squeeze_v3 --discrete  --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble/alex_v4 --gpumem 4600 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network alex_v4 --discrete --scratch --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble/tiny --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network tiny --discrete  --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 


# ------------OFFLINE FACTORS-----
# python dag_train.py -t test_train_only --not_nice --wall_time_train $((30*60)) --number_of_models 2 --load_data_in_ram --network mobile --normalize_over_actions --learning_rate 0.1 --dataset canyon_drone_tiny --max_episodes 30 --discrete --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --histogram_of_weights --histogram_of_activations
# network=squeeze_v1
# # for world in radiator corridor floor poster ceiling blocked_hole doorway arc ; do
# for world in radiator corridor poster ; do
#   python dag_train.py -t $network/${world} --gpumem 3000 --wall_time_train $((10*60*60)) --number_of_models 2 --load_data_in_ram --network $network --normalize_over_actions --dataset $world --max_episodes 1000 --discrete --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --histogram_of_weights --histogram_of_activations
# done


# python condor_offline.py -t reference_radiator --wall_time $((3*60*60)) --dataset radiator --max_episodes 600 --discrete --load_data_in_ram
# python condor_offline.py -t reference_radiator_poster --wall_time $((3*60*60)) --dataset radiator_poster --max_episodes 600 --discrete --load_data_in_ram


# ------------ONLINE-------------
# create dataset
# for i in 10 11 12 ; do
# # for i in $(seq 0 9) ; do
# # for i in 0 ; do
# 	python condor_online.py -t rec_$i --wall_time $((5*60*60)) -w sandbox --robot drone_sim --fsm oracle_drone_fsm -n $((15)) --paramfile params.yaml -ds --save_only_success
# 	# python condor_online.py -t test_$i --wall_time $((2*24*60*60)) -w canyon -w forest -w sandbox --robot drone_sim --fsm oracle_drone_fsm -n $((2)) --paramfile params.yaml -ds -e --save_only_success
# done


# -----------TEST-----------------
# create dataset
# python dag_create_data.py -t test_drone --wall_time_rec $((60*60)) --destination test_drone --not_nice --number_of_recorders 2 --number_of_runs 2 -w corridor --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success --evaluation
# python dag_create_data.py -t test_drone_fail --wall_time_rec $((60*60)) --destination test_drone_fail --not_nice --number_of_recorders 2 --number_of_runs 2 -w corridor --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success --evaluation --minimum_number_of_success 5

# python dag_train_and_evaluate.py -t test_train_eva --wall_time_train $((60*60)) --wall_time_eva $((60*60)) --not_nice --number_of_models 1 --dataset small --max_episodes 2 --paramfile eva_params.yaml --number_of_runs 2 -w esatv1 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --no_batchnorm_learning --speed 1.3
# python dag_train_and_evaluate.py -t test_train_eva --wall_time_train $((60*60)) --wall_time_eva $((60*60)) --not_nice --number_of_models 2 --dataset small --load_data_in_ram --max_episodes 10 --visualize_saliency_of_output --visualize_deep_dream_of_output --paramfile eva_params.yaml --number_of_runs 2 -w esatv1 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --no_batchnorm_learning --speed 1.3
# python dag_train_and_evaluate.py -t test_train_eva_discrete --wall_time_train $((60*60)) --wall_time_eva $((60*60)) --not_nice --number_of_models 2 --dataset small --load_data_in_ram --max_episodes 10 --visualize_saliency_of_output --visualize_deep_dream_of_output --paramfile eva_params.yaml --number_of_runs 5 -w sandbox --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network mobile --no_batchnorm_learning --discrete --speed 1.3


# train offline:
#python condor_offline.py -t test_offline --not_nice --dont_copy --wall_time $((15*60)) --dataset canyon_drone_tiny --max_episodes 3 --batch_size 3


# train and evaluate:
# python dag_train_and_evaluate.py -t test_offline_online --not_nice --wall_time_train $((20*60)) --wall_time_eva $((20*60)) --gpumem 3000 --number_of_models 1 --load_data_in_ram --dataset canyon_drone_tiny --max_episodes 3 --paramfile eva_params.yaml --number_of_runs 1 -w canyon --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --network alex --scratch --speed 1.3

# python dag_create_data.py -t rec_dt_san --wall_time_rec $((3*15*10*60)) --destination doshico_turtle --number_of_recorders 10 --number_of_runs $((5*15)) -w sandbox --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success --evaluation
# test:
# Train and evaluate model
# python dag_train_and_evaluate.py -t doshico_drone --wall_time_train $((15*60*60)) --wall_time_eva $((5*60*60)) --number_of_models 5 --dataset doshico_drone --max_episodes 100 --paramfile eva_params.yaml --number_of_runs 20 -w esat_v1 -w esat_v2 -w sandbox -w canyon -w forest --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --auxiliary_depth --network mobile_nfc --no_batchnorm_learning --speed 1.3
# python dag_train_and_evaluate.py -t doshico_turtle --wall_time_train $((15*60*60)) --wall_time_eva $((5*60*60)) --number_of_models 5 --dataset doshico_turtle --max_episodes 100 --paramfile eva_params.yaml --number_of_runs 20 -w esat_v1 -w esat_v2 -w sandbox -w canyon -w forest --robot turtle_sim --fsm nn_turtle_fsm --evaluation --network mobile_nfc --no_batchnorm_learning --speed 0.8



# python dag_train_and_evaluate.py -t canyon_drone_scratch --not_nice --wall_time_train $((25*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 2 --dataset canyon_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w canyon -w esat_v1 -w esat_v2 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --scratch --grad_mul_weight 1
# python dag_train_and_evaluate.py -t forest_drone_scratch --not_nice --wall_time_train $((25*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 2 --dataset forest_drone --max_episodes 1000 --paramfile eva_params.yaml --number_of_runs 10 -w forest -w esat_v1 -w esat_v2 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --scratch --grad_mul_weight 1
# python dag_train_and_evaluate.py -t forest_drone --wall_time_train $((15*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 10 --dataset forest_drone --max_episodes 50 --paramfile eva_params.yaml --number_of_runs 10 -w forest --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --scratch


# while [ true ] ; do clear; condor_q; sleep 2; done


#----------------ARCHIVE------------------

