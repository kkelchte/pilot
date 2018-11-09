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


# ------------LLL CREATE DATA DOSHICO-------

# python dag_create_data.py -t LLL_doshico_forest --wall_time_rec $((10*60*60)) --destination LLL_doshico_forest --number_of_recorders 10 --number_of_runs 15 -w forest --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 2  --test_len 1

#python dag_create_data.py -t LLL_doshico_canyon --wall_time_rec $((10*60*60)) --destination LLL_doshico_canyon --number_of_recorders 10 --number_of_runs 15 -w canyon --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 2  --test_len 1
#python dag_create_data.py -t LLL_doshico_sandbox --wall_time_rec $((10*60*60)) --destination LLL_doshico_sandbox --number_of_recorders 30 --number_of_runs 15 -w sandbox --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 2  --test_len 1


# ------------LLL DOSHICO-------

# STEP 1: train set on forest and test
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest --rammem 31 --wall_time_train $((3*60*60)) --wall_time_eva $((1*60*60)) --number_of_models 2 --network tiny_v2 --batch_size 64 --load_data_in_ram --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest --max_episodes 100 --discrete --update_importance_weights --paramfile eva_params.yaml --number_of_runs 5 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 -w forest 
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_0001 --rammem 31 --wall_time_train $((3*60*60)) --wall_time_eva $((1*60*60)) --number_of_models 2 --network tiny_v2 --batch_size 64 --load_data_in_ram --learning_rate 0.0001 --optimizer gradientdescent --dataset LLL_doshico_forest --max_episodes 100 --discrete --update_importance_weights --paramfile eva_params.yaml --number_of_runs 5 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 -w forest 

#python dag_train_and_evaluate.py -t LLL_doshico_final/canyon_0001 --rammem 31 --wall_time_train $((3*60*60)) --wall_time_eva $((1*60*60)) --number_of_models 2 --network tiny_v2 --batch_size 64 --load_data_in_ram --learning_rate 0.0001 --optimizer gradientdescent --dataset LLL_doshico_canyon --max_episodes 100 --discrete --update_importance_weights --paramfile eva_params.yaml --number_of_runs 5 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 -w canyon
# python dag_train_and_evaluate.py -t LLL_doshico_final/canyon --rammem 31 --wall_time_train $((3*60*60)) --wall_time_eva $((1*60*60)) --number_of_models 2 --network tiny_v2 --batch_size 64 --load_data_in_ram --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_canyon --max_episodes 1000 --discrete --update_importance_weights --paramfile eva_params.yaml --number_of_runs 5 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 -w canyon


# STEP 2A: train set initialized on forest on canyon with and without lifelonglearning and test in canyon and forest
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_canyon_2/LL_1 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_canyon --max_episodes 1000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --update_importance_weights --lifelonglearning --lll_weight 1 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w canyon
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_canyon_2/LL_10 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_canyon --max_episodes 1000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --update_importance_weights --lifelonglearning --lll_weight 10 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w canyon

# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_canyon_2/LL_10_3000 --rammem 31 --wall_time_train $((30*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_canyon --max_episodes 3000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542  --update_importance_weights --lifelonglearning --lll_weight 10 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w canyon

# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_canyon_2/LL_20 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_canyon --max_episodes 1000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --update_importance_weights --lifelonglearning --lll_weight 20 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w canyon
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_canyon_2/noLL  --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_canyon --max_episodes 1000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --update_importance_weights --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 -w forest -w canyon

# STEP 2B: train set initialized on forest on sandbox with and without lifelonglearning and test in sandbox and forest
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_sandbox_2/LL_10 --wall_time_train $((30*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3 --batch_size 64 --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_sandbox --max_episodes 1000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --update_importance_weights --lifelonglearning --lll_weight 10 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w sandbox
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_sandbox_2/LL_1 --wall_time_train $((30*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3 --batch_size 64 --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_sandbox --max_episodes 1000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --update_importance_weights --lifelonglearning --lll_weight 1 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w sandbox
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_sandbox_2/noLL  --wall_time_train $((30*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3 --batch_size 64 --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_sandbox --max_episodes 1000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --update_importance_weights --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 -w forest -w sandbox

# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_sandbox_small2/LL_10 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3  --network tiny_v2r --load_data_in_ram --batch_size 64 --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_sandbox_small --max_episodes 3000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --discrete --update_importance_weights --lifelonglearning --lll_weight 10 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w sandbox
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_sandbox_small2/LL_1 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3  --network tiny_v2r --load_data_in_ram --batch_size 64 --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_sandbox_small --max_episodes 3000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --discrete --update_importance_weights --lifelonglearning --lll_weight 1 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w sandbox
# python dag_train_and_evaluate.py -t LLL_doshico_final/forest_sandbox_small2/noLL --rammem 31 --wall_time_train $((3*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 3  --network tiny_v2r --load_data_in_ram --batch_size 64 --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_forest_sandbox_small --max_episodes 3000 --checkpoint_path LLL_doshico_final/forest/0/2018-11-02_0542 --discrete --update_importance_weights --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 -w forest -w sandbox


# STEP 3A: train on pretrained on sandbox in canyon
python dag_train_and_evaluate.py -t LLL_doshico_final/sandbox_canyon/noLL --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_sandbox_canyon --max_episodes 1000 --checkpoint_path LLL_doshico_final/sandbox_small_reg/0/2018-11-07_1220 --update_importance_weights                                   --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w canyon
python dag_train_and_evaluate.py -t LLL_doshico_final/sandbox_canyon/LL_1 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_sandbox_canyon --max_episodes 1000 --checkpoint_path LLL_doshico_final/sandbox_small_reg/0/2018-11-07_1220 --update_importance_weights --lifelonglearning --lll_weight 1 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w canyon
python dag_train_and_evaluate.py -t LLL_doshico_final/sandbox_canyon/LL_10 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_sandbox_canyon --max_episodes 1000 --checkpoint_path LLL_doshico_final/sandbox_small_reg/0/2018-11-07_1220 --update_importance_weights --lifelonglearning --lll_weight 10 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w canyon

# STEP 3B: train on pretrained sandbox in forest
python dag_train_and_evaluate.py -t LLL_doshico_final/sandbox_forest/noLL --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_sandbox_forest --max_episodes 1000 --checkpoint_path LLL_doshico_final/sandbox_small_reg/0/2018-11-07_1220 --update_importance_weights                                   --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w forest
python dag_train_and_evaluate.py -t LLL_doshico_final/sandbox_forest/LL_1 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_sandbox_forest --max_episodes 1000 --checkpoint_path LLL_doshico_final/sandbox_small_reg/0/2018-11-07_1220 --update_importance_weights --lifelonglearning --lll_weight 1 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w forest
python dag_train_and_evaluate.py -t LLL_doshico_final/sandbox_forest/LL_10 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_sandbox_forest --max_episodes 1000 --checkpoint_path LLL_doshico_final/sandbox_small_reg/0/2018-11-07_1220 --update_importance_weights --lifelonglearning --lll_weight 10 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w forest

# STEP 4A: train on pretrained canyon in sandbox
python dag_train_and_evaluate.py -t LLL_doshico_final/canyon_sandbox/noLL --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram  --network tiny_v2r --discrete --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_canyon_sandbox --max_episodes 1000 --checkpoint_path LLL_doshico_final/canyon/0/2018-11-05_0403 --update_importance_weights                                   --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w canyon
python dag_train_and_evaluate.py -t LLL_doshico_final/canyon_sandbox/LL_1 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram  --network tiny_v2r --discrete --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_canyon_sandbox --max_episodes 1000 --checkpoint_path LLL_doshico_final/canyon/0/2018-11-05_0403 --update_importance_weights --lifelonglearning --lll_weight 1 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w canyon
python dag_train_and_evaluate.py -t LLL_doshico_final/canyon_sandbox/LL_10 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram  --network tiny_v2r --discrete --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_canyon_sandbox --max_episodes 1000 --checkpoint_path LLL_doshico_final/canyon/0/2018-11-05_0403 --update_importance_weights --lifelonglearning --lll_weight 10 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w sandbox -w canyon

# STEP 4B: train on pretrained canyon in forest
python dag_train_and_evaluate.py -t LLL_doshico_final/canyon_forest/noLL --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_canyon_forest --max_episodes 1000 --checkpoint_path LLL_doshico_final/canyon/0/2018-11-05_0403 --update_importance_weights                                   --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w canyon
python dag_train_and_evaluate.py -t LLL_doshico_final/canyon_forest/LL_1 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_canyon_forest --max_episodes 1000 --checkpoint_path LLL_doshico_final/canyon/0/2018-11-05_0403 --update_importance_weights --lifelonglearning --lll_weight 1 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w canyon
python dag_train_and_evaluate.py -t LLL_doshico_final/canyon_forest/LL_10 --rammem 31 --wall_time_train $((10*60*60)) --wall_time_eva $((3*60*60)) --number_of_models 1 --batch_size 64 --load_data_in_ram --load_config --continue_training --learning_rate 0.001 --optimizer gradientdescent --dataset LLL_doshico_canyon_forest --max_episodes 1000 --checkpoint_path LLL_doshico_final/canyon/0/2018-11-05_0403 --update_importance_weights --lifelonglearning --lll_weight 10 --paramfile eva_params.yaml --number_of_runs 6 --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3  -w forest -w canyon



# STEP 3: train set initialized on forest+canyon on sandbox with and without lifelonglearning and test in sandbox and forest and canyon


# ------------CREATE DATA-------

# PRIMAL EXPERIMENT:

# python dag_create_data.py -t floor_straight --wall_time_rec $((60*60)) --destination floor_straight --number_of_recorders 4 --number_of_runs 5 -w corridor --corridor_bends 0 --corridor_length 4 --extension_config floor --corridor_type empty --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3 --val_len 2  --test_len 2 --x_pos 0 --x_var 0.5 --y_pos 0 --y_var 0.7 --z_pos 1.5 --z_var 0.5 --yaw_or 1.57 --yaw_var 0.523
# python dag_create_data.py -t floor_bended --wall_time_rec $((60*60)) --destination floor_bended --number_of_recorders 4 --number_of_runs 10 -w corridor --corridor_bends 1 --corridor_length 1 --extension_config floor --corridor_type empty --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3 --val_len 4  --test_len 4 --x_pos 0 --x_var 0.5 --y_pos 0 --y_var 0.7 --z_pos 1.5 --z_var 0.5 --yaw_or 1.57 --yaw_var 0.523

# python dag_create_data.py --dont_retry -t test_floor_bended --wall_time_rec $((60*60)) --destination test_floor_bended --number_of_recorders 1 --number_of_runs 1 -w corridor --corridor_bends 1 --corridor_length 1 --extension_config floor --corridor_type empty --robot drone_sim --fsm oracle_drone_fsm --paramfile params.yaml -ds --save_only_success -e --max_depth_rgb_difference 3 --val_len 4  --test_len 4 --x_pos 0 --x_var 0.5 --y_pos 0 --y_var 0.7 --z_pos 1.5 --z_var 0.5 --yaw_or 1.57 --yaw_var 0.523


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
# python dag_train_and_evaluate.py -t naive_ensemble/mobile_imgnet --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network mobile --discrete            --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble/squeeze_v1 --gpumem 2600 --wall_time_train $((10*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network squeeze_v1 --discrete  --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble/squeeze_v3 --gpumem 2600 --wall_time_train $((10*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network squeeze_v3 --discrete  --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble/alex_v4 --gpumem 4600 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network alex_v4 --discrete --scratch --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble/tiny --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network tiny --discrete  --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 


# python dag_train.py -t naive_ensemble/tiny_CAM --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors --max_episodes 1000 --network tiny_CAM --discrete  --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --visualize_control_activation_maps


# python dag_train_and_evaluate.py -t naive_ensemble_10_uni/mobile_imgnet --wall_time_train $((4*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 10 --loss mse --load_data_in_ram --learning_rate 0.05 --dataset all_factors_uni --max_episodes 1000 --network mobile --discrete            --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 

# python dag_train_and_evaluate.py -t naive_ensemble_10_empty_uni/mobile_imgnet --wall_time_train $((15*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 10 --loss mse --learning_rate 0.05 --dataset all_factors_empty_uni --max_episodes 500 --network mobile --discrete            --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 
# python dag_train_and_evaluate.py -t naive_ensemble_10_empty_uni_ou/mobile_imgnet --wall_time_train $((15*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 10 --loss mse --learning_rate 0.05 --dataset all_factors_empty_uni_ou --max_episodes 500 --network mobile --discrete            --visualize_deep_dream_of_output --visualize_saliency_of_output --paramfile eva_params.yaml --number_of_runs 10 -w corridor -w esatv1 --reuse_default_world --robot drone_sim --fsm oracle_nn_drone_fsm --evaluation --speed 1.3 


# ------------OFFLINE FACTORS-----
# python dag_train.py -t test_train_only --not_nice --wall_time_train $((30*60)) --number_of_models 2 --load_data_in_ram --network mobile --normalize_over_actions --learning_rate 0.1 --dataset canyon_drone_tiny --max_episodes 30 --discrete --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --histogram_of_weights --histogram_of_activations
# network=squeeze_v1
# # for world in radiator corridor floor poster ceiling blocked_hole doorway arc ; do
# for world in radiator corridor poster ; do
#   python dag_train.py -t $network/${world} --gpumem 3000 --wall_time_train $((10*60*60)) --number_of_models 2 --load_data_in_ram --network $network --normalize_over_actions --dataset $world --max_episodes 1000 --discrete --scratch --visualize_deep_dream_of_output --visualize_saliency_of_output --histogram_of_weights --histogram_of_activations
# done


# python condor_offline.py -t reference_radiator --wall_time $((3*60*60)) --dataset radiator --max_episodes 600 --discrete --load_data_in_ram
# python condor_offline.py -t reference_radiator_poster --wall_time $((3*60*60)) --dataset radiator_poster --max_episodes 600 --discrete --load_data_in_ram


# ---------LIFELONGLEARNING--------
# python dag_train_and_evaluate.py -t lifelonglearning/domain_A --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.1 --dataset domain_A --max_episodes 1000 --discrete --paramfile eva_params_slow.yaml --number_of_runs 3 -w osb_yellow_barrel --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
# python dag_train_and_evaluate.py -t lifelonglearning/domain_A_actnorm --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --normalize_over_actions --learning_rate 0.1 --dataset domain_A --max_episodes 1000 --discrete --paramfile eva_params_slow.yaml --number_of_runs 3 -w osb_yellow_barrel --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
# python dag_train_and_evaluate.py -t lifelonglearning/domain_B --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.1 --dataset domain_B --max_episodes 1000 --discrete --paramfile eva_params_slow.yaml --number_of_runs 3 -w osb_carton_box --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
# python dag_train_and_evaluate.py -t lifelonglearning/domain_B_actnorm --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --normalize_over_actions --learning_rate 0.1 --dataset domain_B --max_episodes 1000 --discrete --paramfile eva_params_slow.yaml --number_of_runs 3 -w osb_carton_box --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
# python dag_train_and_evaluate.py -t lifelonglearning/domain_C --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.1 --dataset domain_C --max_episodes 1000 --discrete --paramfile eva_params_slow.yaml --number_of_runs 3 -w osb_yellow_barrel_blue --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
# python dag_train_and_evaluate.py -t lifelonglearning/domain_C_actnorm --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --normalize_over_actions --learning_rate 0.1 --dataset domain_C --max_episodes 1000 --discrete --paramfile eva_params_slow.yaml --number_of_runs 3 -w osb_yellow_barrel_blue --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
# python dag_train_and_evaluate.py -t lifelonglearning/domain_B --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.1 --dataset domain_B --max_episodes 1000 --discrete --paramfile eva_params_slow.yaml --number_of_runs 3 -w osb_carton_box --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 
# python dag_train_and_evaluate.py -t lifelonglearning/domain_C --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 3 --loss mse --load_data_in_ram --learning_rate 0.1 --dataset domain_C --max_episodes 1000 --discrete --paramfile eva_params_slow.yaml --number_of_runs 3 -w osb_yellow_barrel_blue --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 

# STEP 1: train in domain A
#for n in tiny_v1 tiny_v2 tiny_v3; do
#  for lr in '01' '001' '0001'; do
#    python dag_train_and_evaluate.py -t LLL/domain_A/$n/$lr --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((2*60*60)) --number_of_models 1 --load_data_in_ram --network $n --learning_rate 0.$lr --update_importance_weights --optimizer gradientdescent --dataset domain_A --max_episodes 300 --discrete --action_bound 0.6 --paramfile eva_params_slow.yaml --number_of_runs 5 -w osb_yellow_barrel --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
#  done
#done

# STEP 2: train in domain C with and without lifelonglearning
#TODO: decide which learning rate for training A gave best results and take that as lr variable
# lr='001'
# for n in tiny_v1 tiny_v2 tiny_v3; do
#    python dag_train_and_evaluate.py -t LLL/domain_Aforest_noLL/$n --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((4*60*60)) --number_of_models 1 --load_data_in_ram --learning_rate 0.0001 --update_importance_weights --optimizer gradientdescent --dataset domain_Aforest --max_episodes 300 --checkpoint_path LLL/domain_A/$n/$lr/0 --load_config --continue_training --paramfile eva_params_slow.yaml --number_of_runs 6 -w osb_yellow_barrel -w forest --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
#    for lambda in 1 10 100 ; do   
#      python dag_train_and_evaluate.py -t LLL/domain_Aforest_LL_${lambda}/$n --rammem 25 --wall_time_train $((3*60*60)) --wall_time_eva $((4*60*60)) --number_of_models 1 --load_data_in_ram --learning_rate 0.0001 --update_importance_weights --optimizer gradientdescent --dataset domain_Aforest --max_episodes 300 --checkpoint_path LLL/domain_A/$n/$lr/0 --load_config --continue_training --lifelonglearning --lll_weight $lambda --paramfile eva_params_slow.yaml --number_of_runs 6 -w osb_yellow_barrel -w forest --robot turtle_sim --fsm nn_turtle_fsm --evaluation --speed 0.3 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
#    done
# done


# ------------CREATE_DATA LLL-------------

# python dag_create_data.py -t rec_barrel_cw --wall_time_rec $((10*60*60)) --destination osb_yellow_barrel_cw --number_of_recorders 4 --number_of_runs 10 -w osb_yellow_barrel --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 1  --test_len 1 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 --min_distance 1
# python dag_create_data.py -t rec_barrel_ccw --wall_time_rec $((10*60*60)) --destination osb_yellow_barrel_ccw --number_of_recorders 4 --number_of_runs 10 -w osb_yellow_barrel --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 1  --test_len 1 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 4.71 --min_distance 1
# python dag_create_data.py -t rec_box_cw --wall_time_rec $((10*60*60)) --destination osb_carton_box_cw --number_of_recorders 4 --number_of_runs 10 -w osb_carton_box --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 1  --test_len 1 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 --min_distance 1
# python dag_create_data.py -t rec_box_ccw --wall_time_rec $((10*60*60)) --destination osb_carton_box_ccw --number_of_recorders 4 --number_of_runs 10 -w osb_carton_box --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 1  --test_len 1 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 4.71 --min_distance 1
# python dag_create_data.py -t rec_blue_barrel_cw --wall_time_rec $((10*60*60)) --destination osb_blue_yellow_barrel_cw --number_of_recorders 4 --number_of_runs 10 -w osb_yellow_barrel_blue --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 1  --test_len 1 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 --min_distance 1
# python dag_create_data.py -t rec_blue_barrel_ccw --wall_time_rec $((10*60*60)) --destination osb_blue_yellow_barrel_cw --number_of_recorders 4 --number_of_runs 10 -w osb_yellow_barrel_blue --robot turtle_sim --fsm oracle_turtle_fsm --paramfile params.yaml -ds --save_only_success -e --val_len 1  --test_len 1 --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 4.71 --min_distance 1




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

