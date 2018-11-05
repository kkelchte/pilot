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

# ------------TRAIN ONLINE------
python condor_online.py -t online_yellow_barrel/default --not_nice --wall_time $((2*60*60)) -w osb_yellow_barrel --robot turtle_sim --fsm nn_turtle_fsm -n $((100)) --paramfile train_params.yaml --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
python condor_online.py -t online_yellow_barrel/old --not_nice --wall_time $((2*60*60)) -w osb_yellow_barrel --robot turtle_sim --fsm nn_turtle_fsm -n $((100)) --paramfile train_params_old.yaml --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 
python condor_online.py -t online_yellow_barrel/lll_1 --not_nice --wall_time $((2*60*60)) -w osb_yellow_barrel --robot turtle_sim --fsm nn_turtle_fsm -n $((100)) --paramfile LLL_train_params.yaml --x_pos 0.45 --x_var 0.15 --yaw_var 1 --yaw_or 1.57 




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

