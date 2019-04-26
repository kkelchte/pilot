#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
source .entrypoint_graph
# source .entrypoint_graph_debug
# source .entrypoint_xpra
# source .entrypoint_xpra_no_build
roscd simulation_supervised/python

#############
# COMMAND

# test data creation
# name="test_collect_esatv3"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 10 --no_training --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8 --stochastic"
# dag_args="--number_of_recorders 1 --destination esatv3_expert --val_len 1 --test_len 1 --min_rgb 1000 --max_rgb 3000"
# condor_args="--wall_time_rec $((10*60*60)) --rammem 7"


# name="test_train_model"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr --evaluate_every -1 --final_evaluation_runs 5"
# pytorch_args="--pause_simulator --online --alpha 1 --prefill --turn_speed 0.8 --speed 0.8 --learning_rate 0.1 \
# --buffer_size 2000 --batch_size 2000 --gradient_steps 2001 --tensorboard --discrete --max_episodes 2000 --loss CrossEntropy --il_weight 1"


# EVALUATE MODEL

# model=log_neural_architectures/tiny_nfc_net_3/1/seed_0
# model=log_neural_architectures/tiny_3d_net_3/1/seed_0
# model=log_neural_architectures/tiny_net/esatv3_expert_200K/1/seed_0
# model=log_neural_architectures/alex_net/esatv3_expert_200K/normalized_output/1/seed_0

# model="discrete_continuous/tinyv3_continuous/seed_0"
# name="test_evaluate_continuous_output"
# model=discrete_continuous/tinyv3_MSE/seed_0
# name="test_evaluate_discrete_output"

# model=log_neural_architectures/alex_net/esatv3_expert_200K/shifted_input/1/seed_0
# name="evaluate_shifted_input"
# model=log_neural_architectures/alex_net/esatv3_expert_200K/normalized_output/1/seed_0
# name="evaluate_normalized_output"

# model='tinyv3_3D_LSTM_net/fbptt/1/seed_0'
# name='evaluate_LSTM/fbptt'

# model='tinyv3_3d_net_2/2/seed_0'
# name='evaluate_3dcnn'

# model='tinyv3_3d_net_1/1/seed_0'
# name='evaluate_3d_1'


# script_args="--z_pos 1 -w esatv3 --random_seed 512 --owr --number_of_runs 10 --graphics --evaluation --python_project pytorch_pilot_beta/pilot"
# pytorch_args=" --online --tensorboard --turn_speed 0.8 --speed 0.8 --checkpoint_path $model  --load_config --continue_training"


# TRAIN MODEL
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 50 --gradient_steps 1 --online --alpha 1 --tensorboard --discrete --max_episodes 10000 --loss CrossEntropy --il_weight 1"

# name="test_standard_rl"
# script_args="--z_pos 1 -w esatv3 --random_seed 512"
# pytorch_args="--pause_simulator --online --alpha 0. --buffer_size 100000 --tensorboard --discrete --max_episodes 80000 --loss CrossEntropy --il_weight 0"

# python run_script.py -t $name $script_args $pytorch_args $extra_args

python run_script.py -pe sing -pp pytorch_pilot/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag test_dag_variance/0_eva  --random_seed 531 --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 2 --evaluation --online --tensorboard --load_config --continue_training
