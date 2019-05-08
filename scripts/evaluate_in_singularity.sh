#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
# source .entrypoint_graph
# source .entrypoint_graph_debug
# source .entrypoint_xpra
source .entrypoint_xpra_no_build
roscd simulation_supervised/python

#############
# COMMAND

# RECOVERY
# name="esatv3_recovery"
# script_args="--owr --z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 2 --no_training --recovery --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8"

# EPSILON-EXPERT
# name="esatv3_epsilon"
# script_args="--owr --z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 2 --no_training --recovery --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8"


# EVALUATE MODEL
# name="test_iteration_3"
# model="DAGGER/dagger_model_3"
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 1 --evaluation  --python_project pytorch_pilot_beta/pilot"
# python run_script.py -t $name $script_args $pytorch_args $extra_args


# TRAIN MODEL
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 10 --gradient_steps 1 --online --alpha 0 --tensorboard --max_episodes 100 --loss MSE --il_weight 1 --clip 1.0 --turn_speed 0.8 --speed 0.8"
# name="test_train_model_2"
# model=DAGGER/5K_concat
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 32 --gradient_steps 10 --online --alpha 0 --tensorboard --max_episodes 11000 --loss MSE --il_weight 1 --clip 1.0 --checkpoint_path $model --load_config"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# python run_script.py -t $name $script_args $pytorch_args $extra_args

# TRAIN ONLINE
name="offline_in_simulation"
pytorch_args="--network tinyv3_3d_net --n_frames 2 --checkpoint_path tinyv3_3d_net_2_continuous_scratch  --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --loss MSE --shifted_input --optimizer SGD --continue_training --clip 1.0"
online_args="--alpha 1 --gradient_steps 10000 --buffer_size 5000 --pause_simulator --prefill"
script_args="--z_pos 1 -w esatv3 --random_seed 512 --evaluate_every -1 --final_evaluation_runs 10 --python_project pytorch_pilot_beta/pilot"
python run_script.py -t $name $pytorch_args $online_args $script_args
