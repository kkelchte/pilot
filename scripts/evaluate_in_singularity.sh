#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
# source .entrypoint_graph
# source .entrypoint_graph_debug
source .entrypoint_xpra
# source .entrypoint_xpra_no_build
roscd simulation_supervised/python

#############
# COMMAND

# BIG LAUNCH OF EVALUATION OF ALL SEEDS OF MODELS
for d in alex_net_normalized_output/0 alex_net_normalized_output/1 alex_net_normalized_output/2 alex_net_normalized_output_5K/0 alex_net_normalized_output_5K/1 alex_net_normalized_output_5K/2 alex_net_pretrained/0 alex_net_pretrained/1 alex_net_pretrained/2 alex_net_pretrained_old/0 alex_net_pretrained_old/1 alex_net_pretrained_old/2 alex_net_reference/0 alex_net_reference/1 alex_net_reference/2 alex_net_reference_5K/0 alex_net_reference_5K/1 alex_net_reference_5K/2 dense_net_pretrained/0 dense_net_pretrained/1 dense_net_pretrained/2 inception_net_pretrained/0 inception_net_pretrained/1 inception_net_pretrained/2 res18_net_pretrained/0 res18_net_pretrained/1 res18_net_pretrained/2 squeeze_net_pretrained/0 squeeze_net_pretrained/1 squeeze_net_pretrained/2 tiny_2concat/0 tiny_2concat/1 tiny_2concat/2 tiny_2concat/3 tiny_2concat/4 tiny_2concat/5 tiny_2concat/6 tiny_2concat/7 tiny_2concat/8 tiny_2concat/9 tiny_2nfc/0 tiny_2nfc/1 tiny_2nfc/2 tiny_2nfc/3 tiny_2nfc/4 tiny_2nfc/5 tiny_2nfc/6 tiny_2nfc/7 tiny_2nfc/8 tiny_2nfc/9 tiny_continuous/0 tiny_continuous/1 tiny_continuous/2 tiny_continuous/3 tiny_continuous/4 tiny_continuous/5 tiny_continuous/6 tiny_continuous/7 tiny_continuous/8 tiny_continuous/9 tiny_discrete_CE/0 tiny_discrete_CE/1 tiny_discrete_CE/2 tiny_discrete_CE/3 tiny_discrete_CE/4 tiny_discrete_CE/5 tiny_discrete_CE/6 tiny_discrete_CE/7 tiny_discrete_CE/8 tiny_discrete_CE/9 tiny_discrete_MSE/0 tiny_discrete_MSE/1 tiny_discrete_MSE/2 tiny_discrete_MSE/3 tiny_discrete_MSE/4 tiny_discrete_MSE/5 tiny_discrete_MSE/6 tiny_discrete_MSE/7 tiny_discrete_MSE/8 tiny_discrete_MSE/9 vgg16_net_pretrained/0 vgg16_net_pretrained/1 vgg16_net_pretrained/2 ; do 
  model="variance_neural_architecture_results/$d"
  name="evaluate_variance/evaluate_$(echo $d | cut -d / -f 1)/$(basename $d)"
  pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation"
  python run_script.py -t $name $script_args $pytorch_args
done


# RECOVERY
# name="esatv3_recovery"
# script_args="--owr --z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 2 --no_training --recovery --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# pytorch_args="--pause_simulator --on_policy --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8"

# EPSILON-EXPERT
# name="esatv3_epsilon"
# script_args="--owr --z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 2 --no_training --recovery --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# pytorch_args="--pause_simulator --on_policy --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8"


# EVALUATE MODEL
# for i in 0 1 2 ; do 

# name="evaluate_res18_new"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation -pp pytorch_pilot_beta/pilot --pause_simulator"
# model="variance_neural_architecture_results/res18_net_pretrained/0"
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python run_script.py -t $name $script_args $pytorch_args $extra_args


# name="evaluate_res18_old"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation -pp pytorch_pilot_beta/pilot --pause_simulator"
# # model="validate_different_seeds_online/seed_0"
# model="log_neural_architectures/res18_net_pretrained/esatv3_expert_200K/01/seed_0"
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python run_script.py -t $name $script_args $pytorch_args $extra_args

# name="evaluate_res18_old_1"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation -pp pytorch_pilot_beta/pilot --pause_simulator"
# # model="validate_different_seeds_online/seed_0"
# model="log_neural_architectures/res18_net_pretrained/esatv3_expert_200K/1/seed_0"
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python run_script.py -t $name $script_args $pytorch_args $extra_args

# name="testing"
# model="variance_neural_architecture_results/res18_net_pretrained/0"
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 1 --evaluation"
# python run_script.py -t $name $script_args $pytorch_args $extra_args
# dag_args="--number_of_models 50"
# condor_args="--wall_time $((10*60)) --gpumem 900 --rammem 7 --cpus 32"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args


# TRAIN MODEL
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 10 --gradient_steps 1 --on_policy --alpha 0 --tensorboard --max_episodes 100 --loss MSE --il_weight 1 --clip 1.0 --turn_speed 0.8 --speed 0.8"
# name="test_train_model_2"
# model=DAGGER/5K_concat
# pytorch_args="--pause_simulator --learning_rate 0.1 --buffer_size 100 --batch_size 32 --gradient_steps 10 --on_policy --alpha 0 --tensorboard --max_episodes 11000 --loss MSE --il_weight 1 --clip 1.0 --checkpoint_path $model --load_config"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# python run_script.py -t $name $script_args $pytorch_args $extra_args

# TRAIN ONLINE
# name="offline_in_simulation"
# pytorch_args="--owr --network tinyv3_3d_net --n_frames 2 --checkpoint_path tinyv3_3d_net_2_continuous_scratch  --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard \
#  --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --loss MSE --shifted_input --optimizer SGD --continue_training --clip 1.0"
# online_args="--alpha 1 --gradient_steps 100 --buffer_size 100 --pause_simulator --min_buffer_size 20"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --evaluate_every -1 --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# python run_script.py -t $name $pytorch_args $online_args $script_args
