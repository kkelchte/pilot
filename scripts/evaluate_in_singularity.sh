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

# See how well this net does in the beginning 
# model="variance_neural_architecture_results/alex_net_normalized_output/0"
# name="opal_long_hours/start"
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 1 --evaluation --final_evaluation_runs 0"
# python run_script.py -t $name $script_args $pytorch_args

run_simulation(){
  name="$1"
  model="$2"
  pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation --final_evaluation_runs 0"
  python run_script.py -t $name $script_args $pytorch_args
}

# DONE
# # - Alex (Scaled)
# run_simulation online_NA_evaluation_extra/Alexnet_Scratch_Scaled log_neural_architectures/alex_net/esatv3_expert_200K/scaled_input/1/seed_0
# # - VGG SGD
# run_simulation online_NA_evaluation_extra/VGG_SGD_Scratch log_neural_architectures/vgg16_net/esatv3_expert_200K/SGD/1/seed_0
# # - VGG Adam
# run_simulation online_NA_evaluation_extra/VGG_Adam_Scratch log_neural_architectures/vgg16_net/esatv3_expert_200K/Adam/00001/seed_0
# # - VGG Adadelta
# run_simulation online_NA_evaluation_extra/VGG_Adadelta_Scratch log_neural_architectures/vgg16_net/esatv3_expert_200K/Adadelta/1/seed_0
# # - Alex pretrained (Shifted)
# run_simulation online_NA_evaluation_extra/Alex_Pre log_neural_architectures/alex_net_pretrained/esatv3_expert_200K/1/seed_0
# # - VGG pretrained SGD
# run_simulation online_NA_evaluation_extra/VGG_Pre_SGD log_neural_architectures/vgg16_net_pretrained/esatv3_expert_200K/SGD/1/seed_0
# # - VGG pretrained Adam
# run_simulation online_NA_evaluation_extra/VGG_Pre_Adam log_neural_architectures/vgg16_net_pretrained/esatv3_expert_200K/Adam/00001/seed_0
# # - VGG pretrained Adadelta
# run_simulation online_NA_evaluation_extra/VGG_Pre_Adadelta log_neural_architectures/vgg16_net_pretrained/esatv3_expert_200K/Adadelta/1/seed_0



# TODO later
# # - Alex Shifted Seed 1
# run_simulation online_NA_evaluation_extra/Alexnet_Scratch_Seed1 log_neural_architectures/alex_net/esatv3_expert_200K/reference_seeds/seed_0
# # - Alex Shifted Seed 2
# run_simulation online_NA_evaluation_extra/Alexnet_Scratch_Seed2 log_neural_architectures/alex_net/esatv3_expert_200K/reference_seeds/seed_0
# # - Alex learning rate 0.01
# run_simulation online_NA_evaluation_extra/Alexnet_Scratch_Seed2 log_neural_architectures/alex_net/esatv3_expert_200K/reference_learningrate/01/seed_0
# # - Alex learning rate 0.001
# run_simulation online_NA_evaluation_extra/Alexnet_Scratch_Seed2 log_neural_architectures/alex_net/esatv3_expert_200K/reference_learningrate/001/seed_0
# # - Alex learning rate 0.0001
# run_simulation online_NA_evaluation_extra/Alexnet_Scratch_Seed2 log_neural_architectures/alex_net/esatv3_expert_200K/reference_learningrate/0001/seed_0

run_simulation opal_long_hours/start variance_neural_architecture_results/alex_net_normalized_output/0
# Alex Finetuned 
run_simulation online_NA_evaluation_extra/Alexnet_Finetuned log_neural_architectures/alex_net_finetune/01/seed_0
# VGG16 Finetuned
run_simulation online_NA_evaluation_extra/VGG16_Finetuned log_neural_architectures/vgg16_net_finetune/1/seed_0
# Inception Finetuned
run_simulation online_NA_evaluation_extra/Inception_Finetuned log_neural_architectures/inception_net_finetune/1/seed_0
# Res18 Finetuned
run_simulation online_NA_evaluation_extra/Res18_Finetuned log_neural_architectures/res18_net_finetune/1/seed_0
# Dense Finetuned
run_simulation online_NA_evaluation_extra/Dense_Finetuned log_neural_architectures/dense_net_finetune/01/seed_0
# Squeeze Finetuned
run_simulation online_NA_evaluation_extra/Squeeze_Finetuned log_neural_architectures/squeeze_net_finetune/1/seed_0

# Alex Pretrained
# VGG16 Pretrained
run_simulation online_NA_evaluation_extra/VGG16_pretrained log_neural_architectures/vgg16_net_pretrained/esatv3_expert_200K/01/seed_0
# Inception Pretrained
run_simulation online_NA_evaluation_extra/Inception_pretrained log_neural_architectures/inception_net_pretrained/esatv3_expert_200K/01/seed_0
# Res18 Pretrained
run_simulation online_NA_evaluation_extra/Res18_pretrained log_neural_architectures/res18_net_pretrained/esatv3_expert_200K/01/seed_0
# Dense Pretrained
run_simulation online_NA_evaluation_extra/Dense_pretrained log_neural_architectures/dense_net_pretrained/esatv3_expert_200K/01/seed_0
# Squeeze Pretrained
run_simulation online_NA_evaluation_extra/Dense_pretrained log_neural_architectures/squeeze_net_pretrained/esatv3_expert_200K/1/seed_0

run_simulation opal_long_hours/end variance_neural_architecture_results/alex_net_normalized_output/0



# See how well this net does in the end  
# model="variance_neural_architecture_results/alex_net_normalized_output/0"
# name="opal_long_hours/end"
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 1 --evaluation --final_evaluation_runs 0"
# python run_script.py -t $name $script_args $pytorch_args

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
