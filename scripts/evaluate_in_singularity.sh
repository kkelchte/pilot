#!/bin/bash
# This scripts evaluate the model in log/testing 2 times in canyon and saves result in log/testing_online
cd /esat/opal/kkelchte/docker_home
# source .entrypoint_graph
# source .entrypoint_graph_debug
# source .entrypoint_xpra
source .entrypoint_xpra_no_build
# roscd simulation_supervised/python
roscd simulation_supervised/python

pwd
#############
# COMMAND

# Collect long corridor dataset USE SIMSUP_BETA FROM ENTROPOINT_GRAPH_DEBUG

# for settings in Black-black-default Blue-blue-default Bricks-bricks-default Green-green-default WoodPallet-woodpallet-default WoodFloor-woodfloor-default Tiled-tiled-default Red-red-default Purple-purple-default OSB-osb-default; do
# for settings in Black-black-default Blue-blue-default Bricks-bricks-default Green-green-default WoodPallet-woodpallet-diffuse WoodFloor-woodfloor-diffuse Tiled-tiled-spot Red-red-default Purple-purple-default OSB-osb-diffuse; do
# for settings in Bricks-bricks-default Green-green-default WoodPallet-woodpallet-diffuse Tiled-tiled-diffuse Red-red-default Purple-purple-default ; do
#   texture="$(echo $settings | cut -d '-' -f 1)"
#   world="$(echo $settings | cut -d '-' -f 2)"
#   light="$(echo $settings | cut -d '-' -f 3)"
#   echo "world $world texture $texture"
#   pytorch_args="--alpha 1 --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9 --yaw_or 1.57"
#   script_args=" -ds --z_pos 1 -w corridor --random_seed 512 --number_of_runs 5 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot/pilot"
#   world_args="--corridor_length 50 --corridor_bends 25 --extension_config corridor_${world} --texture Gazebo/${texture} --lights ${light}_light"
#   python run_script.py -t "varying_corridor/corridor_${world}" $pytorch_args $script_args $world_args
# done


# name="test"
# pytorch_args="--alpha 1 --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9"
# script_args="--recovery -ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot_beta/pilot"
# python run_script.py -t $name $pytorch_args $script_args



###################
# TEST EVALUATION #
###################
# python run_script.py -g -pe sing -pp pytorch_pilot_beta/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag test_online  --save_CAM_images --load_config --on_policy --pause_simulator --evaluation --checkpoint_path chapter_neural_architectures/popular_architectures/inception_net_end-to-end/final/0 --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --copy_dataset --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --scaled_input --learning_rate 0.1
# python run_script.py -g -pe sing -pp pytorch_pilot/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag test_online  --save_CAM_images --load_config --on_policy --pause_simulator --evaluation --checkpoint_path testing --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --copy_dataset --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --scaled_input --learning_rate 0.1


# python run_script.py -g -pe sing -pp pytorch_pilot_beta/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag test_vgg16_SGD  --load_config --on_policy --pause_simulator --evaluation --checkpoint_path chapter_neural_architectures/optimizers/vgg16_SGD_scratch/final/0 --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --copy_dataset --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --scaled_input --learning_rate 0.1
# python run_script.py -pe sing -pp pytorch_pilot/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag testing  --load_config --on_policy --pause_simulator --evaluation --checkpoint_path chapter_policy_learning/how_to_recover/res18_reference/final/1 --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --copy_dataset --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --save_CAM_images --scaled_input --learning_rate 0.1

# python run_script.py -pe sing -pp pytorch_pilot_beta/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag test_on_opal  --load_config --on_policy --pause_simulator --evaluation --checkpoint_path chapter_neural_architectures/optimizers/alex_SGD_pretrained/final/2 --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --normalized_output --max_episodes 10000 --batch_size 32 --loss CrossEntropy --clip 1 --weight_decay 0 --copy_dataset --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --pretrained --optimizer SGD --network vgg16_net --normalized_input --learning_rate 0.1

# python run_script.py -pe sing -pp pytorch_pilot_beta/pilot --log_tag test_stochastic_policy --stochastic --network tinyv3_net --on_policy --pause_simulator --evaluation --checkpoint_path testing --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --scaled_input --learning_rate 0.1


# python run_script.py -pe sing -pp pytorch_pilot/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag test_res18_reference_pretrained/final/1_eva_opal  --load_config --on_policy --pause_simulator --evaluation --checkpoint_path chapter_policy_learning/how_to_recover/res18_reference_pretrained/final/1 --network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --dataset esatv3_expert/2500 --load_data_in_ram --pretrained --learning_rate 0.01 
# sleep 10
# python run_script.py -pe sing -pp pytorch_pilot/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag chapter_policy_learning/how_to_recover/res18_reference_pretrained/final/2_eva_opal  --load_config --on_policy --pause_simulator --evaluation --checkpoint_path chapter_policy_learning/how_to_recover/res18_reference_pretrained/final/2 --network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --dataset esatv3_expert/2500 --load_data_in_ram --pretrained --learning_rate 0.01 
# sleep 10
# python run_script.py -pe sing -pp pytorch_pilot/pilot --summary_dir tensorflow/log/  --data_root pilot_data/  --log_tag chapter_policy_learning/how_to_recover/res18_reference_pretrained/final/3_eva_opal  --load_config --on_policy --pause_simulator --evaluation --checkpoint_path chapter_policy_learning/how_to_recover/res18_reference_pretrained/final/3 --network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --dataset esatv3_expert/2500 --load_data_in_ram --pretrained --learning_rate 0.01 
# sleep 10

# See how well this net does in the beginning 
# model="variance_neural_architecture_results/alex_net_normalized_output/0"
# name="opal_long_hours/start"
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 1 --evaluation --final_evaluation_runs 0"
# python run_script.py -t $name $script_args $pytorch_args

run_simulation(){
  name="$1"
  model="$2"
  pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --pause_simulator"
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot/pilot"
  python run_script.py -t $name $script_args $pytorch_args
}

# python run_script.py -pe sing -pp pytorch_pilot_beta/pilot --summary_dir tensorflow/log/ --owr --data_root pilot_data/ --log_tag testing  --load_config --on_policy --save_CAM_images --checkpoint_path chapter_neural_architectures/data_normalization/alex_scaled_input/learning_rates/lr_01 --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --copy_dataset --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation --learning_rate 0.1 --pause_simulator


# DONE
# for i in 0 1 2 ; do
# 	name="chapter_neural_architectures/output_pretrained/res18_continuous/final/${i}_opal_eva"
# 	model="chapter_neural_architectures/output_pretrained/res18_continuous/final/$i"
# 	run_simulation $name $model
# done

# RECOVERY
# recovery
# name="recovery_extra"
# script_args="--recovery -ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 1  --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot/pilot"
# pytorch_args="--alpha 1 --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9 --on_policy"
# python run_script.py -t $name $script_args $pytorch_args


# name="data_collection/stochastic_extra/gau"
# script_args="-ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 2  --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot/pilot"
# pytorch_args="--alpha 1 --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9 --on_policy --sigma_yaw 0.5 --noise gau"
# python run_script.py -t $name $script_args $pytorch_args

# name="data_collection/stochastic_extra/uni"
# script_args="-ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 2  --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot/pilot"
# pytorch_args="--alpha 1 --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9 --on_policy --sigma_yaw 0.5 --noise uni"
# python run_script.py -t $name $script_args $pytorch_args

name="data_collection/stochastic_extra/ou"
script_args="-ds --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10  --evaluation --final_evaluation_runs 0 --python_project pytorch_pilot/pilot"
pytorch_args="--alpha 1 --pause_simulator --speed 0.8 --turn_speed 0.8 --action_bound 0.9 --on_policy --sigma_yaw 0.1 --noise ou"
python run_script.py -t $name $script_args $pytorch_args

