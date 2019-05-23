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

#------------------------------------------------------------
#
# jobs
#
#------------------------------------------------------------

chapter=neural_architecuters
section=deeplearning101

name="$chapter/$section/alex_skew_input/learning_rates"
pytorch_args="--skew_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"
condor_args="--wall_time $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
python dag_train.py -t $name $pytorch_args $dag_args $condor_args

name="$chapter/$section/alex_skew_input/final"
pytorch_args="--skew_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 1000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --learning_rate 0.1"
script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
condor_args="--wall_time_train $((67200)) --wall_time_eva $((5*5*60+60*10)) --rammem 7 --gpumem_train 1800 --gpumem_eva 900 --copy_dataset --use_greenlist --cpus 16"
dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args

name="$chapter/$section/alex_scaled_input/learning_rates"
pytorch_args="--scaled_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"
condor_args="--wall_time $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
python dag_train.py -t $name $pytorch_args $dag_args $condor_args

name="$chapter/$section/alex_scaled_input/final"
pytorch_args="--scaled_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 1000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --learning_rate 0.1"
script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
condor_args="--wall_time_train $((67200)) --wall_time_eva $((5*5*60+60*10)) --rammem 7 --gpumem_train 1800 --gpumem_eva 900 --copy_dataset --use_greenlist --cpus 16"
dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args

name="$chapter/$section/alex_normalized_input/learning_rates"
pytorch_args="--normalized_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"
condor_args="--wall_time $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
python dag_train.py -t $name $pytorch_args $dag_args $condor_args

name="$chapter/$section/alex_normalized_input/final"
pytorch_args="--normalized_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 1000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --learning_rate 0.1"
script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
condor_args="--wall_time_train $((67200)) --wall_time_eva $((5*5*60+60*10)) --rammem 7 --gpumem_train 1800 --gpumem_eva 900 --copy_dataset --use_greenlist --cpus 16"
dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args

name="$chapter/$section/alex_normalized_output/learning_rates"
pytorch_args="--normalized_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"
condor_args="--wall_time $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
python dag_train.py -t $name $pytorch_args $dag_args $condor_args

name="$chapter/$section/alex_normalized_output/final"
pytorch_args="--normalized_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --tensorboard --max_episodes 1000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --learning_rate 0.1"
script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
condor_args="--wall_time_train $((67200)) --wall_time_eva $((5*5*60+60*10)) --rammem 7 --gpumem_train 1800 --gpumem_eva 900 --copy_dataset --use_greenlist --cpus 16"
dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args

# name="$chapter/$section/vgg16_SGD/learning_rates"
# pytorch_args="--normalized_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"
# condor_args="--wall_time $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# name="$chapter/$section/vgg16_SGD/final"
# pytorch_args="--normalized_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 1000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --learning_rate 0.1"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
# condor_args="--wall_time_train $((67200)) --wall_time_eva $((5*5*60+60*10)) --rammem 7 --gpumem_train 1800 --gpumem_eva 900 --copy_dataset --use_greenlist --cpus 16"
# dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args


#--------------------------- Redo Neural Architectures Experiments
# name="clean/alex_scratch_reference_lr"
# pytorch_args="--skew_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"
# condor_args="--wall_time $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
# dag_args=""
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# name="clean/alex_scratch_reference_train_and_evaluate"
# pytorch_args="--skew_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 1000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --learning_rate 0.1"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
# condor_args="--wall_time_train $((67200)) --wall_time_eva $((5*5*60+60*10)) --rammem 7 --gpumem_train 1800 --gpumem_eva 900 --copy_dataset --use_greenlist --cpus 16"
# dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args


#--------------------------- Create annotated cam maps of testdata

# condor_args="--wall_time $((30*60)) --gpumem 900 --python_project pytorch_pilot_beta/pilot"
# name="CAM_visualizations/0_norelu"
# pytorch_args="--checkpoint_path validate_different_seeds_online/seed_0 --load_config --online --dataset esatv3_test --save_CAM_images --no_training"
# python condor_offline.py -t $name $pytorch_args $condor_args 
# name="CAM_visualizations/1_norelu"
# pytorch_args="--checkpoint_path validate_different_seeds_online/seed_1 --load_config --online --dataset esatv3_test --save_CAM_images --no_training"
# python condor_offline.py -t $name $pytorch_args $condor_args 
# name="CAM_visualizations/2_norelu"
# pytorch_args="--checkpoint_path validate_different_seeds_online/seed_2 --load_config --online --dataset esatv3_test --save_CAM_images --no_training"
# python condor_offline.py -t $name $pytorch_args $condor_args 



#--------------------------- DAG TRAIN AND EVALUATE MODELS NEURAL ARCHITECTURES

# for AR in tinyv3_net tinyv3_3d_net tinyv3_nfc_net ; do
# #   name="input_space/${AR}"
# #   pytorch_args="--network $AR --n_frames 3 --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
# #    --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --shifted_input --optimizer SGD --loss MSE --weight_decay 0 --clip 1"
# #   dag_args="--number_of_models 1"
# #   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
# #   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
#   name="input_space/${AR}_evaluate"
#   model="input_space/${AR}/seed_0"
#   pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time $((30*60)) --gpumem 900 --rammem 15 --cpus 16 --not_nice --use_greenlist"
#   python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
# done


# for NF in 1 2 3 5 8 16 ; do
#   name="number_of_frames/$NF"
#   pytorch_args="--network tinyv3_3d_net --n_frames $NF --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#    --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --shifted_input --optimizer SGD --loss MSE --weight_decay 0 --clip 1"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 6 --gpumem 900 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done


# name="log_neural_architectures/alex_net/esatv3_expert_200K/reference_seeds"
# pytorch_args="--skew_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"
# dag_args="--number_of_models 2 --seeds 456 789"
# condor_args="--wall_time_train $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# for m in seed_0 seed_1 ; do
#   name="log_neural_architectures/alex_net/esatv3_expert_200K/reference_seeds_evaluate/$m"
#   model="log_neural_architectures/alex_net/esatv3_expert_200K/reference_seeds/$m"
#   pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time $((30*60)) --gpumem 900 --rammem 15 --cpus 16 --not_nice --use_greenlist"
#   python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
# done



# for lr in 01 001 0001 ; do 
#   name="log_neural_architectures/alex_net/esatv3_expert_200K/reference_learningrate/$lr"
#   pytorch_args="--skew_input --network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#    --checkpoint_path log_neural_architectures/alex_net_scratch --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.$lr --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((67200)) --rammem 7 --gpumem 1800 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done



# Further fighting variance on condor, check out 1 machine at a time and see if multiple jobs run on the same machine...
# name="fight_condor_variance"
# model="variance_neural_architecture_results/res18_net_pretrained/0"
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --pause_simulator"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 1 --evaluation"
# dag_args="--number_of_models 5"
# condor_args="--wall_time $((15*60)) --gpumem 900 --rammem 15 --cpus 16 --not_nice --use_greenlist"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args


# ### ALEXNET SCRATCH 5K
# name="variance_neural_architecture_results/alex_net_reference_5K"
# pytorch_args="--network alex_net --dataset esatv3_expert_5K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --load_data_in_ram \
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((3600*5)) --wall_time_eva $((60*60)) --rammem 7 --cpus 13 --use_greenlist"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 3 9) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 

# name="variance_neural_architecture_results/alex_net_normalized_output_5K"
# pytorch_args="--network alex_net --dataset esatv3_expert_5K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --load_data_in_ram \
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input --normalized_output"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((3600*5)) --wall_time_eva $((60*60)) --rammem 7 --cpus 13 --use_greenlist"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 2) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 

# ### ALEXNET SCRATCH 200K
# name="variance_neural_architecture_results/alex_net_reference"
# pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 \
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((5*200*60+3600*2)) --wall_time_eva $((60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 2) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 

# name="variance_neural_architecture_results/alex_net_normalized_output"
# pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 \
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input --normalized_output"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((5*200*60+3600*2)) --wall_time_eva $((60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 2) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 

# ### POPULAR ARCHS PRETRAINED


# # 6000 GPU and 1
# # inception, squeeze, vgg16
# for AR in inception_net vgg16_net squeeze_net ; do
#   name="variance_neural_architecture_results/${AR}_pretrained"
#   pytorch_args="--network ${AR} --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --pretrained \
#    --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
#   condor_args="--wall_time_train $((5*200*60+3600*2)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
#   dag_args="--gpumem_train 6000 --gpumem_eva 6000 --model_names $(seq 0 2) --random_numbers $(seq 111 5 161)"
#   python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 
# done

# # 6000 GPU and 01
# # dense
# name="variance_neural_architecture_results/dense_net_pretrained"
# pytorch_args="--network dense_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --pretrained \
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.01 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((5*200*60+3600*2)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 6000 --gpumem_eva 6000 --model_names $(seq 0 2) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 


# # 2000 GPU and 1
# # alex
# name="variance_neural_architecture_results/alex_net_pretrained"
# pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --pretrained --save_CAM_images\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot --pause_simulator"
# condor_args="--wall_time_train $((100*4*60+2*3600)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 2) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 


# # 2000 GPU and 01
# # res18
# name="variance_neural_architecture_results/res18_net_pretrained"
# pytorch_args="--network res18_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --pretrained \
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.01 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((100*4*60+2*3600)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 2) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 


# # ### TINY CONTINUOUS & DISCRETE
# name="variance_neural_architecture_results/tiny_discrete_CE"
# pytorch_args="--network tinyv3_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((100*4*60+2*3600)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 9) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 

# name="variance_neural_architecture_results/tiny_discrete_MSE"
# pytorch_args="--network tinyv3_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((100*4*60+2*3600)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 9) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 

# name="variance_neural_architecture_results/tiny_continuous"
# pytorch_args="--network tinyv3_net --dataset esatv3_expert_200K --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 100 --learning_rate 0.1 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((100*4*60+2*3600)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 9) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 


# # ### TINY 2 CONCAT & 2 SIAMESE
# name="variance_neural_architecture_results/tiny_2concat"
# pytorch_args="--network tinyv3_3d_net --n_frames 2 --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((100*4*60+2*3600)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 9) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args 

# name="variance_neural_architecture_results/tiny_2nfc"
# pytorch_args="--network tinyv3_nfc_net --n_frames 2 --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --shifted_input"
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --python_project pytorch_pilot/pilot"
# condor_args="--wall_time_train $((100*4*60+2*3600)) --wall_time_eva $((2*60*60)) --rammem 7 --cpus 13 --use_greenlist --copy_dataset"
# dag_args="--gpumem_train 1900 --gpumem_eva 1900 --model_names $(seq 0 9) --random_numbers $(seq 111 5 161)"
# python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $condor_args $dag_args




#--------------------------- CONDOR ONLINE
# for i in $(seq 1) ; do
#   name="test_condor_variance/$i"
#   model="log_neural_architectures/discrete_continuous/tinyv3_continuous/seed_0"
#   condor_args=" --wall_time $((60*60)) --gpumem 800 --rammem 7 --cpus 11 --not_nice"
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation --python_project pytorch_pilot/pilot"
#   pytorch_args=" --on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#   python condor_online.py -t $name $condor_args $script_args $pytorch_args
# done
#--------------------------- DAG EVALUATE ONLINE

###### DAGGER
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"
# condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13"

# for mod in 5K_concat 10K_concat dagger_model_1 dagger_model_2 dagger_model_3 ; do 
#   name="DAGGER/evaluate_$mod"
#   model="DAGGER/$mod"
#   pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
#   dag_args="--number_of_models 2"
#   condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13"
#   python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
# done 

###### RECOVERY

# python dag_evaluate.py -t recovery/evaluate_5K_concat --number_of_models 2 --wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13\
#   --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --on_policy --tensorboard --checkpoint_path recovery/5K_concat --load_config --continue_training
# python dag_evaluate.py -t recovery/evaluate_5K --number_of_models 2 --wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13\
#   --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --on_policy --tensorboard --checkpoint_path recovery/5K --load_config --continue_training

###### EPSILON 

# python dag_evaluate.py -t epsilon/evaluate_5K_concat --number_of_models 2 --wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13\
#   --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --on_policy --tensorboard --checkpoint_path epsilon/5K_concat --load_config --continue_training
# python dag_evaluate.py -t epsilon/evaluate_5K --number_of_models 2 --wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13\
#   --z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation --on_policy --tensorboard --checkpoint_path epsilon/5K --load_config --continue_training

###### Redo evaluation of concat
##TinyNet_Concat_2
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"
# condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13 --greenlist andromeda vladimir"
# for model in 200K_continuous  5K_discrete ; do
#   name="test_2concat/test_$model"
#   pytorch_args="--on_policy --tensorboard --checkpoint_path test_2concat/$model --load_config --continue_training"
#   python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
# done

###### REDO 5K_concat for different seeds out of mistrust
# Question: how much does it all depend on the final local minimum???
# name="validate_different_seeds_online"
# pytorch_args="--network tinyv3_3d_net --n_frames 2 --dataset esatv3_expert_5K --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --load_data_in_ram\
#  --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --shifted_input --optimizer SGD --loss MSE --weight_decay 0 --clip 1"
# dag_args="--number_of_models 3"
# condor_args="--wall_time_train $((100*2*60+2*3600)) --rammem 7 --gpumem 900"
# python dag_train.py -t $name $pytorch_args $dag_args $condor_args

# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation -pp pytorch_pilot/pilot"
# dag_args="--number_of_models 1 --use_greenlist"
# condor_args="--wall_time $((1*60*60)) --gpumem 700 --rammem 7 --cpus 13"
# for mod in 0 1 2; do 
#   name="redo_evaluate_different_seeds/$mod"
#   model="validate_different_seeds_online/seed_$mod"
#   pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#   python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
# done
#_________________________________________________________________________________
# Test dag train and evaluate
# for DS in '5K' '10K' '20K' '50K' '100K' '200K' ; do
#   name="datadependency_online_concat/$DS"
#   pytorch_args="--network tinyv3_3d_net --n_frames 2 --dataset esatv3_expert_${DS} --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
#    --checkpoint_path tinyv3_3d_net_2_continuous_scratch --tensorboard --max_episodes 10000 --batch_size 32 --learning_rate 0.1 --shifted_input\
#    --optimizer SGD --loss MSE --weight_decay 0 --clip 1."
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 4 --evaluation --python_project pytorch_pilot/pilot"
#   dag_args="--number_of_models 3"
#   condor_args="--wall_time_train $((6*60*60)) --wall_time_eva $((60*60)) --gpumem 700 --rammem 7 --cpus 13 --use_greenlist"
#   python dag_train_and_evaluate.py -t $name $pytorch_args $script_args $dag_args $condor_args
# done

# for DS in '5K' '10K' '20K' '50K' '100K' '200K' ; do
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation -pp pytorch_pilot/pilot"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time $((1*60*60)) --gpumem 1900 --rammem 7 --cpus 7"
#   for mod in 0 1 2; do 
#     name="datadependency_online_concat_evaluate/$DS/$mod"
#     model="datadependency_online_concat/$DS/$mod"
#     pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training --use_greenlist"
#     python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args
#   done
# done
#_________________________________________________________________________________
# MAS on TinyNet 
# 10000 frames in one hour ==> 50000 in 5hours if 1.5fps 
# 3x slower with savefig

# Train without MAS and see how it 'forgets' along the different runs

# name="continual_learning/5/baseline"
# pytorch_args="--online --dataset forest_trail_dataset --tensorboard --network tinyv3_net --checkpoint_path tinyv3_net_scratch\
#  --buffer_size 100 --min_buffer_size 100 --learning_rate 0.001 --gradient_steps 10 --clip 5.0 --load_data_in_ram\
#  --discrete --weight_decay 0.0005 --buffer_update_rule hard --train_every_N_steps 10"
# dag_args="--number_of_models 1"
# condor_args="--wall_time $((5*3600)) --rammem 7 --gpumem 900 --copy_dataset"
# python condor_offline.py -t $name $pytorch_args $dag_args $condor_args

# for mean in 001 ; do
#   for std in 002 001 ; do 
#     name="continual_learning/5/$mean/$std"
#     pytorch_args="--online --dataset forest_trail_dataset --tensorboard --network tinyv3_net --checkpoint_path tinyv3_net_scratch\
#      --buffer_size 100 --min_buffer_size 100 --learning_rate 0.001 --gradient_steps 10 --clip 5.0 --load_data_in_ram  --buffer_update_rule hard --train_every_N_steps 10\
#      --discrete --loss_window_mean_threshold 0.$mean --loss_window_std_threshold 0.$std --weight_decay 0.0005 --continual_learning_lambda 1 --continual_learning"
#     dag_args="--number_of_models 1"
#     condor_args="--wall_time $((5*3600)) --rammem 7 --gpumem 900 --copy_dataset"
#     python condor_offline.py -t $name $pytorch_args $dag_args $condor_args
#   done
# done


#_________________________________________________________________________________
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"
# condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 13"

# ##TinyNet_Siamese
# name="online_NA_evaluation_redo/TinyNet_Siamese"
# model='log_neural_architectures/tinyv3_nfc_net_3/seed_0'
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##TinyNet_Concat_2
# name="online_NA_evaluation_redo/TinyNet_Concat_2"
# model='tinyv3_3d_net_2/2/seed_0'
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

##TinyNet_LSTM_FBPTT
#name="online_NA_evaluation/TinyNet_LSTM_FBPTT_2"
#model='log_neural_architectures/tinyv3_3D_LSTM_net/fbptt/1/seed_0'
#pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

##TinyNet_LSTM_SBPTT
#name="online_NA_evaluation/TinyNet_LSTM_SBPTT_2"
#model='log_neural_architectures/tinyv3_3D_LSTM_net/sbptt/1/seed_0'
#pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

##TinyNet_LSTM_WBPTT
#name="online_NA_evaluation/TinyNet_LSTM_WBPTT_2"
#model='log_neural_architectures/tinyv3_3D_LSTM_net/wbptt/1/seed_0'
#pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
#python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

## TinyNet_LSTM_WBPTT_init
# name="online_NA_evaluation/TinyNet_LSTM_WBPTT_init"
# model='log_neural_architectures/tinyv3_3D_LSTM_net/wbptt_init/1/seed_0'
# pytorch_args="--on_policy --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

#--------------------------- DAG TRAIN OFFLINE


#   name="alex_net/esatv3_expert_200K/normalized_output/$LR"
#   pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --discrete\
#    --continue_training --checkpoint_path alex_net_scratch --tensorboard --max_episodes 20000 --batch_size 100 --loss CrossEntropy\
#    --learning_rate 0.$LR --normalized_output"
#   dag_args="--number_of_models 1"
#   condor_args="--wall_time_train $((5*200*60+3600*2)) -gpumem 1900 --rammem 7 --copy_dataset"
#   python dag_train.py -t $name $pytorch_args $dag_args $condor_args
# done

#--------------------------- DAG COLLECT DATASET ONLINE

# name="collect_esatv3_stochastic"
# script_args="--z_pos 1 -w esatv3 --random_seed 512  --owr -ds --number_of_runs 10 --no_training --evaluate_every -1 --final_evaluation_runs 0"
# pytorch_args="--pause_simulator --on_policy --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8 --stochastic"
# dag_args="--number_of_recorders 12 --destination esatv3_expert_stochastic --val_len 1 --test_len 1 --min_rgb 2400 --max_rgb 2600"
# condor_args="--wall_time_rec $((10*10*60+3600)) --rammem 6"
# python dag_create_data.py -t $name $script_args $pytorch_args $dag_args $condor_args


watch condor_q
