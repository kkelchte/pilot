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

#--------------------------- CONDOR ONLINE
# for i in $(seq 1) ; do
#   name="test_condor_variance/$i"
#   model="log_neural_architectures/discrete_continuous/tinyv3_continuous/seed_0"
#   condor_args=" --wall_time $((60*60)) --gpumem 800 --rammem 7 --cpus 11 --not_nice"
#   script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 3 --evaluation --python_project pytorch_pilot/pilot"
#   pytorch_args=" --online --tensorboard --checkpoint_path $model --load_config --continue_training"
#   python condor_online.py -t $name $condor_args $script_args $pytorch_args
# done
#--------------------------- DAG EVALUATE ONLINE

condor_args="--wall_time $((2*60*60)) --gpumem 1900 --rammem 7 --cpus 11"
script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
dag_args="--number_of_models 2"
  
# ##AlexNet_Scratch_Reference
# name="evaluate_NA_models/AlexNet_Scratch_Reference"
# model='log_neural_architectures/alex_net_255input/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##AlexNet_Scratch_Shifted Input
# name="evaluate_NA_models/AlexNet_Scratch_Shifted Input"
# model='log_neural_architectures/alex_net/esatv3_expert_200K/shifted_input/1/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

##AlexNet_Scratch_Output_Normalization
name="evaluate_NA_models/AlexNet_Scratch_Output_Normalization_redo"
model='log_neural_architectures/alex_net/esatv3_expert_200K/normalized_output/1/seed_0'
pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

#_________________________________________________________________________________

# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"

# ##AlexNet_Pretrained
# name="evaluate_NA_models/AlexNet_Pretrained"
# model='log_neural_architectures/alex_net_pretrained/esatv3_expert_200K/1/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# condor_args="--wall_time $((2*60*60)) --gpumem 1900 --rammem 7 --cpus 11"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##VGG16_Pretrained
# name="evaluate_NA_models/VGG16_Pretrained"
# model='log_neural_architectures/vgg16_net_pretrained/esatv3_expert_200K/1/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# condor_args="--wall_time $((2*60*60)) --gpumem 6000 --rammem 7 --cpus 11"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##InceptionNet_Pretrained
# name="evaluate_NA_models/InceptionNet_Pretrained"
# model='log_neural_architectures/inception_net_pretrained/esatv3_expert_200K/1/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# condor_args="--wall_time $((2*60*60)) --gpumem 6000 --rammem 7 --cpus 11"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##Res18_Pretrained
# name="evaluate_NA_models/Res18_Pretrained"
# model='log_neural_architectures/res18_net_pretrained/esatv3_expert_200K/01/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# condor_args="--wall_time $((2*60*60)) --gpumem 1900 --rammem 7 --cpus 11"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##DenseNet_Pretrained
# name="evaluate_NA_models/DenseNet_Pretrained"
# model='log_neural_architectures/dense_net_pretrained/esatv3_expert_200K/01/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# condor_args="--wall_time $((2*60*60)) --gpumem 6000 --rammem 7 --cpus 11"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##SqueezeNet_Pretrained
# name="evaluate_NA_models/SqueezeNet_Pretrained"
# model='log_neural_architectures/squeeze_net_pretrained/esatv3_expert_200K/1/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# condor_args="--wall_time $((2*60*60)) --gpumem 6000 --rammem 7 --cpus 11"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# #_________________________________________________________________________________
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"
# condor_args="--wall_time $((2*60*60)) --gpumem 800 --rammem 7 --cpus 11"

# ##TinyNet_Discrete_CE
# name="evaluate_NA_models/TinyNet_Discrete_CE"
# model='log_neural_architectures/discrete_continuous/tinyv3_CE/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##TinyNet_Discrete_MSE
# name="evaluate_NA_models/TinyNet_Discrete_MSE"
# model='log_neural_architectures/discrete_continuous/tinyv3_MSE/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##TinyNet_Continuous
# name="evaluate_NA_models/TinyNet_Continuous"
# model='log_neural_architectures/discrete_continuous/tinyv3_continuous/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

#_________________________________________________________________________________
# script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 10 --evaluation"
# dag_args="--number_of_models 2"
# condor_args="--wall_time $((2*60*60)) --gpumem 900 --rammem 7 --cpus 11"

# ##TinyNet_Siamese
# name="evaluate_NA_models/TinyNet_Siamese"
# model='log_neural_architectures/tinyv3_nfc_net_3/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##TinyNet_Concat_2
# name="evaluate_NA_models/TinyNet_Concat_2"
# model='tinyv3_3d_net_2/2/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##TinyNet_LSTM_FBPTT
# name="evaluate_NA_models/TinyNet_LSTM_FBPTT"
# model='log_neural_architectures/tinyv3_3D_LSTM_net/fbptt/1/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##TinyNet_LSTM_SBPTT
# name="evaluate_NA_models/TinyNet_LSTM_SBPTT"
# model='log_neural_architectures/tinyv3_3D_LSTM_net/sbptt/1/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
# python dag_evaluate.py -t $name $dag_args $condor_args $script_args $pytorch_args

# ##TinyNet_LSTM_WBPTT
# name="evaluate_NA_models/TinyNet_LSTM_WBPTT"
# model='log_neural_architectures/tinyv3_3D_LSTM_net/wbptt/1/seed_0'
# pytorch_args="--online --tensorboard --checkpoint_path $model --load_config --continue_training"
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
# pytorch_args="--pause_simulator --online --alpha 1 --tensorboard --turn_speed 0.8 --speed 0.8 --stochastic"
# dag_args="--number_of_recorders 12 --destination esatv3_expert_stochastic --val_len 1 --test_len 1 --min_rgb 2400 --max_rgb 2600"
# condor_args="--wall_time_rec $((10*10*60+3600)) --rammem 6"
# python dag_create_data.py -t $name $script_args $pytorch_args $dag_args $condor_args


watch condor_q
