#!/bin/bash
chapter=chapter_neural_architectures
section=data_normalization
pytorch_args="--network alex_net --dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9\
 --max_episodes 10000 --batch_size 32 --loss CrossEntropy --optimizer SGD --clip 1 --weight_decay 0"


echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

pretrain(){
  cd ..
  condor_args_pretraining="--wall_time $((24*60*60)) --rammem 7 --gpumem 1800 --copy_dataset"
  python dag_train.py $pytorch_args $condor_args_pretraining -t $*
  cd launch
}
train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --save_CAM_images"
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((30*60+5*5*60+30*60)) --rammem 7 --gpumem_train 1800 --gpumem_eva 1800 --copy_dataset"
  dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

#######################################
# Pretrain for different learning rates
#######################################

# pretrain $chapter/$section/alex_skew_input/learning_rates --skew_input
# pretrain $chapter/$section/alex_scaled_input/learning_rates --scaled_input
# pretrain $chapter/$section/alex_normalized_input/learning_rates --normalized_input
# pretrain $chapter/$section/alex_normalized_output/learning_rates --normalized_output


#######################################
# Set winning learning rate
#######################################

# train $chapter/$section/alex_skew_input/final --skew_input --learning_rate 0.1
# train $chapter/$section/alex_scaled_input/final --scaled_input --learning_rate 0.1
# train $chapter/$section/alex_normalized_input/final --normalized_input --learning_rate 0.1
# train $chapter/$section/alex_normalized_output/final --normalized_output --learning_rate 0.1
# train $chapter/$section/alex_scaled_input_normalized_output/final  --normalized_output --scaled_input --learning_rate 0.1
# train $chapter/$section/alex_skew_input_normalized_output/final  --normalized_output --skew_input --learning_rate 0.1


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
