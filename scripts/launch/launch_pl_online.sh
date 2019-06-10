#!/bin/bash
chapter=chapter_policy_learning
section=online
pytorch_args="--network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input\
 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --pretrained"

 

echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

pretrain(){
  cd ..
  condor_args_pretraining="--wall_time $((24*60*60))  "
  python dag_train.py $pytorch_args $condor_args_pretraining -t $*
  cd launch
}
train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((4*3600))"
  dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

#######################################
# Pretrain for different learning rates
#######################################

# dataset=esatv3_expert/2500
# pretrain $chapter/$section/$dataset/reference/learning_rates --rammem 7 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule nothing --dataset $dataset --load_data_in_ram --gpumem 1800 
# pretrain $chapter/$section/$dataset/hardbuffer/learning_rates --rammem 7 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --load_data_in_ram --gpumem 1800 
# pretrain $chapter/$section/$dataset/continual/std_0002/learning_rates --rammem 7 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --load_data_in_ram --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.03 --loss_window_std_threshold 0.0002 --continual_learning_lambda 1
# pretrain $chapter/$section/$dataset/continual/std_0003/learning_rates --rammem 7 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --load_data_in_ram --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.03 --loss_window_std_threshold 0.0003 --continual_learning_lambda 1

dataset=long_corridor
pretrain $chapter/$section/$dataset/reference/learning_rates --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule nothing --dataset $dataset --gpumem 1800 
pretrain $chapter/$section/$dataset/hardbuffer/learning_rates --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --gpumem 1800 
pretrain $chapter/$section/$dataset/continual/std_004/learning_rates --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.02 --loss_window_std_threshold 0.004 --continual_learning_lambda 1
pretrain $chapter/$section/$dataset/continual/std_003/learning_rates --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.02 --loss_window_std_threshold 0.003 --continual_learning_lambda 1
pretrain $chapter/$section/$dataset/continual/std_002/learning_rates --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.02 --loss_window_std_threshold 0.002 --continual_learning_lambda 1

#######################################
# Set winning learning rate
#######################################

dataset=esatv3_expert/2500
train $chapter/$section/$dataset/reference/final --rammem 7 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule nothing --dataset $dataset --load_data_in_ram --gpumem 1800 --learning_rate 0.001
train $chapter/$section/$dataset/hardbuffer/final --rammem 7 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --load_data_in_ram --gpumem 1800 --learning_rate 0.001
train $chapter/$section/$dataset/continual/std_0002/final --rammem 7 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --load_data_in_ram --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.03 --loss_window_std_threshold 0.0002 --continual_learning_lambda 1 --learning_rate 0.001 
train $chapter/$section/$dataset/continual/std_0003/final --rammem 7 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --load_data_in_ram --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.03 --loss_window_std_threshold 0.0003 --continual_learning_lambda 1 --learning_rate 0.001

# dataset=long_corridor
# train $chapter/$section/$dataset/reference/final --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule nothing --dataset $dataset --gpumem 1800 
# train $chapter/$section/$dataset/hardbuffer/final --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --gpumem 1800 
# train $chapter/$section/$dataset/continual/std_004/final --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.02 --loss_window_std_threshold 0.004 --continual_learning_lambda 1
# train $chapter/$section/$dataset/continual/std_003/final --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.02 --loss_window_std_threshold 0.003 --continual_learning_lambda 1
# train $chapter/$section/$dataset/continual/std_002/final --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset $dataset --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.02 --loss_window_std_threshold 0.002 --continual_learning_lambda 1




sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
