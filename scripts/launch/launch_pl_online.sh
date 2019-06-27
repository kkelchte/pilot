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
# Train online and evaluate online
#######################################


# pretrain $chapter/$section/esatv3_expert/recovery_reference/reference --learning_rate 0.001 --rammem 10 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule nothing --dataset esatv3_expert/recovery_reference --load_data_in_ram --gpumem 1800 
# pretrain $chapter/$section/esatv3_expert/recovery_reference/hardbuffer --learning_rate 0.001 --rammem 10 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset esatv3_expert/recovery_reference --load_data_in_ram --gpumem 1800 
# pretrain $chapter/$section/esatv3_expert/recovery_reference/continual/0003std --learning_rate 0.001 --rammem 10 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset esatv3_expert/recovery_reference --load_data_in_ram --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.005 --loss_window_std_threshold 0.003 --continual_learning_lambda 1 --loss_window_length 10
pretrain $chapter/$section/esatv3_expert/recovery_reference/continual/0003std_redo --learning_rate 0.001 --rammem 10 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset esatv3_expert/recovery_reference --load_data_in_ram --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.005 --loss_window_std_threshold 0.003 --continual_learning_lambda 1 --loss_window_length 10 --python_project pytorch_pilot_beta/pilot


# pretrain $chapter/$section/long_corridor/reference --learning_rate 0.001 --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule nothing --dataset long_corridor --gpumem 1800 
# pretrain $chapter/$section/long_corridor/hardbuffer --learning_rate 0.001 --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset long_corridor --gpumem 1800 
# pretrain $chapter/$section/long_corridor/continual/std_004 --learning_rate 0.001 --rammem 4 --online --min_buffer_size 32 --buffer_size 32 --gradient_steps 3 --buffer_update_rule hard --dataset long_corridor --gpumem 1800 --continual_learning --loss_window_mean_threshold 0.005 --loss_window_std_threshold 0.005 --continual_learning_lambda 1 --loss_window_length 10



sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
