#!/bin/bash
chapter=chapter_domain_shift
section=randomization2
pytorch_args="--network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input\
 --max_episodes 10000 --batch_size 32 --loss MSE --optimizer SGD --clip 1 --weight_decay 0"

echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

pretrain(){
  cd ..
  condor_args_pretraining="--wall_time $((24*60*60)) --gpumem 1900"
  python dag_train.py $pytorch_args $condor_args_pretraining -t $*
  cd launch
}

train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --save_CAM_images"
  condor_args="--wall_time_train $((200*60*3+30*60)) --wall_time_eva $((3*3600)) --rammem 7 --gpumem_train 1800 --gpumem_eva 1800"
  dag_args="--model_names $(seq 1 3) --random_numbers $(seq 56431 56441)"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

##########################################
# Pretrain for different learning rates
##########################################

pretrain $chapter/$section/res18_reference_data/learning_rates --dataset esatv3_expert/transferred_reference --rammem 7 --pretrained --load_data_in_ram
pretrain $chapter/$section/res18_randomized_data/learning_rates --dataset esatv3_expert/randomized --rammem 7 --pretrained --load_data_in_ram


##############################
# Set winning learning rate
##############################

# train $chapter/$section/res18_reference_data/final --dataset esatv3_expert_200K --rammem 5 --pretrained --copy_dataset --learning_rate
# train $chapter/$section/res18_randomized_data/final --dataset esatv3_expert/randomized --rammem 5 --pretrained --copy_dataset --learning_rate 0.01


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
