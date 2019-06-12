#!/bin/bash
chapter=chapter_domain_shift
section=auxiliarydepth2
pytorch_args="--network res18_net --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input\
 --batch_size 32 --loss MSE --optimizer SGD --clip 1 --weight_decay 0"

echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

pretrain(){
  cd ..
  condor_args_pretraining="--wall_time $((24*60*60))"
  python dag_train.py $pytorch_args $condor_args_pretraining -t $*
  cd launch
}

train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --save_CAM_images"
  condor_args="--wall_time_train $((200*60*3+30*60)) --wall_time_eva $((3*3600)) --rammem 7"
  dag_args="--model_names $(seq 1 3) --random_numbers $(seq 56431 56441)"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

##########################################
# Pretrain for different learning rates
##########################################

# pretrain $chapter/$section/reference/learning_rates --dataset esatv3_expert/transferred_reference --rammem 7 --pretrained --load_data_in_ram  --gpumem 1900 --max_episodes 10000
# pretrain $chapter/$section/auxiliarydepth/learning_rates --dataset esatv3_expert/transferred_reference --rammem 12 --pretrained --load_data_in_ram --auxiliary_depth --gpumem 5000 --max_episodes 10000
pretrain $chapter/$section/auxiliarydepth_long/learning_rates --dataset esatv3_expert/transferred_reference --rammem 12 --pretrained --load_data_in_ram --auxiliary_depth --gpumem 5000 --extract_nearest_features --auxiliary_lambda 10  --max_episodes 20000 --save_auxiliary_prediction


##############################
# Set winning learning rate
##############################

# pretrain $chapter/$section/reference/learning_rates --dataset esatv3_expert/transferred_reference --rammem 7 --pretrained --load_data_in_ram  --gpumem_train 1800 --gpumem_eva 1800
# pretrain $chapter/$section/auxiliarydepth/learning_rates --dataset esatv3_expert/transferred_reference --rammem 12 --pretrained --load_data_in_ram --auxiliary_depth --gpumem_train 5000 --gpumem_eva 5000


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
