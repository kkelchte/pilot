#!/bin/bash
chapter=chapter_domain_shift/new
pytorch_args="--turn_speed 0.8 --speed 0.8 --action_bound 0.9 --scaled_input\
 --batch_size 32 --loss MSE --optimizer SGD --clip 1 --weight_decay 0 --normalized_output"

pretrain(){
  cd ..
  condor_args_pretraining="--wall_time $((24*60*60))"
  python dag_train.py $pytorch_args $condor_args_pretraining -t $*
  cd launch
}

##########################################
# Pretrain for different learning rates
##########################################

# REFERENCE
pretrain $chapter/reference/learning_rates --dataset esatv3_expert/transferred_reference --rammem 7 --load_data_in_ram --gpumem 1900 --extract_nearest_features --network res18_net  --max_episodes 10000
# SOLUTION A
pretrain $chapter/intermediate_representation/learning_rates --dataset esatv3_expert/depth --rammem 12 --load_data_in_ram --gpumem 1900 --extract_nearest_features --network res18_depth_net  --max_episodes 20000
# SOLUTION B
pretrain $chapter/randomization/learning_rates --dataset esatv3_expert/randomized --rammem 7 --gpumem 1900 --extract_nearest_features --network res18_net --max_episodes 10000
# SOLUTION C
pretrain $chapter/auxiliarydepth/learning_rates --dataset esatv3_expert/transferred_reference --rammem 12 --load_data_in_ram --gpumem 5000 --extract_nearest_features --network res18_net --max_episodes 20000 --auxiliary_depth --auxiliary_lambda 10 --save_auxiliary_prediction
# SOLUTION D
pretrain $chapter/style_transferred_augmented/learning_rates --dataset esatv3_expert/transferred --rammem 12 --load_data_in_ram --gpumem 1900 --extract_nearest_features --network res18_net --max_episodes 10000
pretrain $chapter/style_transferred/learning_rates --dataset esat_transferred --rammem 7 --load_data_in_ram --gpumem 1900 --extract_nearest_features --network res18_net  --max_episodes 10000


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
