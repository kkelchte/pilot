#!/bin/bash
chapter=chapter_neural_architectures
section=data_dependency
pytorch_args="--turn_speed 0.8 --speed 0.8 --action_bound 0.9 --normalized_output\
 --max_episodes 10000 --batch_size 32 --clip 1 --scaled_input --optimizer SGD --loss MSE --network res18_net"

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
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((30*60+5*5*60+30*60)) --rammem 7 --gpumem_train 1800 --copy_dataset"
  dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

#######################################
# Pretrain for different learning rates
#######################################

pretrain $chapter/$section/200K/learning_rates --dataset esatv3_expert_200K --weight_decay 0
pretrain $chapter/$section/100K/learning_rates --dataset esatv3_expert_100K --weight_decay 0
pretrain $chapter/$section/50K/learning_rates --dataset esatv3_expert_50K --weight_decay 0
pretrain $chapter/$section/20K/learning_rates --dataset esatv3_expert_20K --weight_decay 0
pretrain $chapter/$section/10K/learning_rates --dataset esatv3_expert_10K --weight_decay 0

pretrain $chapter/$section/200K_pretrained/learning_rates --dataset esatv3_expert_200K --weight_decay 0 --pretrained
pretrain $chapter/$section/100K_pretrained/learning_rates --dataset esatv3_expert_100K --weight_decay 0 --pretrained
pretrain $chapter/$section/50K_pretrained/learning_rates --dataset esatv3_expert_50K --weight_decay 0 --pretrained
pretrain $chapter/$section/20K_pretrained/learning_rates --dataset esatv3_expert_20K --weight_decay 0 --pretrained
pretrain $chapter/$section/10K_pretrained/learning_rates --dataset esatv3_expert_10K --weight_decay 0 --pretrained

pretrain $chapter/$section/200K_wd00001/learning_rates --dataset esatv3_expert_200K --weight_decay 0.0001
pretrain $chapter/$section/100K_wd00001/learning_rates --dataset esatv3_expert_100K --weight_decay 0.0001
pretrain $chapter/$section/50K_wd00001/learning_rates --dataset esatv3_expert_50K --weight_decay 0.0001
pretrain $chapter/$section/20K_wd00001/learning_rates --dataset esatv3_expert_20K --weight_decay 0.0001
pretrain $chapter/$section/10K_wd00001/learning_rates --dataset esatv3_expert_10K --weight_decay 0.0001


#######################################
# Set winning learning rate
#######################################
# train $chapter/$section/200K/final --dataset esatv3_expert_200K --weight_decay 0 --learning_rate ???
# train $chapter/$section/100K/final --dataset esatv3_expert_100K --weight_decay 0 --learning_rate ???
# train $chapter/$section/50K/final --dataset esatv3_expert_50K --weight_decay 0 --learning_rate ???
# train $chapter/$section/20K/final --dataset esatv3_expert_20K --weight_decay 0 --learning_rate ???
# train $chapter/$section/10K/final --dataset esatv3_expert_10K --weight_decay 0 --learning_rate ???

# train $chapter/$section/200K_pretrained/final --dataset esatv3_expert_200K --weight_decay 0 --pretrained --learning_rate ???
# train $chapter/$section/100K_pretrained/final --dataset esatv3_expert_100K --weight_decay 0 --pretrained --learning_rate ???
# train $chapter/$section/50K_pretrained/final --dataset esatv3_expert_50K --weight_decay 0 --pretrained --learning_rate ???
# train $chapter/$section/20K_pretrained/final --dataset esatv3_expert_20K --weight_decay 0 --pretrained --learning_rate ???
# train $chapter/$section/10K_pretrained/final --dataset esatv3_expert_10K --weight_decay 0 --pretrained --learning_rate ???

# train $chapter/$section/200K_wd00001/final --dataset esatv3_expert_200K --weight_decay 0.0001 --learning_rate ???
# train $chapter/$section/100K_wd00001/final --dataset esatv3_expert_100K --weight_decay 0.0001 --learning_rate ???
# train $chapter/$section/50K_wd00001/final --dataset esatv3_expert_50K --weight_decay 0.0001 --learning_rate ???
# train $chapter/$section/20K_wd00001/final --dataset esatv3_expert_20K --weight_decay 0.0001 --learning_rate ???
# train $chapter/$section/10K_wd00001/final --dataset esatv3_expert_10K --weight_decay 0.0001 --learning_rate ???

#######################################
# Combine results
#######################################

# LOGFOLDERS="$(for AR in res18_discrete res18_discrete_stochastic ; do printf " chapter_neural_architectures/output/${AR}/final/0"; done)"
# LEGEND="Discrete Discrete_stochastic"
# python combine_results.py --tags validation_accuracy --title Discrete --log_folders $LOGFOLDERS --legend_names $LEGEND --subsample 3

# LOGFOLDERS="$(for AR in res18_continuous res18_continuous_stochastic_wd001 ; do printf " chapter_neural_architectures/output/${AR}/final/0"; done)"
# LEGEND="Continuous Continuous_stochastic"
# python combine_results.py --tags validation_imitation_learning --title Continuous --log_folders $LOGFOLDERS --legend_names $LEGEND --subsample 3


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"