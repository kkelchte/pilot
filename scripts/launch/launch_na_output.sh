#!/bin/bash
chapter=chapter_neural_architectures
section=output
pytorch_args="--dataset esatv3_expert_200K --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --normalized_output\
 --max_episodes 10000 --batch_size 32 --clip 1 --scaled_input --optimizer SGD --pretrained"

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

# pretrain $chapter/$section/res18_discrete/learning_rates --network res18_net --discrete --weight_decay 0 --loss CrossEntropy
# pretrain $chapter/$section/res18_discrete_stochastic/learning_rates --network res18_net --discrete --stochastic  --weight_decay 0 --loss CrossEntropy
# pretrain $chapter/$section/res18_discrete_MSE/learning_rates --network res18_net --discrete --weight_decay 0 --loss MSE

# pretrain $chapter/$section/res18_continuous/learning_rates --network res18_net  --weight_decay 0 --loss MSE
# pretrain $chapter/$section/res18_continuous_stochastic/learning_rates --network res18_net --stochastic  --weight_decay 0 --loss MSE
# pretrain $chapter/$section/res18_continuous_stochastic_wd00001/learning_rates --network res18_net --stochastic --weight_decay 0.00001 --loss MSE
# pretrain $chapter/$section/res18_continuous_stochastic_wd0001/learning_rates --network res18_net --stochastic --weight_decay 0.0001 --loss MSE
# pretrain $chapter/$section/res18_continuous_stochastic_wd001/learning_rates --network res18_net --stochastic --weight_decay 0.001 --loss MSE

#######################################
# Set winning learning rate
#######################################

train $chapter/$section/res18_discrete/final --network res18_net --discrete --weight_decay 0 --loss CrossEntropy --learning_rate 0.01
train $chapter/$section/res18_discrete_stochastic/final_001 --network res18_net --discrete --stochastic  --weight_decay 0 --loss CrossEntropy --learning_rate 0.01
# train $chapter/$section/res18_discrete_MSE/final --network res18_net --discrete --weight_decay 0 --loss MSE --learning_rate 0.01

# train $chapter/$section/res18_continuous/final_redo --network res18_net  --weight_decay 0 --loss MSE --learning_rate 0.1
# # train $chapter/$section/res18_continuous_stochastic/final --network res18_net --stochastic  --weight_decay 0 --loss MSE --learning_rate 0.01
# train $chapter/$section/res18_continuous_stochastic/final --network res18_net --stochastic --weight_decay 0.001 --loss MSE --learning_rate 0.01
#MAYBE
# train $chapter/$section/res18_continuous_stochastic_wd00001/final --network res18_net --stochastic --weight_decay 0.00001 --loss MSE --learning_rate 0.01
# train $chapter/$section/res18_continuous_stochastic_wd0001/final --network res18_net --stochastic --weight_decay 0.0001 --loss MSE --learning_rate 0.01

#######################################
# Combine results
#######################################

# LOGFOLDERS="$(for AR in res18_discrete res18_discrete_stochastic ; do printf " chapter_neural_architectures/output_pretrained/${AR}/final/0"; done)"
# LEGEND="Discrete Discrete_stochastic"
# python combine_results.py --headless --tags validation_accuracy --title Discrete --log_folders $LOGFOLDERS --legend_names $LEGEND --subsample 3

# LOGFOLDERS="$(for AR in res18_discrete_MSE res18_continuous res18_continuous_stochastic ; do printf " chapter_neural_architectures/output_pretrained/${AR}/final/0"; done)"
# LEGEND="Discrete Continuous Continuous_stochastic"
# python combine_results.py --headless --tags validation_imitation_learning --title Continuous --log_folders $LOGFOLDERS --legend_names $LEGEND --subsample 3

# LOGFOLDERS="$(for AR in res18_continuous res18_continuous_stochastic ; do printf " chapter_neural_architectures/output_pretrained/${AR}/final/0"; done)"
# LEGEND="Continuous Continuous_stochastic"
# python combine_results.py --headless --tags validation_imitation_learning --title Continuous --log_folders $LOGFOLDERS --legend_names $LEGEND --subsample 3


sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"
