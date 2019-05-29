#!/bin/bash
chapter=chapter_neural_architectures
section=output
pytorch_args="--dataset esatv3_expert_200K --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --normalized_output\
 --tensorboard --max_episodes 10000 --batch_size 32 --loss CrossEntropy --clip 1 --scaled_input --optimizer SGD"

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

pretrain $chapter/$section/res18_discrete/learning_rates --pretrained --network res18_net --discrete --weight_decay 0
pretrain $chapter/$section/res18_discrete_stochastic/learning_rates --pretrained --network res18_net --discrete --stochastic  --weight_decay 0
pretrain $chapter/$section/res18_continuous/learning_rates --pretrained --network res18_net  --weight_decay 0
pretrain $chapter/$section/res18_continuous_stochastic/learning_rates --pretrained --network res18_net --stochastic  --weight_decay 0
pretrain $chapter/$section/res18_continuous_stochastic_wd0001/learning_rates --pretrained --network res18_net --stochastic --weight_decay 0.0001
pretrain $chapter/$section/res18_continuous_stochastic_wd001/learning_rates --pretrained --network res18_net --stochastic --weight_decay 0.001
pretrain $chapter/$section/res18_continuous_stochastic_wd01/learning_rates --pretrained --network res18_net --stochastic --weight_decay 0.01

#######################################
# Set winning learning rate
#######################################

# train $chapter/$section/alex_SGD_scratch/final --optimizer SGD --network alex_net --learning_rate 0.1
# train $chapter/$section/vgg16_SGD_scratch/final --optimizer SGD --network vgg16_net --learning_rate  0.1
# train $chapter/$section/vgg16_Adam_scratch/final --optimizer Adam --network vgg16_net --learning_rate 0.0001
# train $chapter/$section/vgg16_Adadelta_scratch/final --optimizer Adadelta --network vgg16_net --learning_rate 0.1
# train $chapter/$section/alex_SGD_pretrained/final --pretraining --optimizer SGD --network alex_net --learning_rate 0.1
# train $chapter/$section/vgg16_SGD_pretrained/final --pretraining --optimizer SGD --network vgg16_net --learning_rate 0.1
# train $chapter/$section/vgg16_Adam_pretrained/final --pretraining --optimizer Adam --network vgg16_net --learning_rate 0.00001
# train $chapter/$section/vgg16_Adadelta_pretrained/final --pretraining --optimizer Adadelta --network vgg16_net --learning_rate 0.1

sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"