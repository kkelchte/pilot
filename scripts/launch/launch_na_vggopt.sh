#!/bin/bash
chapter=chapter_neural_architectures
section=optimizers
pytorch_args="--dataset esatv3_expert_200K --discrete --turn_speed 0.8 --speed 0.8 --action_bound 0.9 --normalized_output\
 --max_episodes 10000 --batch_size 32 --loss CrossEntropy --clip 1 --weight_decay 0"

echo "####### chapter: $chapter #######"
echo "####### section: $section #######"

pretrain(){
  cd ..
  condor_args_pretraining="--wall_time $((24*60*60)) --rammem 7 --copy_dataset"
  python dag_train.py $pytorch_args $condor_args_pretraining -t $*
  cd launch
}
train(){
  cd ..
  script_args="--z_pos 1 -w esatv3 --random_seed 512 --number_of_runs 5 --evaluation"
  condor_args="--wall_time_train $((24*60*60)) --wall_time_eva $((2*3600)) --rammem 7 --copy_dataset"
  dag_args="--model_names 0 1 2 --random_numbers 123 456 789"
  python dag_train_and_evaluate.py $pytorch_args $condor_args $dag_args $script_args -t $*
  cd launch
}

#######################################
# Pretrain for different learning rates
#######################################

# pretrain $chapter/$section/alex_SGD_scratch/learning_rates --optimizer SGD --network alex_net  --scaled_input  --gpumem 6000
# pretrain $chapter/$section/vgg16_SGD_scratch/learning_rates --optimizer SGD --network vgg16_net  --scaled_input  --gpumem 6000
# pretrain $chapter/$section/vgg16_Adam_scratch/learning_rates --optimizer Adam --network vgg16_net  --scaled_input  --gpumem 6000
# pretrain $chapter/$section/vgg16_Adadelta_scratch/learning_rates --optimizer Adadelta --network vgg16_net  --scaled_input  --gpumem 6000

# pretrain $chapter/$section/alex_SGD_pretrained/learning_rates --pretrained --optimizer SGD --network alex_net  --scaled_input  --gpumem 1900

#TODO
pretrain $chapter/$section/vgg16_SGD_pretrained/learning_rates --pretrained --optimizer SGD --network vgg16_net  --scaled_input  --gpumem 6000 --python_project pytorch_pilot_beta/pilot
# pretrain $chapter/$section/alex_SGD_pretrained_normalized/learning_rates --pretrained --optimizer SGD --network alex_net  --normalized_input  --gpumem 1900
# pretrain $chapter/$section/vgg16_SGD_pretrained_normalized/learning_rates --pretrained --optimizer SGD --network vgg16_net  --normalized_input  --gpumem 6000

#TODO
# pretrain $chapter/$section/vgg16_Adam_pretrained/learning_rates --pretrained --optimizer Adam --network vgg16_net  --scaled_input  --gpumem 6000
# pretrain $chapter/$section/vgg16_Adadelta_pretrained/learning_rates --pretrained --optimizer Adadelta --network vgg16_net  --scaled_input  --gpumem 6000

#######################################
# Set winning learning rate
#######################################

# train $chapter/$section/alex_SGD_scratch/final --optimizer SGD --network alex_net --learning_rate 0.1  --scaled_input
# train $chapter/$section/vgg16_SGD_scratch/final --optimizer SGD --network vgg16_net --learning_rate  0.1  --scaled_input
# train $chapter/$section/vgg16_Adam_scratch/final --optimizer Adam --network vgg16_net --learning_rate 0.0001  --scaled_input
# train $chapter/$section/vgg16_Adadelta_scratch/final --optimizer Adadelta --network vgg16_net --learning_rate 0.1  --scaled_input

#TODO
# train $chapter/$section/alex_SGD_pretrained/final --pretrained --optimizer SGD --network alex_net  --scaled_input  --gpumem_train 1900 --learning_rate 0.1
# train $chapter/$section/vgg16_SGD_pretrained/final --pretrained --optimizer SGD --network vgg16_net  --scaled_input  --gpumem_train 6000 --learning_rate ...
# train $chapter/$section/alex_SGD_pretrained_normalized/final --pretrained --optimizer SGD --network alex_net --normalized_input  --gpumem_train 1900 --learning_rate ...
# train $chapter/$section/vgg16_SGD_pretrained_normalized/final --pretrained --optimizer SGD --network vgg16_net --normalized_input  --gpumem_train 6000 --learning_rate ...
# # train $chapter/$section/vgg16_Adam_pretrained/final --pretrained --optimizer Adam --network vgg16_net  --scaled_input  --gpumem_train 6000 --learning_rate ...
# # train $chapter/$section/vgg16_Adadelta_pretrained/final --pretrained --optimizer Adadelta --network vgg16_net  --scaled_input  --gpumem_train 6000 --learning_rate ...



sleep 3
condor_q
echo "cd /esat/opal/kkelchte/docker_home/tensorflow/log/$chapter/$section"